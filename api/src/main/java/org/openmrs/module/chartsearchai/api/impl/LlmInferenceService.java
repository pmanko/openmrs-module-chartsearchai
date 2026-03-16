/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.api.impl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.util.Collections;
import java.util.Comparator;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

/**
 * Answers natural language questions about a patient's chart using direct LLM inference.
 * Uses embedding similarity to pre-filter records to the most relevant ones, then sends
 * those to the LLM for reasoning. Falls back to the full chart if no embeddings exist.
 */
@Service("chartSearchAi.llmInferenceService")
public class LlmInferenceService implements ChartSearchService {

	private static final Logger log = LoggerFactory.getLogger(LlmInferenceService.class);

	@Autowired
	private PatientRecordLoader recordLoader;

	@Autowired
	private PatientChartSerializer chartSerializer;

	@Autowired
	@Qualifier("chartSearchAi.embeddingProvider")
	private EmbeddingProvider embeddingProvider;

	@Autowired
	private ChartSearchAiDAO dao;

	@Autowired
	private LlmProvider llmProvider;

	private static final Pattern CITATION_PATTERN = Pattern.compile("\\[(\\d+(?:,\\s*\\d+)*)\\]");

	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = buildChart(patient, question);

		String response = llmProvider.search(chart.getText(), question);

		return new ChartAnswer(response, extractCitedReferences(response, chart.getMappings()));
	}

	@Override
	public ChartAnswer searchStreaming(Patient patient, String question,
			Consumer<String> tokenConsumer) {
		PatientChart chart = buildChart(patient, question);

		String response = llmProvider.searchStreaming(chart.getText(), question, tokenConsumer);

		return new ChartAnswer(response, extractCitedReferences(response, chart.getMappings()));
	}

	private PatientChart buildChart(Patient patient, String question) {
		if (!usePreFilter()) {
			return chartSerializer.serialize(patient);
		}

		List<ChartEmbedding> similar = findSimilar(patient, question,
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		if (similar.isEmpty()) {
			log.debug("No embeddings found, falling back to full chart");
			return chartSerializer.serialize(patient);
		}

		Set<String> relevantKeys = new HashSet<String>();
		for (ChartEmbedding ce : similar) {
			relevantKeys.add(ce.getResourceType() + ":" + ce.getResourceId());
		}

		List<SerializedRecord> allRecords = recordLoader.loadAll(patient);
		List<SerializedRecord> filtered = new ArrayList<SerializedRecord>();
		for (SerializedRecord record : allRecords) {
			if (relevantKeys.contains(record.getResourceType() + ":" + record.getResourceId())) {
				filtered.add(record);
			}
		}

		log.debug("Pre-filtered {} records to {} using embeddings", allRecords.size(), filtered.size());

		return chartSerializer.serialize(filtered);
	}

	private boolean usePreFilter() {
		String mode = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_PRE_FILTER);
		return !"false".equalsIgnoreCase(mode != null ? mode.trim() : "");
	}

	private List<ChartEmbedding> findSimilar(Patient patient, String question, int topK) {
		float[] queryVector;
		try {
			queryVector = embeddingProvider.embed(question);
		}
		catch (Exception e) {
			log.debug("Embedding provider not available, falling back to full chart", e);
			return Collections.emptyList();
		}

		List<ChartEmbedding> allEmbeddings = dao.getByPatient(patient);
		if (allEmbeddings.isEmpty()) {
			return Collections.emptyList();
		}

		List<ScoredEmbedding> scored = new ArrayList<ScoredEmbedding>();
		for (ChartEmbedding ce : allEmbeddings) {
			float[] vector = ce.getEmbeddingVector();
			if (vector.length != queryVector.length) {
				continue;
			}
			double dot = 0, normA = 0, normB = 0;
			for (int i = 0; i < queryVector.length; i++) {
				dot += queryVector[i] * vector[i];
				normA += queryVector[i] * queryVector[i];
				normB += vector[i] * vector[i];
			}
			double denom = Math.sqrt(normA) * Math.sqrt(normB);
			double similarity = denom == 0 ? 0 : dot / denom;
			scored.add(new ScoredEmbedding(ce, similarity));
		}

		Collections.sort(scored, new Comparator<ScoredEmbedding>() {
			@Override
			public int compare(ScoredEmbedding a, ScoredEmbedding b) {
				return Double.compare(b.score, a.score);
			}
		});

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		int limit = Math.min(topK, scored.size());
		for (int i = 0; i < limit; i++) {
			results.add(scored.get(i).embedding);
		}
		return results;
	}

	private static class ScoredEmbedding {

		final ChartEmbedding embedding;

		final double score;

		ScoredEmbedding(ChartEmbedding embedding, double score) {
			this.embedding = embedding;
			this.score = score;
		}
	}

	static List<RecordReference> extractCitedReferences(String answer, List<RecordMapping> mappings) {
		Map<Integer, RecordMapping> indexMap = new HashMap<Integer, RecordMapping>();
		for (RecordMapping mapping : mappings) {
			indexMap.put(mapping.getIndex(), mapping);
		}

		Set<Integer> seen = new LinkedHashSet<Integer>();
		Matcher matcher = CITATION_PATTERN.matcher(answer);
		while (matcher.find()) {
			for (String part : matcher.group(1).split(",")) {
				String trimmed = part.trim();
				try {
					seen.add(Integer.valueOf(trimmed));
				}
				catch (NumberFormatException e) {
					// skip non-numeric parts
				}
			}
		}

		List<RecordReference> references = new ArrayList<RecordReference>();
		for (Integer index : seen) {
			RecordMapping mapping = indexMap.get(index);
			if (mapping != null) {
				references.add(new RecordReference(index, mapping.getResourceType(), mapping.getResourceId()));
			}
		}
		return references;
	}
}
