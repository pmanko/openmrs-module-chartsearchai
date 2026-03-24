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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.function.Consumer;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.openmrs.module.chartsearchai.api.impl.LlmProvider.LlmResponse;
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
	private EmbeddingIndexer embeddingIndexer;

	@Autowired
	private LlmProvider llmProvider;

	private static final String NO_RECORDS_ANSWER = "No clinical records found for this patient.";

	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = buildChart(patient, question);

		if (chart.getText().isEmpty()) {
			return new ChartAnswer(NO_RECORDS_ANSWER, Collections.emptyList());
		}

		LlmResponse response = llmProvider.search(chart.getText(), question);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()));
	}

	@Override
	public ChartAnswer searchStreaming(Patient patient, String question,
			Consumer<String> tokenConsumer) {
		PatientChart chart = buildChart(patient, question);

		if (chart.getText().isEmpty()) {
			tokenConsumer.accept(NO_RECORDS_ANSWER);
			return new ChartAnswer(NO_RECORDS_ANSWER, Collections.emptyList());
		}

		LlmResponse response = llmProvider.searchStreaming(chart.getText(), question, tokenConsumer);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()));
	}

	private PatientChart buildChart(Patient patient, String question) {
		if (!usePreFilter()) {
			return chartSerializer.serialize(patient);
		}

		int topK = getTopK();
		// findSimilar returns null when no embeddings exist (needs indexing),
		// or an empty list when embeddings exist but nothing matched the query.
		List<ChartEmbedding> similar = findSimilar(patient, question, topK);

		if (similar == null) {
			log.info("No embeddings found for patient [id={}], indexing now", patient.getPatientId());
			try {
				embeddingIndexer.indexPatient(patient);
				similar = findSimilar(patient, question, topK);
			}
			catch (Exception e) {
				log.error("Failed to index patient [id={}], falling back to full chart",
						patient.getPatientId(), e);
			}
		}

		if (similar == null) {
			log.debug("Still no embeddings after indexing attempt, falling back to full chart");
			return chartSerializer.serialize(patient);
		}

		if (similar.isEmpty()) {
			log.debug("No records matched the query, returning empty chart");
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
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

		return chartSerializer.serialize(patient, filtered);
	}

	private boolean usePreFilter() {
		String mode = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER);
		return !"false".equalsIgnoreCase(mode != null ? mode.trim() : "");
	}

	private int getTopK() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_TOP_K);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed > 0) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid topK value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K;
	}

	private double getSimilarityRatio() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_SIMILARITY_RATIO);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed > 0 && parsed < 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid similarityRatio value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO;
	}

	private static final Set<String> QUERY_STOPWORDS = loadStopwords("query-stopwords.txt");

	private static Set<String> loadStopwords(String fileName) {
		// Try the OpenMRS application data directory first so admins can customize
		// without recompiling. Fall back to the bundled resource.
		InputStream is = null;
		boolean fromFile = false;
		try {
			File appDataFile = new File(
					org.openmrs.util.OpenmrsUtil.getApplicationDataDirectory(),
					"chartsearchai" + File.separator + fileName);
			if (appDataFile.exists()) {
				is = new FileInputStream(appDataFile);
				fromFile = true;
				log.info("Loading stopwords from {}", appDataFile.getAbsolutePath());
			}
		}
		catch (Exception e) {
			log.debug("Could not load stopwords from application data directory: {}", e.getMessage());
		}

		if (is == null) {
			is = LlmInferenceService.class.getClassLoader().getResourceAsStream(fileName);
			if (is == null) {
				log.warn("Stopwords resource not found: {}, query normalization will be disabled", fileName);
				return Collections.emptySet();
			}
		}

		Set<String> words = new HashSet<String>();
		try (InputStream stream = is) {
			BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
			String line;
			while ((line = reader.readLine()) != null) {
				String trimmed = line.trim();
				if (!trimmed.isEmpty() && !trimmed.startsWith("#")) {
					words.add(trimmed.toLowerCase());
				}
			}
		}
		catch (IOException e) {
			log.warn("Failed to load stopwords from {}: {}", fileName, e.getMessage());
		}

		if (fromFile) {
			log.info("Loaded {} stopwords from application data directory", words.size());
		}
		return Collections.unmodifiableSet(words);
	}

	/**
	 * Removes common stopwords before embedding so that queries like
	 * "any medications?" and "does the patient have any medications?"
	 * produce the same embedding vector and thus the same retrieval results.
	 *
	 * @param question the raw user question
	 */
	static String stripQueryStopwords(String question) {
		String[] words = question.toLowerCase().replaceAll("[?!.,;:]", "").trim().split("\\s+");
		StringBuilder sb = new StringBuilder();
		for (String word : words) {
			if (!word.isEmpty() && !QUERY_STOPWORDS.contains(word)) {
				if (sb.length() > 0) {
					sb.append(" ");
				}
				sb.append(word);
			}
		}
		return sb.length() > 0 ? sb.toString() : question.toLowerCase().trim();
	}

	private List<ChartEmbedding> findSimilar(Patient patient, String question, int topK) {
		String normalizedQuery = stripQueryStopwords(question);
		log.debug("Normalized query for embedding: '{}' -> '{}'", question, normalizedQuery);
		float[] queryVector;
		try {
			queryVector = embeddingProvider.embed(normalizedQuery);
		}
		catch (Exception e) {
			log.debug("Embedding provider not available, falling back to full chart", e);
			return null;
		}

		List<ChartEmbedding> allEmbeddings = dao.getByPatient(patient);
		if (allEmbeddings.isEmpty()) {
			return null;
		}

		List<ScoredEmbedding> scored = new ArrayList<ScoredEmbedding>();
		for (ChartEmbedding ce : allEmbeddings) {
			float[] vector = ce.getEmbeddingVector();
			if (vector.length != queryVector.length) {
				log.warn("Skipping embedding [id={}] with mismatched dimensions ({} vs expected {})",
						ce.getEmbeddingId(), vector.length, queryVector.length);
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

		if (scored.isEmpty()) {
			return Collections.emptyList();
		}

		double topScore = scored.get(0).score;
		// If nothing scores above the absolute floor, the query is unrelated
		// to any record in the chart (e.g. "any teacher?" against clinical data).
		if (topScore < ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR) {
			log.debug("Top score {} is below absolute floor {}, returning empty",
					String.format("%.4f", topScore),
					ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR);
			return Collections.emptyList();
		}

		int limit = Math.min(topK, scored.size());
		double similarityRatio = getSimilarityRatio();
		double ratioFloor = topScore * similarityRatio;
		double minScore;
		// When the top score is high (> 0.50), the query matched a specific
		// record strongly and the ratio-based floor is reliable. When the top
		// score is moderate, queries are fuzzier and OOV vocabulary can depress
		// relevant records — the range-based floor provides needed leniency.
		if (topScore > 0.50) {
			minScore = ratioFloor;
		} else {
			double scoreRange = topScore - scored.get(limit - 1).score;
			double rangeFloor = topScore - similarityRatio * scoreRange;
			minScore = Math.min(ratioFloor, rangeFloor);
		}

		// Adaptive cutoff: find the natural cluster boundary using score gap
		// detection. When the drop between consecutive scores exceeds a
		// multiplier of the running average gap, that signals the end of the
		// relevant cluster. The similarity ratio threshold and topK ceiling
		// still apply as hard limits.
		int adaptiveCutoff = findAdaptiveCutoff(scored, limit, minScore, getScoreGapMultiplier());

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		for (int i = 0; i < adaptiveCutoff; i++) {
			results.add(scored.get(i).embedding);
		}

		StringBuilder scores = new StringBuilder();
		for (int i = 0; i < limit; i++) {
			if (i > 0) {
				scores.append(", ");
			}
			ScoredEmbedding se = scored.get(i);
			scores.append(se.embedding.getResourceType())
					.append(":").append(se.embedding.getResourceId())
					.append("=").append(String.format("%.4f", se.score));
		}
		log.debug("Similarity scores: [{}], threshold: {}, adaptiveCutoff: {}", scores,
				String.format("%.4f", minScore), adaptiveCutoff);
		log.debug("Returning {} of {} candidates (topScore={}, minScore={}, max={})",
				results.size(), scored.size(),
				String.format("%.4f", topScore), String.format("%.4f", minScore), topK);
		return results;
	}

	/**
	 * Finds the adaptive cutoff point in a sorted list of scored embeddings by
	 * detecting a significant gap in similarity scores. Walks the sorted scores
	 * and tracks the running average gap between consecutive entries. When a gap
	 * exceeds the average by more than the configured multiplier, the cluster
	 * boundary is found. Always includes at least
	 * {@link ChartSearchAiConstants#ADAPTIVE_MIN_RECORDS} records (if available
	 * above the similarity floor) so the LLM has enough context.
	 *
	 * @param scored sorted list of scored embeddings (highest score first)
	 * @param limit the maximum number of candidates to consider (topK cap)
	 * @param minScore the absolute similarity floor from the ratio threshold
	 * @param gapMultiplier how many times larger than the average gap a score drop
	 *        must be to trigger a cutoff
	 * @return the number of records to include in the primary cluster
	 */
	static int findAdaptiveCutoff(List<ScoredEmbedding> scored, int limit, double minScore,
			double gapMultiplier) {
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;

		// Count how many records pass the similarity floor
		int aboveFloor = 0;
		for (int i = 0; i < limit; i++) {
			if (scored.get(i).score >= minScore) {
				aboveFloor++;
			} else {
				break;
			}
		}

		if (aboveFloor == 0) {
			return 0;
		}

		// Walk through consecutive scores tracking the running average gap.
		// Once a gap exceeds gapMultiplier * avgGap, cut there. All prior
		// gaps feed the running average so that the baseline reflects the
		// full score distribution, which prevents degenerate seeds (e.g. a
		// single tied-score gap of 0.0) from making the detector overly
		// trigger-happy.
		double gapSum = 0;
		int cutoff = aboveFloor; // default: include everything above the floor
		for (int i = 1; i < aboveFloor; i++) {
			double gap = scored.get(i - 1).score - scored.get(i).score;

			// Only consider cutting after the minimum number of records, and
			// only when we have at least 2 prior gaps to form a meaningful
			// average (i - 1 gaps have been accumulated at this point).
			if (i >= minRecords && i >= 2) {
				double avgGap = gapSum / (i - 1);
				if (gap > avgGap * gapMultiplier) {
					cutoff = i;
					log.debug("Score gap detected at position {}: gap={}, avgGap={}, multiplier={}",
							i, String.format("%.4f", gap), String.format("%.4f", avgGap),
							gapMultiplier);
					break;
				}
			}
			gapSum += gap;
		}

		return cutoff;
	}

	private static double getScoreGapMultiplier() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_SCORE_GAP_MULTIPLIER);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed > 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid scoreGapMultiplier value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER;
	}

	static class ScoredEmbedding {

		final ChartEmbedding embedding;

		final double score;

		ScoredEmbedding(ChartEmbedding embedding, double score) {
			this.embedding = embedding;
			this.score = score;
		}
	}

	static List<RecordReference> extractCitedReferences(List<Integer> citations,
			List<RecordMapping> mappings) {
		Map<Integer, RecordMapping> indexMap = new HashMap<Integer, RecordMapping>();
		for (RecordMapping mapping : mappings) {
			indexMap.put(mapping.getIndex(), mapping);
		}

		Set<Integer> seen = new LinkedHashSet<Integer>();
		for (Integer index : citations) {
			seen.add(index);
		}

		List<RecordReference> references = new ArrayList<RecordReference>();
		for (Integer index : seen) {
			RecordMapping mapping = indexMap.get(index);
			if (mapping != null) {
				references.add(new RecordReference(index, mapping.getResourceType(),
						mapping.getResourceId(), mapping.getDate()));
			} else {
				log.warn("LLM cited record [{}] which does not exist in the provided records", index);
			}
		}
		Collections.sort(references, Comparator.comparing(RecordReference::getDate,
				Comparator.nullsLast(Comparator.reverseOrder())));
		return references;
	}
}
