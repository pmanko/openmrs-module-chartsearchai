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
			queryVector = embeddingProvider.embed(getQueryPrefix() + normalizedQuery);
		}
		catch (Exception e) {
			log.debug("Embedding provider not available, falling back to full chart", e);
			return null;
		}

		List<ChartEmbedding> allEmbeddings = dao.getByPatient(patient);
		if (allEmbeddings.isEmpty()) {
			return null;
		}

		String[] queryTerms = extractQueryTerms(normalizedQuery);
		double keywordWeight = getKeywordWeight();

		double maxBaseScore = 0;

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
			double semanticScore = denom == 0 ? 0 : dot / denom;

			// Additive bonus: keyword overlap can only increase the score,
			// never decrease it. A zero keyword match leaves the semantic
			// score unchanged, avoiding the trap where a weighted average
			// would suppress scores below the absolute similarity floor.
			double keywordScore = computeKeywordScore(queryTerms, ce.getTextContent());
			double baseScore = semanticScore + keywordWeight * keywordScore;

			if (baseScore > maxBaseScore) {
				maxBaseScore = baseScore;
			}

			scored.add(new ScoredEmbedding(ce, baseScore));
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

		if (maxBaseScore < ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR) {
			log.debug("Top score {} is below absolute floor {}, returning empty",
					String.format("%.4f", maxBaseScore),
					ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR);
			return Collections.emptyList();
		}

		// Permissive floor: gap detection handles the real cutoff based on
		// score distribution. The floor just excludes near-zero noise so
		// the gap detector has a clean signal.
		double minScore = ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR / 2;

		// Gap detection considers ALL records above the floor — no topK cap
		// on the search window. The embedding model naturally clusters
		// records by relevance: for specific queries ("HB results"), the
		// relevant records score high and the gap appears early. For broad
		// queries ("list all medications"), all type-matched records cluster
		// together and the gap appears after the entire cluster. The score
		// distribution itself determines how many records to return.
		int adaptiveCutoff = findAdaptiveCutoff(scored, scored.size(),
				minScore, getScoreGapMultiplier(), getMinScoreGap());

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		for (int i = 0; i < adaptiveCutoff; i++) {
			results.add(scored.get(i).embedding);
		}

		int logLimit = Math.min(topK, scored.size());
		StringBuilder scores = new StringBuilder();
		for (int i = 0; i < logLimit; i++) {
			if (i > 0) {
				scores.append(", ");
			}
			ScoredEmbedding se = scored.get(i);
			scores.append(se.embedding.getResourceType())
					.append(":").append(se.embedding.getResourceId())
					.append("=").append(String.format("%.4f", se.score));
		}
		log.debug("Similarity scores: [{}], minScore: {}, adaptiveCutoff: {}",
				scores, String.format("%.4f", minScore), adaptiveCutoff);
		log.debug("Returning {} of {} candidates (topScore={})",
				results.size(), scored.size(),
				String.format("%.4f", maxBaseScore));
		return results;
	}

	/**
	 * Finds the adaptive cutoff point in a sorted list of scored embeddings by
	 * detecting a significant gap in similarity scores. Walks the sorted scores
	 * and tracks the running average gap between consecutive entries. When a gap
	 * exceeds the average by more than the configured multiplier AND exceeds the
	 * minimum absolute gap, the cluster boundary is found. The absolute minimum
	 * prevents premature cutting on small gaps that only appear large relative
	 * to a tight cluster. Always includes at least
	 * {@link ChartSearchAiConstants#ADAPTIVE_MIN_RECORDS} records (if available
	 * above the similarity floor) so the LLM has enough context.
	 *
	 * @param scored sorted list of scored embeddings (highest score first)
	 * @param limit the maximum number of candidates to consider
	 * @param minScore the absolute similarity floor
	 * @param gapMultiplier how many times larger than the average gap a score drop
	 *        must be to trigger a cutoff
	 * @param minGap the minimum absolute gap required to trigger a cutoff,
	 *        regardless of the multiplier condition
	 * @return the number of records to include in the primary cluster
	 */
	static int findAdaptiveCutoff(List<ScoredEmbedding> scored, int limit, double minScore,
			double gapMultiplier, double minGap) {
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
				if (gap > avgGap * gapMultiplier && gap > minGap) {
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

	private static double getMinScoreGap() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_MIN_SCORE_GAP);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid minScoreGap value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP;
	}

	private static double getKeywordWeight() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_KEYWORD_WEIGHT);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid keywordWeight value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT;
	}

	private static String getQueryPrefix() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_QUERY_PREFIX);
		if (value != null && !value.trim().isEmpty()) {
			return value;
		}
		return ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX;
	}

	/**
	 * Extracts content terms from the normalized query for keyword matching.
	 * Returns lowercased terms with length >= 2 (single-letter terms are too
	 * ambiguous to be useful for keyword overlap scoring).
	 */
	static String[] extractQueryTerms(String normalizedQuery) {
		String[] allTerms = normalizedQuery.toLowerCase().split("\\s+");
		List<String> terms = new ArrayList<String>();
		for (String term : allTerms) {
			if (term.length() >= 2) {
				terms.add(term);
			}
		}
		return terms.toArray(new String[0]);
	}

	/**
	 * Computes a keyword overlap score between query terms and record text.
	 * Returns a value between 0.0 (no terms match) and 1.0 (all terms match).
	 * Uses case-insensitive substring matching so that a query term "metformin"
	 * matches "Drug order: Metformin 500mg". Also tries a simple plural stem
	 * (stripping trailing 's') so that "conditions" matches "Condition: ...".
	 *
	 * @param queryTerms the extracted query terms (lowercased)
	 * @param textContent the full serialized record text
	 * @return the fraction of query terms found in the text
	 */
	static double computeKeywordScore(String[] queryTerms, String textContent) {
		if (queryTerms.length == 0 || textContent == null || textContent.isEmpty()) {
			return 0.0;
		}
		String lowerText = textContent.toLowerCase();
		int matched = 0;
		for (String term : queryTerms) {
			if (lowerText.contains(term)) {
				matched++;
			} else if (term.length() > 3 && term.endsWith("s") && !term.endsWith("ss")) {
				// Simple plural stem: "conditions" → "condition",
				// "medications" → "medication", "tests" → "test".
				// Avoids stemming words ending in "ss" like "less", "pass".
				if (lowerText.contains(term.substring(0, term.length() - 1))) {
					matched++;
				}
			}
		}
		return (double) matched / queryTerms.length;
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
