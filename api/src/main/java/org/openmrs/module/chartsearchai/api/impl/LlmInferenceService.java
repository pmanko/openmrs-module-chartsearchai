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
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.LuceneIndexer;
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
	private LuceneIndexer luceneIndexer;

	@Autowired
	private LlmProvider llmProvider;

	private static final String NO_RECORDS_ANSWER = "No clinical records found for this patient.";

	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = buildChart(patient, question);

		if (chart.getText().isEmpty()) {
			String answer = buildNoMatchAnswer(question);
			return new ChartAnswer(answer, Collections.emptyList());
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
			String answer = buildNoMatchAnswer(question);
			tokenConsumer.accept(answer);
			return new ChartAnswer(answer, Collections.emptyList());
		}

		LlmResponse response = llmProvider.searchStreaming(chart.getText(), question, tokenConsumer);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()));
	}

	/**
	 * Builds a specific "no match" answer that names what was asked about,
	 * rather than returning a generic "no records found" message. This
	 * avoids an unnecessary LLM round-trip and gives the clinician a
	 * clear signal about what data is absent.
	 *
	 * @param question the original user question
	 * @return a human-readable answer naming the missing data
	 */
	static String buildNoMatchAnswer(String question) {
		String terms = stripQueryStopwords(question);
		if (terms.isEmpty()) {
			return NO_RECORDS_ANSWER;
		}
		return "There are no records about " + terms + " in this patient's chart.";
	}

	private static final Pattern RECENCY_PATTERN = Pattern.compile(
			"(?:last|past|previous|recent|most recent)\\s+(\\d+)", Pattern.CASE_INSENSITIVE);

	/**
	 * Extracts a numeric recency constraint from the question, e.g. "last 7
	 * visits" returns 7. Returns 0 if no constraint is found.
	 *
	 * @param question the raw user question
	 * @return the recency cap, or 0 if none detected
	 */
	static int extractRecencyCap(String question) {
		Matcher m = RECENCY_PATTERN.matcher(question);
		if (m.find()) {
			try {
				int n = Integer.parseInt(m.group(1));
				return n > 0 ? n : 0;
			}
			catch (NumberFormatException e) {
				return 0;
			}
		}
		return 0;
	}

	/**
	 * Caps the number of records per concept to {@code maxPerConcept}.
	 * Groups records by their concept key (the text stripped of its
	 * trailing numeric value, e.g. "Clinical observation: Test — Weight
	 * (kg): 94.0" becomes "Clinical observation: Test — Weight (kg)").
	 * Since the input list is already sorted most-recent-first, the first
	 * N records per group are the most recent.
	 *
	 * <p>Records whose text does not end with a numeric value (e.g.
	 * conditions, allergies) are treated as unique groups and always
	 * kept — the recency cap only limits repeated measurements.
	 *
	 * @param records the filtered records, sorted most-recent-first
	 * @param maxPerConcept maximum records to keep per concept group
	 * @return the capped list, preserving the original sort order
	 */
	static List<SerializedRecord> capPerConcept(List<SerializedRecord> records,
			int maxPerConcept) {
		Map<String, Integer> groupCounts = new HashMap<String, Integer>();
		List<SerializedRecord> capped = new ArrayList<SerializedRecord>();
		for (SerializedRecord record : records) {
			String key = conceptKey(record.getText());
			int count = groupCounts.getOrDefault(key, 0);
			if (count < maxPerConcept) {
				capped.add(record);
				groupCounts.put(key, count + 1);
			}
		}
		return capped;
	}

	/**
	 * Extracts a concept grouping key from record text by stripping the
	 * trailing numeric value. For example:
	 * <ul>
	 * <li>"Clinical observation: Test — Weight (kg): 94.0"
	 *     → "Clinical observation: Test — Weight (kg)"</li>
	 * <li>"Clinical observation: Test — Systolic Blood Pressure: 137.0"
	 *     → "Clinical observation: Test — Systolic Blood Pressure"</li>
	 * </ul>
	 * If the text does not end with a numeric value, the full text is
	 * returned, making each such record its own group.
	 */
	static String conceptKey(String text) {
		if (text == null) {
			return "";
		}
		// Strip trailing numeric value like ": 94.0" or ": 137.0"
		return text.replaceAll(":\\s*[\\d.]+\\s*$", "").trim();
	}

	private PatientChart buildChart(Patient patient, String question) {
		if (!usePreFilter()) {
			return chartSerializer.serialize(patient);
		}

		if (isLucenePipeline()) {
			return buildChartWithLucene(patient, question);
		}
		return buildChartWithEmbeddings(patient, question);
	}

	private PatientChart buildChartWithEmbeddings(Patient patient, String question) {
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

		return filterAndSerialize(patient, question, relevantKeys);
	}

	private PatientChart buildChartWithLucene(Patient patient, String question) {
		if (!luceneIndexer.hasIndex(patient)) {
			log.info("No Lucene index for patient [id={}], indexing now", patient.getPatientId());
			try {
				luceneIndexer.indexPatient(patient);
			}
			catch (Exception e) {
				log.error("Failed to Lucene-index patient [id={}], falling back to full chart",
						patient.getPatientId(), e);
				return chartSerializer.serialize(patient);
			}
		}

		String normalizedQuery = stripQueryStopwords(question);
		if (normalizedQuery.isEmpty()) {
			return chartSerializer.serialize(patient);
		}

		List<LuceneIndexer.LuceneSearchResult> results = luceneIndexer.search(
				patient, normalizedQuery, getTopK() * 10);

		if (results.isEmpty()) {
			log.debug("Lucene returned no results for query '{}', returning empty chart",
					normalizedQuery);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		Set<String> relevantKeys = new HashSet<String>();
		for (LuceneIndexer.LuceneSearchResult result : results) {
			relevantKeys.add(result.getResourceType() + ":" + result.getResourceId());
		}

		log.debug("Lucene returned {} results for query '{}'",
				relevantKeys.size(), normalizedQuery);

		return filterAndSerialize(patient, question, relevantKeys);
	}

	private PatientChart filterAndSerialize(Patient patient, String question,
			Set<String> relevantKeys) {
		List<SerializedRecord> allRecords = recordLoader.loadAll(patient);
		List<SerializedRecord> filtered = new ArrayList<SerializedRecord>();
		for (SerializedRecord record : allRecords) {
			if (relevantKeys.contains(record.getResourceType() + ":" + record.getResourceId())) {
				filtered.add(record);
			}
		}

		log.debug("Pre-filtered {} records to {} using {}",
				allRecords.size(), filtered.size(),
				isLucenePipeline() ? "Lucene" : "embeddings");

		int recencyCap = extractRecencyCap(question);
		if (recencyCap > 0) {
			filtered = capPerConcept(filtered, recencyCap);
			log.debug("Recency cap {} applied, {} records remain", recencyCap, filtered.size());
		}

		return chartSerializer.serialize(patient, filtered);
	}

	boolean isLucenePipeline() {
		String pipeline = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE);
		return ChartSearchAiConstants.PIPELINE_LUCENE.equalsIgnoreCase(
				pipeline != null ? pipeline.trim() : "");
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
		String[] words = question.toLowerCase().replaceAll("'s\\b", "").replaceAll("[?!.,;:']", "").trim().split("\\s+");
		List<String> contentWords = new ArrayList<String>();
		List<String> allClean = new ArrayList<String>();
		for (String word : words) {
			if (!word.isEmpty()) {
				allClean.add(word);
				if (!QUERY_STOPWORDS.contains(word)) {
					contentWords.add(word);
				}
			}
		}
		if (contentWords.size() >= 2) {
			StringBuilder sb = new StringBuilder();
			for (String w : contentWords) {
				if (sb.length() > 0) {
					sb.append(" ");
				}
				sb.append(w);
			}
			return sb.toString();
		}
		// Too few content words — preserve all cleaned words so the
		// embedding model gets enough context. The full sentence
		// "does the patient have cancer" produces a more specific
		// embedding than the single word "cancer", helping the model
		// differentiate cancer-related records from unrelated ones.
		if (!allClean.isEmpty()) {
			StringBuilder sb = new StringBuilder();
			for (String w : allClean) {
				if (sb.length() > 0) {
					sb.append(" ");
				}
				sb.append(w);
			}
			return sb.toString();
		}
		return question.toLowerCase().trim();
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
		double maxSemanticScore = 0;

		// For short queries (N≤3), require ≥2 term matches for bonus so that
		// single coincidental matches (e.g. "history" in "immunization history"
		// for "history of cancer") are penalized. For longer multi-concept
		// queries (N≥4), matching ANY term is legitimate — the query spans
		// multiple concepts and each record type may only match its own concept.
		double bonusThreshold = queryTerms.length >= 4
				? 1.0 / queryTerms.length
				: queryTerms.length == 0 ? 1.0
				: (double) Math.min(2, queryTerms.length) / queryTerms.length;
		List<ScoredEmbedding> scored = new ArrayList<ScoredEmbedding>();
		for (ChartEmbedding ce : allEmbeddings) {
			float[] vector = ce.getEmbeddingVector();
			if (vector.length != queryVector.length) {
				log.warn("Skipping embedding [id={}] with mismatched dimensions ({} vs expected {})",
						ce.getEmbeddingId(), vector.length, queryVector.length);
				continue;
			}
			double semanticScore = cosineSimilarity(queryVector, vector);

			// Additive keyword bonus increases the score when enough terms
			// match. For N≤2 queries, a partial keyword match (below the
			// bonus threshold) applies a penalty instead, suppressing
			// coincidental single-word overlaps like "history" in
			// "immunization history" for "history of cancer?".
			// Include the embedding prefix in keyword matching so that
			// type-specific terms (e.g. "medication") match the prefix
			// "Medication prescription: ..." on drug orders.
			String keywordText = ChartSearchAiConstants.getEmbeddingPrefix(
					ce.getResourceType(), ce.getTextContent()) + ce.getTextContent();
			double keywordScore = computeKeywordScore(queryTerms, keywordText);
			// Only apply keyword bonus when enough terms match. In multi-word
			// queries, a single coincidental keyword match (e.g. "history" in
			// "immunization history" for "history of cancer") should not inflate
			// the ranking score. Require ≥2 term matches for multi-word queries
			// (N≤3), or any match for multi-concept queries (N≥4).
			double keywordBonus = keywordScore >= bonusThreshold ? keywordScore : 0.0;
			// Partial keyword penalty for short queries (N≤2): when a record
			// matches some keywords but not enough for bonus, penalize it.
			// This suppresses false positives like "Immunization history"
			// matching "history" in "history of cancer?". For longer queries
			// (N≥3), partial matches are common and legitimate (e.g. "disease"
			// in "sexually transmitted disease?" matching HIV Disease), so
			// penalty is not applied.
			double keywordPenalty = 0.0;
			if (queryTerms.length <= 2 && keywordScore > 0 && keywordScore < bonusThreshold) {
				keywordPenalty = keywordScore;
			}
			double baseScore = semanticScore + keywordWeight * keywordBonus
					- keywordWeight * keywordPenalty;

			if (semanticScore > maxSemanticScore) {
				maxSemanticScore = semanticScore;
			}
			if (baseScore > maxBaseScore) {
				maxBaseScore = baseScore;
			}

			scored.add(new ScoredEmbedding(ce, baseScore, keywordScore, semanticScore));
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

		if (maxSemanticScore < ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR) {
			log.debug("Top semantic score {} is below absolute floor {}, returning empty",
					String.format("%.4f", maxSemanticScore),
					ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR);
			return Collections.emptyList();
		}

		// Stricter gate when too few records match any query keyword:
		// check whether the top semantic score is a statistical outlier
		// (z-score ≥ threshold) rather than part of the noise floor.
		// This automatically adapts to any embedding model and dataset
		// — no model-specific magic numbers. A z-score of 2.0 means the
		// best match is in the top 2.3% of the score distribution.
		// Requires at least 30 records for the z-score to be statistically
		// meaningful; with fewer records, the pipeline's other stages
		// (gap detection, ratio floor, topK) provide sufficient filtering.
		if (queryTerms.length > 0
				&& scored.size() >= ChartSearchAiConstants.MIN_RECORDS_FOR_Z_SCORE) {
			int keywordMatchCount = 0;
			for (ScoredEmbedding se : scored) {
				if (se.keywordScore > 0) {
					keywordMatchCount++;
				}
			}
			if (keywordMatchCount < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				double sumSem = 0;
				for (ScoredEmbedding se : scored) {
					sumSem += se.semanticScore;
				}
				double meanSem = sumSem / scored.size();
				double sumSqDiff = 0;
				for (ScoredEmbedding se : scored) {
					double diff = se.semanticScore - meanSem;
					sumSqDiff += diff * diff;
				}
				double stddev = Math.sqrt(sumSqDiff / scored.size());
				double zScore = stddev == 0 ? 0
						: (maxSemanticScore - meanSem) / stddev;
				if (zScore < ChartSearchAiConstants.ZERO_KEYWORD_MIN_Z_SCORE) {
					log.debug("Only {} keyword match(es) and top semantic "
							+ "z-score {} is below threshold {}, "
							+ "returning empty (maxSem={}, mean={}, "
							+ "stddev={})",
							keywordMatchCount,
							String.format("%.2f", zScore),
							ChartSearchAiConstants.ZERO_KEYWORD_MIN_Z_SCORE,
							String.format("%.4f", maxSemanticScore),
							String.format("%.4f", meanSem),
							String.format("%.4f", stddev));
					return Collections.emptyList();
				}
			}
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

		List<ScoredEmbedding> candidates = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < adaptiveCutoff; i++) {
			candidates.add(scored.get(i));
		}

		// Keyword refinement: when gap detection returns a broad set but
		// keyword matches identify a specific subset, prefer those records.
		// This catches cases where the score distribution is too smooth for
		// gap detection (e.g. "conditions" query where condition records get
		// a keyword boost but the gap to non-conditions is < minGap).
		boolean refinementActivated = false;
		if (keywordWeight > 0) {
			List<ScoredEmbedding> refined = refineByKeywords(candidates, queryTerms.length);
			refinementActivated = refined.size() < candidates.size();
			candidates = refined;
		}

		// Post-processing with two paths:
		//
		// REFINEMENT PATH: Keyword refinement identified a relevant subset.
		// Run a second-pass gap detection within the refined set to separate
		// genuine keyword matches from false positives (e.g., Fetishism
		// "Chronic disease management" matching "disease" in STD query).
		// Uses adaptive minGap proportional to max semantic score so the
		// threshold scales with the score range of the current query.
		//
		// NON-REFINEMENT PATH: No keyword subset was found (either all or
		// no records have keywords). Run a sensitive second-pass gap
		// detection on the full candidate set to find tight clusters
		// (e.g., 2 Kaposi sarcoma records for "cancer?" query). Then
		// apply ratio floor + topK as a safety net.
		if (refinementActivated) {
			// The second-pass gap detection exists to split genuine keyword
			// matches from coincidental ones (e.g. "Chronic disease
			// management" matching "disease" in an STD query). But when
			// keyword scores across the refined set are uniform, all
			// candidates matched the same keyword pattern — the second gap
			// would only be splitting on semantic score variance, which is
			// noise for records that are all equally keyword-relevant.
			// Skip the gap when keyword score variance is zero (or near
			// zero due to floating-point), preserving the full refined set.
			double kwMin = Double.MAX_VALUE;
			double kwMax = 0;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore < kwMin) {
					kwMin = se.keywordScore;
				}
				if (se.keywordScore > kwMax) {
					kwMax = se.keywordScore;
				}
			}
			boolean uniformKeywords = (kwMax - kwMin) < 0.01;
			if (!uniformKeywords) {
				double maxSemanticRefined = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > maxSemanticRefined) {
						maxSemanticRefined = se.semanticScore;
					}
				}
				double adaptiveMinGap = Math.max(
						maxSemanticRefined * ChartSearchAiConstants.REFINEMENT_ADAPTIVE_GAP_RATIO,
						ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
				int refinedCutoff = findAdaptiveCutoff(candidates, candidates.size(),
						minScore, getScoreGapMultiplier(), adaptiveMinGap);
				candidates = new ArrayList<ScoredEmbedding>(candidates.subList(0, refinedCutoff));
			}
		} else {
			// Sensitive second-pass: lower multiplier and small absolute
			// minGap to detect tight clusters that the first pass missed.
			int secondCutoff = findAdaptiveCutoff(candidates, candidates.size(),
					minScore, ChartSearchAiConstants.SECOND_PASS_GAP_MULTIPLIER,
					ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
			if (secondCutoff < candidates.size()) {
				candidates = new ArrayList<ScoredEmbedding>(candidates.subList(0, secondCutoff));
			}
			double ratioFloor = maxBaseScore * getSimilarityRatio();
			List<ScoredEmbedding> strict = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : candidates) {
				if (se.score >= ratioFloor) {
					strict.add(se);
				}
			}
			candidates = strict;
			// When every surviving candidate has keyword matches, the
			// combination of gap detection + ratio floor already identified
			// the relevant cluster. TopK would arbitrarily truncate
			// legitimate results (e.g. 40 vitals for a multi-concept query
			// about "BP, weight, and temperature trend"). Only apply topK
			// when some candidates lack keywords — they may be semantic
			// false positives that need capping.
			boolean allHaveKeywords = true;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore == 0) {
					allHaveKeywords = false;
					break;
				}
			}
			if (!allHaveKeywords && candidates.size() > topK) {
				candidates = candidates.subList(0, topK);
			}
		}

		// Inter-candidate coherence filter: remove "topic outliers" that
		// scored similarly to the query by coincidence but are unrelated to
		// the other results. Requires at least 3 candidates to form a
		// meaningful cluster comparison. The filter itself validates that
		// all candidates have embedding vectors, returning unmodified if not.
		if (candidates.size() >= 3) {
			candidates = filterByCoherence(candidates);
		}

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		for (ScoredEmbedding se : candidates) {
			results.add(se.embedding);
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
	 * detecting a significant gap in raw semantic similarity scores. Uses
	 * semantic scores (not keyword-inflated combined scores) so that keyword
	 * bonuses don't create artificial cluster boundaries — this ensures
	 * multi-concept queries (e.g. "blood pressure, weight, and temperature")
	 * don't get split by uneven keyword matching across concepts.
	 * Walks the sorted scores and tracks the running average gap between
	 * consecutive entries. When a gap exceeds the average by more than the
	 * configured multiplier AND exceeds the minimum absolute gap, the cluster
	 * boundary is found. The absolute minimum prevents premature cutting on
	 * small gaps that only appear large relative to a tight cluster. Always
	 * includes at least {@link ChartSearchAiConstants#ADAPTIVE_MIN_RECORDS}
	 * records (if available above the similarity floor) so the LLM has enough
	 * context.
	 *
	 * @param scored sorted list of scored embeddings (highest combined score first)
	 * @param limit the maximum number of candidates to consider
	 * @param minScore the absolute similarity floor (applied to semantic scores)
	 * @param gapMultiplier how many times larger than the average gap a score drop
	 *        must be to trigger a cutoff
	 * @param minGap the minimum absolute gap required to trigger a cutoff,
	 *        regardless of the multiplier condition
	 * @return the number of records to include in the primary cluster
	 */
	static int findAdaptiveCutoff(List<ScoredEmbedding> scored, int limit, double minScore,
			double gapMultiplier, double minGap) {
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;

		// Build a sorted list of semantic scores for gap detection.
		// Records are passed in combined-score order, but gap detection
		// uses semantic scores to find natural clusters without keyword
		// bonus inflation.
		List<Double> semanticSorted = new ArrayList<Double>();
		for (int i = 0; i < limit; i++) {
			semanticSorted.add(scored.get(i).semanticScore);
		}
		Collections.sort(semanticSorted, Collections.reverseOrder());

		// Count how many records pass the similarity floor
		int aboveFloor = 0;
		for (int i = 0; i < semanticSorted.size(); i++) {
			if (semanticSorted.get(i) >= minScore) {
				aboveFloor++;
			} else {
				break;
			}
		}

		if (aboveFloor == 0) {
			return 0;
		}

		// Walk through consecutive semantic scores tracking the running
		// average gap. Once a gap exceeds gapMultiplier * avgGap, cut there.
		// All prior gaps feed the running average so that the baseline
		// reflects the full score distribution.
		double gapSum = 0;
		int cutoff = aboveFloor; // default: include everything above the floor
		for (int i = 1; i < aboveFloor; i++) {
			double gap = semanticSorted.get(i - 1) - semanticSorted.get(i);

			// Only consider cutting after the minimum number of records, and
			// only when we have at least 1 prior gap to compute a baseline
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

	/**
	 * Refines the gap-detected result set by keeping only records that have
	 * strong keyword matches, when those matches identify a clear subset.
	 * This catches cases where the score distribution is too smooth for gap
	 * detection to find a cutoff, but keyword matching provides a
	 * discriminative type signal (e.g., "conditions" matching "Condition:"
	 * in record text).
	 *
	 * <p>The refinement requires at least 1 matching query term for any
	 * query length ({@code minKwScore = 1.0 / queryTermCount}). Any keyword
	 * relevance is sufficient because gap detection is the primary noise
	 * filter — refinement only separates "has keyword evidence" from "has
	 * no keyword evidence". It only
	 * activates when:
	 * <ul>
	 * <li>At least {@link ChartSearchAiConstants#ADAPTIVE_MIN_RECORDS} records
	 * have strong keyword matches (enough to be meaningful)</li>
	 * <li>The keyword-matched records are a proper subset of the candidates
	 * (not all records match, which would mean keywords aren't
	 * discriminative)</li>
	 * </ul>
	 *
	 * @param candidates the gap-detected result set with keyword scores
	 * @param queryTermCount the number of query terms used for scoring
	 * @return the refined set (keyword-matched only) or the original set
	 *         if refinement conditions are not met
	 */
	static List<ScoredEmbedding> refineByKeywords(List<ScoredEmbedding> candidates,
			int queryTermCount) {
		if (queryTermCount == 0) {
			return candidates;
		}
		// Require at least 1 matching query term — any keyword relevance
		// is sufficient for refinement. Gap detection is the primary filter
		// for noise; refinement separates "has keyword evidence" from
		// "has no keyword evidence". This ensures multi-concept queries
		// (e.g. "blood pressure, weight, and temperature") don't drop
		// records matching a single concept like "weight" (1/N terms).
		double minKwScore = 1.0 / queryTermCount;

		List<ScoredEmbedding> keywordMatched = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= minKwScore) {
				keywordMatched.add(se);
			}
		}
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
		if (keywordMatched.size() >= minRecords && keywordMatched.size() < candidates.size()) {
			log.debug("Keyword refinement: keeping {} of {} (minKwScore={})",
					keywordMatched.size(), candidates.size(), String.format("%.3f", minKwScore));
			return keywordMatched;
		}
		return candidates;
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
			if (term.length() >= 2 && !QUERY_STOPWORDS.contains(term)) {
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
		String[] textWords = lowerText.split("\\s+");
		int matched = 0;
		for (String term : queryTerms) {
			boolean termMatched = false;

			// 1. Exact substring match
			if (lowerText.contains(term)) {
				termMatched = true;
			}

			// 2. Plural stem: strip trailing 's'
			if (!termMatched && term.length() > 3
					&& term.endsWith("s") && !term.endsWith("ss")) {
				if (lowerText.contains(term.substring(0, term.length() - 1))) {
					termMatched = true;
				}
			}

			// 3. Morphological stem: trim 2-3 trailing characters to handle
			// derivational variants (allergic/allergy, prescribed/prescription).
			// Uses word-prefix matching instead of substring to avoid false
			// positives from compound words (e.g. "allerg" inside "Photoallergy").
			if (!termMatched && term.length() >= 7) {
				for (int trim = 2; trim <= 3 && !termMatched; trim++) {
					String stem = term.substring(0, term.length() - trim);
					if (stem.length() >= 5) {
						for (String word : textWords) {
							if (word.startsWith(stem)) {
								termMatched = true;
								break;
							}
						}
					}
				}
			}

			if (termMatched) {
				matched++;
			}
		}
		return (double) matched / queryTerms.length;
	}

	/**
	 * Computes cosine similarity between two float vectors. When both
	 * vectors are L2-normalized (as produced by the ONNX embedding
	 * provider), this reduces to a simple dot product.
	 *
	 * @param a first embedding vector
	 * @param b second embedding vector
	 * @return cosine similarity in [-1, 1], or 0 if either vector is empty
	 */
	static double cosineSimilarity(float[] a, float[] b) {
		if (a == null || b == null || a.length == 0 || b.length == 0
				|| a.length != b.length) {
			return 0;
		}
		double dot = 0, normA = 0, normB = 0;
		for (int i = 0; i < a.length; i++) {
			dot += a[i] * b[i];
			normA += a[i] * a[i];
			normB += b[i] * b[i];
		}
		double denom = Math.sqrt(normA) * Math.sqrt(normB);
		return denom == 0 ? 0 : dot / denom;
	}

	/**
	 * Filters the final candidate set by inter-candidate coherence to
	 * remove "topic outliers" — records that scored similarly to the query
	 * by coincidence but are conceptually unrelated to the other results.
	 *
	 * <p>When gap detection and keyword filtering cannot separate correct
	 * from incorrect records (because their query-similarity scores are
	 * nearly identical), this filter uses a <b>second signal</b>: how
	 * similar each candidate is to the OTHER candidates. Records that form
	 * a coherent topic cluster (e.g. two Kaposi sarcoma diagnoses) will
	 * have high mutual similarity, while unrelated records (e.g. a CD4
	 * Count test that happened to score similarly for "cancer?") will be
	 * dissimilar to the cluster and can be detected as outliers.
	 *
	 * <p>Algorithm:
	 * <ol>
	 * <li>Compute pairwise cosine similarity between all candidates'
	 * embedding vectors (O(n²), cheap for small result sets).</li>
	 * <li>For each candidate, compute its average similarity to all other
	 * candidates — its "coherence score".</li>
	 * <li>Sort by coherence descending and apply gap detection: if a
	 * candidate's coherence drops sharply compared to the running average,
	 * it (and all lower-coherence candidates) are removed.</li>
	 * </ol>
	 *
	 * <p>Only applied when there are ≥ 3 candidates with valid embedding
	 * vectors. With fewer candidates, there is no meaningful cluster to
	 * compare against.
	 *
	 * @param candidates the post-filtered candidate set
	 * @return the coherence-filtered subset (preserving original order)
	 */
	static List<ScoredEmbedding> filterByCoherence(List<ScoredEmbedding> candidates) {
		int n = candidates.size();

		// Cache embedding vectors to avoid redundant byte[] → float[]
		// decoding in the O(n²) pairwise loop.
		float[][] vectors = new float[n][];
		boolean allValid = true;
		for (int i = 0; i < n; i++) {
			byte[] raw = candidates.get(i).embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				allValid = false;
				break;
			}
			vectors[i] = candidates.get(i).embedding.getEmbeddingVector();
		}
		if (!allValid) {
			return candidates;
		}

		// Compute pairwise cosine similarities between candidate embeddings
		double[][] pairSim = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				double sim = cosineSimilarity(vectors[i], vectors[j]);
				pairSim[i][j] = sim;
				pairSim[j][i] = sim;
			}
		}

		// Compute coherence: average similarity to all other candidates
		double[] coherence = new double[n];
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int j = 0; j < n; j++) {
				if (j != i) {
					sum += pairSim[i][j];
				}
			}
			coherence[i] = sum / (n - 1);
		}

		// Build index array sorted by coherence descending
		Integer[] sortedIdx = new Integer[n];
		for (int i = 0; i < n; i++) {
			sortedIdx[i] = i;
		}
		Arrays.sort(sortedIdx, new Comparator<Integer>() {

			@Override
			public int compare(Integer a, Integer b) {
				return Double.compare(coherence[b], coherence[a]);
			}
		});

		// Gap detection on coherence scores: find where coherence drops
		// sharply. Uses adaptive minGap proportional to max coherence so
		// the threshold scales with the actual inter-candidate similarity
		// range. The multiplier (2.0) is moderate — we only remove clear
		// outliers, not borderline records.
		double maxCoherence = coherence[sortedIdx[0]];
		double coherenceMinGap = maxCoherence
				* ChartSearchAiConstants.COHERENCE_ADAPTIVE_GAP_RATIO;
		int keepCount = n;
		double gapSum = 0;
		for (int i = 1; i < n; i++) {
			double gap = coherence[sortedIdx[i - 1]] - coherence[sortedIdx[i]];
			if (i >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS && i >= 2) {
				double avgGap = gapSum / (i - 1);
				if (gap > avgGap * ChartSearchAiConstants.COHERENCE_GAP_MULTIPLIER
						&& gap > coherenceMinGap) {
					keepCount = i;
					log.debug(
							"Coherence gap at position {}: gap={}, avgGap={}, "
									+ "removed={} outlier(s)",
							i, String.format("%.4f", gap),
							String.format("%.4f", avgGap), n - i);
					break;
				}
			}
			gapSum += gap;
		}

		if (keepCount == n) {
			return candidates;
		}

		// Build the set of indices to keep, then filter preserving
		// the original combined-score order
		Set<Integer> keepSet = new HashSet<Integer>();
		for (int i = 0; i < keepCount; i++) {
			keepSet.add(sortedIdx[i]);
		}
		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < n; i++) {
			if (keepSet.contains(i)) {
				filtered.add(candidates.get(i));
			}
		}
		return filtered;
	}

	static class ScoredEmbedding {

		final ChartEmbedding embedding;

		final double score;

		final double keywordScore;

		final double semanticScore;

		ScoredEmbedding(ChartEmbedding embedding, double score, double keywordScore,
				double semanticScore) {
			this.embedding = embedding;
			this.score = score;
			this.keywordScore = keywordScore;
			this.semanticScore = semanticScore;
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
