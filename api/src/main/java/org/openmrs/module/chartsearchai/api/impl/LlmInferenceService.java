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
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.ElasticsearchIndexer;
import org.openmrs.module.chartsearchai.api.HybridRetriever;
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
	private ElasticsearchIndexer elasticsearchIndexer;

	@Autowired
	private HybridRetriever hybridRetriever;

	@Autowired
	private LlmProvider llmProvider;


	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = buildChart(patient, question);

		LlmResponse response = llmProvider.search(chart.getText(), question);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()),
				response.getInputTokens(), response.getOutputTokens());
	}

	@Override
	public ChartAnswer searchStreaming(Patient patient, String question,
			Consumer<String> tokenConsumer) {
		PatientChart chart = buildChart(patient, question);

		LlmResponse response = llmProvider.searchStreaming(chart.getText(), question, tokenConsumer);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()),
				response.getInputTokens(), response.getOutputTokens());
	}

	private static final String NUMBER_GROUP =
			"(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)";

	private static final String KEYWORD_GROUP =
			"(?:last|latest|past|previous|recent|most recent)";

	private static final Pattern RECENCY_PATTERN = Pattern.compile(
			KEYWORD_GROUP + "\\s+" + NUMBER_GROUP
			+ "|" + NUMBER_GROUP + "\\s+" + KEYWORD_GROUP,
			Pattern.CASE_INSENSITIVE);

	private static final Map<String, Integer> WORD_NUMBERS;

	static {
		Map<String, Integer> m = new HashMap<String, Integer>();
		m.put("one", 1);
		m.put("two", 2);
		m.put("three", 3);
		m.put("four", 4);
		m.put("five", 5);
		m.put("six", 6);
		m.put("seven", 7);
		m.put("eight", 8);
		m.put("nine", 9);
		m.put("ten", 10);
		WORD_NUMBERS = Collections.unmodifiableMap(m);
	}

	/**
	 * Extracts a numeric recency constraint from the question, e.g. "last 7
	 * visits" or "latest two weights" returns the number. Supports both
	 * digits and word numbers (one through ten). Returns 0 if no constraint
	 * is found.
	 *
	 * @param question the raw user question
	 * @return the recency cap, or 0 if none detected
	 */
	static int extractRecencyCap(String question) {
		Matcher m = RECENCY_PATTERN.matcher(question);
		if (m.find()) {
			// Group 1 = keyword-first ("last 7"), group 2 = number-first ("7 most recent")
			String value = (m.group(1) != null ? m.group(1) : m.group(2)).toLowerCase();
			Integer wordNum = WORD_NUMBERS.get(value);
			if (wordNum != null) {
				return wordNum;
			}
			try {
				int n = Integer.parseInt(value);
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
	 * Groups records by concept key, preserving the original order within each group.
	 * Groups appear in the order their first record is encountered. For example,
	 * interleaved [BP, Weight, BP, Temp, Weight] becomes [BP, BP, Weight, Weight, Temp].
	 * This helps small LLMs process multi-concept queries by reducing the need to
	 * mentally sort interleaved records.
	 */
	static List<SerializedRecord> groupByConcept(List<SerializedRecord> records) {
		Map<String, List<SerializedRecord>> groups = new java.util.LinkedHashMap<String, List<SerializedRecord>>();
		for (SerializedRecord record : records) {
			String key = conceptKey(record.getText());
			groups.computeIfAbsent(key, k -> new ArrayList<SerializedRecord>()).add(record);
		}
		List<SerializedRecord> result = new ArrayList<SerializedRecord>();
		for (List<SerializedRecord> group : groups.values()) {
			result.addAll(group);
		}
		return result;
	}

	/**
	 * Extracts a concept grouping key from record text by stripping the
	 * date prefix and trailing numeric value with optional unit. For example:
	 * <ul>
	 * <li>"Clinical observation: (2025-10-30) Test — Weight (kg): 94.0 kg"
	 *     → "Clinical observation: Test — Weight (kg)"</li>
	 * <li>"Clinical observation: (2025-09-17) Test — Systolic Blood Pressure: 137.0 mmHg"
	 *     → "Clinical observation: Test — Systolic Blood Pressure"</li>
	 * </ul>
	 * If the text does not end with a numeric value, the full text (minus
	 * date) is returned, making each such record its own group.
	 */
	static String conceptKey(String text) {
		if (text == null) {
			return "";
		}
		// Strip date like "(2025-10-30) " that appears after the type prefix
		String stripped = text.replaceAll("\\(\\d{4}-\\d{2}-\\d{2}\\)\\s*", "");
		// Strip trailing numeric value and optional unit like ": 94.0 kg"
		// or ": 36.7 DEG C" or ": 988.0 cells/mmL"
		return stripped.replaceAll(":\\s*[\\d.]+(?:\\s+\\S+)*\\s*$", "").trim();
	}

	private PatientChart buildChart(Patient patient, String question) {
		if (!usePreFilter()) {
			return chartSerializer.serialize(patient);
		}

		if (isHybridPipeline()) {
			return buildChartWithHybrid(patient, question);
		}
		if (isElasticsearchPipeline()) {
			return buildChartWithElasticsearch(patient, question);
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
			log.debug("No records matched the query '{}' for patient [id={}], returning empty chart",
					question, patient.getPatientId());
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		Set<String> relevantKeys = new HashSet<String>();
		for (ChartEmbedding ce : similar) {
			relevantKeys.add(ChartSearchAiUtils.resourceKey(ce.getResourceType(), ce.getResourceId()));
		}

		log.debug("findSimilar returned {} records for query '{}' patient [id={}]: {}",
				similar.size(), question, patient.getPatientId(), relevantKeys);

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
			relevantKeys.add(ChartSearchAiUtils.resourceKey(result.getResourceType(), result.getResourceId()));
		}

		log.debug("Lucene returned {} results for query '{}'",
				relevantKeys.size(), normalizedQuery);

		return filterAndSerialize(patient, question, relevantKeys);
	}

	private PatientChart buildChartWithElasticsearch(Patient patient, String question) {
		if (!elasticsearchIndexer.isAvailable()) {
			log.warn("Elasticsearch not available, falling back to embedding pipeline");
			return buildChartWithEmbeddings(patient, question);
		}

		if (!elasticsearchIndexer.hasIndex(patient)) {
			log.info("No ES index for patient [id={}], indexing now", patient.getPatientId());
			try {
				elasticsearchIndexer.indexPatient(patient);
			}
			catch (Exception e) {
				log.error("Failed to ES-index patient [id={}], falling back to embeddings",
						patient.getPatientId(), e);
				return buildChartWithEmbeddings(patient, question);
			}
		}

		String normalizedQuery = stripQueryStopwords(question);
		if (normalizedQuery.isEmpty()) {
			return chartSerializer.serialize(patient);
		}

		String embeddingInput = prepareEmbeddingInput(question, getQueryPrefix());
		float[] queryVector;
		try {
			queryVector = embeddingProvider.embed(embeddingInput);
		}
		catch (Exception e) {
			log.error("Failed to embed query for ES search, falling back to embeddings", e);
			return buildChartWithEmbeddings(patient, question);
		}

		List<ElasticsearchIndexer.ElasticsearchSearchResult> results =
				elasticsearchIndexer.search(patient, normalizedQuery, queryVector,
						getTopK() * ChartSearchAiConstants.ES_FETCH_MULTIPLIER);

		if (results.isEmpty()) {
			log.debug("Elasticsearch returned no results for query '{}', returning empty chart",
					normalizedQuery);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		// Run ES results through the same filter pipeline as the embedding
		// pipeline to remove noise — RRF always returns up to maxResults
		// from the kNN side even when most are irrelevant.
		List<ElasticsearchIndexer.ElasticsearchSearchResult> filtered =
				filterEsResults(results, queryVector, normalizedQuery);

		if (filtered.isEmpty()) {
			log.debug("Elasticsearch: all {} results filtered out for query '{}'",
					results.size(), normalizedQuery);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		Set<String> relevantKeys = new HashSet<String>();
		for (ElasticsearchIndexer.ElasticsearchSearchResult result : filtered) {
			relevantKeys.add(ChartSearchAiUtils.resourceKey(result.getResourceType(), result.getResourceId()));
		}

		log.debug("Elasticsearch returned {} results ({} after filter pipeline) for query '{}': {}",
				results.size(), relevantKeys.size(), normalizedQuery, relevantKeys);

		return filterAndSerialize(patient, question, relevantKeys);
	}

	/**
	 * Filters Elasticsearch results through the same scoring and gap
	 * detection pipeline used by the embedding retrieval path. Computes
	 * cosine similarity and keyword scores from the returned embedding
	 * vectors and text, then applies filterPipeline() to remove noise.
	 *
	 * @param results the raw ES search results
	 * @param queryVector the embedded query vector
	 * @param normalizedQuery the query after stopword removal (pre-computed
	 *        by the caller to avoid redundant processing)
	 */
	static List<ElasticsearchIndexer.ElasticsearchSearchResult> filterEsResults(
			List<ElasticsearchIndexer.ElasticsearchSearchResult> results,
			float[] queryVector, String normalizedQuery) {
		String[] queryTerms = extractQueryTerms(normalizedQuery);

		// Build parallel arrays for filterPipeline
		List<ElasticsearchIndexer.ElasticsearchSearchResult> valid =
				new ArrayList<ElasticsearchIndexer.ElasticsearchSearchResult>();
		for (ElasticsearchIndexer.ElasticsearchSearchResult r : results) {
			if (r.getEmbedding() != null && r.getEmbedding().length == queryVector.length) {
				valid.add(r);
			}
		}
		if (valid.isEmpty()) {
			return results;
		}

		double[] semanticScores = new double[valid.size()];
		double[] keywordScores = new double[valid.size()];
		ChartEmbedding[] embeddings = new ChartEmbedding[valid.size()];
		for (int i = 0; i < valid.size(); i++) {
			ElasticsearchIndexer.ElasticsearchSearchResult r = valid.get(i);
			semanticScores[i] = ChartSearchAiUtils.cosineSimilarity(
					queryVector, r.getEmbedding());
			keywordScores[i] = r.getText() != null
					? computeKeywordScore(queryTerms, r.getText()) : 0;

			ChartEmbedding ce = new ChartEmbedding();
			ce.setResourceType(r.getResourceType());
			ce.setResourceId(r.getResourceId());
			ce.setEmbeddingVector(r.getEmbedding());
			embeddings[i] = ce;
		}

		List<ChartEmbedding> filtered = filterPipeline(semanticScores,
				keywordScores, embeddings, queryTerms, valid.size(),
				PipelineConfig.defaults());

		Set<String> survivorKeys = new HashSet<String>();
		for (ChartEmbedding ce : filtered) {
			survivorKeys.add(ChartSearchAiUtils.resourceKey(ce.getResourceType(), ce.getResourceId()));
		}

		List<ElasticsearchIndexer.ElasticsearchSearchResult> out =
				new ArrayList<ElasticsearchIndexer.ElasticsearchSearchResult>();
		for (ElasticsearchIndexer.ElasticsearchSearchResult r : results) {
			if (survivorKeys.contains(ChartSearchAiUtils.resourceKey(r.getResourceType(), r.getResourceId()))) {
				out.add(r);
			}
		}
		return out;
	}

	private PatientChart buildChartWithHybrid(Patient patient, String question) {
		try {
			hybridRetriever.ensureIndexed(patient);
		}
		catch (Exception e) {
			log.error("Failed to index patient [id={}] for hybrid pipeline, falling back to full chart",
					patient.getPatientId(), e);
			return chartSerializer.serialize(patient);
		}

		String normalizedQuery = stripQueryStopwords(question);
		if (normalizedQuery.isEmpty()) {
			return chartSerializer.serialize(patient);
		}

		Set<String> relevantKeys;
		try {
			relevantKeys = hybridRetriever.search(
					patient, normalizedQuery, getQueryPrefix(), getTopK());
		}
		catch (Exception e) {
			log.error("Hybrid search failed for patient [id={}], falling back to full chart",
					patient.getPatientId(), e);
			return chartSerializer.serialize(patient);
		}

		if (relevantKeys.isEmpty()) {
			log.debug("Hybrid returned no results for query '{}', returning empty chart",
					normalizedQuery);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		log.debug("Hybrid returned {} results for query '{}'",
				relevantKeys.size(), normalizedQuery);

		return filterAndSerialize(patient, question, relevantKeys);
	}

	private PatientChart filterAndSerialize(Patient patient, String question,
			Set<String> relevantKeys) {
		List<SerializedRecord> allRecords = recordLoader.loadAll(patient);
		List<SerializedRecord> filtered = new ArrayList<SerializedRecord>();
		for (SerializedRecord record : allRecords) {
			if (relevantKeys.contains(ChartSearchAiUtils.resourceKey(record.getResourceType(), record.getResourceId()))) {
				filtered.add(record);
			}
		}

		log.debug("Pre-filtered {} records to {} using {}",
				allRecords.size(), filtered.size(),
				isHybridPipeline() ? "Hybrid" :
						isElasticsearchPipeline() ? "Elasticsearch" :
								isLucenePipeline() ? "Lucene" : "embeddings");

		int recencyCap = extractRecencyCap(question);
		if (recencyCap > 0) {
			filtered = capPerConcept(filtered, recencyCap);
			log.debug("Recency cap {} applied, {} records remain", recencyCap, filtered.size());
		}

		filtered = groupByConcept(filtered);

		return chartSerializer.serialize(patient, filtered);
	}

	boolean isHybridPipeline() {
		String pipeline = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE);
		return ChartSearchAiConstants.PIPELINE_HYBRID.equalsIgnoreCase(
				pipeline != null ? pipeline.trim() : "");
	}

	boolean isElasticsearchPipeline() {
		String pipeline = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE);
		return ChartSearchAiConstants.PIPELINE_ELASTICSEARCH.equalsIgnoreCase(
				pipeline != null ? pipeline.trim() : "");
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

	List<ChartEmbedding> findSimilar(Patient patient, String question) {
		return findSimilar(patient, question, getTopK());
	}

	private List<ChartEmbedding> findSimilar(Patient patient, String question, int topK) {
		List<ChartEmbedding> allEmbeddings = dao.getByPatient(patient);
		if (allEmbeddings.isEmpty()) {
			return null;
		}

		PipelineConfig config = new PipelineConfig(
				getKeywordWeight(), getScoreGapMultiplier(), getMinScoreGap(),
				getGapValidationCosineThreshold(), getSimilarityRatio());

		try {
			return findSimilar(allEmbeddings, embeddingProvider, question, topK,
					getQueryPrefix(), config);
		}
		catch (Exception e) {
			log.debug("Embedding provider not available, falling back to full chart", e);
			return null;
		}
	}

	/**
	 * Static query pipeline that runs the exact same logic as the instance
	 * {@link #findSimilar(Patient, String, int)} but without DAO or Spring
	 * dependencies. Accepts pre-loaded embeddings and an embedding provider,
	 * making it directly callable from integration tests with zero simulation.
	 *
	 * @param allEmbeddings the patient's chart embeddings (as stored by
	 *        {@link EmbeddingIndexer#indexPatient})
	 * @param provider embedding provider for query vectorization
	 * @param question the natural language question
	 * @param topK maximum records to return
	 * @param queryPrefix prefix prepended to the query before embedding
	 * @param config pipeline tuning parameters
	 * @return filtered list of relevant embeddings, or null if empty
	 */
	static List<ChartEmbedding> findSimilar(List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question, int topK,
			String queryPrefix, PipelineConfig config) {
		String normalizedQuery = stripQueryStopwords(question);
		String[] queryTerms = extractQueryTerms(normalizedQuery);
		String embeddingQuery = buildEmbeddingQuery(normalizedQuery);
		float[] queryVector = provider.embed(queryPrefix + embeddingQuery);

		double[] semanticScores = new double[allEmbeddings.size()];
		double[] keywordScores = new double[allEmbeddings.size()];
		ChartEmbedding[] embeddings = new ChartEmbedding[allEmbeddings.size()];
		int validCount = 0;
		for (int i = 0; i < allEmbeddings.size(); i++) {
			ChartEmbedding ce = allEmbeddings.get(i);
			float[] vector = ce.getEmbeddingVector();
			if (vector == null || vector.length != queryVector.length) {
				continue;
			}
			embeddings[validCount] = ce;
			semanticScores[validCount] = cosineSimilarity(queryVector, vector);
			String keywordText = ChartSearchAiUtils.buildPrefixedText(
					ce.getResourceType(), ce.getTextContent());
			keywordScores[validCount] = computeKeywordScore(queryTerms, keywordText);
			validCount++;
		}
		if (validCount < embeddings.length) {
			embeddings = java.util.Arrays.copyOf(embeddings, validCount);
			semanticScores = java.util.Arrays.copyOf(semanticScores, validCount);
			keywordScores = java.util.Arrays.copyOf(keywordScores, validCount);
		}

		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, topK, config);
	}

	/**
	 * Overload that accepts a term count instead of actual query terms.
	 * The keyword-tier subset check is skipped since the actual terms
	 * are not available.
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			int queryTermCount, int topK, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				null, queryTermCount, topK, config);
	}

	/**
	 * Pure filtering pipeline: scores, gates, gap detection, keyword
	 * refinement, coherence filtering, and below-floor rescue. Takes
	 * pre-computed scores and config parameters — no dependency on
	 * OpenMRS Context, making it directly testable.
	 *
	 * @param semanticScores cosine similarity between query and each record
	 * @param keywordScores keyword overlap fraction for each record
	 * @param embeddings the chart embeddings (parallel to score arrays)
	 * @param queryTerms query terms after stopword removal (null-safe:
	 *        when null, the keyword-tier subset check is skipped)
	 * @param topK maximum records to return (safety cap)
	 * @param config pipeline tuning parameters
	 * @return filtered list of relevant embeddings, or empty list
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int topK, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, queryTerms != null ? queryTerms.length : 0,
				topK, config);
	}

	private static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int queryTermCount, int topK,
			PipelineConfig config) {

		double maxBaseScore = 0;
		double maxSemanticScore = 0;

		// Require ≥2 term matches for the keyword bonus so that single
		// coincidental matches (e.g. "history" in "immunization history"
		// for "history of cancer", or "lab" in "Labs ordered." for
		// "lab orders placed resulted") don't inflate scores.
		double bonusThreshold = queryTermCount == 0 ? 1.0
				: (double) Math.min(2, queryTermCount) / queryTermCount;
		List<ScoredEmbedding> scored = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < embeddings.length; i++) {
			double semanticScore = semanticScores[i];
			double keywordScore = keywordScores[i];

			// Additive keyword bonus increases the score when enough terms
			// match. For N≤2 queries, a partial keyword match (below the
			// bonus threshold) applies a penalty instead, suppressing
			// coincidental single-word overlaps like "history" in
			// "immunization history" for "history of cancer?".
			double keywordBonus = keywordScore >= bonusThreshold ? keywordScore : 0.0;
			// Partial keyword penalty for short queries (N≤2): when a record
			// matches some keywords but not enough for bonus, penalize it.
			double keywordPenalty = 0.0;
			if (queryTermCount <= 2 && keywordScore > 0 && keywordScore < bonusThreshold) {
				keywordPenalty = keywordScore;
			}
			double baseScore = semanticScore + config.keywordWeight * keywordBonus
					- config.keywordWeight * keywordPenalty;

			if (semanticScore > maxSemanticScore) {
				maxSemanticScore = semanticScore;
			}
			if (baseScore > maxBaseScore) {
				maxBaseScore = baseScore;
			}

			scored.add(new ScoredEmbedding(embeddings[i], baseScore, keywordScore, semanticScore));
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

		// Floor gate: if neither the best semantic score nor the best
		// combined score reaches the floor, there is no relevance signal.
		double floorScore = Math.max(maxSemanticScore, maxBaseScore);
		if (floorScore < ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR) {
			log.debug("Top score {} (semantic={}, combined={}) is below "
					+ "absolute floor {}, returning empty",
					String.format("%.4f", floorScore),
					String.format("%.4f", maxSemanticScore),
					String.format("%.4f", maxBaseScore),
					ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR);
			return Collections.emptyList();
		}

		int keywordMatchCount = 0;
		for (ScoredEmbedding se : scored) {
			if (se.keywordScore > 0) {
				keywordMatchCount++;
			}
		}

		// Stricter gate when too few records match any query keyword:
		// check whether the top semantic score is a statistical outlier
		// (z-score ≥ threshold) rather than part of the noise floor.
		if (queryTermCount > 0
				&& scored.size() >= ChartSearchAiConstants.MIN_RECORDS_FOR_Z_SCORE) {
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
		double minScore = Math.min(
				ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR / 2,
				maxSemanticScore / 2);

		// Gap detection considers ALL records above the floor — no topK cap.
		int adaptiveCutoff = findAdaptiveCutoff(scored, scored.size(),
				minScore, config.scoreGapMultiplier, config.minScoreGap);

		List<ScoredEmbedding> candidates = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < adaptiveCutoff; i++) {
			candidates.add(scored.get(i));
		}

		// Keyword refinement: when gap detection returns a broad set but
		// keyword matches identify a specific subset, prefer those records.
		// Save pre-refinement candidates for semantic core discovery.
		List<ScoredEmbedding> preRefinementCandidates = candidates;
		boolean refinementActivated = false;
		if (config.keywordWeight > 0) {
			List<ScoredEmbedding> refined = refineByKeywords(candidates, queryTermCount);
			refinementActivated = refined.size() < candidates.size();
			candidates = refined;
		}

		log.debug("Pipeline: adaptiveCutoff={}, afterRefinement={}, refinementActivated={}",
				adaptiveCutoff, candidates.size(), refinementActivated);

		boolean partialKwValidated = false;

		// Post-processing with two paths:
		//
		// REFINEMENT PATH: Keyword refinement identified a relevant subset.
		// Run a second-pass gap detection within the refined set to separate
		// genuine keyword matches from false positives.
		//
		// NON-REFINEMENT PATH: No keyword subset was found. Run a sensitive
		// second-pass gap detection, then ratio floor + topK as a safety net.
		if (refinementActivated) {
			double kwMin = Double.MAX_VALUE;
			double kwMax = 0;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore > 0) {
					if (se.keywordScore < kwMin) {
						kwMin = se.keywordScore;
					}
					if (se.keywordScore > kwMax) {
						kwMax = se.keywordScore;
					}
				}
			}
			if (kwMin == Double.MAX_VALUE) {
				kwMin = kwMax;
			}
			boolean uniformKeywords = (kwMax - kwMin) < 0.01;
			// Partial keyword match: not all query terms appear in any
			// record. The keyword signal is weak — "blood" in "blood
			// pressure" is incidental, not evidence of relevance to
			// "blood problems".
			double bonusThresh = queryTermCount == 0 ? 1.0
					: (double) Math.min(2, queryTermCount) / queryTermCount;
			boolean partialKeywordMatch = uniformKeywords
					&& kwMax < bonusThresh
					&& queryTermCount <= 2;
			log.debug("Pipeline refinement: kwMin={}, kwMax={}, uniform={}, "
					+ "partialKw={}", String.format("%.4f", kwMin),
					 String.format("%.4f", kwMax), uniformKeywords,
					 partialKeywordMatch);
			if (partialKeywordMatch) {
				// Semantic core approach: find non-keyword records with
				// high semantic scores — these represent what the query
				// actually means without keyword bias. Then validate
				// keyword matches against this core.
				List<ScoredEmbedding> nonKeyword =
						new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : preRefinementCandidates) {
					if (se.keywordScore == 0) {
						nonKeyword.add(se);
					}
				}
				Collections.sort(nonKeyword,
						new java.util.Comparator<ScoredEmbedding>() {
					@Override
					public int compare(ScoredEmbedding a,
							ScoredEmbedding b) {
						return Double.compare(b.semanticScore,
								a.semanticScore);
					}
				});

				// Gap detection on non-keyword records to find the
				// semantic core — the cluster of records genuinely
				// relevant to the query by meaning alone.
				List<ScoredEmbedding> semanticCore =
						new ArrayList<ScoredEmbedding>();
				if (nonKeyword.size() >= ChartSearchAiConstants
						.ADAPTIVE_MIN_RECORDS) {
					double maxSemNk = nonKeyword.get(0).semanticScore;
					double nkMinGap = Math.max(
							maxSemNk * ChartSearchAiConstants
									.REFINEMENT_ADAPTIVE_GAP_RATIO,
							ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
					int coreCutoff = findAdaptiveCutoff(nonKeyword,
							nonKeyword.size(),
							ChartSearchAiConstants
									.ABSOLUTE_SIMILARITY_FLOOR,
							config.scoreGapMultiplier, nkMinGap);
					for (int i = 0; i < coreCutoff; i++) {
						semanticCore.add(nonKeyword.get(i));
					}

					log.debug("Semantic core: {} non-keyword"
							+ " records, primary gap at {},"
							+ " core size {}",
							nonKeyword.size(), coreCutoff,
							semanticCore.size());
				}

				if (!semanticCore.isEmpty()) {
					// Expand: find records cosine-coherent with the
					// core AND semantically close to the query.
					// Cosine alone admits template-similar noise
					// (e.g. "Condition: Hypertension" matching
					// "Condition: Anaemia" via shared template).
					// The score floor filters this: semantic score
					// (query cosine) is the only signal where
					// genuinely relevant records consistently
					// outscore template-similar noise.
					double coreMinSem = semanticCore.get(
							semanticCore.size() - 1)
							.semanticScore;
					double scoreFloor = coreMinSem
							* ChartSearchAiConstants
									.SEMANTIC_CORE_SCORE_RATIO;
					List<ScoredEmbedding> expanded =
							new ArrayList<ScoredEmbedding>(
									semanticCore);
					Set<Integer> expandedIds =
							new java.util.HashSet<Integer>();
					for (ScoredEmbedding se : semanticCore) {
						expandedIds.add(
								se.embedding.getResourceId());
					}
					for (ScoredEmbedding se
							: preRefinementCandidates) {
						if (expandedIds.contains(
								se.embedding.getResourceId())) {
							continue;
						}
						if (se.semanticScore < scoreFloor) {
							continue;
						}
						float[] vec = se.embedding
								.getEmbeddingVector();
						double maxCos = 0;
						for (ScoredEmbedding core
								: semanticCore) {
							double cos = cosineSimilarity(vec,
									core.embedding
											.getEmbeddingVector());
							if (cos > maxCos) {
								maxCos = cos;
							}
						}
						if (maxCos >= ChartSearchAiConstants
								.SEMANTIC_CORE_MIN_COSINE) {
							expanded.add(se);
							expandedIds.add(
									se.embedding.getResourceId());
						}
					}
					candidates = expanded;
					log.debug("Semantic core expansion:"
							+ " core={}, floor={}, result={}",
							semanticCore.size(),
							String.format("%.4f", scoreFloor),
							candidates.size());
					partialKwValidated = true;
				} else {
					// No semantic core found — fall back to the
					// semantic ratio floor on keyword matches.
					double maxSemanticKw = 0;
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore > maxSemanticKw) {
							maxSemanticKw = se.semanticScore;
						}
					}
					double semFloor = maxSemanticKw
							* ChartSearchAiConstants
									.REFINEMENT_SEMANTIC_RATIO;
					List<ScoredEmbedding> filtered =
							new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore >= semFloor) {
							filtered.add(se);
						}
					}
					if (filtered.size() >= ChartSearchAiConstants
							.ADAPTIVE_MIN_RECORDS) {
						log.debug("Partial-kw semantic floor: "
								+ "{} -> {} (floor={})",
								candidates.size(), filtered.size(),
								String.format("%.4f", semFloor));
						candidates = filtered;
					}
				}
			} else if (!uniformKeywords) {
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
						minScore, config.scoreGapMultiplier, adaptiveMinGap);
				if (refinedCutoff < candidates.size()) {
					// Validate: if records on both sides of the gap belong
					// to the same broad topic (high cross-boundary cosine),
					// the gap is intra-topic and should not be used as a
					// cutoff. Instead, fall back to a semantic ratio floor
					// that separates genuinely relevant records from
					// coincidental keyword matches (e.g. "blood" in
					// "Blood Oxygen" for a "blood pressure" query).
					if (isGapCoherent(candidates, refinedCutoff,
							config.gapValidationCosineThreshold)) {
						double semanticFloor = maxSemanticRefined
								* ChartSearchAiConstants.REFINEMENT_SEMANTIC_RATIO;
						List<ScoredEmbedding> floored = new ArrayList<ScoredEmbedding>();
						for (ScoredEmbedding se : candidates) {
							if (se.semanticScore >= semanticFloor) {
								floored.add(se);
							}
						}
						log.debug("Refinement gap is intra-topic, "
								+ "using semantic floor {} instead: {} -> {}",
								String.format("%.4f", semanticFloor),
								candidates.size(), floored.size());
						candidates = floored;
					} else if (queryTerms != null) {
						// Gap is NOT intra-topic. Before cutting, check
						// if records below the gap add new concept
						// coverage for a multi-concept query. This
						// rescues e.g. "weight" and "temperature"
						// records that cluster below BP records due to
						// fewer keyword matches.
						List<ScoredEmbedding> rescued =
								filterRedundantKeywordTier(candidates,
										queryTerms, kwMax);
						if (rescued.size() >= refinedCutoff) {
							log.debug("Refinement gap at {} is a "
									+ "concept boundary: rescued {} "
									+ "records with new coverage",
									refinedCutoff, rescued.size()
									- refinedCutoff);
							candidates = rescued;
						} else {
							candidates = new ArrayList<ScoredEmbedding>(
									candidates.subList(0, refinedCutoff));
						}
					} else {
						candidates = new ArrayList<ScoredEmbedding>(
								candidates.subList(0, refinedCutoff));
					}
				} else if (queryTerms != null) {
					// Gap detection found no score gap. Check whether
					// lower-tier records only match query terms already
					// covered by the higher tier. If so, they add no
					// information and are false positives (e.g. a program
					// enrollment matching "active" when conditions already
					// match "active" + "condition"). Records that match
					// unique terms are kept (e.g. "weight" records in a
					// "blood pressure weight temperature" query).
					candidates = filterRedundantKeywordTier(
							candidates, queryTerms, kwMax);
				}
			}
		} else {
			// Sensitive second-pass: same multiplier as the first pass, but
			// with a much lower absolute floor (SECOND_PASS_MIN_GAP vs
			// DEFAULT_MIN_SCORE_GAP). This detects tight clusters that the
			// first pass missed because their gaps were below the 0.10 floor.
			int secondCutoff = findAdaptiveCutoff(candidates, candidates.size(),
					minScore, config.scoreGapMultiplier,
					ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
			if (secondCutoff < candidates.size()) {
				// Validate the gap: if records above and below belong to
				// the same topic (high cross-boundary cosine), the gap is
				// intra-topic and should not be used as a cutoff. This
				// prevents abbreviation queries like "STD" from splitting
				// same-topic records that the embedding model scored
				// differently due to text format variations (e.g.
				// "Diagnosis: HIV Disease" vs "Assessment: HIV Disease").
				if (isGapCoherent(candidates, secondCutoff,
						config.gapValidationCosineThreshold)) {
					log.debug("Second-pass gap at {} is intra-topic, skipping cut",
							secondCutoff);
				} else {
					candidates = new ArrayList<ScoredEmbedding>(
							candidates.subList(0, secondCutoff));
				}
			}
			// Use the lower of maxBaseScore and maxSemanticScore for the
			// floor. Keyword bonuses inflate combined scores for records
			// with coincidental term matches (e.g. "pulse" in "pulse
			// oximeter" inflates SpO2 for "blood pressure and pulse").
			// Keyword penalties deflate combined scores below semantic
			// (e.g. "history" in immunization records for "history of
			// cancer"). Using the min avoids inflation from either side.
			// The per-record check uses min(combined, semantic) so bonuses
			// are stripped (checking semantic) and penalties are preserved
			// (checking combined).
			double ratioFloor = Math.min(maxBaseScore, maxSemanticScore)
					* config.similarityRatio;
			List<ScoredEmbedding> strict = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : candidates) {
				if (Math.min(se.score, se.semanticScore) >= ratioFloor) {
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

		// Phase 1: Outlier removal — remove individual candidates that
		// are topically unrelated to the majority of results. Uses gap
		// detection on inter-candidate coherence scores.
		//
		// At n=3 this is restricted to tight clusters (score ratio ≥ 0.90)
		// because with spread scores the minority candidate may be the
		// only correct result (e.g. Syphilitic Cirrhosis at 83% of top
		// for an STD query — a genuine hit that shouldn't be removed).
		// Skip when the partial-keyword semantic core path already
		// curated the candidate set — candidates were selected from
		// the semantic core and validated keyword reps. Coherence
		// filtering would incorrectly remove minority record types
		// (e.g. hemoglobin lab tests when haemorrhagic disease
		// condition/diagnosis form a tighter pair).
		if (!partialKwValidated && candidates.size() >= 4) {
			candidates = filterByCoherence(candidates);
		} else if (!partialKwValidated && candidates.size() == 3) {
			double topSemantic = candidates.get(0).semanticScore;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore > topSemantic) {
					topSemantic = se.semanticScore;
				}
			}
			double lowestSemantic = candidates.get(0).semanticScore;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore < lowestSemantic) {
					lowestSemantic = se.semanticScore;
				}
			}
			if (topSemantic > 0 && lowestSemantic / topSemantic >= 0.90) {
				candidates = filterByCoherence(candidates);
			}
		}

		// Phase 2: Zero-keyword validation — when no surviving candidate
		// has keyword support, the result set is purely semantic and
		// must pass two orthogonal confidence checks. Keywords are
		// direct evidence of relevance (the record literally contains
		// the queried term) and bypass this gate entirely.
		//
		// The two checks catch complementary noise patterns:
		//
		// - Coherence: are the candidates about the same topic? Rejects
		//   scattered false positives about unrelated topics (e.g.
		//   Female Infertility + Granuloma Annulare for an HIV query).
		//
		// - Z-score: is the model confident? Rejects uniform false
		//   positives about the same irrelevant topic (e.g. 3 pulse
		//   readings for a CD4 query) where mutual coherence is high
		//   but query relevance is low.
		{
			boolean candidatesHaveKeywords = false;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore > 0) {
					candidatesHaveKeywords = true;
					break;
				}
			}
			if (log.isDebugEnabled()) {
				StringBuilder sb = new StringBuilder();
				for (ScoredEmbedding se : candidates) {
					if (sb.length() > 0) sb.append(", ");
					sb.append(se.embedding.getResourceType())
						.append(':').append(se.embedding.getResourceId())
						.append(" sem=").append(String.format("%.4f", se.semanticScore))
						.append(" kw=").append(String.format("%.4f", se.keywordScore));
				}
				log.debug("Phase 2: candidatesHaveKeywords={}, candidates={}: [{}]",
						candidatesHaveKeywords, candidates.size(), sb);
			}
			if (!partialKwValidated && !candidatesHaveKeywords
					&& !candidates.isEmpty()) {
				if (candidates.size() >= 2) {
					candidates = filterByMeanCoherence(candidates, config);
				}
				if (!candidates.isEmpty()
						&& scored.size()
						>= ChartSearchAiConstants.MIN_RECORDS_FOR_Z_SCORE) {
					double sum = 0;
					for (ScoredEmbedding se : scored) {
						sum += se.semanticScore;
					}
					double mean = sum / scored.size();
					double sqSum = 0;
					for (ScoredEmbedding se : scored) {
						double d = se.semanticScore - mean;
						sqSum += d * d;
					}
					double std = Math.sqrt(sqSum / scored.size());

					if (std > 0) {
						// For small clusters (≤3 candidates), use
						// max z-score: each candidate was selected
						// for a reason and a single strong match IS
						// the signal. For larger clusters (≥4), use
						// the median z-score: a single high-scoring
						// false positive can push the max above
						// threshold while the cluster as a whole is
						// noise (e.g. 5 pulse readings for a CD4
						// query where one outlier hits z=2.79 but
						// the median is 2.39). This mirrors Phase 1
						// coherence logic which also treats n≤3
						// differently from n≥4.
						double[] zScores =
								new double[candidates.size()];
						for (int i = 0; i < candidates.size(); i++) {
							zScores[i] = (candidates.get(i).semanticScore
									- mean) / std;
						}
						java.util.Arrays.sort(zScores);
						double zRepresentative = candidates.size() <= 3
								? zScores[zScores.length - 1]
								: zScores[zScores.length / 2];
						log.debug("Zero-keyword z-score check: "
								+ "mean={}, std={}, z={} ({}), "
								+ "threshold={}, candidates={}, scored={}",
								String.format("%.4f", mean),
								String.format("%.4f", std),
								String.format("%.2f", zRepresentative),
								candidates.size() <= 3 ? "max" : "median",
								ChartSearchAiConstants
										.ZERO_KEYWORD_CLUSTER_MIN_Z,
								candidates.size(), scored.size());
						if (zRepresentative < ChartSearchAiConstants
								.ZERO_KEYWORD_CLUSTER_MIN_Z) {
							log.debug("Zero-keyword results rejected");
							candidates = Collections.emptyList();
						}
					}
				}
			}
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
		int semanticCutoff = aboveFloor; // default: include everything above the floor
		for (int i = 1; i < aboveFloor; i++) {
			double gap = semanticSorted.get(i - 1) - semanticSorted.get(i);

			// Only consider cutting after the minimum number of records, and
			// only when we have at least 1 prior gap to compute a baseline
			// average (i - 1 gaps have been accumulated at this point).
			if (i >= minRecords && i >= 2) {
				double avgGap = gapSum / (i - 1);
				if (gap > avgGap * gapMultiplier && gap > minGap) {
					semanticCutoff = i;
					log.debug("Score gap detected at position {}: gap={}, avgGap={}, multiplier={}",
							i, String.format("%.4f", gap), String.format("%.4f", avgGap),
							gapMultiplier);
					break;
				}
			}
			gapSum += gap;
		}

		// Map the semantic cluster back to the combined-score-sorted input.
		// The gap was detected in semantic-score order; find the lowest
		// semantic score in the cluster and include all input records whose
		// semantic score meets this threshold. This ensures the cutoff index
		// aligns with scored[0..cutoff-1] rather than using a position from
		// a differently-sorted list.
		// Find all items in the combined-score list whose semantic score is
		// at or above the cluster boundary. We track the highest index that
		// qualifies so that the returned cutoff is a contiguous prefix
		// (scored[0..cutoff-1]) containing every cluster member.
		double clusterThreshold = semanticSorted.get(semanticCutoff - 1);
		int cutoff = 0;
		for (int i = 0; i < limit; i++) {
			if (scored.get(i).semanticScore >= clusterThreshold) {
				cutoff = i + 1;
			}
		}

		return Math.max(cutoff, Math.min(minRecords, aboveFloor));
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

	/**
	 * Filters out lower-keyword-tier records whose matched query terms
	 * are already fully covered by the higher-keyword tier. This removes
	 * false positives like a program enrollment matching only "active"
	 * when condition records already match "active" + "condition". Records
	 * that match unique terms not covered by the higher tier are kept
	 * (e.g. "weight" records in a "blood pressure weight temperature"
	 * query, since the higher-tier BP records don't match "weight").
	 *
	 * @param candidates the refined candidate set with non-uniform keywords
	 * @param queryTerms the query terms after stopword removal
	 * @param kwMax the maximum keyword score in the candidate set
	 * @return filtered candidates, or the original list if filtering
	 *         would produce fewer than {@code ADAPTIVE_MIN_RECORDS}
	 */
	static List<ScoredEmbedding> filterRedundantKeywordTier(
			List<ScoredEmbedding> candidates, String[] queryTerms,
			double kwMax) {
		// Collect query terms covered by the higher-keyword tier.
		Set<String> coveredTerms = new HashSet<String>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						se.embedding.getTextContent());
				String lower = text.toLowerCase();
				String[] words = lower.split("\\s+");
				for (String term : queryTerms) {
					if (termMatchesText(term, lower, words)) {
						coveredTerms.add(term);
					}
				}
			}
		}

		// Keep lower-tier records only if they match a term the higher
		// tier doesn't cover.
		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				filtered.add(se);
			} else {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						se.embedding.getTextContent());
				String lower = text.toLowerCase();
				String[] words = lower.split("\\s+");
				boolean addsNewCoverage = false;
				for (String term : queryTerms) {
					if (!coveredTerms.contains(term)
							&& termMatchesText(term, lower, words)) {
						addsNewCoverage = true;
						break;
					}
				}
				if (addsNewCoverage) {
					filtered.add(se);
				}
			}
		}
		if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			log.debug("Keyword tier subset filter: {} -> {} "
					+ "(coveredTerms={})",
					candidates.size(), filtered.size(), coveredTerms);
			return filtered;
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

	private static double getGapValidationCosineThreshold() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(
						ChartSearchAiConstants.GP_EMBEDDING_GAP_VALIDATION_COSINE_THRESHOLD);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid gapValidationCosineThreshold value '{}', "
						+ "using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD;
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
	 * Prepares the full embedding input string from a raw question.
	 * This is the production pipeline: strip stopwords, build the
	 * embedding query, and prepend the query prefix. Both production
	 * code and tests should use this method to ensure consistency.
	 *
	 * @param question the raw user question (e.g. "any cancer?")
	 * @param queryPrefix the prefix to prepend (e.g. "" or "search_query: ")
	 * @return the text to pass to the embedding provider
	 */
	static String prepareEmbeddingInput(String question, String queryPrefix) {
		String normalized = stripQueryStopwords(question);
		String embeddingQuery = buildEmbeddingQuery(normalized);
		return queryPrefix + embeddingQuery;
	}

	/**
	 * Builds the query string used for embedding by stripping stopwords
	 * from the normalized query. This removes filler words like "any" or
	 * "does" that dilute the embedding signal, while keeping all
	 * non-stopword tokens (including short terms like numbers).
	 * Falls back to the full normalized query when all words are stopwords.
	 */
	private static String buildEmbeddingQuery(String normalizedQuery) {
		String[] words = normalizedQuery.split("\\s+");
		StringBuilder sb = new StringBuilder();
		for (String w : words) {
			if (!QUERY_STOPWORDS.contains(w.toLowerCase())) {
				if (sb.length() > 0) {
					sb.append(" ");
				}
				sb.append(w);
			}
		}
		return sb.length() > 0 ? sb.toString() : normalizedQuery;
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
			if (termMatchesText(term, lowerText, textWords)) {
				matched++;
			}
		}
		return (double) matched / queryTerms.length;
	}

	/**
	 * Checks whether a single query term matches within the given text
	 * using three matching strategies: exact substring, plural stem,
	 * and morphological stem. This is the shared matching logic used
	 * by both {@link #computeKeywordScore} and the keyword-tier
	 * subset check in the filter pipeline.
	 *
	 * @param term the lowercase query term to match
	 * @param lowerText the lowercase full text to search
	 * @param textWords the lowercase text split into words
	 * @return true if the term matches the text by any strategy
	 */
	static boolean termMatchesText(String term, String lowerText, String[] textWords) {
		// 1. Exact substring match
		if (lowerText.contains(term)) {
			return true;
		}

		// 2. Plural stem: strip trailing 's' and match as a whole word
		// (not substring) so "order" matches "order:" but not "ordered".
		if (term.length() > 3
				&& term.endsWith("s") && !term.endsWith("ss")) {
			String stem = term.substring(0, term.length() - 1);
			for (String word : textWords) {
				// Strip trailing punctuation so "order:" becomes "order"
				String clean = word.replaceAll("[^a-z0-9]+$", "");
				if (clean.equals(stem)) {
					return true;
				}
			}
		}

		// 3. Morphological stem: trim 2-3 trailing characters to handle
		// derivational variants (allergic/allergy, prescribed/prescription).
		// Uses word-prefix matching instead of substring to avoid false
		// positives from compound words (e.g. "allerg" inside "Photoallergy").
		if (term.length() >= 7) {
			for (int trim = 2; trim <= 3; trim++) {
				String stem = term.substring(0, term.length() - trim);
				if (stem.length() >= 5) {
					for (String word : textWords) {
						if (word.startsWith(stem)) {
							return true;
						}
					}
				}
			}
		}

		return false;
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
		return ChartSearchAiUtils.cosineSimilarity(a, b);
	}

	/**
	 * Grows a cluster starting from the initial {@code seedSize} records by
	 * adding candidates whose embedding has cosine similarity &ge; threshold
	 * with at least one existing cluster member. Iterates until no more
	 * records can be added (fixed-point). Records are tested in score order
	 * (highest first) so higher-scoring candidates are prioritized.
	 *
	 * @param candidates all candidates (sorted by combined score descending)
	 * @param seedSize number of initial cluster records (the top seedSize)
	 * @param cosineThreshold minimum cosine similarity to join the cluster
	 * @return the grown cluster, preserving the original score order
	 */
	static List<ScoredEmbedding> growCluster(List<ScoredEmbedding> candidates,
			int seedSize, double cosineThreshold) {
		if (seedSize >= candidates.size()) {
			return candidates;
		}
		boolean[] inCluster = new boolean[candidates.size()];
		float[][] vectors = new float[candidates.size()][];
		for (int i = 0; i < candidates.size(); i++) {
			vectors[i] = candidates.get(i).embedding.getEmbeddingVector();
			inCluster[i] = i < seedSize;
		}

		// Iteratively add candidates that are coherent with the cluster
		boolean added = true;
		while (added) {
			added = false;
			for (int i = seedSize; i < candidates.size(); i++) {
				if (inCluster[i] || vectors[i] == null) {
					continue;
				}
				for (int j = 0; j < candidates.size(); j++) {
					if (!inCluster[j] || vectors[j] == null
							|| vectors[j].length != vectors[i].length) {
						continue;
					}
					if (cosineSimilarity(vectors[i], vectors[j]) >= cosineThreshold) {
						inCluster[i] = true;
						added = true;
						break;
					}
				}
			}
		}

		List<ScoredEmbedding> result = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < candidates.size(); i++) {
			if (inCluster[i]) {
				result.add(candidates.get(i));
			}
		}
		return result;
	}

	/**
	 * Validates whether a gap in the score distribution at the given cutoff
	 * position is intra-topic (both sides belong to the same broad topic)
	 * by computing the average cosine similarity between records just above
	 * and just below the gap boundary. Uses a small window (up to 3
	 * records on each side) to get a robust estimate that doesn't depend
	 * on a single pair.
	 *
	 * <p>When the average cross-boundary cosine meets or exceeds the
	 * threshold, the gap is intra-topic and should NOT be used as a cutoff.
	 * For example, a gap between Respiratory Rate records (high score) and
	 * SpO2 records (slightly lower score) has high cross-boundary cosine
	 * because both are clinical vital sign observations.
	 *
	 * @param scored the sorted candidate list
	 * @param cutoff the gap position (records 0..cutoff-1 above, cutoff+ below)
	 * @param cosineThreshold minimum average cosine for the gap to be intra-topic
	 * @return true if the gap is intra-topic (should NOT cut here)
	 */
	static boolean isGapCoherent(List<ScoredEmbedding> scored, int cutoff,
			double cosineThreshold) {
		if (cutoff <= 0 || cutoff >= scored.size()) {
			return false;
		}

		// Use a small window on each side of the gap for robustness
		int windowSize = Math.min(3, Math.min(cutoff, scored.size() - cutoff));
		double sumCosine = 0;
		int pairs = 0;

		for (int a = cutoff - windowSize; a < cutoff; a++) {
			float[] vecA = scored.get(a).embedding.getEmbeddingVector();
			if (vecA == null) {
				continue;
			}
			for (int b = cutoff; b < cutoff + windowSize; b++) {
				float[] vecB = scored.get(b).embedding.getEmbeddingVector();
				if (vecB == null || vecB.length != vecA.length) {
					continue;
				}
				sumCosine += cosineSimilarity(vecA, vecB);
				pairs++;
			}
		}

		if (pairs == 0) {
			return false;
		}

		double avgCosine = sumCosine / pairs;
		return avgCosine >= cosineThreshold;
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

		// Extract embedding vectors for the shared coherence filter
		float[][] vectors = new float[n][];
		for (int i = 0; i < n; i++) {
			byte[] raw = candidates.get(i).embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				return candidates;
			}
			vectors[i] = candidates.get(i).embedding.getEmbeddingVector();
		}

		boolean[] keep = ChartSearchAiUtils.filterByCoherence(vectors);

		int keepCount = 0;
		for (boolean k : keep) {
			if (k) keepCount++;
		}
		if (keepCount == n) {
			return candidates;
		}

		log.debug("Coherence filter removed {} outlier(s)", n - keepCount);

		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < n; i++) {
			if (keep[i]) {
				filtered.add(candidates.get(i));
			}
		}
		return filtered;
	}

	/**
	 * Checks whether a small set of candidates (n ≤ 3) are mutually
	 * coherent by computing mean pairwise cosine similarity. If the mean
	 * is below the gap validation threshold, all candidates are rejected
	 * as false positives — they are unrelated topics that individually
	 * scored well against the query but are not coherent as a group.
	 *
	 * <p>This complements {@link #filterByCoherence}, which uses gap
	 * detection and cannot remove candidates when coherence values are
	 * uniformly low (stddev ≈ 0). Mean pairwise coherence catches the
	 * case where ALL candidates are false positives (e.g. blood
	 * transfusion, syphilitic cirrhosis, and granuloma annulare for
	 * "has the patient ever been immunized?").
	 */
	static List<ScoredEmbedding> filterByMeanCoherence(
			List<ScoredEmbedding> candidates, PipelineConfig config) {
		boolean allHaveEmbeddings = true;
		for (ScoredEmbedding se : candidates) {
			byte[] raw = se.embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				allHaveEmbeddings = false;
				break;
			}
		}
		if (!allHaveEmbeddings) {
			return candidates;
		}

		double sumCosine = 0;
		int pairCount = 0;
		for (int i = 0; i < candidates.size(); i++) {
			for (int j = i + 1; j < candidates.size(); j++) {
				sumCosine += cosineSimilarity(
						candidates.get(i).embedding.getEmbeddingVector(),
						candidates.get(j).embedding.getEmbeddingVector());
				pairCount++;
			}
		}
		double meanCoherence = sumCosine / pairCount;
		if (meanCoherence < config.gapValidationCosineThreshold) {
			log.debug("{} candidates are not mutually coherent "
					+ "(meanCosine={}, threshold={}), returning empty",
					candidates.size(),
					String.format("%.4f", meanCoherence),
					config.gapValidationCosineThreshold);
			return Collections.emptyList();
		}
		return candidates;
	}

	/**
	 * Rescues records that scored below the similarity floor but are
	 * coherent with the surviving candidate cluster. Computes each
	 * below-floor record's average cosine similarity to the cluster
	 * members and includes it if that coherence is at least as high as
	 * the lowest coherence within the cluster itself. This ensures
	 * rescued records are genuinely topically identical (e.g. temperature
	 * readings for a "vitals" query) rather than unrelated records that
	 * happened to score below the floor.
	 *
	 * @param candidates the post-coherence-filtered cluster
	 * @param scored all scored records (including below-floor ones)
	 * @param adaptiveCutoff the first-pass cutoff (records above the floor)
	 * @return the original candidates plus any rescued below-floor records
	 */
	static List<ScoredEmbedding> rescueBelowFloor(List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored, int adaptiveCutoff) {
		if (adaptiveCutoff >= scored.size()) {
			return candidates;
		}

		// Collect cluster embedding vectors
		float[][] clusterVecs = new float[candidates.size()][];
		boolean allValid = true;
		for (int i = 0; i < candidates.size(); i++) {
			byte[] raw = candidates.get(i).embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				allValid = false;
				break;
			}
			clusterVecs[i] = candidates.get(i).embedding.getEmbeddingVector();
		}
		if (!allValid) {
			return candidates;
		}

		// Compute the minimum within-cluster coherence as the rescue threshold.
		// A below-floor record must be at least as coherent with the cluster
		// as the least coherent existing member.
		double minClusterCoherence = Double.MAX_VALUE;
		for (int i = 0; i < clusterVecs.length; i++) {
			double sum = 0;
			for (int j = 0; j < clusterVecs.length; j++) {
				if (j != i) {
					sum += cosineSimilarity(clusterVecs[i], clusterVecs[j]);
				}
			}
			double coherence = sum / (clusterVecs.length - 1);
			if (coherence < minClusterCoherence) {
				minClusterCoherence = coherence;
			}
		}

		// Identify which records are already in the cluster
		Set<Integer> inCluster = new HashSet<Integer>();
		for (ScoredEmbedding se : candidates) {
			inCluster.add(se.embedding.getEmbeddingId());
		}

		// Check below-floor records for coherence with the cluster
		List<ScoredEmbedding> rescued = new ArrayList<ScoredEmbedding>();
		for (int i = adaptiveCutoff; i < scored.size(); i++) {
			ScoredEmbedding se = scored.get(i);
			if (inCluster.contains(se.embedding.getEmbeddingId())) {
				continue;
			}
			byte[] raw = se.embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				continue;
			}
			float[] vec = se.embedding.getEmbeddingVector();
			double sum = 0;
			for (float[] cv : clusterVecs) {
				sum += cosineSimilarity(vec, cv);
			}
			double coherence = sum / clusterVecs.length;
			if (coherence >= minClusterCoherence) {
				rescued.add(se);
			}
		}

		if (rescued.isEmpty()) {
			return candidates;
		}

		log.debug("Below-floor rescue: recovered {} records "
				+ "(minClusterCoherence={})",
				rescued.size(), String.format("%.4f", minClusterCoherence));
		List<ScoredEmbedding> result = new ArrayList<ScoredEmbedding>(candidates);
		result.addAll(rescued);
		return result;
	}

	/**
	 * Holds all configurable pipeline parameters, decoupling the filtering
	 * logic from the OpenMRS Context so the pipeline can be tested without
	 * a running application server.
	 */
	static class PipelineConfig {
		final double keywordWeight;
		final double scoreGapMultiplier;
		final double minScoreGap;
		final double gapValidationCosineThreshold;
		final double similarityRatio;

		PipelineConfig(double keywordWeight, double scoreGapMultiplier,
				double minScoreGap, double gapValidationCosineThreshold,
				double similarityRatio) {
			this.keywordWeight = keywordWeight;
			this.scoreGapMultiplier = scoreGapMultiplier;
			this.minScoreGap = minScoreGap;
			this.gapValidationCosineThreshold = gapValidationCosineThreshold;
			this.similarityRatio = similarityRatio;
		}

		/** Returns a config using all default constant values. */
		static PipelineConfig defaults() {
			return new PipelineConfig(
					ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
					ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
					ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP,
					ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
					ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO);
		}
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
