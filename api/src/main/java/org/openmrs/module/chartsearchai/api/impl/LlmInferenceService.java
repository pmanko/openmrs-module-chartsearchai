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

import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.FLOOR_RESCUE_MIN_Z_SCORE;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.ElasticsearchIndexer;
import org.openmrs.module.chartsearchai.api.HybridRetriever;
import org.openmrs.module.chartsearchai.api.LuceneIndexer;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile;
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

	/** Matches a definite-article recency phrase without a number, e.g.
	 *  "the latest weight" or "the most recent BP". The definite article
	 *  signals that the user expects a single (the most recent) result,
	 *  unlike bare "latest vital signs" which is a synonym for "recent".
	 *  Implies a cap of 1. */
	private static final Pattern BARE_RECENCY_PATTERN = Pattern.compile(
			"\\bthe\\s+(?:latest|most recent)\\b", Pattern.CASE_INSENSITIVE);

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
	 * digits and word numbers (one through ten). A bare recency keyword
	 * without a number (e.g. "latest weight", "most recent BP") implies 1.
	 * Returns 0 if no constraint is found.
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
		// Bare recency keyword without a number implies cap of 1.
		if (BARE_RECENCY_PATTERN.matcher(question).find()) {
			return 1;
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
	 * Filters {@code allRecords} to those whose resource keys are in
	 * {@code relevantKeys}, then applies the recency cap derived from
	 * {@code question}. The input {@code allRecords} list must already be
	 * sorted most-recent-first so that {@link #capPerConcept} keeps the
	 * latest record per concept group.
	 *
	 * <p>Shared by {@link #filterAndSerialize} (production) and
	 * {@link #findRelevantRecords} (the composed retrieval entry point used
	 * by tests), so both paths apply identical filter + cap semantics.
	 */
	static List<SerializedRecord> filterAndCap(
			List<SerializedRecord> allRecords,
			Set<String> relevantKeys, String question) {
		List<SerializedRecord> filtered = new ArrayList<SerializedRecord>();
		for (SerializedRecord record : allRecords) {
			if (relevantKeys.contains(ChartSearchAiUtils.resourceKey(
					record.getResourceType(), record.getResourceId()))) {
				filtered.add(record);
			}
		}
		int recencyCap = extractRecencyCap(question);
		if (recencyCap > 0) {
			filtered = capPerConcept(filtered, recencyCap);
		}
		return filtered;
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
		List<SerializedRecord> filtered = filterAndCap(allRecords, relevantKeys, question);

		log.debug("Pre-filtered {} records to {} using {}",
				allRecords.size(), filtered.size(),
				isHybridPipeline() ? "Hybrid" :
						isElasticsearchPipeline() ? "Elasticsearch" :
								isLucenePipeline() ? "Lucene" : "embeddings");

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
				getGapValidationCosineThreshold(), getSimilarityRatio(),
				ModelNoiseProfile.conservativeDefault());

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
	 * @param topK target result count — applied only when some candidates
	 *        lack keyword matches; bypassed when every candidate has a
	 *        keyword match
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
			// Strip synonym parentheticals before keyword scoring so
			// that "(syn. Hemoglobin performed on blood)" doesn't cause
			// "blood" to match Haemoglobin records. Synonyms stay in the
			// embedding for semantic matching — only keyword scoring uses
			// the stripped text.
			String keywordText = ChartSearchAiUtils.buildPrefixedText(
					ce.getResourceType(),
					ConceptNameUtil.stripSynonyms(ce.getTextContent()));
			keywordScores[validCount] = computeKeywordScore(queryTerms, keywordText);
			validCount++;
		}
		if (validCount < embeddings.length) {
			embeddings = java.util.Arrays.copyOf(embeddings, validCount);
			semanticScores = java.util.Arrays.copyOf(semanticScores, validCount);
			keywordScores = java.util.Arrays.copyOf(keywordScores, validCount);
		}

		// Compute noise profile from the patient's embeddings so all
		// pipeline thresholds adapt to this embedding model's geometry.
		ModelNoiseProfile noiseProfile =
				ModelNoiseProfile.compute(embeddings);
		log.debug("NoiseProfile: Q1={} median={} mean={} P95={} "
				+ "intraMean={} floor={}",
				String.format("%.4f", noiseProfile.noiseQ1),
				String.format("%.4f", noiseProfile.noiseMedian),
				String.format("%.4f", noiseProfile.noiseMean),
				String.format("%.4f", noiseProfile.noiseP95),
				String.format("%.4f", noiseProfile.intraConceptMean),
				String.format("%.4f", noiseProfile.absoluteSimilarityFloor()));
		PipelineConfig profiledConfig = new PipelineConfig(
				config.keywordWeight, config.scoreGapMultiplier,
				config.minScoreGap,
				config.gapValidationCosineThreshold,
				config.similarityRatio, noiseProfile);

			return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, topK, profiledConfig);
	}

	/**
	 * Composed retrieval pipeline: runs {@link #findSimilar}, then
	 * {@link #filterAndCap}, then {@link #groupByConcept} — the same
	 * post-retrieval sequence that production's {@code filterAndSerialize}
	 * runs before handing records to {@code chartSerializer.serialize}.
	 *
	 * <p>This is the canonical entry point for tests: it exercises every
	 * step of the retrieval pipeline that doesn't require Spring/DAO
	 * infrastructure, so pipeline-assembly bugs (e.g. forgetting to call
	 * one of the post-retrieval helpers) can't slip past tests.
	 *
	 * @param allEmbeddings the patient's chart embeddings (most-recent-first)
	 * @param allRecords the patient's serialized records (most-recent-first
	 *        — used to apply the recency cap in date order)
	 * @param provider embedding provider for query vectorization
	 * @param question the natural language question
	 * @param topK target result count for {@link #findSimilar}
	 * @param queryPrefix prefix prepended to the query before embedding
	 * @param config pipeline tuning parameters
	 * @return filtered, capped, and concept-grouped serialized records,
	 *         or {@code null} if {@code findSimilar} returned {@code null}
	 *         (no embeddings)
	 */
	static List<SerializedRecord> findRelevantRecords(
			List<ChartEmbedding> allEmbeddings,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, String question, int topK,
			String queryPrefix, PipelineConfig config) {
		List<ChartEmbedding> similar = findSimilar(allEmbeddings, provider,
				question, topK, queryPrefix, config);
		if (similar == null) {
			return null;
		}
		if (similar.isEmpty()) {
			return Collections.emptyList();
		}
		Set<String> relevantKeys = new HashSet<String>();
		for (ChartEmbedding ce : similar) {
			relevantKeys.add(ChartSearchAiUtils.resourceKey(
					ce.getResourceType(), ce.getResourceId()));
		}
		return groupByConcept(filterAndCap(allRecords, relevantKeys, question));
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
	 * @param topK target result count — applied only when some candidates
	 *        lack keyword matches; bypassed when every candidate has a
	 *        keyword match (gap detection + ratio floor already identified
	 *        the relevant cluster in that case)
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

		int keywordMatchCount = 0;
		for (ScoredEmbedding se : scored) {
			if (se.keywordScore > 0) {
				keywordMatchCount++;
			}
		}

		// Floor gate: if neither the best semantic score nor the best
		// combined score reaches the floor, attempt a z-score + cluster-
		// density rescue. Returns null to short-circuit with an empty
		// result, or the belowFloorRescued flag to continue.
		Boolean floorResult = applyFloorGate(scored, maxSemanticScore,
				maxBaseScore, queryTermCount, keywordMatchCount, config);
		if (floorResult == null) {
			return Collections.emptyList();
		}
		boolean belowFloorRescued = floorResult;

		// Slim-margin gate: reject suspicious single-record clusters
		// just above the noise floor with zero keyword matches.
		if (!applySlimMarginGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, belowFloorRescued,
				config)) {
			return Collections.emptyList();
		}

		// Initial z-score gate: reject when few records have keyword
		// matches and the top semantic score isn't a statistical outlier.
		// Returns null to short-circuit, or {zScore, threshold} that are
		// reused later by the cluster z-score gate.
		double[] zScoreState = applyInitialZScoreGate(scored,
				maxSemanticScore, queryTermCount, keywordMatchCount);
		if (zScoreState == null) {
			return Collections.emptyList();
		}
		double initialZScore = zScoreState[0];
		double initialZThreshold = zScoreState[1];

		// Gap detection + candidate building: find the semantic cluster
		// boundary and build the candidate set (records above the boundary
		// OR with full keyword match).
		CandidateBuildResult built = buildCandidates(scored,
				maxSemanticScore, bonusThreshold, config);
		double minScore = built.minScore;
		int adaptiveCutoff = built.adaptiveCutoff;
		List<ScoredEmbedding> candidates = built.candidates;

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
		boolean firstPassGapDetected = adaptiveCutoff < scored.size();
		// Set in the non-refinement path after the ratio floor filters
		// candidates. Used by the cluster z-score gate: when the ratio
		// floor alone produces a tight set (< 25% of total), it is
		// structural evidence of a genuine semantic cluster — analogous
		// to gap detection. Without this, umbrella queries like
		// "vital signs" (which map to Pulse, Respiratory Rate, etc.
		// but share no keywords with any record) are incorrectly
		// rejected because the smooth score distribution prevents
		// gap detection from firing.
		int ratioFloorCandidateCount = -1;

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
			// Semantic dominance: non-keyword records have higher
			// semantic scores than keyword-matched records, indicating
			// the keyword matches are likely false positives — e.g.
			// "rate" matching "Respiratory rate" when the query means
			// "heart rate" (= Pulse). When true, the semantic core
			// path should run AND skip expansion to prevent template-
			// similar false positives from re-entering via cosine.
			boolean semanticDominance = false;
			double maxKwSem = 0;
			double maxNonKwSem = 0;
			if (uniformKeywords && kwMax < bonusThresh
					&& queryTermCount > 0) {
				double refMinKw = 1.0 / queryTermCount;
				for (ScoredEmbedding se : preRefinementCandidates) {
					if (se.keywordScore >= refMinKw) {
						if (se.semanticScore > maxKwSem) {
							maxKwSem = se.semanticScore;
						}
					} else {
						if (se.semanticScore > maxNonKwSem) {
							maxNonKwSem = se.semanticScore;
						}
					}
				}
				semanticDominance = maxNonKwSem > maxKwSem;
			}
			boolean partialKeywordMatch = uniformKeywords
					&& kwMax < bonusThresh;
			log.debug("Pipeline refinement: kwMin={}, kwMax={}, uniform={}, "
					+ "partialKw={}, semDom={}",
					 String.format("%.4f", kwMin),
					 String.format("%.4f", kwMax), uniformKeywords,
					 partialKeywordMatch, semanticDominance);
			if (partialKeywordMatch) {
			  // Always compute the semantic core from non-keyword
			  // records. Gap detection tells us whether there is
			  // genuine structure (a cluster boundary separating
			  // relevant non-kw records from noise) vs a uniform
			  // distribution (no cluster = core is unreliable).
			  // This data-driven check replaces the former hardcoded
			  // kwMax >= 0.5 threshold.
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
				boolean coreFallbackUsed = false;
				int coreCutoff = nonKeyword.size();
				if (nonKeyword.size() >= ChartSearchAiConstants
						.ADAPTIVE_MIN_RECORDS) {
					double maxSemNk = nonKeyword.get(0).semanticScore;
					double nkMinGap = Math.max(
							maxSemNk * refinementAdaptiveGapRatio(
									scored),
							secondPassMinGap(scored));
					coreCutoff = findAdaptiveCutoff(nonKeyword,
							nonKeyword.size(),
							config.noiseProfile.absoluteSimilarityFloor(),
							config.scoreGapMultiplier, nkMinGap);

					// Gap-weighted Welch's t²: find the split
					// that maximizes gap × t². The gap factor
					// ensures we cut at actual spacing jumps
					// (not tight pairs). The t² factor ensures
					// the two classes (core vs tail) are well-
					// separated relative to their internal
					// spread. Uses the Bessel correction
					// (sample variance, n−1) which naturally
					// penalizes small classes and requires
					// n ≥ 2 per class — no arbitrary thresholds
					// needed for tiny datasets. The optimal
					// split is fully determined by the score
					// distribution.
					int fisherCut = nonKeyword.size();
					double totalSumF = 0;
					double totalSumSqF = 0;
					for (ScoredEmbedding se : nonKeyword) {
						totalSumF += se.semanticScore;
						totalSumSqF += se.semanticScore
								* se.semanticScore;
					}
					int nF = nonKeyword.size();
					double leftSumF = 0;
					double leftSumSqF = 0;
					double rightSumF = totalSumF;
					double rightSumSqF = totalSumSqF;
					double maxGF = 0;
					for (int i = 1; i < nF; i++) {
						double s = nonKeyword.get(i - 1)
								.semanticScore;
						leftSumF += s;
						leftSumSqF += s * s;
						rightSumF -= s;
						rightSumSqF -= s * s;
						if (i < ChartSearchAiConstants
								.ADAPTIVE_MIN_RECORDS) {
							continue;
						}
						int nR = nF - i;
						// Welch's t² requires n ≥ 2 per
						// class for sample variance to be
						// defined. This also prevents
						// arbitrary cuts that leave a
						// single record in the tail.
						if (nR < 2) {
							break;
						}
						double lm = leftSumF / i;
						double rm = rightSumF / nR;
						// Population variance (σ²)
						double lv = leftSumSqF / i
								- lm * lm;
						double rv = rightSumSqF / nR
								- rm * rm;
						// Bessel correction: sample
						// variance / n = σ² / (n−1).
						// This doubles the denominator
						// contribution for n=2, naturally
						// penalizing small classes.
						double den = lv / (i - 1)
								+ rv / (nR - 1) + 1e-12;
						double diff = lm - rm;
						double t2 = diff * diff / den;
						double bGap = nonKeyword.get(i - 1)
								.semanticScore
								- nonKeyword.get(i)
										.semanticScore;
						double gf = bGap * t2;
						if (gf > maxGF) {
							maxGF = gf;
							fisherCut = i;
						}
					}
					log.debug("Fisher cluster cut at {}"
							+ " (score={})", fisherCut,
							String.format("%.2f", maxGF));
					// Use the tighter of primary gap detection
					// and Fisher cluster detection, but only
					// if the boundary gap is a local increase
					// in spacing (larger than the within-
					// cluster gap immediately above it). This
					// prevents Fisher from cutting at the
					// start of smooth exponential decays where
					// gaps decrease monotonically — there is
					// no cluster boundary in such
					// distributions, just the natural shape of
					// the score falloff.
					if (fisherCut < coreCutoff) {
						double bGapF = nonKeyword
								.get(fisherCut - 1)
								.semanticScore
								- nonKeyword.get(fisherCut)
										.semanticScore;
						double prevGapF = nonKeyword
								.get(fisherCut - 2)
								.semanticScore
								- nonKeyword
										.get(fisherCut - 1)
										.semanticScore;
						if (bGapF > prevGapF) {
							coreFallbackUsed = true;
							coreCutoff = fisherCut;
						} else {
							log.debug("Fisher cut at {} "
									+ "suppressed: boundary"
									+ " gap {}"
									+ " <= prev gap {}",
									fisherCut,
									String.format("%.4f",
											bGapF),
									String.format("%.4f",
											prevGapF));
						}
					}

					for (int i = 0; i < coreCutoff; i++) {
						semanticCore.add(nonKeyword.get(i));
					}

					log.debug("Semantic core: {} non-keyword"
							+ " records, primary gap at {},"
							+ " core size {}",
							nonKeyword.size(), coreCutoff,
							semanticCore.size());
				}

				// Core has structure when gap detection found a
				// boundary (coreCutoff < total non-kw count),
				// meaning the top cluster is a genuine semantic
				// group — not just the entire non-kw distribution.
				boolean coreHasStructure = !semanticCore.isEmpty()
						&& coreCutoff < nonKeyword.size();

				// The core is useful when (a) gap detection found
				// structure AND (b) the core is topically related
				// to the keyword-matched records. Topical
				// relatedness = max cosine between the core's top
				// record and ANY keyword record. Using max (not
				// top-1) is robust to sort order. Threshold is
				// SEMANTIC_CORE_MIN_COSINE (0.55) — the same
				// constant the expansion step uses. Empirically:
				// blood problem: Anaemia ↔ kw max=0.62-0.68
				// (same blood topic, passes); STD: Temperature ↔
				// kw max=0.47-0.50 (unrelated, fails).
				boolean coreRelevant = false;
				if (coreHasStructure && !semanticCore.isEmpty()
						&& !candidates.isEmpty()) {
					// Check if the core's top record is topically
					// related to ANY keyword-matched record. Using
					// the max cosine is robust to which kw record
					// happens to sort first.
					float[] coreVec = semanticCore.get(0)
							.embedding.getEmbeddingVector();
					double maxCoreKwCosine = 0;
					for (ScoredEmbedding se : candidates) {
						double cos = cosineSimilarity(coreVec,
								se.embedding
										.getEmbeddingVector());
						if (cos > maxCoreKwCosine) {
							maxCoreKwCosine = cos;
						}
					}
					coreRelevant = maxCoreKwCosine
							>= ChartSearchAiConstants
									.SEMANTIC_CORE_MIN_COSINE;
					log.debug("Core topical check: maxCos={}"
							+ " vs threshold={}, relevant={}",
							String.format("%.4f",
									maxCoreKwCosine),
							ChartSearchAiConstants
									.SEMANTIC_CORE_MIN_COSINE,
							coreRelevant);
				}
				log.debug("Core validation: structure={}, "
						+ "relevant={}, cutoff={}/{}, semDom={}",
						coreHasStructure, coreRelevant,
						coreCutoff, nonKeyword.size(),
						semanticDominance);

				if (semanticDominance || coreRelevant) {
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
					// When the core was tightened by min-gap
					// cluster detection, use coreMin/coreMax
					// as the expansion ratio so the floor
					// scales with the detected cluster width.
					double coreMinSem = semanticCore.get(
							semanticCore.size() - 1)
							.semanticScore;
					double coreMaxSem = semanticCore.get(0)
							.semanticScore;
					// When coreFallback was used (min-gap cluster
					// detection tightened the core), use the core's
					// own min/max ratio — it reflects the detected
					// cluster's actual width. Otherwise, use the
					// refinement semantic ratio from the broader
					// candidate set, which naturally scales the
					// expansion floor with the score distribution.
					double expansionRatio = coreFallbackUsed
							? (coreMaxSem > 0
									? coreMinSem / coreMaxSem
									: ChartSearchAiConstants
											.SEMANTIC_CORE_SCORE_RATIO)
							: ChartSearchAiConstants
									.SEMANTIC_CORE_SCORE_RATIO;
					double scoreFloor = coreMinSem
							* expansionRatio;
					List<ScoredEmbedding> expanded =
							new ArrayList<ScoredEmbedding>(
									semanticCore);
					Set<Integer> expandedIds =
							new java.util.HashSet<Integer>();
					for (ScoredEmbedding se : semanticCore) {
						expandedIds.add(
								se.embedding.getResourceId());
					}
					// When semantic dominance triggered this
					// path, the core IS the answer — expansion
					// would only add template-similar noise (e.g.
					// BP records for a "heart rate" query). Skip
					// expansion entirely; the core already
					// contains the correct records.
					if (!semanticDominance) {
						for (ScoredEmbedding se
								: preRefinementCandidates) {
							if (expandedIds.contains(
									se.embedding
											.getResourceId())) {
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
								double cos = cosineSimilarity(
										vec,
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
										se.embedding
												.getResourceId());
							}
						}
					}
					candidates = expanded;
					log.debug("Semantic core expansion:"
							+ " core={}, floor={}, result={}",
							semanticCore.size(),
							String.format("%.4f", scoreFloor),
							candidates.size());
					// Cap: generic keywords (e.g. "disease")
					// produce broad cores or over-expansion.
					// Apply topK as a safety cap, same as the
					// non-refinement path, unless every
					// candidate has keyword support.
					if (candidates.size() > topK) {
						boolean allExpHaveKw = true;
						for (ScoredEmbedding se : candidates) {
							if (se.keywordScore == 0) {
								allExpHaveKw = false;
								break;
							}
						}
						if (!allExpHaveKw) {
							candidates =
									new ArrayList<ScoredEmbedding>(
											candidates.subList(
													0, topK));
							log.debug("Semantic core topK cap:"
									+ " {} -> {}", expanded.size(),
									topK);
						}
					}
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
							* refinementSemanticRatio(
									candidates);
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
				} else {
				// No core structure found — fall back to the
				// semantic ratio floor on keyword matches.
				// When gap detection finds no boundary in the
				// non-keyword distribution, the "core" is the
				// entire non-kw set (noise, not a cluster).
				// Apply a ratio floor to the keyword-matched
				// candidates instead.
				double maxSemanticKw = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > maxSemanticKw) {
						maxSemanticKw = se.semanticScore;
					}
				}
				double semFloor = maxSemanticKw
						* config.similarityRatio;
				List<ScoredEmbedding> filtered =
						new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore >= semFloor) {
						filtered.add(se);
					}
				}
				if (filtered.size() >= ChartSearchAiConstants
						.ADAPTIVE_MIN_RECORDS) {
					log.debug("Low-coverage kw semantic floor:"
							+ " {} -> {} (floor={})",
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
						maxSemanticRefined * refinementAdaptiveGapRatio(
								scored),
						secondPassMinGap(scored));
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
								* refinementSemanticRatio(candidates);
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
			// Non-refinement path: second-pass gap detection, ratio
			// floor with concept-pairing rescue, and topK safety net.
			int[] rfccOut = { -1 };
			candidates = applyNonRefinementPath(candidates, scored,
					maxSemanticScore, maxBaseScore, minScore, topK,
					config, rfccOut);
			ratioFloorCandidateCount = rfccOut[0];
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
		// Also skip when every candidate matches ALL query terms
		// (kwScore >= 1.0). Full keyword match is conclusive
		// evidence that the record belongs to the queried category
		// — e.g. "any conditions?" returns all records containing
		// "condition". The coherence filter measures embedding
		// similarity, which penalizes unusual medical terminology
		// (e.g. "Enteroviral vesicular stomatitis") despite the
		// keyword confirming it IS a condition. Partial keyword
		// matches (kwScore < 1.0) still need coherence filtering
		// because the matched term may be incidental (e.g.
		// "disease" in "Chronic disease management" for an STD
		// query).
		boolean allFullKeywordMatch = true;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore < 1.0) {
				allFullKeywordMatch = false;
				break;
			}
		}
		// Compound-keyword match detection: when candidates partition the
		// query into complementary term subsets (union of matched terms
		// covers every query term, yet no single candidate matches all
		// terms), the candidate set is a legitimate multi-concept result
		// — e.g. "HIV and CD4 count" where HIV records match {hiv} and
		// CD4 records match {cd4, count}. Coherence filtering would drop
		// the minority cluster as outliers even though keyword evidence
		// confirms both concepts belong to the queried set.
		boolean isCompoundKeywordMatch = isCompoundKeywordMatch(
				candidates, queryTerms);
		if (!partialKwValidated && !allFullKeywordMatch
				&& !isCompoundKeywordMatch
				&& candidates.size() >= 4) {
			candidates = filterByCoherence(candidates);
		} else if (!partialKwValidated && !allFullKeywordMatch
				&& !isCompoundKeywordMatch
				&& candidates.size() == 3) {
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
				// Small-cluster coherence gate: for ≤3 zero-keyword
				// candidates, apply a stricter coherence threshold
				// derived from the patient's own embedding statistics.
				// Legitimate small clusters are same-concept groups
				// (condition+diagnosis pairs, often with an obs) whose
				// mutual coherence is close to the intra-concept mean
				// (0.93+). False positives are cross-topic clusters
				// (e.g. 3 substance abuse conditions for a "medications"
				// query) whose coherence (0.48–0.79) falls below the
				// intra-concept range. The threshold (intraConceptMean −
				// intraConceptStd) is the lower bound of same-concept
				// coherence, cleanly separating the two groups.
				if (candidates.size() == 3) {
					double intraFloor =
							config.noiseProfile.intraConceptMean
							- config.noiseProfile.intraConceptStd;
					double sumCos = 0;
					int nPairs = 0;
					for (int ci = 0; ci < candidates.size(); ci++) {
						float[] vi = candidates.get(ci).embedding
								.getEmbeddingVector();
						if (vi == null) {
							continue;
						}
						for (int cj = ci + 1;
								cj < candidates.size(); cj++) {
							float[] vj = candidates.get(cj).embedding
									.getEmbeddingVector();
							if (vj == null) {
								continue;
							}
							sumCos += ChartSearchAiUtils
									.cosineSimilarity(vi, vj);
							nPairs++;
						}
					}
					if (nPairs > 0) {
						double mc = sumCos / nPairs;
						if (mc < intraFloor) {
							// Check if removing the lowest-
							// coherence record leaves a same-
							// concept pair above intraFloor.
							// Phase 1 n=3 skips coherence
							// filtering when scores are spread
							// (ratio < 0.90), so a cross-concept
							// outlier can survive into Phase 2.
							// Rather than rejecting everything,
							// extract the coherent same-concept
							// pair if one exists.
							double[] avgCoh = new double[3];
							for (int ci = 0; ci < 3; ci++) {
								double s = 0;
								int cnt = 0;
								float[] vi = candidates.get(ci)
										.embedding
										.getEmbeddingVector();
								if (vi == null) {
									continue;
								}
								for (int cj = 0; cj < 3; cj++) {
									if (ci == cj) {
										continue;
									}
									float[] vj = candidates
											.get(cj).embedding
											.getEmbeddingVector();
									if (vj == null) {
										continue;
									}
									s += ChartSearchAiUtils
											.cosineSimilarity(
													vi, vj);
									cnt++;
								}
								avgCoh[ci] = cnt > 0
										? s / cnt : 0;
							}
							int worstIdx = 0;
							for (int ci = 1; ci < 3; ci++) {
								if (avgCoh[ci]
										< avgCoh[worstIdx]) {
									worstIdx = ci;
								}
							}
							int p1 = -1, p2 = -1;
							for (int ci = 0; ci < 3; ci++) {
								if (ci != worstIdx) {
									if (p1 < 0) {
										p1 = ci;
									} else {
										p2 = ci;
									}
								}
							}
							float[] v1 = candidates.get(p1)
									.embedding
									.getEmbeddingVector();
							float[] v2 = candidates.get(p2)
									.embedding
									.getEmbeddingVector();
							double pairCos = (v1 != null
									&& v2 != null)
									? ChartSearchAiUtils
											.cosineSimilarity(
													v1, v2)
									: 0;
							String cn1 = ConceptNameUtil
									.extractConceptName(
											candidates
													.get(p1)
													.embedding
													.getTextContent());
							String cn2 = ConceptNameUtil
									.extractConceptName(
											candidates
													.get(p2)
													.embedding
													.getTextContent());
							boolean sameConcept =
									cn1 != null
									&& cn1.equals(cn2);
							if (pairCos >= intraFloor
									&& sameConcept) {
								List<ScoredEmbedding> pair =
										new ArrayList<ScoredEmbedding>();
								pair.add(candidates.get(p1));
								pair.add(candidates.get(p2));
								log.debug("Small-cluster gate:"
										+ " removed outlier"
										+ " [{}], kept pair"
										+ " cos={}, concept={}",
										candidates.get(
												worstIdx)
												.embedding
												.getResourceId(),
										String.format("%.4f",
												pairCos),
										cn1);
								candidates = pair;
							} else {
								log.debug("Small-cluster"
										+ " coherence gate:"
										+ " meanCoherence={}"
										+ " < intraFloor={},"
										+ " returning empty",
										String.format("%.4f",
												mc),
										String.format("%.4f",
												intraFloor));
								candidates = Collections
										.emptyList();
							}
						}
					}
				}
				// Skip the z-score gate when first-pass gap detection found
				// a tight cluster. The gap is structural evidence that the
				// model distinguishes these records from the rest. Without
				// this, compressed-distribution models (e.g. MedCPT,
				// std ≈ 0.03) can never reach the z-score threshold because
				// even the most relevant records are < 2.5σ from the mean.
				// A cluster is "tight" when it captures fewer records than
				// the number of unique concepts in the dataset — a genuine
				// query-relevant cluster targets a few concepts, while a
				// broad swath captures records from many different concepts.
				// The z-score gate is redundant when the floor gate
				// already validated signal via z-score rescue — the
				// top score was proven to be a statistical outlier
				// despite being below the absolute floor.
				Set<String> tightCheckConcepts = new HashSet<String>();
				for (ScoredEmbedding se : scored) {
					String cn = ConceptNameUtil.extractConceptName(
							se.embedding.getTextContent());
					if (cn != null) {
						tightCheckConcepts.add(cn);
					}
				}
				int tightThreshold = tightCheckConcepts.size();
				boolean tightClusterDetected = belowFloorRescued
						|| (firstPassGapDetected
								&& adaptiveCutoff < tightThreshold)
						|| (ratioFloorCandidateCount >= 0
								&& ratioFloorCandidateCount
								< tightThreshold
								&& initialZScore
								>= FLOOR_RESCUE_MIN_Z_SCORE
								&& maxSemanticScore
								>= config.noiseProfile.absoluteSimilarityFloor()
										+ config.minScoreGap);
				// Single-candidate structural gate: when the
				// ratio floor itself produced only one candidate
				// (not a topK cap on a larger set) and there is
				// no tight-cluster evidence, the candidate is an
				// isolated noise peak. Legitimate single matches
				// produce tight clusters because the gap detector
				// isolates them from the bulk distribution.
				if (!tightClusterDetected
						&& candidates.size() == 1
						&& ratioFloorCandidateCount == 1) {
					log.debug("Single zero-keyword candidate "
							+ "with no tight-cluster support "
							+ "and ratioFloor=1, returning "
							+ "empty");
					candidates = Collections.emptyList();
				}
				if (!tightClusterDetected
						&& !candidates.isEmpty()
						&& hasStatisticalVariance(scored)) {
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
						// Two-tier Gumbel threshold:
						// K≤3 (max): Gumbel(uniqueConcepts)
						//   — strict; rescued by initial gate
						// K≥4 (median): Gumbel(uniqueConcepts/2)
						//   — the median scales with the N/2-th
						//   order statistic, so halve effective N
						double clusterZThreshold;
						if (candidates.size() <= 3) {
							clusterZThreshold =
									clusterGumbelThreshold(
											scored);
						} else {
							clusterZThreshold =
									medianGumbelThreshold(
											scored);
						}
						log.debug("Zero-keyword z-score check: "
								+ "mean={}, std={}, z={} ({}), "
								+ "threshold={}, candidates={}, scored={}",
								String.format("%.4f", mean),
								String.format("%.4f", std),
								String.format("%.2f", zRepresentative),
								candidates.size() <= 3 ? "max" : "median",
								String.format("%.2f", clusterZThreshold),
								candidates.size(), scored.size());
						if (zRepresentative < clusterZThreshold) {
							// For K≤3, the initial gate may
							// have already validated the signal.
							// If so, defer to that validation
							// rather than rejecting here.
							if (candidates.size() <= 3
									&& initialZThreshold > 0
									&& initialZScore
									>= initialZThreshold) {
								log.debug("Cluster z-score below "
										+ "threshold but initial gate "
										+ "validated signal, keeping");
							} else {
								log.debug(
									"Zero-keyword results rejected");
								candidates =
										Collections.emptyList();
							}
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

		return Math.max(semanticCutoff, Math.min(minRecords, aboveFloor));
	}

	/**
	 * Floor gate — stage 1 of the filter pipeline.
	 *
	 * <p>When the best semantic and combined scores are both below the
	 * embedding model's absolute similarity floor, the candidate set is
	 * below the noise boundary. Attempts a z-score + cluster-density
	 * rescue that recovers vocabulary-mismatch queries (e.g. "how hot
	 * is the patient?" → Temperature records) where the model correctly
	 * ranks relevant records first but cosine similarity is inherently
	 * low.
	 *
	 * @return {@code null} if the gate rejects (the caller should return
	 *         an empty result); otherwise a boxed boolean indicating
	 *         whether the rescue fired. The caller propagates this as
	 *         the {@code belowFloorRescued} flag to downstream gates.
	 */
	static Boolean applyFloorGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, double maxBaseScore,
			int queryTermCount, int keywordMatchCount,
			PipelineConfig config) {
		double floorScore = Math.max(maxSemanticScore, maxBaseScore);
		if (floorScore >= config.noiseProfile.absoluteSimilarityFloor()) {
			return Boolean.FALSE;
		}
		boolean belowFloorRescued = false;
		if (queryTermCount > 0
				&& hasStatisticalVariance(scored)
				&& keywordMatchCount < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			double zScore = computeSemanticZScore(scored, maxSemanticScore);
			double floorRescueZThreshold =
					floorRescueGumbelThreshold(scored);
			if (zScore >= floorRescueZThreshold) {
				// Verify the signal comes from a genuine cluster,
				// not an isolated outlier. Count records within a
				// tight band of the max score — vocabulary-mismatch
				// queries ("how hot" → Temperature) produce many
				// records at similar scores, while false positives
				// ("TB" on a TB-free dataset) only have 1-2 outlier
				// conditions in that band.
				// Data-derived parameters:
				// - band width: 1/uniqueConcepts of the max score
				// - min cluster: average records per concept
				Set<String> floorConcepts = new HashSet<String>();
				for (ScoredEmbedding se : scored) {
					String cn = ConceptNameUtil
							.extractConceptName(
									se.embedding
											.getTextContent());
					if (cn != null) {
						floorConcepts.add(cn);
					}
				}
				int nConcepts = Math.max(2, floorConcepts.size());
				double densityBand = 1.0 / nConcepts;
				int minCluster = (int) Math.ceil(
						(double) scored.size() / nConcepts);
				double densityFloor = maxSemanticScore
						* (1 - densityBand);
				int clusterDensity = 0;
				for (ScoredEmbedding se : scored) {
					if (se.semanticScore >= densityFloor) {
						clusterDensity++;
					}
				}
				belowFloorRescued = clusterDensity >= minCluster;
				log.debug("Floor gate z-score rescue: zScore={}, "
						+ "density={} (band={}, floor={}), "
						+ "minCluster={}, rescued={}",
						String.format("%.2f", zScore),
						clusterDensity,
						String.format("%.4f", densityBand),
						String.format("%.4f", densityFloor),
						minCluster,
						belowFloorRescued);
			}
		}
		if (!belowFloorRescued) {
			log.debug("Top score {} (semantic={}, combined={}) is below "
					+ "absolute floor {}, returning empty",
					String.format("%.4f", floorScore),
					String.format("%.4f", maxSemanticScore),
					String.format("%.4f", maxBaseScore),
					config.noiseProfile.absoluteSimilarityFloor());
			return null;
		}
		return Boolean.TRUE;
	}

	/**
	 * Slim-margin gate — stage 2 of the filter pipeline.
	 *
	 * <p>When the top semantic score is above the absolute floor but
	 * within one {@code minScoreGap} of it, with zero keyword matches,
	 * require at least 2 records above the floor. A single record just
	 * above the floor (e.g. Rash at 0.29 for "any allergies?" with
	 * floor 0.26) is likely a coincidental near-miss, not genuine signal.
	 *
	 * @return {@code true} to continue, {@code false} to short-circuit
	 *         with an empty result
	 */
	static boolean applySlimMarginGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount, boolean belowFloorRescued,
			PipelineConfig config) {
		if (belowFloorRescued || keywordMatchCount != 0 || queryTermCount <= 0) {
			return true;
		}
		double smFloor = config.noiseProfile.absoluteSimilarityFloor();
		if (maxSemanticScore >= smFloor + config.minScoreGap) {
			return true;
		}
		int aboveFloorCount = 0;
		for (ScoredEmbedding se : scored) {
			if (se.semanticScore >= smFloor) {
				aboveFloorCount++;
			}
		}
		if (aboveFloorCount < 2) {
			log.debug("Slim-margin gate: maxSem={} is within "
					+ "{} of floor {}, zero keywords, and only "
					+ "{} record(s) above floor — returning empty",
					String.format("%.4f", maxSemanticScore),
					config.minScoreGap,
					String.format("%.4f", smFloor),
					aboveFloorCount);
			return false;
		}
		return true;
	}

	/**
	 * Initial z-score gate — stage 3 of the filter pipeline.
	 *
	 * <p>When too few records match any query keyword, require the top
	 * semantic score to be a statistical outlier (z-score ≥ Gumbel
	 * threshold). If it isn't, the top score is part of the noise floor
	 * rather than genuine signal.
	 *
	 * <p>Captures the z-score and threshold values for reuse by the
	 * downstream cluster z-score gate — when the initial z-score is
	 * strong and the ratio floor produces a tight cluster, the cluster
	 * gate is skipped as redundant.
	 *
	 * @return {@code null} if the gate rejects (caller returns empty);
	 *         otherwise a 2-element array {@code [zScore, threshold]}.
	 *         Both are {@code -1} when the gate didn't compute them
	 *         (no query terms, no statistical variance, or enough kw
	 *         matches to skip).
	 */
	static double[] applyInitialZScoreGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount) {
		double[] result = { -1, -1 };
		if (queryTermCount <= 0 || !hasStatisticalVariance(scored)) {
			return result;
		}
		if (keywordMatchCount >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			return result;
		}
		double zScore = computeSemanticZScore(scored, maxSemanticScore);
		double threshold = effectiveGumbelThreshold(scored);
		result[0] = zScore;
		result[1] = threshold;
		if (zScore < threshold) {
			log.debug("Only {} keyword match(es) and top semantic "
					+ "z-score {} is below Gumbel threshold {}, "
					+ "returning empty (maxSem={})",
					keywordMatchCount,
					String.format("%.2f", zScore),
					String.format("%.2f", threshold),
					String.format("%.4f", maxSemanticScore));
			return null;
		}
		return result;
	}

	/**
	 * Output of {@link #buildCandidates}: the candidate list plus the
	 * scalar outputs ({@code minScore}, {@code adaptiveCutoff}) that
	 * downstream stages still need.
	 */
	static final class CandidateBuildResult {
		final List<ScoredEmbedding> candidates;
		final int adaptiveCutoff;
		final double minScore;

		CandidateBuildResult(List<ScoredEmbedding> candidates,
				int adaptiveCutoff, double minScore) {
			this.candidates = candidates;
			this.adaptiveCutoff = adaptiveCutoff;
			this.minScore = minScore;
		}
	}

	/**
	 * Gap detection + candidate building — stage 4 of the filter pipeline.
	 *
	 * <p>Computes the permissive {@code minScore} floor, an IQR-based
	 * adaptive {@code minGap} for compressed-distribution models, then
	 * calls {@link #findAdaptiveCutoff} to locate the semantic cluster
	 * boundary. Builds the candidate set from records at or above the
	 * boundary OR with a full keyword match (kwScore ≥ bonusThreshold).
	 * Sorts candidates by combined score.
	 */
	static CandidateBuildResult buildCandidates(
			List<ScoredEmbedding> scored, double maxSemanticScore,
			double bonusThreshold, PipelineConfig config) {
		// Permissive floor: gap detection handles the real cutoff based on
		// score distribution. The floor just excludes near-zero noise so
		// the gap detector has a clean signal.
		double minScore = Math.min(
				config.noiseProfile.absoluteSimilarityFloor() / 2,
				maxSemanticScore / 2);

		// Adaptive min gap: scale the floor to the semantic score range so
		// models with compressed distributions (e.g. MedCPT: 0.60–0.66) can
		// detect gaps that are large relative to their range but small in
		// absolute terms. Uses the interquartile range (Q1 to Q3) to resist
		// outliers at both ends. Capped at the configured minScoreGap so
		// existing behavior for wide-range models (all-MiniLM) is preserved.
		double firstPassMinGap = config.minScoreGap;
		if (scored.size() >= 4) {
			List<Double> semScores = new ArrayList<Double>(scored.size());
			for (ScoredEmbedding se : scored) {
				if (se.semanticScore >= minScore) {
					semScores.add(se.semanticScore);
				}
			}
			Collections.sort(semScores);
			if (semScores.size() >= 4) {
				double q1 = semScores.get(semScores.size() / 4);
				double q3 = semScores.get(3 * semScores.size() / 4);
				double iqr = q3 - q1;
				// Only reduce the min gap when the score distribution is
				// genuinely compressed (IQR < half the configured min gap).
				// This preserves exact existing behavior for wide-range
				// models (all-MiniLM: IQR ~0.10, threshold 0.05) while
				// enabling gap detection for compressed models (MedCPT:
				// IQR ~0.02 < 0.05).
				if (iqr < config.minScoreGap / 2) {
					// Scale IQR by the concept-to-record ratio.
					// This fraction naturally adapts: datasets
					// with many unique concepts relative to total
					// records have larger between-concept gaps as
					// a fraction of IQR, so the threshold should
					// be higher. The ratio typically falls in
					// 0.15-0.40 — similar to the empirical range
					// of gap fractions across datasets.
					Set<String> gapConcepts = new HashSet<String>();
					for (ScoredEmbedding se : scored) {
						String cn = ConceptNameUtil
								.extractConceptName(
										se.embedding
												.getTextContent());
						if (cn != null) {
							gapConcepts.add(cn);
						}
					}
					double fraction = (double) gapConcepts.size()
							/ scored.size();
					firstPassMinGap = iqr * fraction;
				}
			}
		}

		// Gap detection considers ALL records above the floor — no topK cap.
		int adaptiveCutoff = findAdaptiveCutoff(scored, scored.size(),
				minScore, config.scoreGapMultiplier, firstPassMinGap);

		// Build the candidate set from the semantic cluster. The gap was
		// detected in semantic-score order; include all records whose
		// semantic score is at or above the cluster boundary. This handles
		// keyword penalties pushing high-semantic records far down the
		// combined-score list — taking a combined-score prefix would either
		// include too many (old max-index) or miss cluster members (count).
		// Filtering by semantic threshold captures exactly the right records.
		List<ScoredEmbedding> candidates;
		if (adaptiveCutoff >= scored.size()) {
			candidates = new ArrayList<ScoredEmbedding>(scored);
		} else {
			// Find the cluster threshold: the Nth-highest semantic score
			List<Double> semanticDesc = new ArrayList<Double>(scored.size());
			for (ScoredEmbedding se : scored) {
				semanticDesc.add(se.semanticScore);
			}
			Collections.sort(semanticDesc, Collections.reverseOrder());
			double clusterThreshold = semanticDesc.get(adaptiveCutoff - 1);
			candidates = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : scored) {
				// Two independent inclusion signals:
				// 1. Semantic cluster: embedding similarity at or above the
				//    gap-detected boundary.
				// 2. Full keyword match: the record literally contains all
				//    query terms (kwScore >= bonusThreshold). The embedding
				//    model may assign low similarity to obscure medical
				//    terminology (e.g. "Enteroviral vesicular stomatitis"
				//    for a "conditions?" query) but the keyword match is
				//    conclusive evidence the record belongs to the queried
				//    category.
				if (se.semanticScore >= clusterThreshold
						|| (config.keywordWeight > 0
								&& se.keywordScore >= bonusThreshold)) {
					candidates.add(se);
				}
			}
			// Sort by combined score to maintain the expected ordering
			Collections.sort(candidates, new Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.score, a.score);
				}
			});
		}

		return new CandidateBuildResult(candidates, adaptiveCutoff, minScore);
	}

	/**
	 * Non-refinement post-processing path — stage 5b of the filter pipeline.
	 *
	 * <p>Runs when {@code refinementActivated} is false (no keyword subset
	 * was found by {@link #refineByKeywords}). Sequence:
	 * <ol>
	 *   <li>Sensitive second-pass gap detection with a lower absolute
	 *       floor derived from the score distribution's std.</li>
	 *   <li>Ratio floor: keep records whose min(combined, semantic) is at
	 *       least {@code similarityRatio} of the top score.</li>
	 *   <li>Concept-pairing rescue: pull back near-miss records whose
	 *       same-concept partner survived the ratio floor.</li>
	 *   <li>TopK cap: only applied when some candidates have kwScore=0,
	 *       to avoid truncating legitimate multi-concept results.</li>
	 * </ol>
	 *
	 * @param ratioFloorCandidateCountOut receives the strict-floor count
	 *        (before concept-pairing rescue) — used by the cluster
	 *        z-score gate downstream.
	 * @return the post-processed candidate list
	 */
	static List<ScoredEmbedding> applyNonRefinementPath(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored,
			double maxSemanticScore, double maxBaseScore,
			double minScore, int topK, PipelineConfig config,
			int[] ratioFloorCandidateCountOut) {
		// Sensitive second-pass: same multiplier as the first pass, but
		// with a much lower absolute floor derived from the score
		// distribution's std. This detects tight clusters that the
		// first pass missed because their gaps were below the first-pass
		// floor.
		int secondCutoff = findAdaptiveCutoff(candidates, candidates.size(),
				minScore, config.scoreGapMultiplier,
				secondPassMinGap(scored));
		if (secondCutoff < candidates.size()) {
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
		List<ScoredEmbedding> nearMiss = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (Math.min(se.score, se.semanticScore) >= ratioFloor) {
				strict.add(se);
			} else {
				nearMiss.add(se);
			}
		}
		// Capture the strict count before rescue so downstream
		// gates see only candidates that truly passed the floor.
		ratioFloorCandidateCountOut[0] = strict.size();
		// Concept-pairing rescue: when a record barely misses
		// the ratio floor but its same-concept partner survived,
		// the floor is splitting a natural concept pair. Rescue
		// the near-miss if its gap from the floor is within the
		// noise resolution (noiseStd / uniqueConcepts).
		//
		// Guard: only rescue when the strict set contains at
		// least one concept with 2+ records. This distinguishes
		// two floor calibration regimes:
		// - Within-concept: the floor captured complete concept
		//   groups (2+ records), so it's at a level where same-
		//   concept records naturally cluster above it. A
		//   singleton from another concept whose partner barely
		//   misses is likely a split pair → rescue justified.
		// - Inter-concept: every concept has exactly 1 record,
		//   so the floor is at the inter-concept boundary. Each
		//   concept's best record passed; partners scored lower
		//   for a reason → don't second-guess the calibration.
		if (!strict.isEmpty() && !nearMiss.isEmpty()) {
			Map<String, Integer> conceptCounts = new HashMap<String, Integer>();
			Set<String> survivorConcepts = new HashSet<String>();
			for (ScoredEmbedding se : strict) {
				String cn = ConceptNameUtil
						.extractConceptName(se.embedding.getTextContent());
				if (cn != null) {
					survivorConcepts.add(cn);
					Integer count = conceptCounts.get(cn);
					conceptCounts.put(cn, count == null ? 1 : count + 1);
				}
			}
			boolean withinConceptCalibration = false;
			for (int count : conceptCounts.values()) {
				if (count >= 2) {
					withinConceptCalibration = true;
					break;
				}
			}
			if (withinConceptCalibration) {
				Set<String> allConcepts = new HashSet<String>();
				for (ScoredEmbedding se : scored) {
					String cn = ConceptNameUtil
							.extractConceptName(se.embedding.getTextContent());
					if (cn != null) {
						allConcepts.add(cn);
					}
				}
				double rescueTolerance = config.noiseProfile.noiseStd
						/ Math.max(1, allConcepts.size());
				for (ScoredEmbedding se : nearMiss) {
					double eff = Math.min(se.score, se.semanticScore);
					double gap = ratioFloor - eff;
					if (gap > rescueTolerance) {
						continue;
					}
					String cn = ConceptNameUtil
							.extractConceptName(se.embedding.getTextContent());
					if (cn != null && survivorConcepts.contains(cn)) {
						strict.add(se);
						log.debug("Concept-pairing rescue: [{}] gap={}, concept={}",
								se.embedding.getResourceId(),
								String.format("%.4f", gap), cn);
					}
				}
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
		return candidates;
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
		double kwMinScore = Double.MAX_VALUE;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= minKwScore) {
				keywordMatched.add(se);
				if (se.keywordScore < kwMinScore) {
					kwMinScore = se.keywordScore;
				}
			}
		}
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
		if (keywordMatched.size() >= minRecords && keywordMatched.size() < candidates.size()) {
			// Rescue non-keyword records that the semantic model ranks
			// above the WEAKEST keyword tier. In multi-concept queries
			// (e.g. "heart rate and blood pressure"), different keyword
			// tiers match different concepts: BP matches "blood"+
			// "pressure" (strong), Resp matches "rate" (weak). The
			// weakest tier is most likely to be false positives.
			// Non-keyword records that outscore these weak matches
			// semantically are likely the correct interpretation of
			// an unmatched concept (e.g. Pulse for "heart rate").
			double maxMinTierSemantic = 0;
			for (ScoredEmbedding se : keywordMatched) {
				if (se.keywordScore <= kwMinScore + 0.01
						&& se.semanticScore > maxMinTierSemantic) {
					maxMinTierSemantic = se.semanticScore;
				}
			}
			List<ScoredEmbedding> refined =
					new ArrayList<ScoredEmbedding>(keywordMatched);
			int rescued = 0;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore < minKwScore
						&& se.semanticScore > maxMinTierSemantic) {
					refined.add(se);
					rescued++;
				}
			}
			if (rescued > 0) {
				log.debug("Keyword refinement: keeping {} of {} "
						+ "(minKwScore={}, rescued {} semantic-"
						+ "dominant non-keyword records, "
						+ "minTierSem={})",
						refined.size(), candidates.size(),
						String.format("%.3f", minKwScore), rescued,
						String.format("%.4f", maxMinTierSemantic));
			} else {
				log.debug("Keyword refinement: keeping {} of {} "
						+ "(minKwScore={})",
						keywordMatched.size(), candidates.size(),
						String.format("%.3f", minKwScore));
			}
			return refined;
		}
		return candidates;
	}

	/**
	 * Detects whether a candidate set is a compound-query match — i.e.
	 * the union of terms matched across candidates covers every query
	 * term, yet no single candidate matches all terms. This is the
	 * structural signature of a multi-concept query like "HIV and CD4
	 * count" where HIV records match {hiv} and CD4 records match
	 * {cd4, count}: each concept cluster contributes a complementary
	 * subset of the query.
	 *
	 * <p>Single-concept queries never qualify because their relevant
	 * records match every query term (single-term queries trivially,
	 * multi-term queries because all terms describe the same concept).
	 *
	 * <p>Used to bypass {@link #filterByCoherence}, which would otherwise
	 * drop the minority concept cluster as outliers despite keyword
	 * evidence that those records belong to the queried set.
	 *
	 * @param candidates the candidate records
	 * @param queryTerms the query terms after stopword removal
	 *        (returns {@code false} if null or empty)
	 * @return {@code true} if the candidate set collectively covers
	 *         every query term and no single record does so alone
	 */
	static boolean isCompoundKeywordMatch(
			List<ScoredEmbedding> candidates, String[] queryTerms) {
		if (queryTerms == null || queryTerms.length < 2) {
			return false;
		}
		Set<String> unionCoverage = new HashSet<String>();
		boolean anyRecordMatchesAll = false;
		for (ScoredEmbedding se : candidates) {
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			int matchCount = 0;
			for (String term : queryTerms) {
				if (termMatchesText(term, text, words)) {
					unionCoverage.add(term);
					matchCount++;
				}
			}
			if (matchCount == queryTerms.length) {
				anyRecordMatchesAll = true;
			}
		}
		if (anyRecordMatchesAll) {
			return false;
		}
		for (String term : queryTerms) {
			if (!unionCoverage.contains(term)) {
				return false;
			}
		}
		return true;
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
		// Strip synonyms so "(syn. ...)" text doesn't inflate term
		// coverage — matches findSimilar keyword scoring behavior.
		Set<String> coveredTerms = new HashSet<String>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						ConceptNameUtil.stripSynonyms(
								se.embedding.getTextContent()));
				String lower = text.toLowerCase();
				String[] words = lower.split("\\s+");
				for (String term : queryTerms) {
					if (termMatchesText(term, lower, words)) {
						coveredTerms.add(term);
					}
				}
			}
		}

		// Semantic floor for zero-keyword rescued records: if a record
		// has no keyword match at all but was rescued by refineByKeywords
		// for its high semantic score, it represents a concept the
		// keyword vocabulary can't capture (e.g. Pulse for "heart rate").
		// Keep it if its semantic score is within REFINEMENT_SEMANTIC_RATIO
		// of the best candidate.
		double maxSemantic = 0;
		for (ScoredEmbedding se : candidates) {
			if (se.semanticScore > maxSemantic) {
				maxSemantic = se.semanticScore;
			}
		}
		double semanticFloor = maxSemantic
				* refinementSemanticRatio(candidates);

		// Keep lower-tier records only if they match a term the higher
		// tier doesn't cover, OR have zero keyword match but high
		// semantic relevance (rescued synonym/concept mismatch).
		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				filtered.add(se);
			} else if (se.keywordScore == 0
					&& se.semanticScore >= semanticFloor) {
				// Zero-keyword record with strong semantic signal —
				// the embedding model says this is relevant but no
				// query term appears in the text. This happens when
				// a clinical concept uses different terminology
				// (e.g. "Pulse" for "heart rate").
				filtered.add(se);
			} else {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						ConceptNameUtil.stripSynonyms(
								se.embedding.getTextContent()));
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
		final ModelNoiseProfile noiseProfile;

		PipelineConfig(double keywordWeight, double scoreGapMultiplier,
				double minScoreGap, double gapValidationCosineThreshold,
				double similarityRatio) {
			this(keywordWeight, scoreGapMultiplier, minScoreGap,
					gapValidationCosineThreshold, similarityRatio,
					ModelNoiseProfile.conservativeDefault());
		}

		PipelineConfig(double keywordWeight, double scoreGapMultiplier,
				double minScoreGap, double gapValidationCosineThreshold,
				double similarityRatio,
				ModelNoiseProfile noiseProfile) {
			this.keywordWeight = keywordWeight;
			this.scoreGapMultiplier = scoreGapMultiplier;
			this.minScoreGap = minScoreGap;
			this.gapValidationCosineThreshold = gapValidationCosineThreshold;
			this.similarityRatio = similarityRatio;
			this.noiseProfile = noiseProfile;
		}

		/** Returns a config using all default constant values. */
		static PipelineConfig defaults() {
			return new PipelineConfig(
					ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
					ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
					ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP,
					ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
					ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO,
					ModelNoiseProfile.conservativeDefault());
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

	/**
	 * Computes the z-score of the top semantic score relative to the full
	 * score distribution. A high z-score means the best match is a
	 * statistical outlier — genuine signal rather than noise floor.
	 */
	private static double computeSemanticZScore(List<ScoredEmbedding> scored,
			double maxSemanticScore) {
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
		return stddev == 0 ? 0 : (maxSemanticScore - meanSem) / stddev;
	}

	/**
	 * Gumbel extreme value threshold adjusted for correlated observations.
	 * Semantic scores have two levels of correlation:
	 * <ol>
	 *   <li>Records within the same concept produce nearly identical
	 *       scores (reducing N to uniqueConcepts)</li>
	 *   <li>Concepts within the same domain (e.g. all medical concepts)
	 *       share embedding space structure, partially correlating their
	 *       query cosines (further reducing effectiveN)</li>
	 * </ol>
	 * The effective degrees of freedom is estimated as
	 * ln(uniqueConcepts) — the log captures the second-level
	 * correlation from domain structure.
	 */
	static double effectiveGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				Math.log(Math.max(2, uniqueConcepts.size())));
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the cluster z-score gate
	 * (Phase 2). Uses the raw unique concept count as effective N —
	 * after gap detection and coherence filtering isolate a candidate
	 * cluster, the surviving concepts act more independently than
	 * in the full score distribution (where intra-domain correlation
	 * is high). This produces thresholds of ~2.5-2.7 for typical
	 * datasets (30-45 unique concepts), matching the discrimination
	 * needed to separate genuine clusters from noise.
	 */
	static double clusterGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0, uniqueConcepts.size());
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the cluster z-score gate
	 * when the representative is the MEDIAN (K≥4 candidates). The
	 * median of K top-scoring records corresponds to the (N/2)-th
	 * order statistic, so the effective N for the Gumbel formula is
	 * halved. This produces thresholds of ~2.15-2.50 for typical
	 * datasets, appropriately more lenient than the max-based
	 * threshold since the median is naturally lower than the max.
	 */
	static double medianGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				uniqueConcepts.size() / 2.0);
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the floor rescue z-score
	 * gate. Uses uniqueConcepts^(2/3) as effective N — an
	 * interpolation between the initial gate's heavy-correlation
	 * model (ln(N)) and the cluster gate's independence model (N).
	 * The floor rescue operates in an intermediate regime: it checks
	 * a single score (like the initial gate) but must be stricter
	 * because it's allowing below-floor results through. A secondary
	 * cluster density check provides an additional safety net.
	 * Produces thresholds of ~2.0-2.2 for typical datasets
	 * (20-45 unique concepts).
	 */
	/**
	 * Computes a data-derived similarity floor from the score
	 * distribution: mean + std. Scores below this level are not
	 * even one standard deviation above the dataset mean, indicating
	 * no distinctive signal for this query. This adapts to any
	 * embedding model's score range — models with compressed
	 * distributions (e.g. 0.5-0.7) get higher floors, while models
	 * with wide distributions (e.g. 0.1-0.5) get lower floors.
	 */
	static double floorRescueGumbelThreshold(
			List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				Math.pow(Math.max(2, uniqueConcepts.size()),
						2.0 / 3.0));
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Computes a data-derived minimum gap for the second-pass gap
	 * detector from the median of consecutive sorted-score gaps.
	 * The median gap represents the "typical" spacing between
	 * adjacent scores in this distribution. Using half the median
	 * gap as the floor ensures we only detect gaps that are
	 * meaningfully larger than typical spacing, while staying
	 * sensitive enough for compressed distributions.
	 */
	static double secondPassMinGap(List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return 0;
		}
		List<Double> sortedScores = new ArrayList<Double>(
				scored.size());
		for (ScoredEmbedding se : scored) {
			sortedScores.add(se.semanticScore);
		}
		Collections.sort(sortedScores, Collections.reverseOrder());
		List<Double> gaps = new ArrayList<Double>();
		for (int i = 1; i < sortedScores.size(); i++) {
			double gap = sortedScores.get(i - 1)
					- sortedScores.get(i);
			if (gap > 0) {
				gaps.add(gap);
			}
		}
		if (gaps.isEmpty()) {
			return 0;
		}
		Collections.sort(gaps);
		double medianGap = gaps.get(gaps.size() / 2);
		return medianGap / 2.0;
	}

	/**
	 * Computes a data-derived adaptive gap ratio for the refinement
	 * path's second-pass gap detection. Uses the standard deviation
	 * of the top quartile of scores divided by the max score. The
	 * top quartile captures the region where refinement gaps matter
	 * — its std measures the natural spread among the highest-scoring
	 * records. Dividing by max normalizes across score scales.
	 */
	static double refinementAdaptiveGapRatio(
			List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return 0.10;
		}
		List<Double> sortedScores = new ArrayList<Double>(
				scored.size());
		for (ScoredEmbedding se : scored) {
			sortedScores.add(se.semanticScore);
		}
		Collections.sort(sortedScores, Collections.reverseOrder());
		double max = sortedScores.get(0);
		if (max <= 0) {
			return 0.10;
		}
		// Top quartile standard deviation
		int topN = Math.max(4, sortedScores.size() / 4);
		double sum = 0;
		for (int i = 0; i < topN; i++) {
			sum += sortedScores.get(i);
		}
		double mean = sum / topN;
		double sqSum = 0;
		for (int i = 0; i < topN; i++) {
			double d = sortedScores.get(i) - mean;
			sqSum += d * d;
		}
		double std = Math.sqrt(sqSum / topN);
		double ratio = std / max;
		return Math.max(0.05, Math.min(0.15, ratio));
	}

	/**
	 * Computes a data-derived semantic ratio for refinement filtering.
	 * Returns 1 - CV (coefficient of variation) of the candidate
	 * scores, clamped to [0.50, 0.90]. When scores are tightly
	 * clustered (low CV → ratio near 1.0), most candidates are
	 * relevant. When scores are spread (high CV → ratio near 0.50),
	 * only the top candidates matter. This replaces the hardcoded
	 * REFINEMENT_SEMANTIC_RATIO (0.70).
	 */
	static double refinementSemanticRatio(
			List<ScoredEmbedding> candidates) {
		if (candidates.size() < 2) {
			return 0.70;
		}
		List<Double> scores = new ArrayList<Double>(
				candidates.size());
		for (ScoredEmbedding se : candidates) {
			scores.add(se.semanticScore);
		}
		Collections.sort(scores);
		double max = scores.get(scores.size() - 1);
		if (max <= 0) {
			return 0.70;
		}
		// Use Q1/max as the semantic ratio. Q1 captures the lower
		// bound of the "bulk" of the distribution, excluding the
		// bottom quartile (likely noise). This naturally adapts:
		// tight sets (all scores similar) → high ratio (strict
		// floor); wide multi-concept sets → low ratio (permissive).
		// Clamped to [0.60, 0.90] — records below 60% of the best
		// score are noise; above 90% is over-strict for any set.
		double q1 = scores.get(scores.size() / 4);
		double ratio = q1 / max;
		return Math.max(0.60, Math.min(0.90, ratio));
	}

	/**
	 * Returns true if the score distribution has enough variance for
	 * z-score statistics to be meaningful. Replaces the hardcoded
	 * MIN_RECORDS_FOR_Z_SCORE (30) with a data-driven check: the
	 * standard deviation must be non-trivial (> 1e-6) and there must
	 * be at least 4 records (minimum for IQR-based statistics).
	 */
	private static boolean hasStatisticalVariance(List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return false;
		}
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
		return std > 1e-6;
	}
}
