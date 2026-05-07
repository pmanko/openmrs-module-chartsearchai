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
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
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

	@Autowired
	@Qualifier("chartSearchAi.chartCache")
	private ChartCache chartCache;

	private static final int NOISE_PROFILE_CACHE_MAX_SIZE = 100;

	private final ConcurrentHashMap<String, ModelNoiseProfile> noiseProfileCache =
			new ConcurrentHashMap<String, ModelNoiseProfile>();

	/**
	 * Bundles the pipeline result list with the computed noise profile so the
	 * caller can cache the profile for reuse on subsequent queries.
	 */
	static class FindSimilarResult {

		final List<ChartEmbedding> records;

		final ModelNoiseProfile noiseProfile;

		final int keywordMatchCount;

		FindSimilarResult(List<ChartEmbedding> records,
				ModelNoiseProfile noiseProfile) {
			this(records, noiseProfile, -1);
		}

		FindSimilarResult(List<ChartEmbedding> records,
				ModelNoiseProfile noiseProfile,
				int keywordMatchCount) {
			this.records = records;
			this.noiseProfile = noiseProfile;
			this.keywordMatchCount = keywordMatchCount;
		}
	}

	private void invalidateNoiseProfileCache(Patient patient) {
		String prefix = patient.getPatientId() + ":";
		for (String key : noiseProfileCache.keySet()) {
			if (key.startsWith(prefix)) {
				noiseProfileCache.remove(key);
			}
		}
	}

	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = buildChart(patient, question);
		LlmResponse response = llmProvider.search(chartTextOrPlaceholder(chart), question);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()),
				response.getInputTokens(), response.getOutputTokens(),
				response.getCachedTokens());
	}

	@Override
	public void warmup(Patient patient) {
		if (!isWarmupEnabled()) {
			return;
		}
		// Skip the upstream chart-serialization cost when the active engine
		// gains nothing from warmup (e.g. remote APIs that cache themselves).
		if (!llmProvider.supportsWarmup()) {
			return;
		}
		// Pre-filter pipelines build a different prompt prefix for each query (the
		// records sent depend on the question), so a chart-only warmup wouldn't match
		// what a real query produces — the KV cache would not be reused.
		if (usePreFilter()) {
			return;
		}
		PatientChart chart = buildChart(patient, "");
		llmProvider.warmup(chartTextOrPlaceholder(chart));
	}

	/**
	 * Substitutes a placeholder when the chart has no records, so the LLM produces a
	 * query-specific "no records" answer instead of one based on demographics alone.
	 */
	private static String chartTextOrPlaceholder(PatientChart chart) {
		return chart.getMappings().isEmpty() ? "(No relevant records found)" : chart.getText();
	}

	@Override
	public ChartAnswer searchStreaming(Patient patient, String question,
			Consumer<String> tokenConsumer) {
		PatientChart chart = buildChart(patient, question);
		LlmResponse response = llmProvider.searchStreaming(
				chartTextOrPlaceholder(chart), question, tokenConsumer);

		return new ChartAnswer(response.getAnswer(),
				extractCitedReferences(response.getCitations(), chart.getMappings()),
				response.getInputTokens(), response.getOutputTokens(),
				response.getCachedTokens());
	}

	static int extractRecencyCap(String question) {
		return QueryPreprocessor.extractRecencyCap(question);
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
	 * Shared post-retrieval pipeline called by both production
	 * ({@link #filterAndSerialize}) and test ({@link #findRelevantRecords})
	 * paths. Single source of truth for: filterAndCap, concept-name rescue,
	 * post-cap precision filter, and groupByConcept.
	 *
	 * @param allRecords        all patient records
	 * @param relevantKeys      keys selected by the retrieval stage
	 * @param question          the user's natural-language question
	 * @param pipelineSize      records returned by findSimilar (for detecting
	 *                          recency-cap reduction); pass -1 to skip
	 *                          rescue/post-cap filter
	 * @param keywordMatchCount keyword matches from the pipeline; pass -1 to
	 *                          skip rescue/post-cap filter
	 * @param provider          embedding provider (null skips rescue/filter)
	 * @param queryPrefix       prefix for embedding queries
	 * @return filtered, rescued, and concept-grouped records
	 */
	static List<SerializedRecord> postRetrievalPipeline(
			List<SerializedRecord> allRecords,
			Set<String> relevantKeys,
			String question,
			int pipelineSize,
			int keywordMatchCount,
			EmbeddingProvider provider,
			String queryPrefix) {
		List<SerializedRecord> filtered = filterAndCap(
				allRecords, relevantKeys, question);

		boolean recencyCapReduced = pipelineSize > 0
				&& filtered.size() < pipelineSize / 2;

		// Step 1: Single-record concept-name rescue (e.g. Height
		// for "latest BMI" when only Weight survived the cap).
		if (recencyCapReduced && keywordMatchCount == 0
				&& provider != null && filtered.size() == 1) {
			String normalizedQ = stripQueryStopwords(question);
			String embQ = QueryPreprocessor.buildEmbeddingQuery(normalizedQ);
			float[] qVec = provider.embedQuery(
					queryPrefix + embQ);
			filtered = conceptNameRescueRecords(filtered,
					allRecords, provider, qVec, question);
		}

		// Step 2: Post-cap concept-name precision filter — remove
		// irrelevant concepts from zero-keyword queries (e.g. SpO2
		// for "BMI"). For keyword-matching queries, the keyword
		// scoring already provides precision — the multi-concept
		// rescue handles cleanup for comma-list queries.
		if (recencyCapReduced && keywordMatchCount == 0
				&& provider != null
				&& filtered.size() >= 3
				&& filtered.size() <= 10) {
			String normalizedQ = stripQueryStopwords(question);
			filtered = applyPostCapConceptFilter(filtered,
					provider, normalizedQ, queryPrefix);
		}

		// Step 3: Multi-concept rescue for comma-separated queries
		// (e.g. "blood pressure, weight, and temperature"). Runs
		// AFTER the precision filter so rescued concepts aren't
		// contaminated by noise that should have been filtered.
		if (provider != null && question.contains(",")
				&& !filtered.isEmpty()) {
			filtered = multiConceptRescue(filtered, allRecords,
					provider, question, queryPrefix);
		}

		// Step 4: Weak-match cleanup for multi-concept queries.
		// When a record's concept name matches fewer query terms
		// than another record matching the same term, it's a weak
		// match (e.g. "Blood Oxygen Saturation" matches only
		// "blood", while "Systolic Blood Pressure" matches both
		// "blood" and "pressure"). Drop weak matches.
		if (question.contains(",") && !filtered.isEmpty()) {
			String normalizedQ = stripQueryStopwords(question);
			String[] qTerms = extractQueryTerms(normalizedQ);
			if (qTerms.length > 1) {
				filtered = dropWeakMatches(filtered, qTerms);
			}
		}

		return groupByConcept(filtered);
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
			PatientChart cached = chartCache.get(patient);
			if (cached != null) {
				return cached;
			}
			PatientChart chart = chartSerializer.serialize(patient);
			chartCache.put(patient, chart);
			return chart;
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
		// findSimilar returns null when no embeddings exist (needs indexing),
		// or an empty list when embeddings exist but nothing matched the query.
		FindSimilarResult fsResult = findSimilarResult(patient, question);

		if (fsResult == null || fsResult.records == null) {
			log.info("No embeddings found for patient [id={}], indexing now", patient.getPatientId());
			try {
				embeddingIndexer.indexPatient(patient);
				// Invalidate cached noise profile — embeddings have changed
				invalidateNoiseProfileCache(patient);
				fsResult = findSimilarResult(patient, question);
			}
			catch (Exception e) {
				log.error("Failed to index patient [id={}], falling back to full chart",
						patient.getPatientId(), e);
			}
		}

		if (fsResult == null || fsResult.records == null) {
			log.debug("Still no embeddings after indexing attempt, falling back to full chart");
			return chartSerializer.serialize(patient);
		}

		List<ChartEmbedding> similar = fsResult.records;

		if (similar.isEmpty()) {
			log.debug("No records matched the query '{}' for patient [id={}], returning empty chart",
					question, patient.getPatientId());
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		Set<String> relevantKeys = new HashSet<String>();
		for (ChartEmbedding ce : similar) {
			relevantKeys.add(ChartSearchAiUtils.resourceKey(ce.getResourceType(), ce.getResourceId()));
		}

		log.warn("findSimilar returned {} records for query '{}' patient [id={}]: {}",
				similar.size(), question, patient.getPatientId(), relevantKeys);

		return filterAndSerialize(patient, question, relevantKeys,
				similar.size(), fsResult.keywordMatchCount);
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
			queryVector = embeddingProvider.embedQuery(embeddingInput);
		}
		catch (Exception e) {
			log.error("Failed to embed query for ES search, falling back to embeddings", e);
			return buildChartWithEmbeddings(patient, question);
		}

		// ES search: RRF combines BM25 keyword matches with kNN semantic
		// matches. Used to identify candidates that may have keyword signal
		// the pure-embedding path would miss (e.g. "allergy" → "Patient
		// allergy:" via BM25).
		List<ElasticsearchIndexer.ElasticsearchSearchResult> esResults =
				elasticsearchIndexer.search(patient, normalizedQuery, queryVector,
						getTopK() * ChartSearchAiConstants.ES_FETCH_MULTIPLIER);

		// Run the FULL embedding filter pipeline on all patient embeddings —
		// not just the ES-returned subset. filterPipeline uses adaptive
		// gap-based filtering that needs the full score distribution to
		// distinguish signal from noise. Running it on the pre-filtered ES
		// subset produces a compressed distribution where even relevant
		// records (e.g. Kaposi for "cancer?") can't be separated from noise.
		List<ElasticsearchIndexer.ElasticsearchSearchResult> allEsEmbeddings =
				elasticsearchIndexer.fetchAllPatientEmbeddings(patient.getPatientId());
		List<ChartEmbedding> allEmbeddings = esResultsToChartEmbeddings(allEsEmbeddings);

		List<ChartEmbedding> pipelineFiltered = null;
		if (!allEmbeddings.isEmpty()) {
			FindSimilarResult fsResult = findSimilar(allEmbeddings,
					embeddingProvider, question, getQueryPrefix(),
					new PipelineConfig(getKeywordWeight(),
							getScoreGapMultiplier(), getMinScoreGap(),
							getGapValidationCosineThreshold(),
							getSimilarityRatio()));
			pipelineFiltered = fsResult.records;
		}

		// Use only the pipeline-filtered records — the full-corpus
		// findSimilar already includes keyword scoring, so BM25-only
		// matches from ES are also captured. The old filterEsResults
		// (conservative default) is no longer in the union because its
		// loose thresholds let through off-topic kNN matches (e.g. BP
		// records for a "cd4 count" query).
		Set<String> relevantKeys = new HashSet<String>();
		if (pipelineFiltered != null) {
			for (ChartEmbedding ce : pipelineFiltered) {
				relevantKeys.add(ChartSearchAiUtils.resourceKey(
						ce.getResourceType(), ce.getResourceId()));
			}
		}

		if (relevantKeys.isEmpty()) {
			log.debug("Elasticsearch: no records survived filtering for query '{}'",
					normalizedQuery);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		log.debug("Elasticsearch returned {} ES results, {} after full pipeline for query '{}': {}",
				esResults.size(), relevantKeys.size(), normalizedQuery, relevantKeys);

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
	/**
	 * Converts ES search results to {@link ChartEmbedding} objects for use
	 * with the embedding pipeline's {@link #findSimilar} method. The ES
	 * index stores the PREFIXED text ({@code "Clinical observation: ..."})
	 * but {@code findSimilar} re-prefixes via {@code buildPrefixedText},
	 * so we must strip the structural prefix here to avoid double-prefixing
	 * that corrupts keyword scoring and concept-name extraction.
	 */
	private static List<ChartEmbedding> esResultsToChartEmbeddings(
			List<ElasticsearchIndexer.ElasticsearchSearchResult> esResults) {
		Set<String> prefixes = ChartSearchAiUtils.getAllEmbeddingPrefixes();
		List<ChartEmbedding> out = new ArrayList<ChartEmbedding>(esResults.size());
		for (ElasticsearchIndexer.ElasticsearchSearchResult r : esResults) {
			if (r.getEmbedding() == null) {
				continue;
			}
			ChartEmbedding ce = new ChartEmbedding();
			ce.setResourceType(r.getResourceType());
			ce.setResourceId(r.getResourceId());
			ce.setEmbeddingVector(r.getEmbedding());
			// Strip the structural prefix that ES stores so findSimilar
			// doesn't double-prefix when calling buildPrefixedText.
			String text = r.getText();
			if (text != null) {
				for (String pfx : prefixes) {
					if (text.startsWith(pfx)) {
						text = text.substring(pfx.length());
						break;
					}
				}
			}
			ce.setTextContent(text);
			out.add(ce);
		}
		return out;
	}

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

		List<ChartEmbedding> filtered = EmbeddingRankingPipeline.filterPipeline(semanticScores,
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
		return filterAndSerialize(patient, question, relevantKeys, -1, -1);
	}

	private PatientChart filterAndSerialize(Patient patient, String question,
			Set<String> relevantKeys, int pipelineSize,
			int keywordMatchCount) {
		List<SerializedRecord> allRecords = recordLoader.loadAll(patient);

		List<SerializedRecord> filtered = postRetrievalPipeline(
				allRecords, relevantKeys, question,
				pipelineSize, keywordMatchCount,
				embeddingProvider, getQueryPrefix());

		log.debug("Pre-filtered {} records to {} using {}",
				allRecords.size(), filtered.size(),
				isHybridPipeline() ? "Hybrid" :
						isElasticsearchPipeline() ? "Elasticsearch" :
								isLucenePipeline() ? "Lucene" : "embeddings");

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
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER, "false");
		return !"false".equalsIgnoreCase(mode.trim());
	}

	static boolean isWarmupEnabled() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_WARMUP_ENABLED, "true");
		return !"false".equalsIgnoreCase(value.trim());
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

	static String stripQueryStopwords(String question) {
		return QueryPreprocessor.stripQueryStopwords(question);
	}

	List<ChartEmbedding> findSimilar(Patient patient, String question) {
		FindSimilarResult result = findSimilarResult(patient, question);
		return result != null ? result.records : null;
	}

	FindSimilarResult findSimilarResult(Patient patient, String question) {
		List<ChartEmbedding> allEmbeddings = dao.getByPatient(patient);
		if (allEmbeddings.isEmpty()) {
			return null;
		}

		// Select model-specific pipeline config by fingerprinting the
		// model (embedding a sentinel string). Falls back to path-based
		// detection, then L6-v2 defaults.
		String modelIdentity = embeddingProvider != null
				? embeddingProvider.identifyModel() : null;
		String modelName = embeddingProvider != null
				? embeddingProvider.getModelName() : null;
		PipelineConfig baseConfig = PipelineConfig.forModel(
				modelIdentity != null ? modelIdentity : modelName);

		// Reuse cached noise profile for this patient+model when
		// available, avoiding ~30 redundant ONNX embeddings per query.
		String cacheKey = patient.getPatientId()
				+ ":" + (modelIdentity != null ? modelIdentity : modelName);
		ModelNoiseProfile cachedProfile = noiseProfileCache.get(cacheKey);

		// Use model-specific defaults, but allow global property overrides
		// when an admin has explicitly customized them (i.e., the GP value
		// differs from the L6-v2 default that ships as the initial value).
		PipelineConfig config = PipelineConfig.buildEffective(baseConfig,
				getKeywordWeight(), getScoreGapMultiplier(),
				getMinScoreGap(), getGapValidationCosineThreshold(),
				getSimilarityRatio(), cachedProfile);
		log.warn("Model identity={}, config: kwWeight={}, gapMult={}, "
				+ "minGap={}, gapCosThresh={}, simRatio={}, floorZScore={}",
				modelIdentity, config.keywordWeight, config.scoreGapMultiplier,
				config.minScoreGap, config.gapValidationCosineThreshold,
				config.similarityRatio, config.floorRescueMinZScore);

		try {
			FindSimilarResult result = findSimilar(allEmbeddings,
					embeddingProvider, question, getQueryPrefix(), config);
			if (result.noiseProfile != null) {
				if (noiseProfileCache.size() >= NOISE_PROFILE_CACHE_MAX_SIZE) {
					// Evict an arbitrary entry to stay within bounds.
					// Use an iterator with a hasNext() guard so a concurrent
					// eviction that empties the cache does not throw
					// NoSuchElementException.
					Iterator<String> it = noiseProfileCache.keySet().iterator();
					if (it.hasNext()) {
						noiseProfileCache.remove(it.next());
					}
				}
				noiseProfileCache.put(cacheKey, result.noiseProfile);
			}
			return result;
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
	 * @param queryPrefix prefix prepended to the query before embedding
	 * @param config pipeline tuning parameters
	 * @return filtered list of relevant embeddings, or null if empty
	 */
	/** @deprecated topK is no longer used by the pipeline. Use the overload without it. */
	static List<ChartEmbedding> findSimilar(List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question, int topK,
			String queryPrefix, PipelineConfig config) {
		return findSimilarWithProfile(allEmbeddings, provider, question,
				queryPrefix, config).records;
	}

	static FindSimilarResult findSimilar(List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		return findSimilarWithProfile(allEmbeddings, provider, question,
				queryPrefix, config);
	}

	static FindSimilarResult findSimilarWithProfile(
			List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		String normalizedQuery = stripQueryStopwords(question);
		String[] queryTerms = extractQueryTerms(normalizedQuery);
		String embeddingQuery = QueryPreprocessor.buildEmbeddingQuery(normalizedQuery);
		float[] queryVector = provider.embedQuery(queryPrefix + embeddingQuery);

		// Identify "type indicator" query terms — terms that appear in
		// any structural embedding prefix (e.g. "medication" matches
		// "Medication prescription:", "test" matches "Lab or diagnostic
		// test:", "allergy" matches "Patient allergy:"). For these
		// terms, only matches in the prefix portion of a record's text
		// count — body-text matches are demoted as coincidental
		// narrative mentions (e.g. an encounter note saying "Medication
		// adjusted" doesn't make it a medication record; "Test —
		// Haemoglobin" doesn't make it a test order). Content terms
		// that don't appear in any prefix (e.g. "azithromycin",
		// "diabetes") still match freely in the body. The prefix
		// vocabulary is the static set defined by getEmbeddingPrefix —
		// using the global vocabulary (not just dataset-present
		// prefixes) ensures consistent behavior across datasets.
		Set<String> typeIndicatorTerms = new HashSet<String>();
		if (queryTerms.length > 0) {
			for (String prefix
					: ChartSearchAiUtils.getAllEmbeddingPrefixes()) {
				String lowerPrefix = prefix.toLowerCase();
				String[] prefixWords = lowerPrefix.split("\\s+");
				for (String term : queryTerms) {
					if (typeIndicatorTerms.contains(term)) {
						continue;
					}
					if (termMatchesText(term, lowerPrefix, prefixWords)) {
						typeIndicatorTerms.add(term);
					}
				}
			}
		}

		// IDF-based term filtering: compute document frequency for
		// each query term. Terms matching > 20% of records are too
		// generic (e.g. "normal", "range", "time") and would flood
		// keyword scoring with noise. Replace queryTerms with the
		// filtered set for keyword scoring only (semantic embedding
		// still uses the full query).
		String[] kwTerms = queryTerms;
		if (queryTerms.length > 2) {
			int totalDocs = 0;
			int[] docFreq = new int[queryTerms.length];
			for (int i = 0; i < allEmbeddings.size(); i++) {
				ChartEmbedding ce = allEmbeddings.get(i);
				if (ce.getEmbeddingVector() == null) continue;
				totalDocs++;
				String text = (ce.getTextContent() != null
						? ce.getTextContent() : "").toLowerCase();
				for (int t = 0; t < queryTerms.length; t++) {
					if (text.contains(queryTerms[t])) {
						docFreq[t]++;
					}
				}
			}
			if (totalDocs > 0) {
				double threshold = totalDocs * 0.2;
				List<String> filtered = new ArrayList<String>();
				int genericCount = 0;
				for (int t = 0; t < queryTerms.length; t++) {
					if (docFreq[t] <= threshold) {
						filtered.add(queryTerms[t]);
					} else {
						genericCount++;
					}
				}
				// Only activate when the query is dominated by
				// generic terms (> 50% filtered). This targets
				// long queries like "HB results over time, are
				// values moving toward normal range" without
				// affecting shorter, focused queries.
				if (!filtered.isEmpty()
						&& genericCount > queryTerms.length / 2) {
					kwTerms = filtered.toArray(
							new String[filtered.size()]);
					log.warn("IDF filter: {} -> {} terms (dropped "
							+ "generic terms from {})",
							queryTerms.length, kwTerms.length,
							java.util.Arrays.toString(queryTerms));
				}
			}
		}

		double[] semanticScores = new double[allEmbeddings.size()];
		double[] keywordScores = new double[allEmbeddings.size()];
		ChartEmbedding[] embeddings = new ChartEmbedding[allEmbeddings.size()];
		int validCount = 0;
		// First pass: collect valid records and compute semantic scores.
		// Keyword scoring is deferred until after concept-similarity
		// expansion so the (possibly replaced) kwTerms are scored.
		for (int i = 0; i < allEmbeddings.size(); i++) {
			ChartEmbedding ce = allEmbeddings.get(i);
			float[] vector = ce.getEmbeddingVector();
			if (vector == null || vector.length != queryVector.length) {
				continue;
			}
			embeddings[validCount] = ce;
			semanticScores[validCount] = cosineSimilarity(queryVector, vector);
			validCount++;
		}
		if (validCount < embeddings.length) {
			embeddings = java.util.Arrays.copyOf(embeddings, validCount);
		}
		// Concept-similarity expansion: when the query phrasing isn't
		// in any record but the embedding model recognises a single
		// patient-chart concept as a near-synonym (e.g. "heart rate"
		// → "Pulse"), replace kwTerms with the concept's tokens so
		// keyword scoring can anchor on real record vocabulary. See
		// expandKwTermsViaConceptSimilarity for the activation rule.
		String[] expandedKwTerms = expandKwTermsViaConceptSimilarity(
				kwTerms, queryVector, embeddings, provider, config);
		if (expandedKwTerms != kwTerms) {
			kwTerms = expandedKwTerms;
			// Type-indicator restriction was computed from the original
			// query terms. After replacement the new terms are concept
			// names; the user did not literally use type-indicator
			// vocabulary, so clear the restriction set to keep keyword
			// matching unconstrained on the replacement tokens.
			typeIndicatorTerms = new HashSet<String>();
		}
		// Second pass: compute keyword scores with the (possibly
		// expanded) kwTerms.
		for (int i = 0; i < embeddings.length; i++) {
			ChartEmbedding ce = embeddings[i];
			// Strip synonym parentheticals before keyword scoring so
			// that "(syn. Hemoglobin performed on blood)" doesn't cause
			// "blood" to match Haemoglobin records. Synonyms stay in the
			// embedding for semantic matching — only keyword scoring uses
			// the stripped text.
			String body = ConceptNameUtil.stripSynonyms(ce.getTextContent());
			String keywordText = ChartSearchAiUtils.buildPrefixedText(
					ce.getResourceType(), body);
			if (typeIndicatorTerms.isEmpty()) {
				keywordScores[i] = computeKeywordScore(
						kwTerms, keywordText);
			} else {
				keywordScores[i] = computeKeywordScoreRestricted(
						kwTerms, keywordText, body,
						typeIndicatorTerms);
			}
		}
		if (validCount < semanticScores.length) {
			semanticScores = java.util.Arrays.copyOf(semanticScores, validCount);
			keywordScores = java.util.Arrays.copyOf(keywordScores, validCount);
		}

		// Compute noise profile using hint-stripped re-embeddings for
		// cross-concept similarity, so the profile stays stable across
		// category-hint enrichment. O(C) extra embeddings (~30ms).
		// Reuse pre-computed profile from config when available
		// (e.g. test harnesses that run many queries on the same dataset).
		ModelNoiseProfile noiseProfile = config.noiseProfile != null
				&& !config.noiseProfile.isConservativeDefault()
				? config.noiseProfile
				: ModelNoiseProfile.compute(embeddings, provider);
		log.warn("NoiseProfile: Q1={} median={} mean={} P95={} "
				+ "intraMean={} floor={}",
				String.format("%.4f", noiseProfile.noiseQ1),
				String.format("%.4f", noiseProfile.noiseMedian),
				String.format("%.4f", noiseProfile.noiseMean),
				String.format("%.4f", noiseProfile.noiseP95),
				String.format("%.4f", noiseProfile.intraConceptMean),
				String.format("%.4f", noiseProfile.absoluteSimilarityFloor()));
		PipelineConfig profiledConfig =
				config.withNoiseProfile(noiseProfile);

		int[] gapCutoffOut = new int[1];
		List<ChartEmbedding> pipelineResult = EmbeddingRankingPipeline.filterPipeline(
				semanticScores, keywordScores, embeddings,
				queryTerms, queryTerms.length, profiledConfig,
				gapCutoffOut);
		int gapCutoff = gapCutoffOut[0];

		int keywordMatchCount = 0;
		for (int i = 0; i < validCount; i++) {
			if (keywordScores[i] > 0) {
				keywordMatchCount++;
			}
		}

		// Query truncation retry: when the pipeline returns empty for
		// a long query (>3 terms), the extra terms may dilute the
		// embedding and flood keyword scoring. Retry with just the
		// first 2 core terms and a fresh embedding. The short query
		// "HB results" finds records that "What are this patient's
		// HB results over time...normal range?" misses.
		if ((pipelineResult == null || pipelineResult.isEmpty())
				&& queryTerms.length > 3) {
			String[] coreTerms = java.util.Arrays.copyOf(
					queryTerms, Math.min(2, queryTerms.length));
			String coreQuery = String.join(" ", coreTerms);
			float[] coreVector = provider.embedQuery(
					queryPrefix + coreQuery);

			double[] coreSemantic = new double[validCount];
			double[] coreKw = new double[validCount];
			for (int i = 0; i < validCount; i++) {
				coreSemantic[i] = cosineSimilarity(
						coreVector, embeddings[i].getEmbeddingVector());
				String body = ConceptNameUtil.stripSynonyms(
						embeddings[i].getTextContent());
				String kwText = ChartSearchAiUtils.buildPrefixedText(
						embeddings[i].getResourceType(), body);
				coreKw[i] = computeKeywordScore(coreTerms, kwText);
			}

			int[] retryGapCutoff = new int[1];
			pipelineResult = EmbeddingRankingPipeline.filterPipeline(coreSemantic, coreKw,
					embeddings, coreTerms, coreTerms.length,
					profiledConfig, retryGapCutoff);
			if (pipelineResult != null && !pipelineResult.isEmpty()) {
				gapCutoff = retryGapCutoff[0];
				log.warn("Query truncation retry: '{}' -> '{}', "
						+ "found {} results",
						String.join(" ", queryTerms), coreQuery,
						pipelineResult.size());
				// Update keyword match count and semantic scores
				// for downstream gates
				keywordMatchCount = 0;
				for (int i = 0; i < validCount; i++) {
					if (coreKw[i] > 0) keywordMatchCount++;
					semanticScores[i] = coreSemantic[i];
				}
			}
		}

		// Dilution rescue: long queries (>3 terms) may dilute the
		// embedding by averaging specific terms ("hb") with generic
		// ones ("time", "normal", "range"). Embed the first 2 terms
		// as a focused query and check if the improvement in max
		// similarity exceeds 1 standard deviation of the focused
		// score distribution — a statistically significant signal
		// gain that indicates the full query lost a concept to
		// dilution. If so, run the focused pipeline and merge small,
		// disjoint results back in.
		if (pipelineResult != null && !pipelineResult.isEmpty()
				&& queryTerms.length > 3) {
			double maxFullSem = 0;
			for (int i = 0; i < validCount; i++) {
				if (semanticScores[i] > maxFullSem) {
					maxFullSem = semanticScores[i];
				}
			}
			String[] coreTerms2 = java.util.Arrays.copyOf(
					queryTerms, Math.min(2, queryTerms.length));
			String coreQuery2 = String.join(" ", coreTerms2);
			float[] coreVector2 = provider.embedQuery(
					queryPrefix + coreQuery2);
			double[] coreSem2 = new double[validCount];
			double maxCoreSem = 0;
			double sumCoreSem = 0;
			double sumCoreSemSq = 0;
			for (int i = 0; i < validCount; i++) {
				double sim = cosineSimilarity(coreVector2,
						embeddings[i].getEmbeddingVector());
				coreSem2[i] = sim;
				if (sim > maxCoreSem) maxCoreSem = sim;
				sumCoreSem += sim;
				sumCoreSemSq += sim * sim;
			}
			double meanCore = validCount > 0
					? sumCoreSem / validCount : 0;
			double stdCore = validCount > 1
					? Math.sqrt(Math.max(0,
							sumCoreSemSq / validCount
									- meanCore * meanCore))
					: 0;
			if (maxCoreSem - maxFullSem > stdCore) {
				double[] coreKw2 = new double[validCount];
				for (int i = 0; i < validCount; i++) {
					String body = ConceptNameUtil.stripSynonyms(
							embeddings[i].getTextContent());
					String kwText =
							ChartSearchAiUtils.buildPrefixedText(
									embeddings[i].getResourceType(),
									body);
					coreKw2[i] = computeKeywordScore(
							coreTerms2, kwText);
				}
				int[] mergeGapCutoff = new int[1];
				List<ChartEmbedding> mergeResult = EmbeddingRankingPipeline.filterPipeline(
						coreSem2, coreKw2, embeddings, coreTerms2,
						coreTerms2.length, profiledConfig,
						mergeGapCutoff);
				if (mergeResult != null
						&& !mergeResult.isEmpty()
						&& mergeResult.size()
								<= ChartSearchAiConstants
										.ADAPTIVE_MIN_RECORDS) {
					Set<ChartEmbedding> existing =
							Collections.newSetFromMap(
									new IdentityHashMap<
											ChartEmbedding,
											Boolean>());
					existing.addAll(pipelineResult);
					boolean anyOverlap = false;
					for (ChartEmbedding ce : mergeResult) {
						if (existing.contains(ce)) {
							anyOverlap = true;
							break;
						}
					}
					if (!anyOverlap) {
						List<ChartEmbedding> merged =
								new ArrayList<>(pipelineResult);
						merged.addAll(mergeResult);
						log.warn("Dilution rescue: merged {}"
								+ " focused results into {} "
								+ "pipeline results for '{}'",
								mergeResult.size(),
								pipelineResult.size(),
								coreQuery2);
						pipelineResult = merged;
					}
				}
			}
		}

		// When type indicators are present, remove straggler records whose
		// structural prefix doesn't match any type indicator term. Only
		// fires when >75% of pipeline results already match — this targets
		// the case where a few wrong-type records leak through on semantic
		// similarity alone (e.g. "CRITICALLY_HIGH" obs for "active
		// conditions") without over-filtering mixed-type result sets.
		int typeMatchCount = -1;
		if (!typeIndicatorTerms.isEmpty()
				&& pipelineResult != null && !pipelineResult.isEmpty()) {
			typeMatchCount = 0;
			for (ChartEmbedding ce : pipelineResult) {
				if (EmbeddingRankingPipeline.matchesTypeIndicator(ce, typeIndicatorTerms)) {
					typeMatchCount++;
				}
			}
			if (typeMatchCount > 0
					&& typeMatchCount > pipelineResult.size() * 2 / 3) {
				int beforeFilter = pipelineResult.size();
				List<ChartEmbedding> filtered = new ArrayList<>();
				for (ChartEmbedding ce : pipelineResult) {
					if (EmbeddingRankingPipeline.matchesTypeIndicator(ce, typeIndicatorTerms)) {
						filtered.add(ce);
					}
				}
				pipelineResult = filtered;
				if (pipelineResult.size() != beforeFilter) {
					log.warn("Type-indicator filter {}: {} -> {}",
							typeIndicatorTerms, beforeFilter,
							pipelineResult.size());
				}
			}
		}

		// Concept-name re-ranking for zero-keyword queries: drop
		// concept outliers whose name-level similarity to the query
		// is significantly below the cluster of other concepts.
		// When no query term matches any record text, the retrieval
		// relies entirely on embedding similarity. Shared prefixes
		// like "Vital signs / Finding —" can dominate scores, making
		// all vitals look equally relevant. Re-ranking by concept
		// name adds precision by comparing each concept name directly
		// against the query embedding.
		boolean noTypeMatch = !typeIndicatorTerms.isEmpty()
				&& typeMatchCount == 0;
		if (pipelineResult != null && !pipelineResult.isEmpty()
				&& keywordMatchCount == 0 && queryTerms.length > 0) {
			List<ChartEmbedding> preGateCandidates = pipelineResult;
			int beforeRerank = pipelineResult.size();
			pipelineResult = rerankByConceptName(pipelineResult,
					embeddings, validCount, queryVector, provider,
					noiseProfile, profiledConfig, noTypeMatch);
			if (pipelineResult.size() != beforeRerank) {
				StringBuilder kept = new StringBuilder();
				for (ChartEmbedding ce : pipelineResult) {
					if (kept.length() > 0) kept.append(", ");
					kept.append(ce.getResourceType()).append(":")
							.append(ce.getResourceId());
				}
				log.warn("Concept-name rerank {}: {} -> {} [{}]",
						java.util.Arrays.toString(queryTerms),
						beforeRerank, pipelineResult.size(), kept);
			}

			// Type-indicator rescue: when the gate rejects all
			// candidates but type indicators exist, rescue
			// candidates whose record type matches the indicated
			// type from the pre-gate set.
			if (pipelineResult.isEmpty()
					&& !typeIndicatorTerms.isEmpty()) {
				List<ChartEmbedding> rescued = new ArrayList<>();
				for (ChartEmbedding ce : preGateCandidates) {
					if (EmbeddingRankingPipeline.matchesTypeIndicator(ce,
							typeIndicatorTerms)) {
						rescued.add(ce);
					}
				}
				if (!rescued.isEmpty()) {
					pipelineResult = rescued;
					log.warn("Type-indicator rescue after "
							+ "concept-name gate: 0 -> {} for {}",
							rescued.size(), typeIndicatorTerms);
				}
			}

		}

		// No-type-match check: when type indicators are present but
		// ALL pipeline results are wrong type AND no records of the
		// indicated type exist in the dataset, the patient has no
		// records of the queried type. Return empty ONLY when the
		// pipeline results' body text also doesn't mention any type
		// indicator term — this distinguishes pure semantic overlap
		// (Asthma ↔ allergies: body says "Condition: Asthma", no
		// mention of "allergy") from cross-type relevance (Fetishism
		// obs with "Medication adjusted": body mentions "medication",
		// so it IS about medications despite the wrong prefix).
		if (noTypeMatch
				&& pipelineResult != null
				&& !pipelineResult.isEmpty()) {
			// Check if any result's body text mentions a type
			// indicator term — if so, the records are cross-type
			// relevant and should be kept.
			boolean bodyMentionsType = false;
			for (ChartEmbedding ce : pipelineResult) {
				String body = (ce.getTextContent() != null
						? ce.getTextContent() : "").toLowerCase();
				String[] bodyWords = body.split("\\s+");
				for (String term : typeIndicatorTerms) {
					if (termMatchesText(term, body, bodyWords)) {
						bodyMentionsType = true;
						break;
					}
				}
				if (bodyMentionsType) break;
			}
			if (!bodyMentionsType) {
				List<ChartEmbedding> typeMatching =
						new ArrayList<>();
				for (int i = 0; i < validCount; i++) {
					if (EmbeddingRankingPipeline.matchesTypeIndicator(embeddings[i],
							typeIndicatorTerms)) {
						typeMatching.add(embeddings[i]);
					}
				}
				if (typeMatching.isEmpty()) {
					log.warn("No-type-match empty: {} wrong-type "
							+ "results, 0 type-matching in "
							+ "dataset — returning empty for {}",
							pipelineResult.size(),
							typeIndicatorTerms);
					pipelineResult = Collections.emptyList();
				} else if (typeMatching.size()
						<= ChartSearchAiConstants
								.ADAPTIVE_MIN_RECORDS) {
					log.warn("No-type-match replacement: "
							+ "replacing {} wrong-type with "
							+ "{} type-matching records for {}",
							pipelineResult.size(),
							typeMatching.size(),
							typeIndicatorTerms);
					pipelineResult = typeMatching;
				}
			}
		}

		// Type-indicator full-scan rescue: when the pipeline returns
		// empty but type indicators exist, scan ALL embeddings for
		// records matching the indicated type. Only rescue when the
		// matching set is small (<=5 records) — a rare type like
		// "allergy" with 1-2 records is likely a genuine miss, but
		// a common type like "condition" with 20+ records means the
		// pipeline correctly determined no specific match.
		if ((pipelineResult == null || pipelineResult.isEmpty())
				&& !typeIndicatorTerms.isEmpty()) {
			List<ChartEmbedding> rescued = new ArrayList<>();
			for (int i = 0; i < validCount; i++) {
				if (EmbeddingRankingPipeline.matchesTypeIndicator(embeddings[i],
						typeIndicatorTerms)) {
					rescued.add(embeddings[i]);
				}
			}
			if (!rescued.isEmpty()
					&& rescued.size()
					<= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				pipelineResult = rescued;
				log.warn("Type-indicator full-scan rescue: "
						+ "found {} records for {}",
						rescued.size(), typeIndicatorTerms);
			}
		}

		// Concept-name floor check: when pipeline results have no
		// keyword anchoring (either kwCount=0 globally or kwCount>0
		// but no pipeline result has positive keyword score), the
		// results rely entirely on embedding similarity. If all
		// results share a single concept whose name is below the
		// noise floor, return empty (e.g. "COPD exacerbations"
		// returning 3× Fetishism with sim=0.12).
		if (!pipelineResult.isEmpty()
				&& queryTerms.length > 0) {
			boolean pipelineHasKwAnchoring = false;
			if (keywordMatchCount > 0) {
				Set<ChartEmbedding> kwFloorSet =
						Collections.newSetFromMap(
								new IdentityHashMap<
										ChartEmbedding, Boolean>());
				kwFloorSet.addAll(pipelineResult);
				for (int i = 0; i < validCount; i++) {
					if (kwFloorSet.contains(embeddings[i])
							&& keywordScores[i] > 0) {
						pipelineHasKwAnchoring = true;
						break;
					}
				}
			}
			if (!pipelineHasKwAnchoring) {
				Set<String> seenConcepts = new HashSet<>();
				for (ChartEmbedding ce : pipelineResult) {
					String name = ConceptNameUtil.extractConceptName(
							ce.getTextContent());
					if (name != null) {
						seenConcepts.add(name);
					}
				}
				if (seenConcepts.size() == 1) {
					// Single concept: floor check
					String conceptName =
							seenConcepts.iterator().next();
					float[] nameVec =
							provider.embedQuery(conceptName);
					double conceptSim = cosineSimilarity(
							queryVector, nameVec);
					double floorWithMargin = noiseProfile
							.absoluteSimilarityFloor()
							* profiledConfig.conceptFloorMargin;
					if (conceptSim < floorWithMargin) {
						log.warn("Concept-name floor check: "
								+ "sim({})={} < floor*{}={}, "
								+ "returning empty for {}",
								conceptName,
								String.format("%.4f", conceptSim),
								String.format("%.2f",
										profiledConfig.conceptFloorMargin),
								String.format("%.4f",
										floorWithMargin),
								java.util.Arrays.toString(
										queryTerms));
						pipelineResult = Collections.emptyList();
					}
				}
			}
		}

		// Gap-saturation check: when kwCount=0 and the gap analysis
		// found no meaningful gap in the score distribution (i.e.
		// gapCutoff exceeds the saturation threshold), the embedding
		// scores are undifferentiated — every record looks equally
		// relevant. This catches queries about topics not in the
		// patient's chart (e.g. "COPD exacerbations" on a
		// vitals-dominated patient: gapCutoff=158/160=99%).
		// Legitimate queries have clear gaps (80/160=50%).
		if (keywordMatchCount == 0
				&& !pipelineResult.isEmpty()
				&& validCount > 0
				&& profiledConfig.gapSaturationThreshold > 0) {
			double gapRatio = (double) gapCutoff / validCount;
			if (gapRatio > profiledConfig.gapSaturationThreshold) {
				log.warn("Gap-saturation check: "
						+ "gapCutoff={}/{} ({}) — "
						+ "returning empty for {}",
						gapCutoff, validCount,
						String.format("%.0f%%",
								gapRatio * 100),
						java.util.Arrays.toString(
								queryTerms));
				pipelineResult = Collections.emptyList();
			}
		}

		return new FindSimilarResult(pipelineResult, noiseProfile,
				keywordMatchCount);
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
	 * @param queryPrefix prefix prepended to the query before embedding
	 * @param config pipeline tuning parameters
	 * @return filtered, capped, and concept-grouped serialized records,
	 *         or {@code null} if {@code findSimilar} returned {@code null}
	 *         (no embeddings)
	 */
	/** @deprecated topK is no longer used by the pipeline. Use the overload without it. */
	static List<SerializedRecord> findRelevantRecords(
			List<ChartEmbedding> allEmbeddings,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, String question, int topK,
			String queryPrefix, PipelineConfig config) {
		return findRelevantRecords(allEmbeddings, allRecords, provider,
				question, queryPrefix, config);
	}

	static List<SerializedRecord> findRelevantRecords(
			List<ChartEmbedding> allEmbeddings,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		FindSimilarResult fsResult = findSimilar(allEmbeddings, provider,
				question, queryPrefix, config);
		List<ChartEmbedding> similar = fsResult.records;
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

		log.warn("findRelevantRecords: {} embeddings -> {} keys",
				similar.size(), relevantKeys.size());

		return postRetrievalPipeline(allRecords, relevantKeys, question,
				similar.size(), fsResult.keywordMatchCount,
				provider,
				ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX);
	}

	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, PipelineConfig config) {
		return EmbeddingRankingPipeline.filterPipeline(semanticScores,
				keywordScores, embeddings, queryTerms, config);
	}


	static int findAdaptiveCutoff(List<ScoredEmbedding> scored, int limit, double minScore,
			double gapMultiplier, double minGap) {
		return EmbeddingRankingPipeline.findAdaptiveCutoff(scored, limit, minScore,
				gapMultiplier, minGap);
	}

	static boolean applySlimMarginGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount, boolean belowFloorRescued,
			PipelineConfig config) {
		return EmbeddingRankingPipeline.applySlimMarginGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, belowFloorRescued, config);
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

	/**
	 * Re-ranks pipeline results by concept name relevance. For each unique
	 * concept in the result set, embeds the concept name with the query
	 * encoder and scores it against the query vector. Drops concepts whose
	 * name-level similarity falls below the median, keeping only the most
	 * query-relevant concepts.
	 *
	 * <p>This is a precision pass: the initial retrieval (full enriched
	 * text) provides recall, while this re-ranking filters out false
	 * positives caused by shared prefixes (e.g. "Vital signs / Finding —"
	 * making all vitals score similarly for a "BMI" query).
	 */
	/**
	 * Post-cap concept-name precision filter for recency-capped results.
	 * When a recency cap (e.g. "latest BMI") drastically reduces the
	 * set, irrelevant concepts from the same time period can survive.
	 * Scores each unique concept name against the query and drops
	 * concepts below candidate mean - std.
	 */
	/**
	 * Concept-name rescue for post-cap results. When the recency cap
	 * reduces a set to 1-2 records for a zero-keyword query, related
	 * concepts may have been lost (e.g. "Height" for "BMI" when only
	 * "Weight" survived). Scans all records' concept names, scores
	 * them against the query, and rescues the most recent record for
	 * concepts scoring above the existing candidates' minimum score.
	 */
	/**
	 * Multi-concept rescue for comma-separated queries. When a query
	 * lists concepts like "blood pressure, weight, and temperature"
	 * and some are missing from results, rescues the most recent
	 * record for each missing concept by checking which query terms
	 * appear in concept names of allRecords but not in the results.
	 */
	/**
	 * Drops records whose concept name is a weak match for the query.
	 * A weak match is one where the concept name matches fewer query
	 * terms than another record's concept name for the same term.
	 * E.g. "Blood Oxygen Saturation" matches only "blood" (1 term),
	 * while "Systolic Blood Pressure" matches "blood" + "pressure"
	 * (2 terms). SpO2 is weaker and gets dropped.
	 */
	static List<SerializedRecord> dropWeakMatches(
			List<SerializedRecord> records, String[] queryTerms) {
		// Score each unique concept name by how many query terms it matches
		Map<String, Integer> conceptTermCount = new HashMap<>();
		for (SerializedRecord r : records) {
			String cn = ConceptNameUtil.extractConceptName(r.getText());
			if (cn == null) continue;
			if (conceptTermCount.containsKey(cn)) continue;
			String lowerCn = cn.toLowerCase();
			String[] cnWords = lowerCn.split("\\s+");
			int count = 0;
			for (String term : queryTerms) {
				if (termMatchesText(term, lowerCn, cnWords)) {
					count++;
				}
			}
			conceptTermCount.put(cn, count);
		}

		// For each query term, find the max term count among concepts
		// that match it
		Map<String, Integer> termMaxCount = new HashMap<>();
		for (Map.Entry<String, Integer> entry
				: conceptTermCount.entrySet()) {
			String lowerCn = entry.getKey().toLowerCase();
			String[] cnWords = lowerCn.split("\\s+");
			for (String term : queryTerms) {
				if (termMatchesText(term, lowerCn, cnWords)) {
					Integer prev = termMaxCount.get(term);
					if (prev == null || entry.getValue() > prev) {
						termMaxCount.put(term, entry.getValue());
					}
				}
			}
		}

		// Drop records whose concept matches a term but with fewer
		// total term matches than the best record for that term
		List<SerializedRecord> result = new ArrayList<>();
		for (SerializedRecord r : records) {
			String cn = ConceptNameUtil.extractConceptName(r.getText());
			if (cn == null) {
				result.add(r);
				continue;
			}
			Integer myCount = conceptTermCount.get(cn);
			if (myCount == null || myCount == 0) {
				result.add(r);
				continue;
			}
			// Check if this concept is the best for at least one
			// of the terms it matches
			String lowerCn = cn.toLowerCase();
			String[] cnWords = lowerCn.split("\\s+");
			boolean bestForSomeTerm = false;
			for (String term : queryTerms) {
				if (termMatchesText(term, lowerCn, cnWords)) {
					Integer max = termMaxCount.get(term);
					if (max != null && myCount >= max) {
						bestForSomeTerm = true;
						break;
					}
				}
			}
			if (bestForSomeTerm) {
				result.add(r);
			}
		}

		if (result.size() < records.size()) {
			log.warn("Weak-match cleanup: {} -> {}", records.size(),
					result.size());
		}
		return result.isEmpty() ? records : result;
	}

	static List<SerializedRecord> multiConceptRescue(
			List<SerializedRecord> filtered,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, String question,
			String queryPrefix) {
		String normalizedQ = stripQueryStopwords(question);
		String[] queryTerms = extractQueryTerms(normalizedQ);
		if (queryTerms.length < 3) {
			return filtered;
		}

		// Find which query terms are covered by current results
		Set<String> coveredTerms = new HashSet<String>();
		for (SerializedRecord r : filtered) {
			String text = (r.getText() != null
					? r.getText() : "").toLowerCase();
			String[] textWords = text.split("\\s+");
			for (String term : queryTerms) {
				if (termMatchesText(term, text, textWords)) {
					coveredTerms.add(term);
				}
			}
		}

		// For each uncovered term, check if it appears in a concept
		// name of any record (not just body text). Require 3+ concept
		// name matches to confirm it's a real missing concept.
		List<SerializedRecord> rescued = new ArrayList<>(filtered);
		Set<String> rescuedConcepts = new HashSet<>();
		for (SerializedRecord r : filtered) {
			String cn = ConceptNameUtil.extractConceptName(r.getText());
			if (cn != null) rescuedConcepts.add(cn);
		}

		int recencyCap = extractRecencyCap(question);

		for (String term : queryTerms) {
			if (coveredTerms.contains(term)) {
				continue;
			}
			// Collect ALL records for each matching concept name.
			List<SerializedRecord> matches = new ArrayList<>();
			for (SerializedRecord r : allRecords) {
				String conceptName =
						ConceptNameUtil.extractConceptName(r.getText());
				if (conceptName == null
						|| rescuedConcepts.contains(conceptName)) {
					continue;
				}
				String lowerName = conceptName.toLowerCase();
				String[] nameWords = lowerName.split("\\s+");
				if (termMatchesText(term, lowerName, nameWords)) {
					matches.add(r);
				}
			}
			if (!matches.isEmpty()) {
				String cn = ConceptNameUtil.extractConceptName(
						matches.get(0).getText());
				rescuedConcepts.add(cn);
				// Apply recency cap to just the rescued concept's
				// records before adding, so "last 3 visits" keeps
				// 3 per concept without restructuring the existing
				// filtered set.
				if (recencyCap > 0) {
					matches = capPerConcept(matches, recencyCap);
				}
				rescued.addAll(matches);
			}
		}

		if (rescued.size() != filtered.size()) {
			log.warn("Multi-concept rescue: {} -> {} for '{}'",
					filtered.size(), rescued.size(), question);
		}
		return rescued;
	}

	static List<SerializedRecord> conceptNameRescueRecords(
			List<SerializedRecord> filtered,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, float[] queryVector,
			String question) {
		// Score existing candidates' concept names
		Set<String> existingConcepts = new HashSet<>();
		double minExistingScore = Double.MAX_VALUE;
		for (SerializedRecord r : filtered) {
			String name = ConceptNameUtil.extractConceptName(
					r.getText());
			if (name != null && !existingConcepts.contains(name)) {
				existingConcepts.add(name);
				float[] nameVec = provider.embedQuery(name);
				double score = ChartSearchAiUtils.cosineSimilarity(
						queryVector, nameVec);
				if (score < minExistingScore) {
					minExistingScore = score;
				}
			}
		}

		// Score all unique concept names across allRecords
		Map<String, Double> conceptScores = new HashMap<>();
		Map<String, SerializedRecord> mostRecent = new HashMap<>();
		for (SerializedRecord r : allRecords) {
			String name = ConceptNameUtil.extractConceptName(
					r.getText());
			if (name == null || existingConcepts.contains(name)) {
				continue;
			}
			if (!conceptScores.containsKey(name)) {
				float[] nameVec = provider.embedQuery(name);
				conceptScores.put(name,
						ChartSearchAiUtils.cosineSimilarity(
								queryVector, nameVec));
			}
			// allRecords is most-recent-first, so first occurrence
			// is the most recent record for each concept.
			if (!mostRecent.containsKey(name)) {
				mostRecent.put(name, r);
			}
		}

		// Rescue concepts scoring within the top quartile of ALL
		// dataset concept-name scores. Using minExistingScore is too
		// strict when only 1 record exists (e.g. Weight for "BMI" —
		// Height scores lower but should still be rescued).
		double[] allScores = new double[conceptScores.size()
				+ existingConcepts.size()];
		int ai = 0;
		for (double s : conceptScores.values()) {
			allScores[ai++] = s;
		}
		// Include existing scores in the distribution
		for (SerializedRecord r : filtered) {
			String name = ConceptNameUtil.extractConceptName(
					r.getText());
			if (name != null) {
				float[] nameVec = provider.embedQuery(name);
				allScores[ai++] = ChartSearchAiUtils.cosineSimilarity(
						queryVector, nameVec);
			}
		}
		if (ai < allScores.length) {
			allScores = java.util.Arrays.copyOf(allScores, ai);
		}
		java.util.Arrays.sort(allScores);
		int q3Start = allScores.length * 3 / 4;
		double q3Sum = 0;
		for (int i = q3Start; i < allScores.length; i++) {
			q3Sum += allScores[i];
		}
		double rescueThreshold = q3Sum / (allScores.length - q3Start);

		// Find the single best-scoring non-existing concept above
		// the rescue threshold. Limiting to 1 prevents over-rescuing
		// (e.g. Haemoglobin for "BMI" when only Height is needed).
		String bestRescueName = null;
		double bestRescueScore = 0;
		for (Map.Entry<String, Double> entry
				: conceptScores.entrySet()) {
			if (entry.getValue() >= rescueThreshold
					&& entry.getValue() > bestRescueScore) {
				bestRescueScore = entry.getValue();
				bestRescueName = entry.getKey();
			}
		}

		int recencyCap = extractRecencyCap(question);
		List<SerializedRecord> rescued = new ArrayList<>(filtered);
		List<String> rescuedNames = new ArrayList<>();
		if (bestRescueName != null) {
			SerializedRecord r = mostRecent.get(bestRescueName);
			if (r != null) {
				rescued.add(r);
				rescuedNames.add(bestRescueName);
			}
		}

		// Apply recency cap to rescued records too
		if (recencyCap > 0 && rescued.size() > filtered.size()) {
			rescued = capPerConcept(rescued, recencyCap);
		}

		if (rescued.size() > filtered.size()) {
			log.warn("Concept-name rescue: {} -> {} (threshold={}, "
					+ "rescued: {})",
					filtered.size(), rescued.size(),
					String.format("%.4f", rescueThreshold),
					rescuedNames);
		}
		return rescued;
	}

	static List<SerializedRecord> applyPostCapConceptFilter(
			List<SerializedRecord> records,
			EmbeddingProvider provider, String normalizedQuery,
			String queryPrefix) {
		Map<String, List<SerializedRecord>> byConcept =
				new java.util.LinkedHashMap<String, List<SerializedRecord>>();
		for (SerializedRecord r : records) {
			String name = ConceptNameUtil.extractConceptName(r.getText());
			if (name == null) {
				name = r.getResourceType() + ":" + r.getResourceId();
			}
			byConcept.computeIfAbsent(name,
					k -> new ArrayList<SerializedRecord>()).add(r);
		}
		if (byConcept.size() < 2) {
			return records;
		}

		String embeddingQuery = QueryPreprocessor.buildEmbeddingQuery(normalizedQuery);
		float[] queryVector = provider.embedQuery(
				queryPrefix + embeddingQuery);
		double[] candScores = new double[byConcept.size()];
		String[] candNames = new String[byConcept.size()];
		int ci = 0;
		for (String name : byConcept.keySet()) {
			float[] nameVec = provider.embedQuery(name);
			candScores[ci] = ChartSearchAiUtils.cosineSimilarity(
					queryVector, nameVec);
			candNames[ci] = name;
			ci++;
		}

		double sum = 0;
		for (double s : candScores) sum += s;
		double mean = sum / candScores.length;
		double sqSum = 0;
		for (double s : candScores) {
			sqSum += (s - mean) * (s - mean);
		}
		double std = Math.sqrt(sqSum / candScores.length);
		if (std < 1e-9) {
			return records;
		}
		// Use mean - 0.5*std: tighter than mean - std (which lets
		// "Serum glucose" through for BMI on THIRD) but not as
		// aggressive as mean (which drops "Height" for BMI on FULL).
		double threshold = mean - 0.5 * std;

		List<SerializedRecord> result = new ArrayList<>();
		for (int i = 0; i < candScores.length; i++) {
			if (candScores[i] >= threshold) {
				result.addAll(byConcept.get(candNames[i]));
			}
		}
		if (result.size() < records.size() && !result.isEmpty()) {
			log.warn("Post-cap concept filter: {} -> {} (threshold={})",
					records.size(), result.size(),
					String.format("%.4f", threshold));
			return result;
		}
		return records;
	}

	static List<ChartEmbedding> rerankByConceptName(
			List<ChartEmbedding> candidates,
			ChartEmbedding[] allEmbeddings, int allCount,
			float[] queryVector, EmbeddingProvider provider,
			ModelNoiseProfile noiseProfile,
			PipelineConfig config) {
		return rerankByConceptName(candidates, allEmbeddings, allCount,
				queryVector, provider, noiseProfile, config, false);
	}

	/**
	 * @param forceGate when true, skip the small-result bypass so the
	 *        z-score outlier gate always runs. Used when type indicators
	 *        are present but zero pipeline results match the indicated
	 *        type — the z-score gate uses ALL concept names (not just
	 *        candidates) so it is statistically valid even for small sets.
	 */
	static List<ChartEmbedding> rerankByConceptName(
			List<ChartEmbedding> candidates,
			ChartEmbedding[] allEmbeddings, int allCount,
			float[] queryVector, EmbeddingProvider provider,
			ModelNoiseProfile noiseProfile,
			PipelineConfig config, boolean forceGate) {
		// Group candidates by concept name
		Map<String, List<ChartEmbedding>> byConcept =
				new java.util.LinkedHashMap<String, List<ChartEmbedding>>();
		for (ChartEmbedding ce : candidates) {
			String name = ConceptNameUtil.extractConceptName(
					ce.getTextContent());
			if (name == null) {
				name = ce.getResourceType() + ":" + ce.getResourceId();
			}
			byConcept.computeIfAbsent(name,
					k -> new ArrayList<ChartEmbedding>()).add(ce);
		}

		// Collect ALL unique concept names in the dataset
		Set<String> allConceptNames =
				new java.util.LinkedHashSet<String>();
		for (int i = 0; i < allCount; i++) {
			String name = ConceptNameUtil.extractConceptName(
					allEmbeddings[i].getTextContent());
			if (name != null) {
				allConceptNames.add(name);
			}
		}
		allConceptNames.addAll(byConcept.keySet());

		// Score ALL concept names against the query
		Map<String, Double> conceptScores =
				new HashMap<String, Double>();
		for (String name : allConceptNames) {
			float[] nameVec = provider.embedQuery(name);
			conceptScores.put(name,
					ChartSearchAiUtils.cosineSimilarity(
							queryVector, nameVec));
		}

		// Small-result bypass: skip the gate when the result
		// set is below the model's configured minimum. Dual-
		// encoder models (MedCPT, minCandidates=2) produce
		// reliable concept-name z-scores even for small sets;
		// single-encoder models (L6-v2, minCandidates=10) need
		// larger sets AND at least 3 candidate concepts because
		// their z-scores for vocabulary-mismatch queries are
		// unreliable. Bypassed when forceGate is true (e.g.
		// type indicators present but zero results match the
		// indicated type — the z-score gate uses the full
		// concept-name distribution so it remains valid).
		if (!forceGate
				&& (candidates.size()
						< config.conceptNameGateMinCandidates
				|| byConcept.size()
						< config.conceptNameGateMinCandidates)) {
			return candidates;
		}

		// Full-distribution concept-name outlier gate: check if
		// any CANDIDATE concept name is a statistical outlier
		// among ALL concept names in the dataset. Uses N^(3/4)
		// as effective degrees of freedom. If no candidate stands
		// out, the pipeline found coincidental embedding overlap.
		if (allConceptNames.size() >= 5) {
			double[] cnScores = new double[conceptScores.size()];
			int ci = 0;
			for (double s : conceptScores.values()) {
				cnScores[ci++] = s;
			}
			double cnSum = 0;
			for (double s : cnScores) cnSum += s;
			double cnMean = cnSum / cnScores.length;
			double cnSqSum = 0;
			for (double s : cnScores) {
				cnSqSum += (s - cnMean) * (s - cnMean);
			}
			double cnStd = Math.sqrt(cnSqSum / cnScores.length);
			if (cnStd > 1e-9) {
				double maxCandidateScore = 0;
				for (String name : byConcept.keySet()) {
					Double s = conceptScores.get(name);
					if (s != null && s > maxCandidateScore) {
						maxCandidateScore = s;
					}
				}
				double candidateZ = (maxCandidateScore - cnMean)
						/ cnStd;
				double effectiveN = Math.pow(cnScores.length,
					0.75);
				double zThreshold = Math.sqrt(
						2 * Math.log(Math.max(2, effectiveN)));
				if (candidateZ < zThreshold) {
					log.warn("Concept-name outlier gate: z={}"
							+ " < threshold={} (best={}, N={})",
							String.format("%.2f", candidateZ),
							String.format("%.2f", zThreshold),
							String.format("%.4f",
									maxCandidateScore),
							cnScores.length);
					return Collections.emptyList();
				}
			}
		}

		// Compute mean and std of concept-name scores, then keep
		// concepts within 1 std of the max. This adapts to each
		// query's score distribution without a fixed threshold.
		double[] scores = new double[conceptScores.size()];
		int idx = 0;
		for (double s : conceptScores.values()) {
			scores[idx++] = s;
		}
		double sum = 0;
		for (double s : scores) sum += s;
		double mean = sum / scores.length;
		double sqSum = 0;
		for (double s : scores) sqSum += (s - mean) * (s - mean);
		double std = Math.sqrt(sqSum / scores.length);

		if (std < 1e-9) {
			return candidates;
		}

		// Drop only outliers: concepts scoring below mean - std.
		// This preserves gradients (most concepts within 1 std of
		// mean) while dropping clear false positives (concepts far
		// below the cluster center).
		double threshold = mean - std;

		List<ChartEmbedding> reranked = new ArrayList<ChartEmbedding>();
		for (Map.Entry<String, List<ChartEmbedding>> entry
				: byConcept.entrySet()) {
			if (conceptScores.get(entry.getKey()) >= threshold) {
				reranked.addAll(entry.getValue());
			}
		}

		// If nothing was dropped, return original
		if (reranked.size() == candidates.size()) {
			return candidates;
		}

		log.warn("Concept-name re-ranking: {} concepts, mean={}, "
				+ "std={}, threshold={}, kept={} of {} records",
				byConcept.size(),
				String.format("%.4f", mean),
				String.format("%.4f", std),
				String.format("%.4f", threshold),
				reranked.size(), candidates.size());

		return reranked.isEmpty() ? candidates : reranked;
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

	static String[] expandKwTermsViaConceptSimilarity(String[] kwTerms,
			float[] queryVector, ChartEmbedding[] embeddings,
			EmbeddingProvider provider, PipelineConfig config) {
		return EmbeddingRankingPipeline.expandKwTermsViaConceptSimilarity(
				kwTerms, queryVector, embeddings, provider, config);
	}

	private static String getQueryPrefix() {
		String value = org.openmrs.api.context.Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_QUERY_PREFIX);
		if (value != null && !value.trim().isEmpty()) {
			return value;
		}
		return ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX;
	}

	static String prepareEmbeddingInput(String question, String queryPrefix) {
		return QueryPreprocessor.prepareEmbeddingInput(question, queryPrefix);
	}

	static String[] extractQueryTerms(String normalizedQuery) {
		return QueryPreprocessor.extractQueryTerms(normalizedQuery);
	}

	static double computeKeywordScore(String[] queryTerms, String textContent) {
		return SimilarityAndScoringEngine.computeKeywordScore(queryTerms, textContent);
	}

	static double computeKeywordScoreRestricted(String[] queryTerms,
			String prefixedText, String body,
			Set<String> typeIndicatorTerms) {
		return SimilarityAndScoringEngine.computeKeywordScoreRestricted(
				queryTerms, prefixedText, body, typeIndicatorTerms);
	}

	static boolean termMatchesText(String term, String lowerText, String[] textWords) {
		return SimilarityAndScoringEngine.termMatchesText(term, lowerText, textWords);
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

	static List<ScoredEmbedding> growCluster(List<ScoredEmbedding> candidates,
			int seedSize, double cosineThreshold) {
		return EmbeddingRankingPipeline.growCluster(candidates, seedSize, cosineThreshold);
	}

	static boolean isGapCoherent(List<ScoredEmbedding> scored, int cutoff,
			double cosineThreshold) {
		return EmbeddingRankingPipeline.isGapCoherent(scored, cutoff, cosineThreshold);
	}

	static List<ScoredEmbedding> rescueBelowFloor(List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored, int adaptiveCutoff) {
		return EmbeddingRankingPipeline.rescueBelowFloor(candidates, scored, adaptiveCutoff);
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

	private static double computeSemanticZScore(List<ScoredEmbedding> scored,
			double maxSemanticScore) {
		return SimilarityAndScoringEngine.computeSemanticZScore(scored, maxSemanticScore);
	}

	static double effectiveGumbelThreshold(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.effectiveGumbelThreshold(scored);
	}

	static double clusterGumbelThreshold(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.clusterGumbelThreshold(scored);
	}

	static double medianGumbelThreshold(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.medianGumbelThreshold(scored);
	}

	static double floorRescueGumbelThreshold(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.floorRescueGumbelThreshold(scored);
	}

	static double secondPassMinGap(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.secondPassMinGap(scored);
	}

	static double refinementAdaptiveGapRatio(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.refinementAdaptiveGapRatio(scored);
	}

	static double refinementSemanticRatio(List<ScoredEmbedding> candidates) {
		return SimilarityAndScoringEngine.refinementSemanticRatio(candidates);
	}

	private static boolean hasStatisticalVariance(List<ScoredEmbedding> scored) {
		return SimilarityAndScoringEngine.hasStatisticalVariance(scored);
	}
}
