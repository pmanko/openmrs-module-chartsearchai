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
		List<ChartEmbedding> pipelineResult = filterPipeline(
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
			pipelineResult = filterPipeline(coreSemantic, coreKw,
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
				List<ChartEmbedding> mergeResult = filterPipeline(
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
				if (matchesTypeIndicator(ce, typeIndicatorTerms)) {
					typeMatchCount++;
				}
			}
			if (typeMatchCount > 0
					&& typeMatchCount > pipelineResult.size() * 2 / 3) {
				int beforeFilter = pipelineResult.size();
				List<ChartEmbedding> filtered = new ArrayList<>();
				for (ChartEmbedding ce : pipelineResult) {
					if (matchesTypeIndicator(ce, typeIndicatorTerms)) {
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
					if (matchesTypeIndicator(ce,
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
					if (matchesTypeIndicator(embeddings[i],
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
				if (matchesTypeIndicator(embeddings[i],
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

	/**
	 * Overload that accepts a term count instead of actual query terms.
	 * The keyword-tier subset check is skipped since the actual terms
	 * are not available.
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			int queryTermCount, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				null, queryTermCount,  config);
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
	 *        lack keyword matches; bypassed when every candidate has a
	 *        keyword match (gap detection + ratio floor already identified
	 *        the relevant cluster in that case)
	 * @param config pipeline tuning parameters
	 * @return filtered list of relevant embeddings, or empty list
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, queryTerms != null ? queryTerms.length : 0,
				config);
	}

	private static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int queryTermCount,
			PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, queryTermCount, config, null);
	}

	/**
	 * @param outGapCutoff if non-null, receives the gap-analysis candidate
	 *        count in [0]. Used by the noise-saturation check to detect
	 *        queries where the dataset has no genuine signal.
	 */
	private static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int queryTermCount,
			PipelineConfig config, int[] outGapCutoff) {

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

		// Keyword flooding guard: when > 50% of records match keywords,
		// the matching terms are too generic to distinguish relevant
		// from irrelevant records (e.g. "normal", "range", "time" in
		// "What are this patient's HB results over time..."). Zero
		// out keyword scores and re-sort by semantic score only.
		if (keywordMatchCount > scored.size() / 2
				&& queryTermCount > 2) {
			log.warn("Keyword flooding: {}/{} records match keywords,"
					+ " zeroing keyword scores for {}",
					keywordMatchCount, scored.size(),
					java.util.Arrays.toString(queryTerms));
			maxBaseScore = 0;
			for (int i = 0; i < scored.size(); i++) {
				ScoredEmbedding se = scored.get(i);
				scored.set(i, new ScoredEmbedding(se.embedding,
						se.semanticScore, 0, se.semanticScore));
				if (se.semanticScore > maxBaseScore) {
					maxBaseScore = se.semanticScore;
				}
			}
			keywordMatchCount = 0;
			Collections.sort(scored, new Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.score, a.score);
				}
			});
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
				maxSemanticScore, queryTermCount, keywordMatchCount,
				config.floorRescueMinZScore);
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
		if (outGapCutoff != null && outGapCutoff.length > 0) {
			outGapCutoff[0] = adaptiveCutoff;
		}
		List<ScoredEmbedding> candidates = built.candidates;

		// Keyword refinement: when gap detection returns a broad set but
		// keyword matches identify a specific subset, prefer those records.
		// Save pre-refinement candidates for semantic core discovery.
		List<ScoredEmbedding> preRefinementCandidates = candidates;
		boolean refinementActivated = false;
		boolean selectiveKwRescued = false;
		if (config.keywordWeight > 0) {
			List<ScoredEmbedding> refined = refineByKeywords(candidates, queryTermCount);
			refinementActivated = refined.size() < candidates.size();
			// Partial-match semantic floor: when keyword refinement
			// kept only PARTIAL matches (no record matches all query
			// terms) AND the best semantic score is below the noise
			// floor, the keyword overlap is coincidental text — the
			// embedding model sees no topical relevance. Remove
			// those records so the non-refinement path's floor gates
			// can reject them cleanly.
			if (refinementActivated) {
				double rMaxSem = 0;
				double rMaxKw = 0;
				for (ScoredEmbedding se : refined) {
					if (se.semanticScore > rMaxSem) {
						rMaxSem = se.semanticScore;
					}
					if (se.keywordScore > rMaxKw) {
						rMaxKw = se.keywordScore;
					}
				}
				// Partial-match semantic floor: when keyword refinement
				// kept only PARTIAL matches (no record matches all query
				// terms) AND the best semantic score is below the noise
				// floor, the keyword overlap is coincidental — the
				// embedding model sees no topical relevance. The floor
				// is max(absoluteFloor, noiseMean - 1 std): for models
				// with wide distributions (L6-v2, noiseMean=0.26) this
				// stays at the absolute floor; for compressed models
				// (MedCPT, noiseMean=0.92) it rises to catch scores
				// that look high in absolute terms but are below the
				// noise baseline.
				double partialKwFloor = Math.max(
						config.noiseProfile.absoluteSimilarityFloor(),
						config.noiseProfile.noiseMean
								- config.noiseProfile.noiseStd);
				// Selective-keyword exception: when the refined
				// set is very small relative to the corpus, the
				// keyword match is highly selective — only a
				// handful of records contain the matching term.
				// Keep ONLY the keyword-matched records and skip
				// the refinement expansion path. Expansion into
				// compressed score spaces (e.g. MedCPT) would flood
				// non-keyword records; the keyword evidence is the
				// sole structural signal. The absolute floor
				// (ADAPTIVE_MIN_RECORDS) handles small charts; on
				// large charts (corpus >
				// LARGE_CORPUS_SELECTIVE_RESCUE_MIN) the gate
				// extends to a fractional threshold so a 6/462
				// match (1.3 %) still counts as selective even
				// though it exceeds the absolute floor of 2.
				int selectiveKwMaxRecords =
						ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
				if (scored.size()
						> ChartSearchAiConstants.LARGE_CORPUS_SELECTIVE_RESCUE_MIN) {
					selectiveKwMaxRecords = Math.max(
							selectiveKwMaxRecords,
							(int) Math.ceil(scored.size()
									* ChartSearchAiConstants.LARGE_CORPUS_SELECTIVE_KW_FRACTION));
				}
				boolean selectiveKwMatch = refined.size()
						<= selectiveKwMaxRecords;
				if (selectiveKwMatch && rMaxKw < bonusThreshold
						&& rMaxSem < partialKwFloor) {
					// Keep the keyword-matched records but route
					// through the non-refinement path (which has
					// ratio-floor and z-score guards) instead of the
					// refinement expansion path (which would flood
					// on compressed score models).
					refinementActivated = false;
					selectiveKwRescued = true;
					log.debug("Selective-keyword rescue: keeping {} "
							+ "records via non-refinement path "
							+ "(maxSem={}, floor={})",
							refined.size(),
							String.format("%.4f", rMaxSem),
							String.format("%.4f", partialKwFloor));
				} else if (rMaxKw < bonusThreshold
						&& rMaxSem < partialKwFloor) {
					Set<Integer> badIds =
							new HashSet<Integer>();
					for (ScoredEmbedding se : refined) {
						badIds.add(
								se.embedding.getResourceId());
					}
					List<ScoredEmbedding> cleaned =
							new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (!badIds.contains(
								se.embedding
										.getResourceId())) {
							cleaned.add(se);
						}
					}
					log.warn("Partial-match semantic floor: "
							+ "removing {} below-floor partial "
							+ "kw records (maxSem={}, floor={})",
							badIds.size(),
							String.format("%.4f", rMaxSem),
							String.format("%.4f", partialKwFloor));
					refined = cleaned;
					refinementActivated = false;
				}
			}
			candidates = refined;
		}

		log.warn("Pipeline stages {}: maxSem={}, maxBase={}, floor={}, kwCount={}, "
				+ "termCount={}, gapCutoff={}, candidates={}, refined={}",
				java.util.Arrays.toString(queryTerms), String.format("%.4f", maxSemanticScore),
				String.format("%.4f", maxBaseScore),
				String.format("%.4f", config.noiseProfile.absoluteSimilarityFloor()),
				keywordMatchCount, queryTermCount,
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
		// second-pass gap detection, then ratio floor
		if (refinementActivated) {
			// Refinement path: keyword refinement found a relevant
			// subset. See applyRefinementPath for the partial-keyword
			// semantic-core and non-uniform gap-detection branches.
			boolean[] pkvOut = { false };
			candidates = applyRefinementPath(candidates,
					preRefinementCandidates, scored, queryTerms,
					queryTermCount, bonusThreshold, minScore, 
					config, pkvOut);
			partialKwValidated = pkvOut[0];
		} else {
			// Non-refinement path: second-pass gap detection, ratio
			// floor with concept-pairing rescue.
			int[] rfccOut = { -1 };
			candidates = applyNonRefinementPath(candidates, scored,
					maxSemanticScore, maxBaseScore, minScore, 
					config, rfccOut);
			ratioFloorCandidateCount = rfccOut[0];
		}

		// Keyword-dominance rescue: when all post-processed candidates
		// have keyword matches (kw > 0) and the set is small, the
		// embedding model's lexical-overlap bias may have inflated
		// keyword-matching records' semantic scores above topically-
		// related records that use different vocabulary (e.g.
		// "Temperature: 37.7" for a "fever" query). Scan the full
		// scored list for non-keyword records that are (a) above the
		// ratio floor and (b) individually coherent with the keyword
		// set above a data-derived threshold: sqrt(noiseMean *
		// noiseP95) — the geometric mean of the noise center and
		// ceiling, scaled by keyword confidence. This threshold is
		// entirely derived from the patient's own cross-concept
		// embedding statistics.
		if (candidates.size()
				<= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS
				&& !candidates.isEmpty()) {
			boolean allKw = true;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore == 0) {
					allKw = false;
					break;
				}
			}
			if (allKw) {
				double kwMaxSem = 0;
				double kwMaxScore = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > kwMaxSem) {
						kwMaxSem = se.semanticScore;
					}
					if (se.keywordScore > kwMaxScore) {
						kwMaxScore = se.keywordScore;
					}
				}
				double kwConf = queryTermCount > 0
						? kwMaxScore * queryTermCount
								/ Math.max(bonusThreshold, 0.01)
						: 1.0;
				double perRecordThreshold = Math.sqrt(
						config.noiseProfile.noiseMean
						* config.noiseProfile.noiseP95) * kwConf;
				// When the selective-keyword rescue saved these
				// candidates from below the noise floor, their
				// semantic scores are systematically lower than
				// the noise baseline. Non-keyword records that are
				// topically related will be similarly depressed.
				// Widen the merge floor by squaring the ratio
				// (0.98² ≈ 0.96) to compensate. The coherence
				// check guards precision. Outside the selective
				// rescue path, the standard ratio floor applies.
				double rescueRatio = selectiveKwRescued
						? config.similarityRatio
								* config.similarityRatio
						: config.similarityRatio;
				double mergeFloor = kwMaxSem * rescueRatio;
				Set<Integer> inCandidates = new HashSet<Integer>();
				for (ScoredEmbedding se : candidates) {
					inCandidates.add(
							se.embedding.getResourceId());
				}

				// For the selective-keyword rescue path, compute a
				// distribution-based coherence threshold instead of
				// the fixed noise-profile one. The fixed threshold
				// (sqrt(noiseMean * noiseP95)) uses doc-doc stats
				// which, for compressed models, can reject related-
				// but-different concepts (e.g. "Nonunion of fracture"
				// vs "Crushing injury": coherence 0.935 < threshold
				// 0.940). The distribution-based approach computes
				// each record's coherence with the candidates and
				// sets the threshold at mean + std — records that are
				// statistical outliers in coherence are genuinely
				// related. This naturally adapts: compressed models
				// get a tighter band (std ~0.01), wide-range models
				// get a wider band (std ~0.15).
				double effectiveThreshold = perRecordThreshold;
				if (selectiveKwRescued) {
					List<Double> cohValues =
							new ArrayList<Double>();
					for (ScoredEmbedding se : scored) {
						if (inCandidates.contains(
								se.embedding.getResourceId())
								|| se.keywordScore > 0) {
							continue;
						}
						float[] vec = se.embedding
								.getEmbeddingVector();
						if (vec == null) {
							continue;
						}
						double cs = 0;
						int cn = 0;
						for (ScoredEmbedding kw : candidates) {
							float[] kwVec = kw.embedding
									.getEmbeddingVector();
							if (kwVec != null
									&& kwVec.length
									== vec.length) {
								cs += ChartSearchAiUtils
										.cosineSimilarity(
												vec, kwVec);
								cn++;
							}
						}
						if (cn > 0) {
							cohValues.add(cs / cn);
						}
					}
					if (cohValues.size() >= 4) {
						double cohSum = 0;
						for (double v : cohValues) {
							cohSum += v;
						}
						double cohMean = cohSum
								/ cohValues.size();
						double cohSqSum = 0;
						for (double v : cohValues) {
							double d = v - cohMean;
							cohSqSum += d * d;
						}
						double cohStd = Math.sqrt(cohSqSum
								/ cohValues.size());
						effectiveThreshold = cohMean + cohStd / 2;
						log.debug("Selective rescue coherence "
								+ "threshold: mean={}, std={}, "
								+ "effective={} (fixed was {})",
								String.format("%.4f", cohMean),
								String.format("%.4f", cohStd),
								String.format("%.4f",
										effectiveThreshold),
								String.format("%.4f",
										perRecordThreshold));
					}
				}

				List<ScoredEmbedding> rescued =
						new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : scored) {
					if (inCandidates.contains(
							se.embedding.getResourceId())
							|| se.keywordScore > 0
							|| se.semanticScore < mergeFloor) {
						continue;
					}
					float[] vec = se.embedding
							.getEmbeddingVector();
					if (vec == null) {
						continue;
					}
					double cosSum = 0;
					int cnt = 0;
					for (ScoredEmbedding kw : candidates) {
						float[] kwVec = kw.embedding
								.getEmbeddingVector();
						if (kwVec != null
								&& kwVec.length == vec.length) {
							cosSum += ChartSearchAiUtils
									.cosineSimilarity(
											vec, kwVec);
							cnt++;
						}
					}
					if (cnt > 0
							&& cosSum / cnt
							>= effectiveThreshold) {
						rescued.add(se);
					}
				}
				if (!rescued.isEmpty()) {
					// Concept-pairing for rescued records (only
					// in the selective-keyword rescue path):
					// when a condition is rescued, its diagnosis
					// partner (or vice versa) may have missed the
					// mergeFloor by a small margin. Pull in
					// partners whose concept name matches a rescued
					// record — these are the same clinical finding
					// recorded as different resource types.
					Set<String> rescuedConcepts =
							new HashSet<String>();
					Set<Integer> rescuedIds =
							new HashSet<Integer>();
					for (ScoredEmbedding se : rescued) {
						rescuedIds.add(
								se.embedding.getResourceId());
						String cn = ConceptNameUtil
								.extractConceptName(
										se.embedding
												.getTextContent());
						if (cn != null) {
							rescuedConcepts.add(cn);
						}
					}
					if (selectiveKwRescued
						&& !rescuedConcepts.isEmpty()) {
						for (ScoredEmbedding se : scored) {
							int rid = se.embedding
									.getResourceId();
							if (inCandidates.contains(rid)
									|| rescuedIds.contains(rid)) {
								continue;
							}
							String cn = ConceptNameUtil
									.extractConceptName(
											se.embedding
													.getTextContent());
							if (cn != null
									&& rescuedConcepts
											.contains(cn)) {
								rescued.add(se);
								rescuedIds.add(rid);
							}
						}
					}
					log.warn("Keyword-dominance rescue: merging "
							+ "{} non-kw records (threshold={}, "
							+ "kwConf={}) with {} kw records",
							rescued.size(),
							String.format("%.4f",
									effectiveThreshold),
							String.format("%.2f", kwConf),
							candidates.size());
					List<ScoredEmbedding> merged =
							new ArrayList<ScoredEmbedding>(
									candidates);
					merged.addAll(rescued);
					candidates = merged;
					partialKwValidated = true;
				}
			}
		}

		// Phase 1: Outlier removal via coherence gap detection, guarded
		// by full-keyword-match and compound-keyword-match bypasses.
		candidates = applyOutlierRemovalPhase1(candidates, queryTerms,
				partialKwValidated);
		// Phase 2: Zero-keyword validation — mean coherence, small-
		// cluster coherence gate, and cluster z-score gate. Bypassed
		// when any candidate has keyword evidence.
		candidates = applyZeroKeywordValidationPhase2(candidates,
				scored, config, partialKwValidated, belowFloorRescued,
				firstPassGapDetected, adaptiveCutoff,
				ratioFloorCandidateCount, initialZScore,
				initialZThreshold, maxSemanticScore);

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		for (ScoredEmbedding se : candidates) {
			results.add(se.embedding);
		}

		// Safety net: if all downstream gates (gap detection, coherence,
		// phase 2 z-score) rejected every record, but:
		// 1. The initial z-score gate validated the top score as a
		//    genuine statistical outlier (≥ 2x threshold), AND
		// 2. The top semantic score clears floor + 2*minScoreGap,
		// return the top record as a last-resort fallback. In a clinical
		// system, silently dropping a record that passed both absolute
		// and statistical validation is more dangerous than returning
		// a single potentially marginal result.
		if (results.isEmpty() && !scored.isEmpty()
				&& initialZScore >= 0
				&& initialZScore >= 2 * initialZThreshold
				&& scored.get(0).semanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ 2 * config.minScoreGap) {
			log.warn("Safety net: all gates rejected {} records but top "
					+ "semantic score {} >= floor+2*gap {} and z-score "
					+ "{} >= 2*threshold {}, returning top record as "
					+ "fallback",
					scored.size(),
					String.format("%.4f", scored.get(0).semanticScore),
					String.format("%.4f",
							config.noiseProfile.absoluteSimilarityFloor()
									+ 2 * config.minScoreGap),
					String.format("%.2f", initialZScore),
					String.format("%.2f", 2 * initialZThreshold));
			results = Collections.singletonList(scored.get(0).embedding);
		}

		// Multi-concept rescue: when a query mentions multiple distinct
		// concepts (e.g. "blood pressure, weight, and temperature"),
		// one concept can dominate keyword scoring (BP matches 2 terms)
		// and push others below the gap. Rescue the best record for
		// each query term that (a) has no matching candidate, (b) has
		// keyword-matching records in the full set, and (c) has at
		// least 3 such records (confirming it's a real concept, not
		// a coincidental text match).

		int logLimit = Math.min(20, scored.size());
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
		log.warn("Similarity scores: [{}], minScore: {}, adaptiveCutoff: {}",
				scores, String.format("%.4f", minScore), adaptiveCutoff);
		log.warn("Returning {} {} of {} candidates (topScore={})",
				java.util.Arrays.toString(queryTerms), results.size(), scored.size(),
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
				log.warn("Floor gate z-score rescue: zScore={}, "
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
			// Allow a single record if its score is in the upper half
			// of the slim-margin zone [floor + gap/2, floor + gap).
			// Records barely above the floor are likely noise; records
			// further above are more likely genuine.
			if (aboveFloorCount == 1
					&& maxSemanticScore >= smFloor + config.minScoreGap / 2) {
				if (log.isDebugEnabled()) {
					log.debug("Slim-margin gate: single record at {} is in "
							+ "upper half of margin zone [{}, {}), allowing",
							String.format("%.4f", maxSemanticScore),
							String.format("%.4f", smFloor + config.minScoreGap / 2),
							String.format("%.4f", smFloor + config.minScoreGap));
				}
				return true;
			}
			log.warn("Slim-margin gate: maxSem={} is within "
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

	static double[] applyInitialZScoreGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount) {
		return applyInitialZScoreGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, -1);
	}

	static double[] applyInitialZScoreGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount, double maxZScoreThreshold) {
		double[] result = { -1, -1 };
		if (queryTermCount <= 0 || !hasStatisticalVariance(scored)) {
			return result;
		}
		if (keywordMatchCount >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			return result;
		}
		double zScore = computeSemanticZScore(scored, maxSemanticScore);
		double threshold = effectiveGumbelThreshold(scored);
		// Allow model-specific configs to cap the threshold for
		// compressed-score models where the Gumbel threshold is
		// too aggressive.
		if (maxZScoreThreshold > 0 && threshold > maxZScoreThreshold) {
			threshold = maxZScoreThreshold;
		}
		result[0] = zScore;
		result[1] = threshold;
		if (zScore < threshold) {
			log.debug("Only {} keyword match(es) and top semantic "
					+ "z-score {} is below threshold {}, "
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
	 * Refinement post-processing path — stage 5a of the filter pipeline.
	 *
	 * <p>Runs when {@code refinementActivated} is true (keyword refinement
	 * identified a relevant subset). Computes uniform-keyword and
	 * semantic-dominance signals, then branches into the partial-keyword
	 * semantic-core path or the non-uniform second-pass gap detection path.
	 *
	 * @param partialKwValidatedOut set to {@code true} when the semantic
	 *        core expansion path fired — used downstream to skip
	 *        coherence filtering that would double-filter the core.
	 * @return the refined candidate list
	 */
	static List<ScoredEmbedding> applyRefinementPath(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> preRefinementCandidates,
			List<ScoredEmbedding> scored,
			String[] queryTerms, int queryTermCount,
			double bonusThreshold, double minScore, 
			PipelineConfig config, boolean[] partialKwValidatedOut) {
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
		boolean semanticDominance = false;
		double maxKwSem = 0;
		double maxNonKwSem = 0;
		if (uniformKeywords && kwMax < bonusThreshold
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
				&& kwMax < bonusThreshold;
		log.warn("Pipeline refinement: kwMin={}, kwMax={}, uniform={}, partialKw={}, semDom={}",
				String.format("%.4f", kwMin),
				String.format("%.4f", kwMax), uniformKeywords,
				partialKeywordMatch, semanticDominance);
		if (partialKeywordMatch) {
			List<ScoredEmbedding> nonKeyword = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : preRefinementCandidates) {
				if (se.keywordScore == 0) {
					nonKeyword.add(se);
				}
			}
			Collections.sort(nonKeyword,
					new java.util.Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.semanticScore, a.semanticScore);
				}
			});
			List<ScoredEmbedding> semanticCore = new ArrayList<ScoredEmbedding>();
			boolean coreFallbackUsed = false;
			int coreCutoff = nonKeyword.size();
			if (nonKeyword.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				double maxSemNk = nonKeyword.get(0).semanticScore;
				double nkMinGap = Math.max(
						maxSemNk * refinementAdaptiveGapRatio(scored),
						secondPassMinGap(scored));
				coreCutoff = findAdaptiveCutoff(nonKeyword, nonKeyword.size(),
						config.noiseProfile.absoluteSimilarityFloor(),
						config.scoreGapMultiplier, nkMinGap);
				int fisherCut = nonKeyword.size();
				double totalSumF = 0;
				double totalSumSqF = 0;
				for (ScoredEmbedding se : nonKeyword) {
					totalSumF += se.semanticScore;
					totalSumSqF += se.semanticScore * se.semanticScore;
				}
				int nF = nonKeyword.size();
				double leftSumF = 0;
				double leftSumSqF = 0;
				double rightSumF = totalSumF;
				double rightSumSqF = totalSumSqF;
				double maxGF = 0;
				for (int i = 1; i < nF; i++) {
					double s = nonKeyword.get(i - 1).semanticScore;
					leftSumF += s;
					leftSumSqF += s * s;
					rightSumF -= s;
					rightSumSqF -= s * s;
					if (i < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
						continue;
					}
					int nR = nF - i;
					if (nR < 2) {
						break;
					}
					double lm = leftSumF / i;
					double rm = rightSumF / nR;
					double lv = leftSumSqF / i - lm * lm;
					double rv = rightSumSqF / nR - rm * rm;
					double den = lv / (i - 1) + rv / (nR - 1) + 1e-12;
					double diff = lm - rm;
					double t2 = diff * diff / den;
					double bGap = nonKeyword.get(i - 1).semanticScore
							- nonKeyword.get(i).semanticScore;
					double gf = bGap * t2;
					if (gf > maxGF) {
						maxGF = gf;
						fisherCut = i;
					}
				}
				log.debug("Fisher cluster cut at {} (score={})",
						fisherCut, String.format("%.2f", maxGF));
				if (fisherCut < coreCutoff) {
					double bGapF = nonKeyword.get(fisherCut - 1).semanticScore
							- nonKeyword.get(fisherCut).semanticScore;
					double prevGapF = nonKeyword.get(fisherCut - 2).semanticScore
							- nonKeyword.get(fisherCut - 1).semanticScore;
					if (bGapF > prevGapF) {
						coreFallbackUsed = true;
						coreCutoff = fisherCut;
					} else {
						log.debug("Fisher cut at {} suppressed: boundary gap {} <= prev gap {}",
								fisherCut,
								String.format("%.4f", bGapF),
								String.format("%.4f", prevGapF));
					}
				}
				for (int i = 0; i < coreCutoff; i++) {
					semanticCore.add(nonKeyword.get(i));
				}
				log.warn("Semantic core: {} non-keyword records, primary gap at {}, core size {}",
						nonKeyword.size(), coreCutoff, semanticCore.size());
				// Coherence-outlier concept pruning: when the core
				// contains ≥3 distinct concepts (each with ≥2 members,
				// i.e. condition+diagnosis pairs), gap detection may
				// have admitted a topical neighbour that the embedding
				// model placed inside the cluster but whose connection
				// to the rest is no stronger than typical cross-concept
				// noise. Drop any concept whose mean inter-concept
				// cosine to non-pair core members is at or below the
				// patient's own noiseMean (e.g. CVA inside a
				// "musculoskeletal injuries" core that's otherwise
				// Nonunion + Achilles + Crushing injury). Generic —
				// uses only the patient's pairwise embedding
				// statistics, no concept-name knowledge.
				semanticCore = pruneCoherenceOutlierConcepts(
						semanticCore, config.noiseProfile);
			}
			boolean coreHasStructure = !semanticCore.isEmpty()
					&& coreCutoff < nonKeyword.size();
			boolean coreRelevant = false;
			if (coreHasStructure && !semanticCore.isEmpty()
					&& !candidates.isEmpty()) {
				float[] coreVec = semanticCore.get(0).embedding.getEmbeddingVector();
				double maxCoreKwCosine = 0;
				for (ScoredEmbedding se : candidates) {
					double cos = cosineSimilarity(coreVec,
							se.embedding.getEmbeddingVector());
					if (cos > maxCoreKwCosine) {
						maxCoreKwCosine = cos;
					}
				}
				coreRelevant = maxCoreKwCosine
						>= ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE;
				log.warn("Core topical check: maxCos={} vs threshold={}, relevant={}",
						String.format("%.4f", maxCoreKwCosine),
						ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE,
						coreRelevant);
			}
			log.warn("Core validation: structure={}, relevant={}, cutoff={}/{}, semDom={}",
					coreHasStructure, coreRelevant,
					coreCutoff, nonKeyword.size(), semanticDominance);
			if (semanticDominance || coreRelevant) {
				if (!semanticCore.isEmpty()) {
					double coreMinSem = semanticCore.get(
							semanticCore.size() - 1).semanticScore;
					double coreMaxSem = semanticCore.get(0).semanticScore;
					double expansionRatio = coreFallbackUsed
							? (coreMaxSem > 0
									? coreMinSem / coreMaxSem
									: ChartSearchAiConstants.SEMANTIC_CORE_SCORE_RATIO)
							: ChartSearchAiConstants.SEMANTIC_CORE_SCORE_RATIO;
					double scoreFloor = coreMinSem * expansionRatio;
					List<ScoredEmbedding> expanded =
							new ArrayList<ScoredEmbedding>(semanticCore);
					Set<Integer> expandedIds = new java.util.HashSet<Integer>();
					for (ScoredEmbedding se : semanticCore) {
						expandedIds.add(se.embedding.getResourceId());
					}
					if (!semanticDominance) {
						for (ScoredEmbedding se : preRefinementCandidates) {
							if (expandedIds.contains(se.embedding.getResourceId())) {
								continue;
							}
							if (se.semanticScore < scoreFloor) {
								continue;
							}
							float[] vec = se.embedding.getEmbeddingVector();
							double maxCos = 0;
							for (ScoredEmbedding core : semanticCore) {
								double cos = cosineSimilarity(vec,
										core.embedding.getEmbeddingVector());
								if (cos > maxCos) {
									maxCos = cos;
								}
							}
							if (maxCos >= ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE) {
								expanded.add(se);
								expandedIds.add(se.embedding.getResourceId());
							}
						}
					}
					// Concept-pair rescue for keyword-matched anchors:
					// when a kw-matched record from the refinement input
					// shares its concept name with a record already in
					// expanded, the only reason it got dropped is the
					// cosine-to-core boundary cutting through a
					// same-concept pair (e.g. Crushing-injury condition
					// survived expansion at cos≈0.56 but its diagnosis
					// sibling failed at cos 0.5494). Restore the missing
					// partner — its keyword evidence is identical and
					// its embedding is by construction near the
					// survivor's, so the cluster is more coherent with
					// it included than split.
					Set<String> expandedConcepts = new HashSet<String>();
					for (ScoredEmbedding se : expanded) {
						String cn = ConceptNameUtil.extractConceptName(
								se.embedding.getTextContent());
						if (cn != null) {
							expandedConcepts.add(cn);
						}
					}
					for (ScoredEmbedding se : candidates) {
						if (expandedIds.contains(se.embedding.getResourceId())) {
							continue;
						}
						if (se.keywordScore <= 0) {
							continue;
						}
						String cn = ConceptNameUtil.extractConceptName(
								se.embedding.getTextContent());
						if (cn != null && expandedConcepts.contains(cn)) {
							expanded.add(se);
							expandedIds.add(se.embedding.getResourceId());
							log.debug("Concept-pair rescue: re-added kw-matched [{}] (concept={})",
									se.embedding.getResourceId(), cn);
						}
					}
					candidates = expanded;
					log.debug("Semantic core expansion: core={}, floor={}, result={}",
							semanticCore.size(),
							String.format("%.4f", scoreFloor),
							candidates.size());
					partialKwValidatedOut[0] = true;
				} else {
					double maxSemanticKw = 0;
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore > maxSemanticKw) {
							maxSemanticKw = se.semanticScore;
						}
					}
					double semFloor = maxSemanticKw
							* refinementSemanticRatio(candidates);
					List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore >= semFloor) {
							filtered.add(se);
						}
					}
					if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
						log.debug("Partial-kw semantic floor: {} -> {} (floor={})",
								candidates.size(), filtered.size(),
								String.format("%.4f", semFloor));
						candidates = filtered;
					}
				}
			} else {
				double maxSemanticKw = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > maxSemanticKw) {
						maxSemanticKw = se.semanticScore;
					}
				}
				double semFloor = maxSemanticKw * config.similarityRatio;
				List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore >= semFloor) {
						filtered.add(se);
					}
				}
				if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
					log.debug("Low-coverage kw semantic floor: {} -> {} (floor={})",
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
					maxSemanticRefined * refinementAdaptiveGapRatio(scored),
					secondPassMinGap(scored));
			int refinedCutoff = findAdaptiveCutoff(candidates,
					candidates.size(), minScore,
					config.scoreGapMultiplier, adaptiveMinGap);
			if (refinedCutoff < candidates.size()) {
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
					log.debug("Refinement gap is intra-topic, using semantic floor {} instead: {} -> {}",
							String.format("%.4f", semanticFloor),
							candidates.size(), floored.size());
					candidates = floored;
				} else if (queryTerms != null) {
					List<ScoredEmbedding> rescued =
							filterRedundantKeywordTier(candidates,
									queryTerms, kwMax, bonusThreshold);
					if (rescued.size() >= refinedCutoff) {
						log.debug("Refinement gap at {} is a concept boundary: rescued {} records with new coverage",
								refinedCutoff, rescued.size() - refinedCutoff);
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
				candidates = filterRedundantKeywordTier(
						candidates, queryTerms, kwMax, bonusThreshold);
			}
		}
		return candidates;
	}

	/**
	 * Phase 1 (outlier removal) — stage 6 of the filter pipeline.
	 *
	 * <p>Removes individual candidates that are topically unrelated to
	 * the majority via {@link #filterByCoherence}. Skipped when:
	 * <ul>
	 *   <li>The partial-keyword semantic-core path already curated the
	 *       set (would double-filter).</li>
	 *   <li>Every candidate matches ALL query terms — keyword evidence
	 *       is conclusive, coherence is redundant.</li>
	 *   <li>The set is a compound-keyword match — coherence would drop
	 *       the minority concept cluster (see
	 *       {@link #isCompoundKeywordMatch}).</li>
	 * </ul>
	 *
	 * <p>At n=3 coherence filtering only fires when scores are tightly
	 * clustered (min/max ratio ≥ 0.90); a spread at n=3 means the
	 * minority candidate may be the only correct result.
	 */
	static List<ScoredEmbedding> applyOutlierRemovalPhase1(
			List<ScoredEmbedding> candidates, String[] queryTerms,
			boolean partialKwValidated) {
		boolean allFullKeywordMatch = true;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore < 1.0) {
				allFullKeywordMatch = false;
				break;
			}
		}
		boolean isCompoundKeywordMatch = isCompoundKeywordMatch(
				candidates, queryTerms);
		double bonusThreshold = queryTerms.length == 0 ? 1.0
				: (double) Math.min(2, queryTerms.length) / queryTerms.length;
		if (!partialKwValidated && !allFullKeywordMatch
				&& !isCompoundKeywordMatch
				&& candidates.size() >= 4) {
			return preserveUniqueCoverage(candidates,
					filterByCoherence(candidates), queryTerms,
					bonusThreshold);
		}
		if (!partialKwValidated && !allFullKeywordMatch
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
				return preserveUniqueCoverage(candidates,
						filterByCoherence(candidates), queryTerms,
						bonusThreshold);
			}
		}
		return candidates;
	}

	/**
	 * After Phase 1's coherence filter runs, restore any candidates whose
	 * keyword evidence covers a query term NOT matched by the survivors.
	 * Coherence filtering uses only embedding geometry and can drop
	 * legitimate cross-concept matches in multi-concept queries (e.g.
	 * Weight records in "blood pressure and weight" — Weight is less
	 * coherent with the BP cluster but matches "weight" which BP doesn't
	 * cover). Records whose keyword matches duplicate the survivors'
	 * coverage (e.g. SpO2 also matching "blood" in a "blood pressure"
	 * context) remain filtered — the survivors already account for that
	 * term, so the dropped record adds no new query coverage.
	 */
	static List<ScoredEmbedding> preserveUniqueCoverage(
			List<ScoredEmbedding> original,
			List<ScoredEmbedding> filtered, String[] queryTerms,
			double bonusThreshold) {
		if (queryTerms == null || queryTerms.length == 0) {
			return filtered;
		}
		Set<Integer> filteredIds = new HashSet<Integer>();
		Set<String> coveredTerms = new HashSet<String>();
		for (ScoredEmbedding se : filtered) {
			filteredIds.add(se.embedding.getResourceId());
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			for (String term : queryTerms) {
				if (termMatchesText(term, text, words)) {
					coveredTerms.add(term);
				}
			}
		}
		List<ScoredEmbedding> result =
				new ArrayList<ScoredEmbedding>(filtered);
		Set<Integer> restoredIds = new HashSet<Integer>(filteredIds);
		// Pass 1: restore records that cover query terms NOT already
		// covered by coherence survivors (existing behavior).
		for (ScoredEmbedding se : original) {
			if (se.keywordScore <= 0
					|| restoredIds.contains(
							se.embedding.getResourceId())) {
				continue;
			}
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			for (String term : queryTerms) {
				if (!coveredTerms.contains(term)
						&& termMatchesText(term, text, words)) {
					result.add(se);
					restoredIds.add(se.embedding.getResourceId());
					break;
				}
			}
		}
		// Pass 2: restore records with strong keyword relevance
		// (keywordScore >= bonusThreshold) even when their matched
		// terms overlap with survivors. Multi-concept queries like
		// "opportunistic infections in HIV" produce candidates from
		// different clinical domains (HIV, TB) that both match the
		// same query terms but don't cohere — coherence filtering
		// drops the less-coherent group even though it's genuinely
		// relevant. The bonus threshold (min(2,N)/N query terms)
		// ensures only records with strong keyword evidence are
		// restored, preventing noisy partial matches (e.g. "blood"
		// matching "blood pressure" for "blood problems" won't
		// trigger this — bonusThreshold=1.0 for 2-term queries).
		for (ScoredEmbedding se : original) {
			if (restoredIds.contains(se.embedding.getResourceId())) {
				continue;
			}
			if (se.keywordScore >= bonusThreshold) {
				result.add(se);
				restoredIds.add(se.embedding.getResourceId());
			}
		}
		return result;
	}

	/**
	 * Phase 2 (zero-keyword validation) — stage 7 of the filter pipeline.
	 *
	 * <p>When no surviving candidate has keyword support, the result set
	 * is purely semantic and must pass two orthogonal confidence checks
	 * (mean coherence and cluster z-score) plus a small-cluster
	 * coherence gate and a single-candidate structural gate. Candidates
	 * with any keyword match bypass this phase entirely.
	 *
	 * <p>The z-score gate is skipped when tight-cluster evidence already
	 * validated signal — captured via {@code belowFloorRescued},
	 * {@code firstPassGapDetected} vs concept count, or the ratio-floor
	 * tight-cluster signal.
	 */
	static List<ScoredEmbedding> applyZeroKeywordValidationPhase2(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored, PipelineConfig config,
			boolean partialKwValidated, boolean belowFloorRescued,
			boolean firstPassGapDetected, int adaptiveCutoff,
			int ratioFloorCandidateCount, double initialZScore,
			double initialZThreshold, double maxSemanticScore) {
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
		log.debug("Phase2 entry: candidates={}, hasKw={}, partialKwValidated={}, belowFloorRescued={}, gapDetected={}, ratioFloorCount={}",
				candidates.size(), candidatesHaveKeywords, partialKwValidated,
				belowFloorRescued, firstPassGapDetected, ratioFloorCandidateCount);
		if (partialKwValidated || candidatesHaveKeywords || candidates.isEmpty()) {
			return candidates;
		}
		if (candidates.size() >= 2) {
			int preMeanCoherence = candidates.size();
			candidates = filterByMeanCoherence(candidates, config);
			log.debug("Phase2 meanCoherence: {} → {}", preMeanCoherence, candidates.size());
		}
		// Small-cluster coherence gate: for ≤3 zero-keyword
		// candidates, apply a stricter coherence threshold derived
		// from the patient's own embedding statistics.
		if (candidates.size() == 3) {
			double intraFloor = config.noiseProfile.intraConceptMean
					- config.noiseProfile.intraConceptStd;
			double sumCos = 0;
			int nPairs = 0;
			for (int ci = 0; ci < candidates.size(); ci++) {
				float[] vi = candidates.get(ci).embedding.getEmbeddingVector();
				if (vi == null) {
					continue;
				}
				for (int cj = ci + 1; cj < candidates.size(); cj++) {
					float[] vj = candidates.get(cj).embedding.getEmbeddingVector();
					if (vj == null) {
						continue;
					}
					sumCos += ChartSearchAiUtils.cosineSimilarity(vi, vj);
					nPairs++;
				}
			}
			if (nPairs > 0) {
				double mc = sumCos / nPairs;
				if (mc < intraFloor) {
					// Extract the coherent same-concept pair if one exists.
					double[] avgCoh = new double[3];
					for (int ci = 0; ci < 3; ci++) {
						double s = 0;
						int cnt = 0;
						float[] vi = candidates.get(ci).embedding.getEmbeddingVector();
						if (vi == null) {
							continue;
						}
						for (int cj = 0; cj < 3; cj++) {
							if (ci == cj) {
								continue;
							}
							float[] vj = candidates.get(cj).embedding.getEmbeddingVector();
							if (vj == null) {
								continue;
							}
							s += ChartSearchAiUtils.cosineSimilarity(vi, vj);
							cnt++;
						}
						avgCoh[ci] = cnt > 0 ? s / cnt : 0;
					}
					int worstIdx = 0;
					for (int ci = 1; ci < 3; ci++) {
						if (avgCoh[ci] < avgCoh[worstIdx]) {
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
					float[] v1 = candidates.get(p1).embedding.getEmbeddingVector();
					float[] v2 = candidates.get(p2).embedding.getEmbeddingVector();
					double pairCos = (v1 != null && v2 != null)
							? ChartSearchAiUtils.cosineSimilarity(v1, v2)
							: 0;
					String cn1 = ConceptNameUtil.extractConceptName(
							candidates.get(p1).embedding.getTextContent());
					String cn2 = ConceptNameUtil.extractConceptName(
							candidates.get(p2).embedding.getTextContent());
					boolean sameConcept = cn1 != null && cn1.equals(cn2);
					if (pairCos >= intraFloor && sameConcept) {
						List<ScoredEmbedding> pair = new ArrayList<ScoredEmbedding>();
						pair.add(candidates.get(p1));
						pair.add(candidates.get(p2));
						log.debug("Small-cluster gate: removed outlier [{}], kept pair cos={}, concept={}",
								candidates.get(worstIdx).embedding.getResourceId(),
								String.format("%.4f", pairCos), cn1);
						candidates = pair;
					} else {
						log.debug("Small-cluster coherence gate: meanCoherence={} < intraFloor={}, returning empty",
								String.format("%.4f", mc),
								String.format("%.4f", intraFloor));
						candidates = Collections.emptyList();
					}
				}
			}
		}
		// Tight-cluster detection: the z-score gate is skipped when
		// structural signals already confirm the cluster.
		Set<String> tightCheckConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String cn = ConceptNameUtil.extractConceptName(se.embedding.getTextContent());
			if (cn != null) {
				tightCheckConcepts.add(cn);
			}
		}
		int tightThreshold = tightCheckConcepts.size();
		// Count distinct concepts in the candidate set entering
		// Phase 2 — used by the multi-concept umbrella bypass below.
		Set<String> p2CandidateConcepts = new HashSet<String>();
		for (ScoredEmbedding se : candidates) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn != null) {
				p2CandidateConcepts.add(cn);
			}
		}
		// Ratio-floor selectivity guard: the ratio floor must
		// have been genuinely selective — keeping at most 25% of
		// all scored records. Compressed-score models (e.g.
		// MedCPT) have such narrow score ranges that a 0.98
		// ratio floor can pass 25%+ of records despite no real
		// match. When the ratio floor isn't selective, the set
		// is just the upper tail of a smooth noise distribution,
		// not a genuine cluster.
		boolean ratioFloorSelective = ratioFloorCandidateCount >= 0
				&& ratioFloorCandidateCount <= scored.size() / 4;
		boolean tightClusterDetected = belowFloorRescued
				|| (firstPassGapDetected && adaptiveCutoff < tightThreshold)
				|| (ratioFloorSelective
						&& ratioFloorCandidateCount < tightThreshold
						&& initialZScore >= config.floorRescueMinZScore
						&& maxSemanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ config.minScoreGap)
				// Multi-concept umbrella bypass: when the ratio floor
				// produced a tight cluster spanning 3+ distinct concepts
				// AND the initial z-score passed its own gate, the concept
				// diversity provides the structural confidence that
				// floorRescueMinZScore provides for single-concept
				// clusters. The initial gate already validated that the
				// top semantic score is a genuine outlier; the multi-concept
				// spread confirms the cluster is not a coincidental
				// grouping. This rescues broad-category queries like
				// "vital signs" where the z-score is strong but just
				// below the 2.0 threshold.
				|| (ratioFloorSelective
						&& ratioFloorCandidateCount < tightThreshold
						&& p2CandidateConcepts.size() >= 3
						&& initialZThreshold > 0
						&& initialZScore >= initialZThreshold
						&& maxSemanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ config.minScoreGap);
		// Single-candidate structural gate.
		if (!tightClusterDetected
				&& candidates.size() == 1
				&& ratioFloorCandidateCount == 1) {
			log.debug("Single zero-keyword candidate with no tight-cluster support and ratioFloor=1, returning empty");
			candidates = Collections.emptyList();
		}
		if (!tightClusterDetected && !candidates.isEmpty()
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
				double[] zScores = new double[candidates.size()];
				for (int i = 0; i < candidates.size(); i++) {
					zScores[i] = (candidates.get(i).semanticScore - mean) / std;
				}
				java.util.Arrays.sort(zScores);
				double zRepresentative = candidates.size() <= 3
						? zScores[zScores.length - 1]
						: zScores[zScores.length / 2];
				double clusterZThreshold;
				if (candidates.size() <= 3) {
					clusterZThreshold = clusterGumbelThreshold(scored);
				} else {
					clusterZThreshold = medianGumbelThreshold(scored);
				}
				log.debug("Phase2 z-score: mean={}, std={}, z={} ({}), threshold={}, tightCluster={}, candidates={}",
						String.format("%.4f", mean),
						String.format("%.4f", std),
						String.format("%.2f", zRepresentative),
						candidates.size() <= 3 ? "max" : "median",
						String.format("%.2f", clusterZThreshold),
						tightClusterDetected, candidates.size());
				if (zRepresentative < clusterZThreshold) {
					if (candidates.size() <= 3
							&& initialZThreshold > 0
							&& initialZScore >= initialZThreshold) {
						log.debug("Phase2: z-score below threshold but initial gate validated, keeping");
					} else {
						log.debug("Phase2: REJECTED by z-score gate (z={} < threshold={})",
								String.format("%.2f", zRepresentative),
								String.format("%.2f", clusterZThreshold));
						candidates = Collections.emptyList();
					}
				}
			}
		}
		return candidates;
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
			double minScore, PipelineConfig config,
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
		//
		// Uniform-keyword exception: when every candidate has the same
		// keyword score, the penalty/bonus is applied equally, so it
		// cancels out in relative ranking but makes the absolute floor
		// artificially tighter (for a uniform penalty) or looser (for
		// a uniform bonus). Fall back to pure-semantic comparison so
		// relative ordering among candidates with identical keyword
		// evidence isn't distorted by the shared adjustment.
		boolean uniformCandidateKw = true;
		double firstKw = candidates.isEmpty() ? 0
				: candidates.get(0).keywordScore;
		for (ScoredEmbedding se : candidates) {
			if (Math.abs(se.keywordScore - firstKw) > 1e-9) {
				uniformCandidateKw = false;
				break;
			}
		}
		// Uniform-keyword tight-cluster bypass: when every candidate
		// shares the same (non-zero) keyword score AND their semantic
		// scores form a tight cluster — quantified by comparing the
		// candidate set's absolute semantic-score spread to the
		// patient's own background noise std — the keyword evidence
		// is uniform across the set and gap detection has already
		// separated them from the noise. The ratio floor in this
		// regime would arbitrarily drop the lower-scoring members of
		// a cohesive same-topic group (e.g. all six "substance abuse"
		// condition+diagnosis pairs with sem 0.52–0.67 and uniform
		// kw=0.5). Skip the floor when the spread is tight.
		//
		// Threshold derivation: candidate scores are statistically
		// indistinguishable when their spread sits within 2× the
		// cross-concept noise std (the standard 95% z-band). This is
		// data-derived per patient via {@link ModelNoiseProfile} —
		// patients with diverse concepts get a wider tolerance,
		// patients with narrow concept distributions get a tighter
		// one, both tracking the model's own discriminative
		// resolution rather than a hardcoded fraction.
		boolean tightUniformCluster = false;
		if (uniformCandidateKw && firstKw > 0
				&& candidates.size()
				>= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			double maxSem = 0;
			double minSem = Double.MAX_VALUE;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore > maxSem) maxSem = se.semanticScore;
				if (se.semanticScore < minSem) minSem = se.semanticScore;
			}
			double spread = maxSem - minSem;
			double noiseSpreadBand = 2.0 * config.noiseProfile.noiseStd;
			if (maxSem > 0 && spread <= noiseSpreadBand) {
				tightUniformCluster = true;
				log.debug("Uniform-keyword tight cluster detected: spread={}, noiseBand={}, kw={}, n={} — bypassing ratio floor",
						String.format("%.4f", spread),
						String.format("%.4f", noiseSpreadBand),
						String.format("%.3f", firstKw),
						candidates.size());
			}
		}
		double ratioFloor = uniformCandidateKw
				? maxSemanticScore * config.similarityRatio
				: Math.min(maxBaseScore, maxSemanticScore)
						* config.similarityRatio;
		List<ScoredEmbedding> strict = new ArrayList<ScoredEmbedding>();
		List<ScoredEmbedding> nearMiss = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (tightUniformCluster) {
				strict.add(se);
				continue;
			}
			double check = uniformCandidateKw ? se.semanticScore
					: Math.min(se.score, se.semanticScore);
			if (check >= ratioFloor) {
				strict.add(se);
			} else {
				nearMiss.add(se);
			}
		}
		// Capture the strict count before rescue so downstream
		// gates see only candidates that truly passed the floor.
		ratioFloorCandidateCountOut[0] = strict.size();
		log.debug("Non-refinement: ratioFloor={}, strict={}, nearMiss={}, total={}",
				String.format("%.4f", ratioFloor), strict.size(),
				nearMiss.size(), candidates.size());
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
		// when some candidates lack keywords — they may be semantic
		// false positives that need capping.
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
	/**
	 * Removes concept-pair outliers from a semantic core by comparing
	 * each concept's average inter-concept cosine to the patient's
	 * cross-concept noise baseline. A concept whose mean cosine to
	 * non-pair core members is at or below {@code noiseProfile.noiseMean}
	 * is, by construction, statistically indistinguishable from a
	 * random unrelated pair — gap detection let it into the cluster
	 * because its similarity to the query happened to fall in the
	 * cluster's score range, but its embedding is no closer to the
	 * other cluster members than chance. Drop it as a topical
	 * neighbour rather than a true cluster member.
	 *
	 * <p>The cutoff is {@code noiseProfile.noiseMean} — entirely
	 * data-derived from the patient's own cross-concept pairwise
	 * statistics. No σ multipliers, no calibrated thresholds, no
	 * concept-name knowledge.
	 *
	 * <p>Only fires when there are ≥ 3 distinct concepts in the core
	 * and ≥ 2 of them have ≥ 2 members (otherwise there's no
	 * meaningful concept-pair structure to analyse).
	 */
	static List<ScoredEmbedding> pruneCoherenceOutlierConcepts(
			List<ScoredEmbedding> core,
			org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile noiseProfile) {
		if (core.size() < 4) {
			return core;
		}
		// Group core members by concept name.
		Map<String, List<ScoredEmbedding>> byConcept =
				new java.util.LinkedHashMap<String, List<ScoredEmbedding>>();
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null) {
				cn = "__unnamed_" + se.embedding.getResourceId();
			}
			List<ScoredEmbedding> list = byConcept.get(cn);
			if (list == null) {
				list = new ArrayList<ScoredEmbedding>();
				byConcept.put(cn, list);
			}
			list.add(se);
		}
		if (byConcept.size() < 3) {
			return core;
		}
		int multiMember = 0;
		for (List<ScoredEmbedding> g : byConcept.values()) {
			if (g.size() >= 2) {
				multiMember++;
			}
		}
		if (multiMember < 2) {
			return core;
		}
		// For each concept group, compute average pairwise cosine
		// from each of its members to every core member NOT in the
		// same concept group.
		Map<String, Double> conceptInterAvg =
				new java.util.LinkedHashMap<String, Double>();
		for (Map.Entry<String, List<ScoredEmbedding>> e : byConcept.entrySet()) {
			String concept = e.getKey();
			List<ScoredEmbedding> group = e.getValue();
			double sum = 0;
			int count = 0;
			for (ScoredEmbedding member : group) {
				float[] mv = member.embedding.getEmbeddingVector();
				if (mv == null) {
					continue;
				}
				for (ScoredEmbedding other : core) {
					String ocn = ConceptNameUtil.extractConceptName(
							other.embedding.getTextContent());
					if (ocn != null && ocn.equals(concept)) {
						continue;
					}
					float[] ov = other.embedding.getEmbeddingVector();
					if (ov == null) {
						continue;
					}
					sum += cosineSimilarity(mv, ov);
					count++;
				}
			}
			if (count > 0) {
				conceptInterAvg.put(concept, sum / count);
			}
		}
		if (conceptInterAvg.size() < 3) {
			return core;
		}
		// Drop concepts whose inter-concept cosine to the rest of the
		// core sits at or below the patient's noise baseline — they're
		// statistically indistinguishable from random unrelated pairs.
		double cutoff = noiseProfile.noiseMean;
		Set<String> drop = new HashSet<String>();
		for (Map.Entry<String, Double> e : conceptInterAvg.entrySet()) {
			if (e.getValue() <= cutoff) {
				drop.add(e.getKey());
			}
		}
		if (drop.isEmpty()) {
			return core;
		}
		// Don't prune if it would leave too few members.
		int kept = 0;
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null || !drop.contains(cn)) {
				kept++;
			}
		}
		if (kept < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			return core;
		}
		List<ScoredEmbedding> pruned = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null || !drop.contains(cn)) {
				pruned.add(se);
			}
		}
		log.debug("Coherence-outlier concept pruning: dropped {} concepts ({}), kept {} of {} core members (cutoff=noiseMean={})",
				drop.size(), drop, pruned.size(), core.size(),
				String.format("%.4f", cutoff));
		return pruned;
	}

	static List<ScoredEmbedding> filterRedundantKeywordTier(
			List<ScoredEmbedding> candidates, String[] queryTerms,
			double kwMax, double bonusThreshold) {
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
		// semantic relevance (rescued synonym/concept mismatch), OR
		// have strong keyword relevance (kwScore >= bonusThreshold).
		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				filtered.add(se);
			} else if (se.keywordScore >= bonusThreshold) {
				// Strong keyword match (e.g. TB matching 2/3 terms of
				// "opportunistic infections in HIV") — genuinely
				// relevant even when terms overlap with the top tier.
				// Without this, multi-concept queries lose cross-domain
				// records that share query terms with the dominant
				// cluster.
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

	private static final Set<String> CONCEPT_EXPANSION_FUNCTION_WORDS =
			Collections.unmodifiableSet(new HashSet<String>(
					java.util.Arrays.asList(
							"of", "to", "and", "or", "the", "a", "an",
							"in", "on", "for", "with", "by", "as")));

	/**
	 * Embedding-based query expansion. When the user's query phrasing
	 * doesn't appear verbatim in any record but a concept name in the
	 * patient's chart is semantically very close to the query (per the
	 * embedding model's query encoder), replaces {@code kwTerms} with
	 * that concept's tokens so keyword scoring can anchor on the
	 * actual record vocabulary.
	 *
	 * <p>Bridges the synonym gap that the OpenMRS concept dictionary
	 * doesn't capture: e.g. the demo Pulse concept lists "HR" but not
	 * "heart rate", yet MedCPT's query encoder maps "heart rate" within
	 * 0.83 cosine of "Pulse". After replacement the partial-match gate
	 * sees a clean full keyword match and stops rejecting pulse
	 * records.</p>
	 *
	 * <p>Activation rule (all required, designed to avoid breaking
	 * queries that already work):</p>
	 * <ol>
	 *   <li>{@code kwTerms.length} between 1 and
	 *   {@link ChartSearchAiConstants#CONCEPT_EXPANSION_MAX_KW_TERMS}
	 *   — longer queries are typically multi-concept and unsafe to
	 *   collapse to one synonym.</li>
	 *   <li>No record contains every {@code kwTerms} substring — if a
	 *   full match is already possible, original keyword scoring is
	 *   doing the right thing and replacement risks losing it.</li>
	 *   <li>The closest concept's similarity AND its margin over the
	 *   second-best concept satisfy one of two model-tuned paths:
	 *     <ol type="a">
	 *       <li>Very-high absolute similarity (≥
	 *       {@code config.conceptExpansionVeryHighMinSim}) AND modest
	 *       margin (≥ {@code config.conceptExpansionVeryHighMinMargin}
	 *       above second-best). Catches strong synonym matches.</li>
	 *       <li>Moderate similarity (≥
	 *       {@code config.conceptExpansionModerateMinSim}) AND larger
	 *       margin (≥ {@code config.conceptExpansionModerateMinMargin}).
	 *       Protects against false expansion when several concepts
	 *       cluster near the threshold (e.g. "allergies" matching
	 *       multiple allergy-related concepts).</li>
	 *     </ol>
	 *   Multi-concept queries (e.g. "blood pressure and pulse" where SBP
	 *   and Pulse both score ~0.92) fail both because their top two
	 *   concepts cluster.</li>
	 *   <li>The concept's tokens differ from {@code kwTerms} (set
	 *   inequality) — else expansion is a no-op.</li>
	 * </ol>
	 *
	 * <p>Cost: one query-encoder call per unique concept name in the
	 * patient's chart (typically 20-40), plus one cosine pass. Returns
	 * the same reference as {@code kwTerms} when no expansion applies,
	 * so callers can detect via identity comparison.</p>
	 */
	static String[] expandKwTermsViaConceptSimilarity(String[] kwTerms,
			float[] queryVector, ChartEmbedding[] embeddings,
			org.openmrs.module.chartsearchai.embedding.EmbeddingProvider provider,
			PipelineConfig config) {
		if (kwTerms.length == 0
				|| kwTerms.length > ChartSearchAiConstants.CONCEPT_EXPANSION_MAX_KW_TERMS
				|| embeddings == null || embeddings.length == 0
				|| provider == null || queryVector == null) {
			return kwTerms;
		}
		// Skip when at least one record already contains every kwTerm —
		// keyword scoring is already finding full matches, replacing
		// would only narrow what we already capture correctly.
		for (ChartEmbedding ce : embeddings) {
			String text = ce.getTextContent();
			if (text == null) continue;
			String lower = text.toLowerCase();
			boolean allFound = true;
			for (String t : kwTerms) {
				if (!lower.contains(t)) {
					allFound = false;
					break;
				}
			}
			if (allFound) {
				return kwTerms;
			}
		}
		// Collect unique concept names from the records.
		Set<String> conceptNames = new java.util.LinkedHashSet<String>();
		for (ChartEmbedding ce : embeddings) {
			String name = ConceptNameUtil.extractConceptName(ce.getTextContent());
			if (name != null) {
				String trimmed = name.trim();
				if (!trimmed.isEmpty()) {
					conceptNames.add(trimmed);
				}
			}
		}
		if (conceptNames.isEmpty()) {
			return kwTerms;
		}
		// Find the closest concept (by query-encoder cosine) AND track
		// the second-best score to assess "clear winner" margin.
		String bestConcept = null;
		double bestSim = -1;
		double secondSim = -1;
		for (String name : conceptNames) {
			float[] vec;
			try {
				vec = provider.embedQuery(name);
			} catch (Exception e) {
				continue;
			}
			if (vec == null || vec.length != queryVector.length) {
				continue;
			}
			double sim = cosineSimilarity(queryVector, vec);
			if (sim > bestSim) {
				secondSim = bestSim;
				bestSim = sim;
				bestConcept = name;
			} else if (sim > secondSim) {
				secondSim = sim;
			}
		}
		double margin = bestSim - secondSim;
		boolean veryHighWithModestMargin =
				bestSim >= config.conceptExpansionVeryHighMinSim
				&& margin >= config.conceptExpansionVeryHighMinMargin;
		boolean moderateWithLargeMargin =
				bestSim >= config.conceptExpansionModerateMinSim
				&& margin >= config.conceptExpansionModerateMinMargin;
		if (bestConcept == null
				|| (!veryHighWithModestMargin && !moderateWithLargeMargin)) {
			return kwTerms;
		}
		// Vocabulary-overlap guard: below the bypass threshold, require
		// at least one original kwTerm to appear in the concept name.
		// This blocks "allergies → Asthma" (no shared word, model just
		// considers them adjacent) while still allowing strong
		// substring/synonym matches like "anaemia → Anemia",
		// "pulse rate → Pulse", "depression → Mild depressive episode",
		// and abbreviations whose similarity exceeds the bypass ("TB →
		// Tuberculosis"). Without this, queries about absent categories
		// silently get redirected to topic-adjacent concepts.
		if (bestSim < config.conceptExpansionVocabBypassThreshold) {
			String conceptLower = bestConcept.toLowerCase();
			boolean hasOverlap = false;
			for (String t : kwTerms) {
				if (t.length() >= ChartSearchAiConstants.CONCEPT_EXPANSION_MIN_OVERLAP_LENGTH
						&& conceptLower.contains(t)) {
					hasOverlap = true;
					break;
				}
			}
			if (!hasOverlap) {
				return kwTerms;
			}
		}
		// Tokenise the concept name. Strip non-alphanumerics, drop
		// single-character tokens that would match almost any text.
		String[] rawTokens = bestConcept.toLowerCase()
				.replaceAll("[^a-z0-9 ]+", " ").trim().split("\\s+");
		// Filter the concept tokens to keep only specific, content-bearing
		// terms:
		//  1. Drop English function words (see
		//     CONCEPT_EXPANSION_FUNCTION_WORDS). These appear in concept
		//     names like "Method of family planning" but as keyword
		//     anchors they would match far too many records.
		//  2. IDF filter: drop tokens whose document frequency exceeds
		//     CONCEPT_EXPANSION_IDF_FRACTION of the corpus. Threshold
		//     has a floor of 1 so very small corpora (unit tests) don't
		//     lose every token.
		int idfThreshold = Math.max(1, (int) Math.ceil(embeddings.length
				* ChartSearchAiConstants.CONCEPT_EXPANSION_IDF_FRACTION));
		List<String> conceptTokens = new ArrayList<String>();
		for (String t : rawTokens) {
			if (t.length() < 2) continue;
			if (CONCEPT_EXPANSION_FUNCTION_WORDS.contains(t)) continue;
			int df = 0;
			for (ChartEmbedding ce : embeddings) {
				String text = ce.getTextContent();
				if (text != null && text.toLowerCase().contains(t)) {
					df++;
				}
			}
			if (df > 0 && df <= idfThreshold) {
				conceptTokens.add(t);
			}
		}
		if (conceptTokens.isEmpty()) {
			return kwTerms;
		}
		// Skip the no-op case: expansion produces the same set as the
		// original kwTerms. Narrowing — e.g. dropping "rate" from
		// ["pulse","rate"] down to ["pulse"] — IS a meaningful change
		// because keyword score for a pulse-only record goes from 1/2
		// (no bonus) to 1/1 (full bonus), unlocking the gate.
		Set<String> existing = new HashSet<String>();
		for (String t : kwTerms) existing.add(t.toLowerCase());
		Set<String> conceptSet = new HashSet<String>(conceptTokens);
		if (existing.equals(conceptSet)) {
			return kwTerms;
		}
		log.warn("Concept-similarity expansion: kwTerms {} -> {} "
				+ "(concept '{}' sim={} margin={})",
				java.util.Arrays.toString(kwTerms),
				conceptTokens, bestConcept,
				String.format("%.4f", bestSim),
				String.format("%.4f", margin));
		return conceptTokens.toArray(new String[0]);
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

	/**
	 * Resource types that are structurally equivalent in the data model.
	 * Conditions and diagnoses both represent clinical findings about a
	 * patient's health. When a type indicator matches one, the other
	 * should also be kept — they are two representations of the same
	 * concept type.
	 */
	private static final java.util.Map<String, String> SISTER_RESOURCE_TYPES;
	static {
		java.util.Map<String, String> m = new java.util.HashMap<>();
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION,
				ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS);
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS,
				ChartSearchAiConstants.RESOURCE_TYPE_CONDITION);
		SISTER_RESOURCE_TYPES = Collections.unmodifiableMap(m);
	}

	/**
	 * Checks whether a chart embedding's structural prefix contains any of
	 * the given type indicator terms. Uses the actual text content to derive
	 * the correct prefix (important for ORDER records whose prefix depends
	 * on body text, e.g. "Drug order:" → "Medication prescription:").
	 * Also checks sister resource types: condition ↔ diagnosis.
	 */
	private static boolean matchesTypeIndicator(ChartEmbedding ce,
			Set<String> typeIndicatorTerms) {
		String textContent = ce.getTextContent() != null
				? ce.getTextContent() : "";
		String fullPrefixed = ChartSearchAiUtils.buildPrefixedText(
				ce.getResourceType(), textContent);
		String prefix = fullPrefixed.substring(0,
				fullPrefixed.length() - textContent.length())
				.toLowerCase();
		String[] prefixWords = prefix.split("\\s+");
		for (String term : typeIndicatorTerms) {
			if (termMatchesText(term, prefix, prefixWords)) {
				return true;
			}
		}
		// Check sister resource type — condition ↔ diagnosis are
		// structurally equivalent (both represent clinical findings).
		String sisterType = SISTER_RESOURCE_TYPES.get(
				ce.getResourceType());
		if (sisterType != null) {
			String sisterPrefix = ChartSearchAiUtils.buildPrefixedText(
					sisterType, "").toLowerCase();
			String[] sisterWords = sisterPrefix.split("\\s+");
			for (String term : typeIndicatorTerms) {
				if (termMatchesText(term, sisterPrefix, sisterWords)) {
					return true;
				}
			}
		}
		return false;
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
	 * Variant of {@link #computeKeywordScore} where {@code typeIndicatorTerms}
	 * may only match in the structural prefix portion of the text
	 * (i.e. {@code prefixedText} minus {@code body}). Other terms match
	 * in the full text as usual. This prevents type-indicator words like
	 * "medication" from matching narrative body text such as "Medication
	 * adjusted" in encounter notes that aren't medication records.
	 */
	static double computeKeywordScoreRestricted(String[] queryTerms,
			String prefixedText, String body,
			Set<String> typeIndicatorTerms) {
		if (queryTerms.length == 0 || prefixedText == null
				|| prefixedText.isEmpty()) {
			return 0.0;
		}
		String lowerFull = prefixedText.toLowerCase();
		String[] fullWords = lowerFull.split("\\s+");
		String prefix = body == null
				? prefixedText
				: prefixedText.substring(0,
						prefixedText.length() - body.length());
		String lowerPrefix = prefix.toLowerCase();
		String[] prefixWords = lowerPrefix.split("\\s+");
		// When type indicators exist, first check if this record
		// matches ANY of them. If it doesn't, this record is the
		// wrong type — non-type-indicator matches (e.g. "active"
		// matching "Status: ACTIVE" on an obs record when the
		// query asks for "conditions") are coincidental and
		// should not contribute to keyword scoring.
		boolean matchesType = false;
		for (String term : queryTerms) {
			if (typeIndicatorTerms.contains(term)
					&& termMatchesText(term, lowerPrefix,
							prefixWords)) {
				matchesType = true;
				break;
			}
		}
		if (!matchesType) {
			return 0.0;
		}
		int matched = 0;
		for (String term : queryTerms) {
			boolean isTypeIndicator = typeIndicatorTerms.contains(term);
			if (isTypeIndicator) {
				if (termMatchesText(term, lowerPrefix, prefixWords)) {
					matched++;
				}
			} else {
				if (termMatchesText(term, lowerFull, fullWords)) {
					matched++;
				}
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

		// 4. Shared-prefix match for long compound medical terms:
		// "immunocompromised" and "immunodeficiency" share the root
		// "immuno" but diverge after that — no simple stemmer catches
		// this. For terms AND text words both >= 12 characters, match
		// if they share a prefix of >= 6 characters. The length
		// constraints prevent false positives on short common words
		// (e.g. "immunization" at 12 chars wouldn't false-match
		// because its prefix "immuni" differs from "immuno").
		if (term.length() >= 12) {
			String termPrefix = term.substring(0, 6);
			for (String word : textWords) {
				String clean = word.replaceAll("[^a-z0-9]+$", "");
				if (clean.length() >= 12 && clean.startsWith(termPrefix)) {
					return true;
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
