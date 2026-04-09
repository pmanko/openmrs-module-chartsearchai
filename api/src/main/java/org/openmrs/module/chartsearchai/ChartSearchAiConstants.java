/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai;

public class ChartSearchAiConstants {

	public static final String PRIV_QUERY_PATIENT_DATA = "AI Query Patient Data";

	public static final String PRIV_VIEW_AUDIT_LOGS = "View AI Audit Logs";

	public static final String GP_EMBEDDING_MODEL_FILE_PATH = "chartsearchai.embedding.modelFilePath";

	public static final String GP_LLM_MODEL_FILE_PATH = "chartsearchai.llm.modelFilePath";

	public static final String GP_EMBEDDING_PRE_FILTER = "chartsearchai.embedding.preFilter";

	public static final String GP_EMBEDDING_VOCAB_FILE_PATH = "chartsearchai.embedding.vocabFilePath";

	public static final String GP_EMBEDDING_TOP_K = "chartsearchai.embedding.topK";

	public static final int DEFAULT_RETRIEVAL_TOP_K = 10;

	public static final String GP_EMBEDDING_SIMILARITY_RATIO = "chartsearchai.embedding.similarityRatio";

	public static final double DEFAULT_SIMILARITY_RATIO = 0.80;

	public static final String GP_EMBEDDING_SCORE_GAP_MULTIPLIER = "chartsearchai.embedding.scoreGapMultiplier";

	public static final double DEFAULT_SCORE_GAP_MULTIPLIER = 2.5;

	public static final String GP_EMBEDDING_KEYWORD_WEIGHT = "chartsearchai.embedding.keywordWeight";

	public static final double DEFAULT_KEYWORD_WEIGHT = 0.3;

	public static final String GP_EMBEDDING_TYPE_BOOST_FACTOR = "chartsearchai.embedding.typeBoostFactor";

	public static final double DEFAULT_TYPE_BOOST_FACTOR = 1.0;

	public static final String GP_EMBEDDING_QUERY_PREFIX = "chartsearchai.embedding.queryPrefix";

	public static final String DEFAULT_QUERY_EMBEDDING_PREFIX = "";

	public static final String GP_EMBEDDING_MAX_SEQUENCE_LENGTH = "chartsearchai.embedding.maxSequenceLength";

	public static final int DEFAULT_MAX_SEQUENCE_LENGTH = 256;

	public static final int ADAPTIVE_MIN_RECORDS = 2;

	public static final double ABSOLUTE_SIMILARITY_FLOOR = 0.25;

	/** Minimum z-score (standard deviations above mean) required for the
	 * top semantic score when fewer than {@link #ADAPTIVE_MIN_RECORDS}
	 * records match any query keyword. Without keyword corroboration, the
	 * top semantic score must be a statistical outlier — not just part of
	 * the noise floor. A z-score of 1.5 means the best match is in the
	 * top ~6.7% of the score distribution, indicating genuine semantic
	 * affinity rather than the embedding model grouping similar record
	 * types together (e.g. all pulse records scoring ~0.26 for "CD4
	 * count"). This threshold automatically adapts to any embedding
	 * model and dataset size since it is relative to the score
	 * distribution. */
	public static final double ZERO_KEYWORD_MIN_Z_SCORE = 1.5;

	/** Minimum z-score for zero-keyword candidates. After gap
	 * detection and coherence filtering, surviving zero-keyword
	 * candidates may be noise — records about an irrelevant topic
	 * (e.g. pulse readings for a CD4 query) that scored slightly
	 * above the dataset baseline. For small clusters (≤3
	 * candidates) the max z-score is checked; for larger clusters
	 * (≥4) the median z-score is used. This prevents a single
	 * high-scoring false positive from pushing the max above
	 * threshold while the cluster as a whole is noise. */
	public static final double ZERO_KEYWORD_CLUSTER_MIN_Z = 2.5;

	/** Minimum number of records required for the z-score gate to
	 * activate. Below this threshold, the score distribution has too
	 * few data points for the z-score to be statistically meaningful. */
	public static final int MIN_RECORDS_FOR_Z_SCORE = 30;

	public static final String GP_EMBEDDING_MIN_SCORE_GAP = "chartsearchai.embedding.minScoreGap";

	/**
	 * Minimum absolute gap between consecutive scores required for the adaptive
	 * cutoff detector to trigger. Prevents premature cutting when a relatively
	 * large gap (compared to a tight cluster's running average) is still small
	 * in absolute terms — e.g. a 0.07 gap inside a cluster that spans from
	 * 0.60 down to 0.30 should not be treated as a relevance boundary.
	 */
	public static final double DEFAULT_MIN_SCORE_GAP = 0.10;

	/** Fraction of the IQR used as the adaptive minimum gap for the
	 * first-pass gap detector when the score distribution is compressed
	 * (IQR &lt; minScoreGap / 2). Value of 0.30 means a gap must be at
	 * least 30% of the IQR to trigger. */
	public static final double MIN_GAP_RANGE_FRACTION = 0.30;

	/** Absolute coherence floor below which a candidate is a true outlier.
	 * If a candidate flagged for removal by the coherence gap detector has
	 * average pairwise coherence at or above this value, the cut is
	 * suppressed — the candidate genuinely belongs to the same topic.
	 * Prevents duplicate/near-duplicate embeddings (identical text → cosine
	 * 1.0 between them) from inflating the coherence range and making a
	 * same-topic record look like an outlier. Empirically: same-topic
	 * candidates removed incorrectly have coherence ~0.91+, while true
	 * cross-topic outliers have coherence ~0.49-. Value of 0.70 sits
	 * well between these ranges. */
	public static final double COHERENCE_SAME_TOPIC_FLOOR = 0.70;

	/** Minimum gap for the second-pass gap detection. The second pass
	 * reuses the primary gap multiplier ({@link #DEFAULT_SCORE_GAP_MULTIPLIER})
	 * but with a much lower absolute floor than the first pass
	 * ({@link #DEFAULT_MIN_SCORE_GAP}). This is what makes it "sensitive":
	 * both passes require the same relative anomaly (2.5× the running
	 * average), but the second pass can detect gaps as small as 0.01
	 * whereas the first pass ignores anything below 0.10. */
	public static final double SECOND_PASS_MIN_GAP = 0.01;

	/** Fraction of max semantic score used as the adaptive minimum gap in
	 * the refinement path's second-pass gap detection. Scales the threshold
	 * with the score range of the current query. */
	public static final double REFINEMENT_ADAPTIVE_GAP_RATIO = 0.10;

	/** Minimum fraction of the top semantic score that a keyword-matched
	 * record must have to be considered genuinely relevant when the
	 * refinement path's gap detection finds an intra-topic gap (validated
	 * by cross-boundary cosine). This separates records that are
	 * semantically close to the query intent (e.g. Weight for a "blood
	 * pressure, weight, and temperature" query) from coincidental keyword
	 * matches (e.g. Blood Oxygen matching "blood"). */
	public static final double REFINEMENT_SEMANTIC_RATIO = 0.70;

	/** Minimum cosine similarity between a keyword-matched record and
	 * the semantic core (non-keyword records found by gap detection)
	 * for the keyword match to be considered topically relevant.
	 * Keyword reps below this threshold are incidental matches —
	 * e.g. SpO2 matching "blood" in "blood oxygen saturation" when
	 * the query is about blood problems. Value of 0.55 sits between
	 * same-topic cross-type pairs (~0.65-0.80) and cross-topic pairs
	 * (~0.40-0.50). */
	public static final double SEMANTIC_CORE_MIN_COSINE = 0.55;

	/** Fraction of the semantic core's minimum score used as the
	 * relevance floor for expansion candidates. Semantic score
	 * (query cosine) is the only signal where genuinely relevant
	 * records consistently outscore template-similar noise — all
	 * inter-record cosine metrics (max, avg, min to core) fail
	 * because the embedding model conflates record-type similarity
	 * with content similarity. Value of 0.80 means expansion
	 * candidates must be within 20% of the core's relevance level. */
	public static final double SEMANTIC_CORE_SCORE_RATIO = 0.80;


	/** Gap multiplier for inter-candidate coherence outlier detection.
	 * Moderate sensitivity — only removes clear topic outliers. */
	public static final double COHERENCE_GAP_MULTIPLIER = 2.0;

	/** Fraction of max coherence used as the adaptive minimum gap for
	 * coherence outlier detection. */
	public static final double COHERENCE_ADAPTIVE_GAP_RATIO = 0.20;

	/** Reference candidate count for coherence gap ratio calibration.
	 * {@link #COHERENCE_ADAPTIVE_GAP_RATIO} was calibrated for candidate
	 * sets of this size. For smaller sets, the gap ratio is scaled up by
	 * √((REFERENCE−1) / (n−1)) to account for the higher variance in
	 * coherence estimates (fewer pairwise comparisons per candidate).
	 * Value of 5 means: for n=4 the scale is modest (√(4/3)≈1.15);
	 * for n≥5 the scale is 1.0 (no inflation). The coherence filter
	 * is not called for n&lt;4 (too few pairwise comparisons). */
	public static final int COHERENCE_REFERENCE_N = 5;


	public static final String GP_EMBEDDING_GAP_VALIDATION_COSINE_THRESHOLD =
			"chartsearchai.embedding.gapValidationCosineThreshold";

	/** Cosine similarity threshold for validating whether a gap in the
	 * score distribution is intra-topic (records on both sides belong to
	 * the same broad topic, e.g. different vital sign types) or
	 * inter-topic (a real relevance boundary). When the average cosine
	 * between the records just above and below the gap meets or exceeds
	 * this threshold, the gap is considered intra-topic and the second-pass
	 * cut is skipped. Value of 0.47 was determined empirically: medical
	 * records within the same broad category (all vitals) typically have
	 * inter-record cosine >= 0.47, while cross-category pairs (vital vs
	 * encounter note or condition) are typically below. */
	public static final double DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD = 0.47;

	public static final String GP_RETRIEVAL_PIPELINE = "chartsearchai.retrieval.pipeline";

	public static final String PIPELINE_EMBEDDING = "embedding";

	public static final String PIPELINE_LUCENE = "lucene";

	public static final String PIPELINE_ELASTICSEARCH = "elasticsearch";

	public static final String PIPELINE_HYBRID = "hybrid";

	/** Fetch multiplier for the Elasticsearch pipeline. The ES pipeline
	 * requests {@code topK * ES_FETCH_MULTIPLIER} results from the search
	 * engine, then passes all of them through the filter pipeline. This
	 * gives keyword rescue enough candidates to find all matching records
	 * — without it, RRF's fixed window would limit keyword rescue to
	 * only the top-K results, missing records that the embedding pipeline
	 * (which scores ALL records) would find. */
	public static final int ES_FETCH_MULTIPLIER = 10;

	/** RRF window size — number of top results from each retriever
	 * considered during rank fusion. Matches the Elasticsearch pipeline's
	 * setting so both pipelines produce comparable results. */
	public static final int RRF_RANK_WINDOW_SIZE = 100;

	/** RRF rank constant (k) — controls how quickly lower-ranked results
	 * lose influence: score = 1 / (k + rank). Higher k produces more
	 * uniform weighting across ranks. */
	public static final int RRF_RANK_CONSTANT = 60;

	/** Minimum cosine similarity for kNN results to be considered relevant.
	 * Uses the same value as {@link #ABSOLUTE_SIMILARITY_FLOOR}
	 * so both pipelines use the same relevance bar. */
	public static final double KNN_MIN_SIMILARITY = ABSOLUTE_SIMILARITY_FLOOR;

	/** Z-score threshold for the kNN fallback gate. When BM25 returns no
	 * keyword matches, kNN results must score this many standard deviations
	 * above the mean of all patient scores to be considered relevant. */
	public static final double KNN_FALLBACK_Z_SCORE = 2.5;

	public static final String GP_AUDIT_LOG_RETENTION_DAYS = "chartsearchai.auditLogRetentionDays";

	public static final int DEFAULT_AUDIT_LOG_RETENTION_DAYS = 90;

	public static final String GP_LLM_ENGINE = "chartsearchai.llm.engine";

	public static final String LLM_ENGINE_LOCAL = "local";

	public static final String LLM_ENGINE_REMOTE = "remote";

	public static final String GP_LLM_REMOTE_ENDPOINT_URL = "chartsearchai.llm.remote.endpointUrl";

	public static final String RP_LLM_REMOTE_API_KEY = "chartsearchai.llm.remote.apiKey";

	public static final String GP_LLM_REMOTE_MODEL_NAME = "chartsearchai.llm.remote.modelName";

	public static final String GP_LLM_CHAT_TEMPLATE = "chartsearchai.llm.chatTemplate";

	public static final String GP_SYSTEM_PROMPT = "chartsearchai.llm.systemPrompt";

	public static final String GP_LLM_TIMEOUT_SECONDS = "chartsearchai.llm.timeoutSeconds";

	public static final int DEFAULT_LLM_TIMEOUT_SECONDS = 120;

	public static final String GP_LLM_IDLE_TIMEOUT_MINUTES = "chartsearchai.llm.idleTimeoutMinutes";

	public static final int DEFAULT_LLM_IDLE_TIMEOUT_MINUTES = 30;

	public static final int DEFAULT_MAX_TOKENS = 2048;

	public static final float DEFAULT_REPEAT_PENALTY = 1.0f;

	public static final String GP_RATE_LIMIT_PER_MINUTE = "chartsearchai.rateLimitPerMinute";

	public static final int DEFAULT_RATE_LIMIT_PER_MINUTE = 10;

	public static final String GP_CACHE_TTL_MINUTES = "chartsearchai.cacheTtlMinutes";

	public static final int DEFAULT_CACHE_TTL_MINUTES = 0;

	public static final int DEFAULT_CACHE_MAX_SIZE = 100;

	// Resource type identifiers used in embeddings and citations
	public static final String RESOURCE_TYPE_OBS = "obs";

	public static final String RESOURCE_TYPE_CONDITION = "condition";

	public static final String RESOURCE_TYPE_ALLERGY = "allergy";

	public static final String RESOURCE_TYPE_DIAGNOSIS = "diagnosis";

	public static final String RESOURCE_TYPE_ORDER = "order";

	public static final String RESOURCE_TYPE_PROGRAM = "program";

	public static final String RESOURCE_TYPE_MEDICATION_DISPENSE = "medication_dispense";

	private ChartSearchAiConstants() {
	}
}
