/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.api;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.apache.http.HttpHost;
import org.apache.http.util.EntityUtils;
import org.elasticsearch.client.Request;
import org.elasticsearch.client.Response;
import org.elasticsearch.client.ResponseException;
import org.elasticsearch.client.RestClient;
import org.openmrs.Patient;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * Indexes patient clinical records into Elasticsearch for hybrid retrieval
 * using Reciprocal Rank Fusion (RRF) to combine BM25 text search with kNN
 * dense vector search. Selected via the {@code chartsearchai.retrieval.pipeline}
 * global property set to {@code elasticsearch}.
 *
 * <p>Requires an Elasticsearch 8.14+ or OpenSearch 2.19+ instance
 * configured in the OpenMRS runtime properties
 * ({@code hibernate.search.backend.uris}). Uses the low-level REST
 * client already on the classpath via Hibernate Search. The backend
 * type (Elasticsearch vs OpenSearch) is auto-detected on first use.
 *
 * <p>Each document stores the patient ID, resource type, resource ID,
 * the prefixed text (for BM25), and the embedding vector (for kNN).
 * The RRF retriever fuses rankings from both signals so that queries
 * like "any cancer?" find semantic matches (e.g. Kaposi sarcoma) even
 * when the literal term is absent from the records.
 */
@Component("elasticsearchIndexer")
public class ElasticsearchIndexer implements Closeable {

	private static final Logger log = LoggerFactory.getLogger(ElasticsearchIndexer.class);

	static final String INDEX_NAME = "chartsearchai-patient-records";

	static final String FIELD_PATIENT_ID = "patient_id";

	static final String FIELD_RESOURCE_TYPE = "resource_type";

	static final String FIELD_RESOURCE_ID = "resource_id";

	static final String FIELD_TEXT = "text";

	static final String FIELD_EMBEDDING = "embedding";

	/** RRF window size — number of top results from each retriever
	 * considered during rank fusion. */
	static final int RRF_RANK_WINDOW_SIZE = 100;

	/** RRF rank constant (k) — controls how quickly lower-ranked results
	 * lose influence: score = 1 / (k + rank). Higher k produces more
	 * uniform weighting across ranks. */
	static final int RRF_RANK_CONSTANT = 60;

	/** Number of approximate nearest-neighbor candidates evaluated per
	 * shard before selecting the top-k. Higher values improve recall
	 * at the cost of latency. */
	static final int KNN_NUM_CANDIDATES = 100;

	static final String SEARCH_PIPELINE_NAME = "chartsearchai-rrf";

	enum BackendType { ELASTICSEARCH, OPENSEARCH }

	@Autowired
	private PatientRecordLoader recordLoader;

	@Autowired
	@Qualifier("chartSearchAi.embeddingProvider")
	private EmbeddingProvider embeddingProvider;

	private volatile RestClient client;

	private volatile boolean indexCreated;

	private volatile boolean pipelineCreated;

	private volatile BackendType backendType;

	private final Object lock = new Object();

	private final ObjectMapper mapper = new ObjectMapper();

	/**
	 * Returns the ES RestClient, creating it lazily from runtime properties.
	 * Returns null if Elasticsearch is not configured.
	 */
	RestClient getClient() {
		if (client == null) {
			synchronized (lock) {
				if (client == null) {
					String esUri = getElasticsearchUri();
					if (esUri == null) {
						return null;
					}
					try {
						HttpHost host = HttpHost.create(esUri);
						client = RestClient.builder(host).build();
					}
					catch (Exception e) {
						log.error("Failed to create Elasticsearch client for URI: {}", esUri, e);
						return null;
					}
				}
			}
		}
		return client;
	}

	/**
	 * Allows tests to inject a pre-built client.
	 */
	void setClient(RestClient client) {
		this.client = client;
	}

	/**
	 * Allows tests to set the backend type directly.
	 */
	void setBackendType(BackendType type) {
		this.backendType = type;
	}

	BackendType getBackendType() {
		return backendType;
	}

	/**
	 * Auto-detects whether the backend is Elasticsearch or OpenSearch
	 * by checking for the {@code distribution} field in the root response.
	 */
	void detectBackendType() throws IOException {
		if (backendType != null) {
			return;
		}
		RestClient c = getClient();
		if (c == null) {
			return;
		}
		Response response = c.performRequest(new Request("GET", "/"));
		String body = EntityUtils.toString(response.getEntity());
		JsonNode root = mapper.readTree(body);
		JsonNode distribution = root.path("version").path("distribution");
		if (!distribution.isMissingNode() && "opensearch".equals(distribution.asText())) {
			backendType = BackendType.OPENSEARCH;
			log.info("Detected OpenSearch backend");
		} else {
			backendType = BackendType.ELASTICSEARCH;
			log.info("Detected Elasticsearch backend");
		}
	}

	/**
	 * Returns true if Elasticsearch is configured in the OpenMRS runtime
	 * properties and the cluster is reachable.
	 */
	public boolean isAvailable() {
		RestClient c = getClient();
		if (c == null) {
			return false;
		}
		try {
			Response response = c.performRequest(new Request("GET", "/_cluster/health"));
			return response.getStatusLine().getStatusCode() == 200;
		}
		catch (IOException e) {
			log.debug("Elasticsearch not reachable", e);
			return false;
		}
	}

	/**
	 * Ensures the index exists with the correct mapping. Creates it on
	 * first call, detecting embedding dimensions from the provider.
	 */
	void ensureIndex() throws IOException {
		detectBackendType();
		if (indexCreated) {
			// Verify the index still exists to handle external deletion
			RestClient c = getClient();
			if (c != null) {
				try {
					c.performRequest(new Request("HEAD", "/" + INDEX_NAME));
					return;
				}
				catch (ResponseException e) {
					if (e.getResponse().getStatusLine().getStatusCode() == 404) {
						indexCreated = false;
					} else {
						throw e;
					}
				}
			}
		}
		synchronized (lock) {
			if (indexCreated) {
				return;
			}
			RestClient c = getClient();
			if (c == null) {
				throw new IOException("Elasticsearch not configured");
			}

			// Check if index already exists
			try {
				c.performRequest(new Request("HEAD", "/" + INDEX_NAME));
				if (hasMismatchedMapping(c)) {
					log.info("Index '{}' has incompatible mapping for {} backend, recreating",
							INDEX_NAME, backendType);
					c.performRequest(new Request("DELETE", "/" + INDEX_NAME));
				} else {
					indexCreated = true;
					ensureSearchPipeline();
					return;
				}
			}
			catch (ResponseException e) {
				if (e.getResponse().getStatusLine().getStatusCode() != 404) {
					throw e;
				}
			}

			// Detect embedding dimensions
			int dims = embeddingProvider.getDimensions();
			if (dims <= 0) {
				embeddingProvider.embed("dimension detection");
				dims = embeddingProvider.getDimensions();
			}
			if (dims <= 0) {
				throw new IOException("Unable to detect embedding dimensions");
			}

			Request createReq = new Request("PUT", "/" + INDEX_NAME);
			createReq.setJsonEntity(buildIndexMapping(dims));
			c.performRequest(createReq);
			indexCreated = true;
			log.info("Created index '{}' with {} dimensions (backend: {})",
					INDEX_NAME, dims, backendType);
			ensureSearchPipeline();
		}
	}

	/**
	 * Creates the RRF search pipeline on OpenSearch if it doesn't exist.
	 * No-op for Elasticsearch (which uses the retriever API instead).
	 */
	void ensureSearchPipeline() throws IOException {
		if (backendType != BackendType.OPENSEARCH || pipelineCreated) {
			return;
		}
		RestClient c = getClient();
		if (c == null) {
			return;
		}
		Request req = new Request("PUT", "/_search/pipeline/" + SEARCH_PIPELINE_NAME);
		req.setJsonEntity(buildSearchPipelineBody());
		c.performRequest(req);
		pipelineCreated = true;
		log.info("Created OpenSearch search pipeline '{}'", SEARCH_PIPELINE_NAME);
	}

	/**
	 * Checks if the existing index mapping is incompatible with the
	 * detected backend. OpenSearch requires {@code knn_vector};
	 * Elasticsearch requires {@code dense_vector}.
	 */
	private boolean hasMismatchedMapping(RestClient c) {
		try {
			Request req = new Request("GET", "/" + INDEX_NAME + "/_mapping");
			Response response = c.performRequest(req);
			String body = EntityUtils.toString(response.getEntity());
			JsonNode root = mapper.readTree(body);
			String embeddingType = root.path(INDEX_NAME).path("mappings")
					.path("properties").path(FIELD_EMBEDDING).path("type").asText("");
			if (backendType == BackendType.OPENSEARCH) {
				return !"knn_vector".equals(embeddingType);
			}
			return !"dense_vector".equals(embeddingType);
		}
		catch (Exception e) {
			log.debug("Failed to check index mapping, assuming mismatch", e);
			return true;
		}
	}

	/**
	 * Builds the OpenSearch search pipeline JSON body for RRF.
	 */
	String buildSearchPipelineBody() throws IOException {
		ObjectNode body = mapper.createObjectNode();
		ArrayNode processors = body.putArray("phase_results_processors");
		ObjectNode processor = processors.addObject().putObject("score-ranker-processor");
		ObjectNode combination = processor.putObject("combination");
		combination.put("technique", "rrf");
		combination.put("rank_constant", RRF_RANK_CONSTANT);
		return mapper.writeValueAsString(body);
	}

	/**
	 * Builds the index creation JSON with mappings for text (BM25) and
	 * dense_vector/knn_vector fields, adapting to the detected backend.
	 */
	String buildIndexMapping(int dims) throws IOException {
		ObjectNode body = mapper.createObjectNode();

		if (backendType == BackendType.OPENSEARCH) {
			body.putObject("settings").put("index.knn", true);
		}

		ObjectNode mappings = body.putObject("mappings");
		ObjectNode properties = mappings.putObject("properties");

		properties.putObject(FIELD_PATIENT_ID).put("type", "integer");
		properties.putObject(FIELD_RESOURCE_TYPE).put("type", "keyword");
		properties.putObject(FIELD_RESOURCE_ID).put("type", "integer");
		properties.putObject(FIELD_TEXT).put("type", "text").put("analyzer", "english");

		ObjectNode embeddingField = properties.putObject(FIELD_EMBEDDING);
		if (backendType == BackendType.OPENSEARCH) {
			embeddingField.put("type", "knn_vector");
			embeddingField.put("dimension", dims);
			ObjectNode method = embeddingField.putObject("method");
			method.put("name", "hnsw");
			method.put("space_type", "cosinesimil");
			method.put("engine", "lucene");
		} else {
			embeddingField.put("type", "dense_vector");
			embeddingField.put("dims", dims);
			embeddingField.put("index", true);
			embeddingField.put("similarity", "cosine");
		}

		return mapper.writeValueAsString(body);
	}

	/**
	 * Full index of a patient's chart. Deletes existing documents for the
	 * patient and re-indexes all clinical data with both text and embeddings.
	 */
	public void indexPatient(Patient patient) {
		RestClient c = getClient();
		if (c == null) {
			log.warn("Elasticsearch not configured, skipping indexing for patient [id={}]",
					patient.getPatientId());
			return;
		}

		log.info("Elasticsearch: indexing patient [id={}]", patient.getPatientId());
		List<SerializedRecord> records = recordLoader.loadAll(patient);

		try {
			ensureIndex();
			deletePatientIndex(patient);

			if (records.isEmpty()) {
				return;
			}

			int bulkBatchSize = 100;
			StringBuilder bulk = new StringBuilder();
			int indexed = 0;
			int batchCount = 0;
			for (SerializedRecord record : records) {
				try {
					String prefixedText = ChartSearchAiConstants.getEmbeddingPrefix(
							record.getResourceType(), record.getText()) + record.getText();
					float[] embedding = embeddingProvider.embed(prefixedText);

					String docId = patient.getPatientId() + "_"
							+ record.getResourceType() + "_" + record.getResourceId();

					ObjectNode action = mapper.createObjectNode();
					action.putObject("index")
							.put("_index", INDEX_NAME)
							.put("_id", docId);
					bulk.append(mapper.writeValueAsString(action)).append('\n');

					ObjectNode doc = mapper.createObjectNode();
					doc.put(FIELD_PATIENT_ID, patient.getPatientId());
					doc.put(FIELD_RESOURCE_TYPE, record.getResourceType());
					doc.put(FIELD_RESOURCE_ID, record.getResourceId());
					doc.put(FIELD_TEXT, prefixedText);
					ArrayNode embArr = doc.putArray(FIELD_EMBEDDING);
					for (float v : embedding) {
						embArr.add(v);
					}
					bulk.append(mapper.writeValueAsString(doc)).append('\n');
					indexed++;
					batchCount++;
				}
				catch (Exception e) {
					log.warn("Failed to prepare ES document for {} [id={}]: {}",
							record.getResourceType(), record.getResourceId(), e.getMessage());
				}

				if (batchCount >= bulkBatchSize) {
					sendBulkRequest(c, bulk.toString(), patient.getPatientId());
					bulk.setLength(0);
					batchCount = 0;
				}
			}

			if (batchCount > 0) {
				sendBulkRequest(c, bulk.toString(), patient.getPatientId());
			}

			c.performRequest(new Request("POST", "/" + INDEX_NAME + "/_refresh"));

			log.info("Elasticsearch: indexed {} of {} records for patient [id={}]",
					indexed, records.size(), patient.getPatientId());
		}
		catch (IOException e) {
			log.error("Elasticsearch: failed to index patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	private void sendBulkRequest(RestClient c, String body, int patientId) throws IOException {
		Request bulkReq = new Request("POST", "/_bulk");
		bulkReq.setJsonEntity(body);
		Response response = c.performRequest(bulkReq);
		try {
			String responseBody = EntityUtils.toString(response.getEntity());
			JsonNode responseJson = mapper.readTree(responseBody);
			if (responseJson.has("errors") && responseJson.get("errors").asBoolean()) {
				log.warn("Elasticsearch bulk index had errors for patient [id={}]", patientId);
			}
		}
		finally {
			EntityUtils.consumeQuietly(response.getEntity());
		}
	}

	/**
	 * Deletes all Elasticsearch documents for a patient.
	 */
	public void deletePatientIndex(Patient patient) {
		RestClient c = getClient();
		if (c == null) {
			return;
		}
		try {
			Request req = new Request("POST", "/" + INDEX_NAME + "/_delete_by_query");
			req.setJsonEntity(buildPatientTermQuery(patient.getPatientId()));
			c.performRequest(req);
		}
		catch (ResponseException e) {
			if (e.getResponse().getStatusLine().getStatusCode() == 404) {
				return;
			}
			log.error("Elasticsearch: failed to delete index for patient [id={}]",
					patient.getPatientId(), e);
		}
		catch (IOException e) {
			log.error("Elasticsearch: failed to delete index for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	/**
	 * Returns true if the Elasticsearch index has any documents for the given patient.
	 */
	public boolean hasIndex(Patient patient) {
		RestClient c = getClient();
		if (c == null) {
			return false;
		}
		try {
			Request req = new Request("POST", "/" + INDEX_NAME + "/_count");
			req.setJsonEntity(buildPatientTermQuery(patient.getPatientId()));
			Response response = c.performRequest(req);
			String body = EntityUtils.toString(response.getEntity());
			JsonNode json = mapper.readTree(body);
			return json.has("count") && json.get("count").asInt() > 0;
		}
		catch (Exception e) {
			log.debug("Elasticsearch: failed to check index for patient [id={}]",
					patient.getPatientId(), e);
			return false;
		}
	}

	/**
	 * Hybrid search using Reciprocal Rank Fusion (RRF) combining BM25 text
	 * search and kNN dense vector search in a single Elasticsearch query.
	 * Returns results ordered by the fused relevance score.
	 *
	 * @param patient the patient to filter by
	 * @param queryText the user's question (used for BM25 text matching)
	 * @param queryVector the embedded query vector (used for kNN similarity)
	 * @param maxResults maximum number of results to return
	 * @return matching results with resource type, ID, and RRF score
	 */
	public List<ElasticsearchSearchResult> search(Patient patient, String queryText,
			float[] queryVector, int maxResults) {
		List<ElasticsearchSearchResult> results = new ArrayList<ElasticsearchSearchResult>();
		RestClient c = getClient();
		if (c == null) {
			return results;
		}

		try {
			detectBackendType();
			ensureSearchPipeline();
			String searchBody = buildSearchQuery(patient.getPatientId(),
					queryText, queryVector, maxResults);
			String endpoint = "/" + INDEX_NAME + "/_search";
			if (backendType == BackendType.OPENSEARCH) {
				endpoint += "?search_pipeline=" + SEARCH_PIPELINE_NAME;
			}
			Request req = new Request("POST", endpoint);
			req.setJsonEntity(searchBody);
			Response response = c.performRequest(req);
			String responseBody = EntityUtils.toString(response.getEntity());
			JsonNode responseJson = mapper.readTree(responseBody);

			JsonNode hits = responseJson.path("hits").path("hits");
			for (JsonNode hit : hits) {
				JsonNode src = hit.path("_source");
				if (!src.has(FIELD_RESOURCE_TYPE) || !src.has(FIELD_RESOURCE_ID)) {
					log.warn("Elasticsearch: skipping hit with missing fields");
					continue;
				}
				results.add(new ElasticsearchSearchResult(
						src.get(FIELD_RESOURCE_TYPE).asText(),
						src.get(FIELD_RESOURCE_ID).asInt(),
						hit.has("_score") ? hit.get("_score").floatValue() : 0f));
			}
		}
		catch (Exception e) {
			log.error("Elasticsearch: search failed for patient [id={}] query '{}'",
					patient.getPatientId(), queryText, e);
		}

		return results;
	}

	/**
	 * Builds the hybrid search query JSON, adapting to the detected backend.
	 * Elasticsearch uses the retriever API (ES 8.14+); OpenSearch uses
	 * a hybrid query with a search pipeline for RRF (OS 2.19+).
	 */
	String buildSearchQuery(int patientId, String queryText,
			float[] queryVector, int maxResults) throws IOException {
		if (backendType == BackendType.OPENSEARCH) {
			return buildOpenSearchQuery(patientId, queryText, queryVector, maxResults);
		}
		return buildElasticsearchQuery(patientId, queryText, queryVector, maxResults);
	}

	/**
	 * Builds the RRF hybrid search query using the Elasticsearch
	 * retriever API (ES 8.14+).
	 */
	String buildElasticsearchQuery(int patientId, String queryText,
			float[] queryVector, int maxResults) throws IOException {
		ObjectNode body = mapper.createObjectNode();
		body.put("size", maxResults);

		ArrayNode source = body.putArray("_source");
		source.add(FIELD_RESOURCE_TYPE);
		source.add(FIELD_RESOURCE_ID);

		ObjectNode retriever = body.putObject("retriever");
		ObjectNode rrf = retriever.putObject("rrf");
		rrf.put("rank_window_size", RRF_RANK_WINDOW_SIZE);
		rrf.put("rank_constant", RRF_RANK_CONSTANT);

		ArrayNode retrievers = rrf.putArray("retrievers");

		// BM25 text retriever
		ObjectNode bm25Retriever = retrievers.addObject().putObject("standard");
		ObjectNode bm25Query = bm25Retriever.putObject("query").putObject("bool");
		bm25Query.putArray("must").addObject()
				.putObject("match").put(FIELD_TEXT, queryText);
		bm25Query.putArray("filter").addObject()
				.putObject("term").put(FIELD_PATIENT_ID, patientId);

		// kNN vector retriever
		ObjectNode knnRetriever = retrievers.addObject().putObject("knn");
		knnRetriever.put("field", FIELD_EMBEDDING);
		ArrayNode qv = knnRetriever.putArray("query_vector");
		for (float v : queryVector) {
			qv.add(v);
		}
		knnRetriever.put("k", Math.min(maxResults, KNN_NUM_CANDIDATES));
		knnRetriever.put("num_candidates", KNN_NUM_CANDIDATES);
		knnRetriever.putObject("filter")
				.putObject("term").put(FIELD_PATIENT_ID, patientId);

		return mapper.writeValueAsString(body);
	}

	/**
	 * Builds the hybrid search query for OpenSearch (OS 2.19+).
	 * Uses the hybrid query type with BM25 and kNN sub-queries;
	 * RRF is applied via the search pipeline set on the request.
	 */
	String buildOpenSearchQuery(int patientId, String queryText,
			float[] queryVector, int maxResults) throws IOException {
		ObjectNode body = mapper.createObjectNode();
		body.put("size", maxResults);

		ArrayNode source = body.putArray("_source");
		source.add(FIELD_RESOURCE_TYPE);
		source.add(FIELD_RESOURCE_ID);

		ObjectNode hybrid = body.putObject("query").putObject("hybrid");
		ArrayNode queries = hybrid.putArray("queries");

		// BM25 text query
		ObjectNode bm25Bool = queries.addObject().putObject("bool");
		bm25Bool.putArray("must").addObject()
				.putObject("match").put(FIELD_TEXT, queryText);
		bm25Bool.putArray("filter").addObject()
				.putObject("term").put(FIELD_PATIENT_ID, patientId);

		// kNN vector query
		ObjectNode knnOuter = queries.addObject().putObject("knn");
		ObjectNode knnInner = knnOuter.putObject(FIELD_EMBEDDING);
		ArrayNode qv = knnInner.putArray("vector");
		for (float v : queryVector) {
			qv.add(v);
		}
		knnInner.put("k", Math.min(maxResults, KNN_NUM_CANDIDATES));
		knnInner.putObject("filter")
				.putObject("term").put(FIELD_PATIENT_ID, patientId);

		return mapper.writeValueAsString(body);
	}

	/**
	 * Builds a JSON term query for filtering by patient ID, used by
	 * {@link #deletePatientIndex} and {@link #hasIndex}.
	 */
	private String buildPatientTermQuery(int patientId) throws IOException {
		ObjectNode root = mapper.createObjectNode();
		root.putObject("query").putObject("term")
				.put(FIELD_PATIENT_ID, patientId);
		return mapper.writeValueAsString(root);
	}

	private String getElasticsearchUri() {
		Properties props = Context.getRuntimeProperties();
		if (props == null) {
			return null;
		}
		String uri = props.getProperty("hibernate.search.backend.uris");
		if (uri == null || uri.trim().isEmpty()) {
			return null;
		}
		String[] uris = uri.split(",");
		return uris[0].trim();
	}

	/**
	 * Re-indexes the patient if the current retrieval pipeline is Elasticsearch
	 * and the patient already has indexed data. Called by AOP advice classes
	 * after patient data changes.
	 */
	public void reindexIfActive(Patient patient) {
		if (patient == null) {
			return;
		}
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_ELASTICSEARCH.equalsIgnoreCase(
				pipeline != null ? pipeline.trim() : "")) {
			return;
		}
		if (hasIndex(patient)) {
			indexPatient(patient);
		}
	}

	/**
	 * Deletes the patient's Elasticsearch index if the current retrieval
	 * pipeline is Elasticsearch. Called by AOP advice after patient merges.
	 */
	public void deleteIfActive(Patient patient) {
		if (patient == null) {
			return;
		}
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_ELASTICSEARCH.equalsIgnoreCase(
				pipeline != null ? pipeline.trim() : "")) {
			return;
		}
		deletePatientIndex(patient);
	}

	@Override
	public void close() throws IOException {
		synchronized (lock) {
			if (client != null) {
				client.close();
				client = null;
			}
			indexCreated = false;
			pipelineCreated = false;
			backendType = null;
		}
	}

	/**
	 * A single Elasticsearch search result with resource metadata and score.
	 */
	public static class ElasticsearchSearchResult {

		private final String resourceType;

		private final int resourceId;

		private final float score;

		public ElasticsearchSearchResult(String resourceType, int resourceId, float score) {
			this.resourceType = resourceType;
			this.resourceId = resourceId;
			this.score = score;
		}

		public String getResourceType() {
			return resourceType;
		}

		public int getResourceId() {
			return resourceId;
		}

		public float getScore() {
			return score;
		}
	}
}
