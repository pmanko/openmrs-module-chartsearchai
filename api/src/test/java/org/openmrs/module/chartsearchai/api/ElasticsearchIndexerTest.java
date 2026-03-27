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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class ElasticsearchIndexerTest {

	private ElasticsearchIndexer indexer;

	private ObjectMapper mapper;

	@BeforeEach
	public void setUp() {
		indexer = new ElasticsearchIndexer();
		mapper = new ObjectMapper();
	}

	@Test
	public void buildIndexMapping_shouldIncludeTextAndDenseVectorFields() throws IOException {
		String json = indexer.buildIndexMapping(384);
		JsonNode root = mapper.readTree(json);

		JsonNode properties = root.path("mappings").path("properties");
		assertFalse(properties.isMissingNode(), "Should have properties node");

		assertEquals("integer", properties.path("patient_id").path("type").asText());
		assertEquals("keyword", properties.path("resource_type").path("type").asText());
		assertEquals("integer", properties.path("resource_id").path("type").asText());
		assertEquals("text", properties.path("text").path("type").asText());
		assertEquals("english", properties.path("text").path("analyzer").asText());
		assertEquals("dense_vector", properties.path("embedding").path("type").asText());
		assertEquals(384, properties.path("embedding").path("dims").asInt());
		assertTrue(properties.path("embedding").path("index").asBoolean());
		assertEquals("cosine", properties.path("embedding").path("similarity").asText());
	}

	@Test
	public void buildIndexMapping_shouldUseDimensionsFromProvider() throws IOException {
		String json768 = indexer.buildIndexMapping(768);
		JsonNode root = mapper.readTree(json768);
		assertEquals(768, root.path("mappings").path("properties")
				.path("embedding").path("dims").asInt());
	}

	@Test
	public void buildSearchQuery_shouldContainRrfRetrieverWithBm25AndKnn() throws IOException {
		float[] queryVector = new float[] { 0.1f, 0.2f, 0.3f };
		String json = indexer.buildSearchQuery(42, "cancer history", queryVector, 10);
		JsonNode root = mapper.readTree(json);

		assertEquals(10, root.path("size").asInt());

		// Should request only resource_type and resource_id in _source
		JsonNode source = root.path("_source");
		assertTrue(source.isArray());
		assertEquals(2, source.size());

		// RRF retriever structure
		JsonNode rrf = root.path("retriever").path("rrf");
		assertFalse(rrf.isMissingNode(), "Should have rrf retriever");
		assertEquals(ElasticsearchIndexer.RRF_RANK_WINDOW_SIZE, rrf.path("rank_window_size").asInt());
		assertEquals(ElasticsearchIndexer.RRF_RANK_CONSTANT, rrf.path("rank_constant").asInt());

		JsonNode retrievers = rrf.path("retrievers");
		assertEquals(2, retrievers.size(), "Should have BM25 and kNN retrievers");

		// BM25 retriever
		JsonNode bm25 = retrievers.get(0).path("standard");
		assertFalse(bm25.isMissingNode());
		JsonNode bm25Bool = bm25.path("query").path("bool");
		assertEquals("cancer history",
				bm25Bool.path("must").get(0).path("match").path("text").asText());
		assertEquals(42,
				bm25Bool.path("filter").get(0).path("term").path("patient_id").asInt());

		// kNN retriever
		JsonNode knn = retrievers.get(1).path("knn");
		assertFalse(knn.isMissingNode());
		assertEquals("embedding", knn.path("field").asText());
		assertEquals(10, knn.path("k").asInt());
		assertEquals(ElasticsearchIndexer.KNN_NUM_CANDIDATES, knn.path("num_candidates").asInt());
		assertEquals(42, knn.path("filter").path("term").path("patient_id").asInt());

		// query_vector should match input
		JsonNode qv = knn.path("query_vector");
		assertEquals(3, qv.size());
		assertEquals(0.1, qv.get(0).doubleValue(), 0.001);
		assertEquals(0.2, qv.get(1).doubleValue(), 0.001);
		assertEquals(0.3, qv.get(2).doubleValue(), 0.001);
	}

	@Test
	public void buildSearchQuery_shouldFilterByPatientInBothRetrievers() throws IOException {
		float[] queryVector = new float[] { 0.5f };
		String json = indexer.buildSearchQuery(99, "medications", queryVector, 5);
		JsonNode root = mapper.readTree(json);

		JsonNode retrievers = root.path("retriever").path("rrf").path("retrievers");

		// BM25 filter
		int bm25PatientId = retrievers.get(0).path("standard").path("query")
				.path("bool").path("filter").get(0).path("term")
				.path("patient_id").asInt();
		assertEquals(99, bm25PatientId);

		// kNN filter
		int knnPatientId = retrievers.get(1).path("knn").path("filter")
				.path("term").path("patient_id").asInt();
		assertEquals(99, knnPatientId);
	}

	@Test
	public void buildSearchQuery_shouldCapKnnKAtNumCandidates() throws IOException {
		float[] queryVector = new float[] { 0.1f };
		// Request more results than KNN_NUM_CANDIDATES
		String json = indexer.buildSearchQuery(1, "test", queryVector, 200);
		JsonNode root = mapper.readTree(json);

		JsonNode knn = root.path("retriever").path("rrf").path("retrievers")
				.get(1).path("knn");
		assertEquals(ElasticsearchIndexer.KNN_NUM_CANDIDATES, knn.path("k").asInt(),
				"k should be capped at num_candidates when maxResults exceeds it");
		assertEquals(ElasticsearchIndexer.KNN_NUM_CANDIDATES, knn.path("num_candidates").asInt());
	}

	@Test
	public void isAvailable_shouldReturnFalseWhenClientIsNull() {
		// No client set — ES not configured
		assertFalse(indexer.isAvailable());
	}

	@Test
	public void hasIndex_shouldReturnFalseWhenClientIsNull() {
		org.openmrs.Patient patient = new org.openmrs.Patient();
		patient.setPatientId(1);
		assertFalse(indexer.hasIndex(patient));
	}

	@Test
	public void search_shouldReturnEmptyWhenClientIsNull() {
		org.openmrs.Patient patient = new org.openmrs.Patient();
		patient.setPatientId(1);
		float[] queryVector = new float[] { 0.1f, 0.2f };
		List<ElasticsearchIndexer.ElasticsearchSearchResult> results =
				indexer.search(patient, "test", queryVector, 10);
		assertNotNull(results);
		assertTrue(results.isEmpty());
	}

	@Test
	public void elasticsearchSearchResult_shouldExposeFields() {
		ElasticsearchIndexer.ElasticsearchSearchResult result =
				new ElasticsearchIndexer.ElasticsearchSearchResult("obs", 42, 1.5f);
		assertEquals("obs", result.getResourceType());
		assertEquals(42, result.getResourceId());
		assertEquals(1.5f, result.getScore(), 0.001f);
	}
}
