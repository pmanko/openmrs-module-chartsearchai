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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.embedding.StubEmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;

public class HybridRetrieverTest {

	@Test
	public void fuseRRF_shouldRankDocumentAppearingInBothListsHigher() {
		List<String> bm25 = Arrays.asList("obs:1", "condition:2", "obs:3");
		List<String> knn = Arrays.asList("condition:2", "obs:4", "obs:1");

		List<String> fused = HybridRetriever.fuseRRF(bm25, knn,
				ChartSearchAiConstants.RRF_RANK_CONSTANT, 10);

		// "condition:2" is rank 2 in BM25 (1/62) and rank 1 in kNN (1/61)
		// "obs:1" is rank 1 in BM25 (1/61) and rank 3 in kNN (1/63)
		// condition:2 combined = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
		// obs:1 combined = 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226
		assertEquals("condition:2", fused.get(0));
		assertEquals("obs:1", fused.get(1));
	}

	@Test
	public void fuseRRF_shouldIncludeResultsFromBothLists() {
		List<String> bm25 = Arrays.asList("obs:1", "obs:2");
		List<String> knn = Arrays.asList("obs:3", "obs:4");

		List<String> fused = HybridRetriever.fuseRRF(bm25, knn,
				ChartSearchAiConstants.RRF_RANK_CONSTANT, 10);

		assertEquals(4, fused.size());
		assertTrue(fused.contains("obs:1"));
		assertTrue(fused.contains("obs:2"));
		assertTrue(fused.contains("obs:3"));
		assertTrue(fused.contains("obs:4"));
	}

	@Test
	public void fuseRRF_shouldRespectMaxResults() {
		List<String> bm25 = Arrays.asList("obs:1", "obs:2", "obs:3");
		List<String> knn = Arrays.asList("obs:4", "obs:5", "obs:6");

		List<String> fused = HybridRetriever.fuseRRF(bm25, knn,
				ChartSearchAiConstants.RRF_RANK_CONSTANT, 2);

		assertEquals(2, fused.size());
	}

	@Test
	public void fuseRRF_shouldHandleEmptyLists() {
		List<String> empty = Collections.emptyList();

		assertEquals(0, HybridRetriever.fuseRRF(empty, empty, 60, 10).size());

		List<String> bm25 = Arrays.asList("obs:1", "obs:2");
		List<String> fused = HybridRetriever.fuseRRF(bm25, empty, 60, 10);
		assertEquals(2, fused.size());
		assertEquals("obs:1", fused.get(0));
	}

	@Test
	public void fuseRRF_shouldHandleSingleElementLists() {
		List<String> bm25 = Arrays.asList("obs:1");
		List<String> knn = Arrays.asList("obs:1");

		List<String> fused = HybridRetriever.fuseRRF(bm25, knn, 60, 10);

		assertEquals(1, fused.size());
		assertEquals("obs:1", fused.get(0));
	}

	@Test
	public void fuseRRF_rrfConstantShouldMatchElasticsearchPipeline() {
		assertEquals(100, ChartSearchAiConstants.RRF_RANK_WINDOW_SIZE);
		assertEquals(60, ChartSearchAiConstants.RRF_RANK_CONSTANT);
		assertEquals(ElasticsearchIndexer.RRF_RANK_WINDOW_SIZE,
				ChartSearchAiConstants.RRF_RANK_WINDOW_SIZE);
		assertEquals(ElasticsearchIndexer.RRF_RANK_CONSTANT,
				ChartSearchAiConstants.RRF_RANK_CONSTANT);
	}

	@Test
	public void rankByCosineSimilarity_shouldReturnTopNBySimilarity() {
		StubEmbeddingProvider provider = new StubEmbeddingProvider();

		List<SerializedRecord> records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null),
				new SerializedRecord("obs", 20,
						"Test — Systolic Blood Pressure: 137.0", null),
				new SerializedRecord("obs", 21,
						"Test — Diastolic Blood Pressure: 67.0", null),
				new SerializedRecord("condition", 30,
						"Condition: Hypertension. Status: ACTIVE", null));

		List<ChartEmbedding> embeddings = EmbeddingIndexer.buildEmbeddings(records, provider);
		float[] queryVector = provider.embed("blood pressure");

		List<String> ranked = HybridRetriever.rankByCosineSimilarity(
				embeddings, queryVector, 2);

		assertEquals(2, ranked.size());
		// Blood pressure records should rank higher than conditions
		assertTrue(ranked.contains("obs:20") || ranked.contains("obs:21"),
				"Top 2 should include a blood pressure record, got: " + ranked);
	}

	@Test
	public void fuseRRF_shouldBoostDocumentsInBothListsOverSingleListEntries() {
		// A document in both lists at low rank should beat a document in
		// only one list at high rank, due to the double RRF contribution
		List<String> bm25 = Arrays.asList("obs:1", "obs:2", "obs:3", "obs:BOTH");
		List<String> knn = Arrays.asList("obs:4", "obs:5", "obs:6", "obs:BOTH");

		List<String> fused = HybridRetriever.fuseRRF(bm25, knn, 60, 10);

		// obs:BOTH is rank 4 in both lists: 2 * (1/64) = 0.03125
		// obs:1 is rank 1 in BM25 only: 1/61 = 0.01639
		int bothIdx = fused.indexOf("obs:BOTH");
		int singleIdx = fused.indexOf("obs:1");
		assertTrue(bothIdx < singleIdx,
				"Document in both lists should rank higher than single-list documents");
	}

	@Test
	public void rankByCosineSimilarity_shouldHandleEmptyEmbeddings() {
		float[] queryVector = new float[] { 1.0f, 0.0f, 0.0f };
		List<String> ranked = HybridRetriever.rankByCosineSimilarity(
				new ArrayList<ChartEmbedding>(), queryVector, 10);
		assertEquals(0, ranked.size());
	}
}
