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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.embedding.StubEmbeddingProvider;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
import org.openmrs.module.chartsearchai.eval.EvalMetrics;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Eval suite for retrieval quality. Uses {@link StubEmbeddingProvider}
 * to test the scoring, gap detection, and keyword refinement pipeline
 * against the 153-record patient dataset.
 */
public class RetrievalQualityEvalTest {

	private static final Logger log = LoggerFactory.getLogger(RetrievalQualityEvalTest.class);

	// Same dataset used in LlmInferenceServiceTest
	private static final String[] DATASET = LlmInferenceServiceTest.FULL_PATIENT_DATASET;

	private static EvalDataset evalDataset;

	private static EmbeddingProvider embeddingProvider;

	private static List<ChartEmbedding> allEmbeddings;

	@BeforeAll
	static void setup() throws IOException {
		evalDataset = EvalDataset.load("eval/retrieval-eval-dataset.json");
		embeddingProvider = new StubEmbeddingProvider();
		allEmbeddings = buildEmbeddings();
	}

	private static List<ChartEmbedding> buildEmbeddings() {
		List<ChartEmbedding> embeddings = new ArrayList<>();
		Date now = new Date();
		for (int i = 0; i < DATASET.length; i++) {
			String text = DATASET[i];
			String resourceType = inferResourceType(text);
			int index = i + 1;

			ChartEmbedding ce = new ChartEmbedding();
			ce.setEmbeddingId(index);
			ce.setResourceType(resourceType);
			ce.setResourceId(index * 100);
			ce.setTextContent(text);
			String prefixed = ChartSearchAiConstants.getEmbeddingPrefix(resourceType, text) + text;
			ce.setEmbeddingVector(embeddingProvider.embed(prefixed));
			ce.setDateCreated(now);
			embeddings.add(ce);
		}
		return embeddings;
	}

	private static String inferResourceType(String text) {
		if (text.startsWith("Medication prescription:") || text.startsWith("Lab test order:")) {
			return "order";
		}
		if (text.startsWith("Medical condition:")) {
			return "condition";
		}
		if (text.startsWith("Clinical diagnosis:")) {
			return "diagnosis";
		}
		if (text.startsWith("Patient allergy:")) {
			return "allergy";
		}
		if (text.startsWith("Program enrollment:")) {
			return "program";
		}
		return "obs";
	}

	@Test
	public void retrievalRecall_shouldMeetMinimumThreshold() {
		int totalCases = 0;
		double totalRecall = 0;

		for (EvalCase evalCase : evalDataset.getCases()) {
			totalCases++;

			String normalizedQuery = LlmInferenceService.stripQueryStopwords(evalCase.getQuestion());
			String[] queryTerms = LlmInferenceService.extractQueryTerms(normalizedQuery);
			float[] queryVector = embeddingProvider.embed(normalizedQuery);

			double keywordWeight = 0.3;
			double bonusThreshold = queryTerms.length >= 4
					? 1.0 / queryTerms.length
					: queryTerms.length == 0 ? 1.0
					: (double) Math.min(2, queryTerms.length) / queryTerms.length;

			List<LlmInferenceService.ScoredEmbedding> scored = new ArrayList<>();
			for (ChartEmbedding ce : allEmbeddings) {
				float[] vector = ce.getEmbeddingVector();
				double semanticScore = LlmInferenceService.cosineSimilarity(queryVector, vector);
				String keywordText = ChartSearchAiConstants.getEmbeddingPrefix(
						ce.getResourceType(), ce.getTextContent()) + ce.getTextContent();
				double keywordScore = LlmInferenceService.computeKeywordScore(queryTerms, keywordText);
				double keywordBonus = keywordScore >= bonusThreshold ? keywordScore : 0.0;
				double keywordPenalty = 0.0;
				if (queryTerms.length <= 2 && keywordScore > 0 && keywordScore < bonusThreshold) {
					keywordPenalty = keywordScore;
				}
				double baseScore = semanticScore + keywordWeight * keywordBonus
						- keywordWeight * keywordPenalty;
				scored.add(new LlmInferenceService.ScoredEmbedding(ce, baseScore, keywordScore, semanticScore));
			}

			Collections.sort(scored, new Comparator<LlmInferenceService.ScoredEmbedding>() {
				@Override
				public int compare(LlmInferenceService.ScoredEmbedding a,
						LlmInferenceService.ScoredEmbedding b) {
					return Double.compare(b.score, a.score);
				}
			});

			int topK = Math.min(30, scored.size());
			List<Integer> retrievedIndices = new ArrayList<>();
			for (int i = 0; i < topK; i++) {
				retrievedIndices.add(scored.get(i).embedding.getEmbeddingId());
			}

			double recall = EvalMetrics.recall(retrievedIndices, evalCase.getExpectedRecordIndices());
			totalRecall += recall;

			log.info("[{}] recall@{}={} expected={} top5={}",
					evalCase.getId(), topK, String.format("%.3f", recall),
					evalCase.getExpectedRecordIndices(),
					retrievedIndices.subList(0, Math.min(5, retrievedIndices.size())));
		}

		double avgRecall = totalCases > 0 ? totalRecall / totalCases : 0;
		log.info("Retrieval eval: avgRecall@30={} across {} cases",
				String.format("%.3f", avgRecall), totalCases);

		assertTrue(avgRecall >= 0.3,
				"Average retrieval recall@30 should be >= 0.3 but was "
				+ String.format("%.3f", avgRecall));
	}

	@Test
	public void keywordScoring_shouldMatchExpectedTerms() {
		String[] terms = {"cd4", "count"};
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Clinical observation: Test — CD4 Count: 988.0");
		assertTrue(score > 0, "Keyword score should be > 0 for matching terms");

		double noMatchScore = LlmInferenceService.computeKeywordScore(terms,
				"Clinical observation: Test — Weight (kg): 94.0");
		assertTrue(noMatchScore < score,
				"Non-matching record should score lower");
	}
}
