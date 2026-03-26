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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;

public class LlmInferenceServiceTest {

	@Test
	public void extractCitedReferences_shouldExtractReferencesFromCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456, null),
				new RecordMapping(2, "order", 201, null));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 2), mappings);

		assertEquals(2, result.size());
		assertEquals("obs", result.get(0).getResourceType());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
		assertEquals("order", result.get(1).getResourceType());
		assertEquals(Integer.valueOf(201), result.get(1).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldReturnEmptyWhenNoCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456, null));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Collections.emptyList(), mappings);

		assertTrue(result.isEmpty());
	}

	@Test
	public void extractCitedReferences_shouldDeduplicateRepeatedCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456, null));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 1), mappings);

		assertEquals(1, result.size());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldHandleMultipleCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 101, null),
				new RecordMapping(2, "obs", 102, null),
				new RecordMapping(3, "obs", 103, null));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 2, 3), mappings);

		assertEquals(3, result.size());
		assertEquals(Integer.valueOf(101), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(102), result.get(1).getResourceId());
		assertEquals(Integer.valueOf(103), result.get(2).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldIgnoreNumbersNotInMappings() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 10, null));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 99), mappings);

		assertEquals(1, result.size());
		assertEquals(Integer.valueOf(10), result.get(0).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldSortByDateMostRecentFirst() {
		Date jan = makeDate(2025, 1, 10);
		Date mar = makeDate(2025, 3, 15);
		Date feb = makeDate(2025, 2, 20);

		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "condition", 50, jan),
				new RecordMapping(2, "order", 30, mar),
				new RecordMapping(3, "obs", 999, feb));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 2, 3), mappings);

		assertEquals(3, result.size());
		assertEquals(Integer.valueOf(30), result.get(0).getResourceId());
		assertEquals(mar, result.get(0).getDate());
		assertEquals(Integer.valueOf(999), result.get(1).getResourceId());
		assertEquals(feb, result.get(1).getDate());
		assertEquals(Integer.valueOf(50), result.get(2).getResourceId());
		assertEquals(jan, result.get(2).getDate());
	}

	@Test
	public void extractCitedReferences_shouldPutNullDatesLast() {
		Date recent = makeDate(2025, 3, 1);

		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 100, null),
				new RecordMapping(2, "obs", 200, recent));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				Arrays.asList(1, 2), mappings);

		assertEquals(2, result.size());
		assertEquals(Integer.valueOf(200), result.get(0).getResourceId());
		assertEquals(recent, result.get(0).getDate());
		assertEquals(Integer.valueOf(100), result.get(1).getResourceId());
		assertNull(result.get(1).getDate());
	}

	@Test
	public void stripQueryStopwords_shouldNormalizeDifferentPhrasingsToSameResult() {
		assertEquals(
				LlmInferenceService.stripQueryStopwords("any medications?"),
				LlmInferenceService.stripQueryStopwords("does the patient have any medications?"));
	}

	@Test
	public void stripQueryStopwords_shouldPreserveContentWords() {
		assertEquals("medications hypertension",
				LlmInferenceService.stripQueryStopwords("any medications for hypertension?"));
	}

	@Test
	public void stripQueryStopwords_shouldReturnOriginalWhenAllStopwords() {
		String result = LlmInferenceService.stripQueryStopwords("does the patient have any?");
		assertTrue(!result.isEmpty());
	}

	@Test
	public void stripQueryStopwords_shouldHandleMixedCase() {
		assertEquals("medications",
				LlmInferenceService.stripQueryStopwords("Does The Patient Have Any Medications?"));
	}

	@Test
	public void stripQueryStopwords_shouldNormalizeCurrentAndLatestToSameResult() {
		assertEquals(
				LlmInferenceService.stripQueryStopwords("What is the current CD4 Count?"),
				LlmInferenceService.stripQueryStopwords("What is the latest CD4 Count?"));
	}

	@Test
	public void defaultSimilarityRatio_shouldBeBetweenZeroAndOne() {
		assertTrue(ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO > 0);
		assertTrue(ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO < 1);
	}

	@Test
	public void findAdaptiveCutoff_shouldDetectGapInScores() {
		// Scores: 0.95, 0.93, 0.91, [gap], 0.75, 0.73
		// The gap from 0.91 to 0.75 (0.16) is much larger than the avg gap so far (0.02)
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.95),
				makeScoredEmbedding(0.93),
				makeScoredEmbedding(0.91),
				makeScoredEmbedding(0.75),
				makeScoredEmbedding(0.73));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.70, 2.5, 0.0);

		assertEquals(3, cutoff, "Should cut at position 3 where the large gap occurs");
	}

	@Test
	public void findAdaptiveCutoff_shouldReturnAllWhenScoresAreUniform() {
		// Scores evenly spaced — no outlier gap
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.88),
				makeScoredEmbedding(0.86),
				makeScoredEmbedding(0.84),
				makeScoredEmbedding(0.82));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.70, 2.5, 0.0);

		assertEquals(5, cutoff, "Should include all records when scores are evenly spaced");
	}

	@Test
	public void findAdaptiveCutoff_shouldRespectSimilarityFloor() {
		// 3 records above floor, 2 below
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.88),
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.60),
				makeScoredEmbedding(0.55));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.80, 2.5, 0.0);

		assertEquals(3, cutoff, "Should not include records below the similarity floor");
	}

	@Test
	public void findAdaptiveCutoff_shouldReturnAllWhenBothAboveFloor() {
		// Both records above the floor — no gap detection needed with only 2
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.50));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 2, 0.40, 2.5, 0.0);

		assertEquals(2, cutoff, "Should include all records when both are above the floor");
	}

	@Test
	public void findAdaptiveCutoff_shouldHandleSingleRecord() {
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.90));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 1, 0.70, 2.5, 0.0);

		assertEquals(1, cutoff);
	}

	@Test
	public void findAdaptiveCutoff_shouldNotCutBeforeMinimumRecords() {
		// Huge gap at i=2 (0.94 -> 0.50 = 0.44). Without the i >= minRecords
		// guard, this would trigger at i=2 (cutoff=2, only 2 records). With
		// minRecords=2, the check is deferred past i=1. The large gap at i=2
		// triggers the cut, yielding 2 records.
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.95),
				makeScoredEmbedding(0.94),
				makeScoredEmbedding(0.50),
				makeScoredEmbedding(0.48),
				makeScoredEmbedding(0.46));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.40, 2.5, 0.0);

		assertTrue(cutoff >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS,
				"Should not cut below the minimum record count");
	}

	@Test
	public void findAdaptiveCutoff_shouldHandleIdenticalScores() {
		// All scores identical — every gap is 0, no cutoff detected
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.85));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 4, 0.70, 2.5, 0.0);

		assertEquals(4, cutoff, "Should include all records when scores are identical");
	}

	@Test
	public void findAdaptiveCutoff_shouldDetectGapAfterIdenticalScores() {
		// 4 identical scores then a drop — gap is infinite relative to avg of 0
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.90),
				makeScoredEmbedding(0.60));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.50, 2.5, 0.0);

		assertEquals(4, cutoff, "Should cut where scores drop after a plateau");
	}

	@Test
	public void findAdaptiveCutoff_shouldNotFalselyCutOnTiedScores() {
		// Positions 1 and 2 are tied (gap=0). A narrow baseline seeded only
		// from this zero gap would make ANY subsequent non-zero gap trigger.
		// The running average includes the 0.10 gap at i=1, giving a realistic
		// baseline: avgGap = (0.10 + 0.00) / 2 = 0.05, threshold = 0.125.
		// The 0.01 gap at i=3 is well below this, so no cut.
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.95),
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.85),
				makeScoredEmbedding(0.84),
				makeScoredEmbedding(0.83));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.70, 2.5, 0.0);

		assertEquals(5, cutoff, "Tied scores should not cause false gap detection");
	}

	@Test
	public void findAdaptiveCutoff_shouldRespectHigherMultiplier() {
		// Same scores as the basic gap test, but with a very high multiplier
		// that prevents the gap from triggering
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.95),
				makeScoredEmbedding(0.93),
				makeScoredEmbedding(0.91),
				makeScoredEmbedding(0.75),
				makeScoredEmbedding(0.73));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.70, 999.0, 0.0);

		assertEquals(5, cutoff, "Very high multiplier should effectively disable gap detection");
	}

	@Test
	public void computeKeywordScore_shouldReturnOneWhenAllTermsMatch() {
		String[] terms = { "metformin", "500mg" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Drug order: Metformin 500mg. Dose: 1.0 Tablet(s) Oral twice daily");
		assertEquals(1.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldReturnZeroWhenNoTermsMatch() {
		String[] terms = { "penicillin", "allergy" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Drug order: Metformin 500mg. Dose: 1.0 Tablet(s)");
		assertEquals(0.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldReturnPartialScore() {
		String[] terms = { "metformin", "allergy" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Drug order: Metformin 500mg");
		assertEquals(0.5, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldBeCaseInsensitive() {
		String[] terms = { "metformin" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Drug order: METFORMIN 500mg");
		assertEquals(1.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldHandleEmptyTerms() {
		assertEquals(0.0, LlmInferenceService.computeKeywordScore(
				new String[0], "Some text"), 0.001);
	}

	@Test
	public void computeKeywordScore_shouldHandleNullText() {
		String[] terms = { "test" };
		assertEquals(0.0, LlmInferenceService.computeKeywordScore(terms, null), 0.001);
	}

	@Test
	public void computeKeywordScore_shouldMatchSubstrings() {
		String[] terms = { "medication" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Dispensed: Metformin 500mg. Status: Completed. medications given");
		assertEquals(1.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldMatchPluralToSingular() {
		// "conditions" (plural) should match "Condition:" (singular) in the text
		String[] terms = { "conditions" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Condition: Nonparalytic stroke. Status: ACTIVE");
		assertEquals(1.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldNotStemShortWords() {
		// "as" (length 2) should not be stemmed to "a"
		String[] terms = { "as" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Condition: Type 2 Diabetes. Status: ACTIVE");
		assertEquals(0.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldNotStemDoubleS() {
		// "pass" should not be stemmed to "pas"
		String[] terms = { "pass" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"some text without the word");
		assertEquals(0.0, score, 0.001);
	}

	@Test
	public void computeKeywordScore_shouldMatchMedicationInPrefixedText() {
		// When findSimilar prepends the embedding prefix, "medication" (stemmed
		// from "medications") should match "Medication prescription: " prefix.
		String[] terms = { "medications", "prescribed", "started" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Medication prescription: Drug order: Metformin 500mg. Dose: 1.0 Tablet(s)");
		assertTrue(score >= 1.0 / 3 - 0.001,
				"'medications' should match 'Medication' in the prefix text via stemming");
	}

	@Test
	public void computeKeywordScore_shouldMatchTestInPrefixedText() {
		// "tests" (stemmed to "test") should match "Lab or diagnostic test: " prefix
		String[] terms = { "tests", "ordered" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Lab or diagnostic test: Test order: CD4 count. Action: NEW");
		assertTrue(score >= 1.0 / 2 - 0.001,
				"'tests' should match 'test' in the prefix text via stemming");
	}

	@Test
	public void findAdaptiveCutoff_shouldNotCutOnSmallAbsoluteGap() {
		// Tight cluster: 0.55, 0.54, 0.53, then a 0.07 gap to 0.46, 0.45
		// Relative to avgGap=0.01, 0.07 is 7x the average (triggers at 2.5x).
		// But 0.07 < minGap of 0.10, so we should NOT cut.
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.55),
				makeScoredEmbedding(0.54),
				makeScoredEmbedding(0.53),
				makeScoredEmbedding(0.46),
				makeScoredEmbedding(0.45));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.40, 2.5, 0.10);

		assertEquals(5, cutoff,
				"Should not cut when gap is below absolute minimum even if above relative threshold");
	}

	@Test
	public void findAdaptiveCutoff_shouldCutOnLargeAbsoluteGap() {
		// Same tight cluster but bigger gap: 0.55, 0.54, 0.53, [0.15 gap], 0.38, 0.37
		// 0.15 > avgGap*2.5 AND 0.15 > 0.10 → should cut.
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbedding(0.55),
				makeScoredEmbedding(0.54),
				makeScoredEmbedding(0.53),
				makeScoredEmbedding(0.38),
				makeScoredEmbedding(0.37));

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, 5, 0.30, 2.5, 0.10);

		assertEquals(3, cutoff,
				"Should cut when gap exceeds both relative and absolute thresholds");
	}

	@Test
	public void extractQueryTerms_shouldFilterOutSingleCharacterTerms() {
		String[] terms = LlmInferenceService.extractQueryTerms("a medications b");
		assertEquals(1, terms.length);
		assertEquals("medications", terms[0]);
	}

	@Test
	public void extractQueryTerms_shouldLowercaseTerms() {
		String[] terms = LlmInferenceService.extractQueryTerms("Metformin Dose");
		assertEquals(2, terms.length);
		assertEquals("metformin", terms[0]);
		assertEquals("dose", terms[1]);
	}

	@Test
	public void extractQueryTerms_shouldHandleEmptyInput() {
		String[] terms = LlmInferenceService.extractQueryTerms("");
		assertEquals(0, terms.length);
	}

	private static LlmInferenceService.ScoredEmbedding makeScoredEmbedding(double score) {
		return makeScoredEmbedding(score, 0.0);
	}

	private static LlmInferenceService.ScoredEmbedding makeScoredEmbedding(double score,
			double keywordScore) {
		ChartEmbedding ce = new ChartEmbedding();
		ce.setResourceType("obs");
		ce.setTextContent("Test — Example: value");
		return new LlmInferenceService.ScoredEmbedding(ce, score, keywordScore);
	}

	@Test
	public void refineByKeywords_shouldFilterToKeywordMatchedSubset() {
		// 3 records have strong keyword matches (2/6 terms), 2 don't.
		// With queryTermCount=6, threshold = min(2, max(1, 2))/6 = 0.333...
		double kwMatch = 2.0 / 6; // exact fraction avoids floating-point mismatch
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, kwMatch),
				makeScoredEmbedding(0.65, kwMatch),
				makeScoredEmbedding(0.60, 0.0),
				makeScoredEmbedding(0.55, kwMatch),
				makeScoredEmbedding(0.50, 0.0));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(3, refined.size(),
				"Should keep only keyword-matched records when they form a proper subset");
	}

	@Test
	public void refineByKeywords_shouldReturnAllWhenAllHaveKeywordMatches() {
		// All records have keyword matches — keywords aren't discriminative
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, 0.50),
				makeScoredEmbedding(0.65, 0.33),
				makeScoredEmbedding(0.60, 0.33));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(3, refined.size(),
				"Should return all records when keywords aren't discriminative");
	}

	@Test
	public void refineByKeywords_shouldReturnAllWhenNoneHaveKeywordMatches() {
		// No keyword matches — no signal to refine on
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, 0.0),
				makeScoredEmbedding(0.65, 0.0),
				makeScoredEmbedding(0.60, 0.0));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(3, refined.size(),
				"Should return all records when no keyword signal exists");
	}

	@Test
	public void refineByKeywords_shouldReturnAllWhenTooFewKeywordMatches() {
		// Only 1 keyword match — below ADAPTIVE_MIN_RECORDS (2)
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, 0.33),
				makeScoredEmbedding(0.65, 0.0),
				makeScoredEmbedding(0.60, 0.0),
				makeScoredEmbedding(0.55, 0.0));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(4, refined.size(),
				"Should return all records when too few have keyword matches");
	}

	@Test
	public void refineByKeywords_shouldIgnoreIncidentalSingleTermMatches() {
		// 2 records match 2/6 terms, 4 records match only 1/6.
		// Threshold = min(2, max(1, 2))/6 = 0.333. Only the 2 strong matches should pass.
		double strongMatch = 2.0 / 6; // 0.333...
		double weakMatch = 1.0 / 6;   // 0.166...
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, strongMatch),
				makeScoredEmbedding(0.65, weakMatch),
				makeScoredEmbedding(0.60, weakMatch),
				makeScoredEmbedding(0.55, strongMatch),
				makeScoredEmbedding(0.50, weakMatch),
				makeScoredEmbedding(0.45, weakMatch));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(2, refined.size(),
				"Should not include records with only incidental single-term matches");
	}

	// --------------------------------------------------------
	// Pipeline regression tests: verify the full scoring →
	// gap detection → keyword refinement chain produces
	// correct results for representative clinical query types.
	// These tests should break if ANY future change causes a
	// regression for a previously working query type.
	// --------------------------------------------------------

	/**
	 * Simulates the full retrieval pipeline: combines semantic and keyword
	 * scores, sorts, applies gap detection, then keyword refinement.
	 * Returns the number of records in the final result set.
	 */
	private static int simulatePipeline(double[] semanticScores, double[] keywordScores,
			double keywordWeight, int queryTermCount) {
		double minScore = ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR / 2;
		double gapMultiplier = ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER;
		double minGap = ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP;

		List<LlmInferenceService.ScoredEmbedding> scored =
				new ArrayList<LlmInferenceService.ScoredEmbedding>();
		for (int i = 0; i < semanticScores.length; i++) {
			double baseScore = semanticScores[i] + keywordWeight * keywordScores[i];
			scored.add(makeScoredEmbedding(baseScore, keywordScores[i]));
		}

		Collections.sort(scored, new Comparator<LlmInferenceService.ScoredEmbedding>() {
			@Override
			public int compare(LlmInferenceService.ScoredEmbedding a,
					LlmInferenceService.ScoredEmbedding b) {
				return Double.compare(b.score, a.score);
			}
		});

		int cutoff = LlmInferenceService.findAdaptiveCutoff(scored, scored.size(),
				minScore, gapMultiplier, minGap);

		List<LlmInferenceService.ScoredEmbedding> candidates =
				new ArrayList<LlmInferenceService.ScoredEmbedding>();
		for (int i = 0; i < cutoff; i++) {
			candidates.add(scored.get(i));
		}

		if (keywordWeight > 0) {
			candidates = LlmInferenceService.refineByKeywords(candidates, queryTermCount);
		}

		int maxResults = ChartSearchAiConstants.DEFAULT_MAX_RESULTS;
		if (candidates.size() > maxResults) {
			candidates = candidates.subList(0, maxResults);
		}

		return candidates.size();
	}

	@Test
	public void pipeline_specificLabQuery_shouldReturnOnlyMatchingRecords() {
		// Models: "latest CD4 count" — 3 terms, CD4 records match 2/3.
		// Gap detection should separate the 3 high-scoring CD4 records from
		// the lower-scoring non-CD4 records. Keyword refinement should not
		// interfere because all gap-detected records have keyword matches.
		double[] semantic = { 0.72, 0.70, 0.68, 0.35, 0.33, 0.31, 0.29, 0.27 };
		double[] keyword  = { 0.67, 0.67, 0.67, 0.00, 0.00, 0.00, 0.00, 0.00 };

		int result = simulatePipeline(semantic, keyword, 0.3, 3);

		assertEquals(3, result,
				"Specific lab query should return only matching records via gap detection");
	}

	@Test
	public void pipeline_broadCategoryQuery_shouldReturnKeywordMatchedRecords() {
		// Models: "active conditions first recorded resolved escalated" — 6 terms,
		// conditions match 2/6 ("condition" + "active"). Scores overlap with
		// non-conditions so gap detection can't separate them. Keyword refinement
		// should filter to the 10 condition records.
		double[] semantic = new double[25];
		double[] keyword = new double[25];
		// 10 conditions: semantic 0.55..0.37, keyword 2/6 = 0.333...
		for (int i = 0; i < 10; i++) {
			semantic[i] = 0.55 - i * 0.02;
			keyword[i] = 2.0 / 6;
		}
		// 15 non-conditions: semantic 0.50..0.22, keyword 0
		for (int i = 0; i < 15; i++) {
			semantic[10 + i] = 0.50 - i * 0.02;
			keyword[10 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 6);

		assertEquals(10, result,
				"Broad category query should return keyword-matched records when gap detection fails");
	}

	@Test
	public void pipeline_genericQuery_shouldReturnBroadSet() {
		// Models: "tell me about this patient" — 2 terms after stopwords,
		// no keyword matches. Gap detection should return all records with
		// no gap in the smooth distribution.
		double[] semantic = new double[12];
		double[] keyword = new double[12];
		for (int i = 0; i < 12; i++) {
			semantic[i] = 0.40 - i * 0.01;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 2);

		assertEquals(12, result,
				"Generic query with no keyword matches should return full gap-detected set");
	}

	@Test
	public void pipeline_incidentalKeywordMatches_shouldNotOverFilter() {
		// Models: "HB results values moving normal range" — 6 terms.
		// HB records match 2 terms ("hb" + one other). Some vital signs match
		// only "normal" (1 term = 0.17). Keyword refinement threshold = 0.33
		// should exclude the single-term "normal" matches. Gap detection should
		// separate HB from everything else via the keyword boost.
		double[] semantic = new double[15];
		double[] keyword = new double[15];
		// 4 HB records: high semantic, keyword 2/6 = 0.33
		for (int i = 0; i < 4; i++) {
			semantic[i] = 0.65 - i * 0.02;
			keyword[i] = 2.0 / 6;
		}
		// 6 records matching just "normal": medium semantic, keyword 1/6 = 0.17
		for (int i = 0; i < 6; i++) {
			semantic[4 + i] = 0.48 - i * 0.02;
			keyword[4 + i] = 1.0 / 6;
		}
		// 5 records with no keyword match: low semantic
		for (int i = 0; i < 5; i++) {
			semantic[10 + i] = 0.35 - i * 0.02;
			keyword[10 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 6);

		assertTrue(result <= 4,
				"Should not include records with only incidental 'normal' keyword matches; got " + result);
	}

	@Test
	public void pipeline_gapDetectionWorks_keywordRefinementShouldNotInterfere() {
		// Models: a query where gap detection correctly separates relevant
		// records (with keyword matches) from irrelevant (without). The
		// keyword refinement should NOT further reduce the set because all
		// gap-detected records already have keyword matches.
		double[] semantic = new double[10];
		double[] keyword = new double[10];
		// 5 relevant: high semantic, keyword 2/3 = 0.67
		for (int i = 0; i < 5; i++) {
			semantic[i] = 0.70 - i * 0.02;
			keyword[i] = 2.0 / 3;
		}
		// 5 irrelevant: low semantic, no keyword match
		for (int i = 0; i < 5; i++) {
			semantic[5 + i] = 0.35 - i * 0.02;
			keyword[5 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 3);

		assertEquals(5, result,
				"When gap detection works, keyword refinement should not reduce the result set");
	}

	@Test
	public void pipeline_smoothDistributionNoKeywords_shouldReturnAllAboveFloor() {
		// Models: "vital signs" — semantic similarity separates relevant
		// records but no keyword matches (record text doesn't contain
		// "vital" or "signs"). With smooth scores and no keyword signal,
		// the pipeline should return everything above the floor.
		double[] semantic = new double[8];
		double[] keyword = new double[8];
		for (int i = 0; i < 8; i++) {
			semantic[i] = 0.50 - i * 0.02;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 2);

		assertEquals(8, result,
				"Smooth distribution with no keywords should return all records above floor");
	}

	@Test
	public void pipeline_keywordWeightZero_shouldDisableRefinement() {
		// When keywordWeight is 0, keyword refinement should be completely
		// disabled. Even if some records would have keyword matches, the
		// pipeline should return the full gap-detected set.
		double[] semantic = new double[8];
		double[] keyword = new double[8];
		for (int i = 0; i < 3; i++) {
			semantic[i] = 0.55 - i * 0.02;
			keyword[i] = 0.50;
		}
		for (int i = 0; i < 5; i++) {
			semantic[3 + i] = 0.48 - i * 0.02;
			keyword[3 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.0, 4);

		assertEquals(8, result,
				"With keywordWeight=0, keyword refinement should be disabled");
	}

	@Test
	public void pipeline_medicationQueryWithDrugOrders_shouldReturnOnlyMedications() {
		// Models: "medications prescribed started" — 3 terms. Drug orders
		// match "medication" via the embedding prefix (1/3 = 0.33). With
		// the revised threshold for 3-term queries (min(2, max(1, 1))/3 =
		// 0.33), a single strong match suffices. Conditions and findings
		// score moderately on semantic similarity but have no keyword match.
		double[] semantic = new double[15];
		double[] keyword = new double[15];
		// 3 drug orders: match "medication" via prefix → 1/3 = 0.33
		for (int i = 0; i < 3; i++) {
			semantic[i] = 0.50 - i * 0.02;
			keyword[i] = 1.0 / 3;
		}
		// 12 conditions/findings: no keyword match
		for (int i = 0; i < 12; i++) {
			semantic[3 + i] = 0.45 - i * 0.02;
			keyword[3 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 3);

		assertEquals(3, result,
				"Medication query should return only drug orders when they match via prefix");
	}

	@Test
	public void pipeline_medicationQueryNoDrugOrders_shouldReturnContext() {
		// Models: "medications prescribed started" — 3 terms. Patient has NO
		// drug orders, so no records match keyword "medication". The pipeline
		// should return the gap-detected set as context so the LLM can say
		// "no medications found."
		double[] semantic = new double[10];
		double[] keyword = new double[10];
		for (int i = 0; i < 10; i++) {
			semantic[i] = 0.40 - i * 0.01;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 3);

		assertEquals(10, result,
				"When no records match keywords, should return full gap-detected set as context");
	}

	@Test
	public void refineByKeywords_shortQuerySingleMatchShouldPass() {
		// For 3-term queries, the threshold = min(2, max(1, 1))/3 = 0.33.
		// A record matching 1/3 terms should pass (single discriminative match).
		double kwMatch = 1.0 / 3;
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, kwMatch),
				makeScoredEmbedding(0.65, kwMatch),
				makeScoredEmbedding(0.60, 0.0),
				makeScoredEmbedding(0.55, 0.0),
				makeScoredEmbedding(0.50, 0.0));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 3);

		assertEquals(2, refined.size(),
				"3-term query: single-match records should pass the threshold");
	}

	@Test
	public void refineByKeywords_longQuerySingleMatchShouldFail() {
		// For 6-term queries, threshold = min(2, max(1, 2))/6 = 0.33.
		// A record matching only 1/6 terms (0.17) should NOT pass.
		double weakMatch = 1.0 / 6;
		double strongMatch = 2.0 / 6;
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbedding(0.70, strongMatch),
				makeScoredEmbedding(0.65, weakMatch),
				makeScoredEmbedding(0.60, strongMatch),
				makeScoredEmbedding(0.55, weakMatch),
				makeScoredEmbedding(0.50, 0.0));

		List<LlmInferenceService.ScoredEmbedding> refined =
				LlmInferenceService.refineByKeywords(candidates, 6);

		assertEquals(2, refined.size(),
				"6-term query: single-match records (0.17) should NOT pass the threshold (0.33)");
	}

	@Test
	public void pipeline_noKeywordMatchesSmoothDistribution_shouldCapToMaxResults() {
		// Models: "HB results over time" — the patient has NO HB results.
		// All 93 records are vital signs with smooth, similar semantic scores
		// and ZERO keyword matches. Gap detection won't trigger (smooth
		// distribution), keyword refinement won't activate (0 matches < 2).
		// The max results cap should prevent dumping the entire chart.
		int recordCount = 93;
		double[] semantic = new double[recordCount];
		double[] keyword = new double[recordCount];
		for (int i = 0; i < recordCount; i++) {
			semantic[i] = 0.42 - i * 0.002;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 7);

		assertEquals(ChartSearchAiConstants.DEFAULT_MAX_RESULTS, result,
				"When no keywords match and gap detection can't discriminate, "
						+ "the max results cap should limit output");
	}

	@Test
	public void pipeline_keywordRefinementShouldNotBeClippedByMaxResults() {
		// When keyword refinement successfully narrows to a subset that is
		// smaller than DEFAULT_MAX_RESULTS, the cap should not interfere.
		// Models: 10 conditions out of 25 total records.
		double[] semantic = new double[25];
		double[] keyword = new double[25];
		for (int i = 0; i < 10; i++) {
			semantic[i] = 0.55 - i * 0.02;
			keyword[i] = 2.0 / 6;
		}
		for (int i = 0; i < 15; i++) {
			semantic[10 + i] = 0.45 - i * 0.02;
			keyword[10 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 6);

		assertEquals(10, result,
				"Keyword refinement result should not be clipped when it's within the cap");
	}

	private static Date makeDate(int year, int month, int day) {
		Calendar cal = Calendar.getInstance();
		cal.set(year, month - 1, day, 0, 0, 0);
		cal.set(Calendar.MILLISECOND, 0);
		return cal.getTime();
	}
}
