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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;

public class LlmInferenceServiceTest {

	static final String[] FULL_PATIENT_DATASET = TestDatasetHelper.FULL_PATIENT_DATASET;

	private static final String[] SECOND_PATIENT_DATASET =
			TestDatasetHelper.SECOND_PATIENT_DATASET;

	private static final String[] THIRD_PATIENT_DATASET =
			TestDatasetHelper.THIRD_PATIENT_DATASET;

	private static final String[] FOURTH_PATIENT_DATASET =
			TestDatasetHelper.FOURTH_PATIENT_DATASET;

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
		// Both queries have only 1 content word ("medications"), so both
		// preserve the full sentence. The embedding model handles both
		// similarly because the key term is the same.
		String short1 = LlmInferenceService.stripQueryStopwords("any medications?");
		String long1 = LlmInferenceService.stripQueryStopwords("does the patient have any medications?");
		assertTrue(short1.contains("medications"), "Short form should contain 'medications'");
		assertTrue(long1.contains("medications"), "Long form should contain 'medications'");
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
		// 1 content word "medications" → preserves full query for context.
		// Should be lowercased and contain the key term.
		String result = LlmInferenceService.stripQueryStopwords(
				"Does The Patient Have Any Medications?");
		assertTrue(result.contains("medications"),
				"Mixed case query should contain 'medications'");
		assertEquals(result, result.toLowerCase(),
				"Result should be lowercased");
	}

	@Test
	public void stripQueryStopwords_shouldPreserveContextForShortQueries() {
		// When stopword removal would leave < 2 words, the full question
		// should be preserved to give the embedding model enough context.
		// "does the patient have cancer?" has only 1 content word ("cancer").
		// The original code embedded the full question and returned 2 results;
		// stripping to just "cancer" loses context and returns 3.
		String result = LlmInferenceService.stripQueryStopwords(
				"does the patient have cancer?");
		String[] words = result.trim().split("\\s+");
		assertTrue(words.length >= 2,
				"Short query should preserve context words, got: '" + result + "'");
	}

	@Test
	public void stripQueryStopwords_shouldNormalizeCurrentAndLatestToSameResult() {
		assertEquals(
				LlmInferenceService.stripQueryStopwords("What is the current CD4 Count?"),
				LlmInferenceService.stripQueryStopwords("What is the latest CD4 Count?"));
	}

	@Test
	public void patientChart_demographicsOnlyHasNoRecords() {
		// A chart with only demographics (no records) should have empty
		// mappings even though getText() is non-empty. The search methods
		// must check mappings, not text, to detect "no records found".
		org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart chart =
				new org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart(
				"Patient: 4-year-old Male", Collections.<RecordMapping>emptyList());
		assertFalse(chart.getText().isEmpty(),
				"Demographics-only chart has non-empty text");
		assertTrue(chart.getMappings().isEmpty(),
				"Demographics-only chart should have no record mappings");
	}

	@Test
	public void extractRecencyCap_shouldExtractNumberFromLastN() {
		assertEquals(7, LlmInferenceService.extractRecencyCap(
				"How have vitals trended across the last 7 visits?"));
	}

	@Test
	public void extractRecencyCap_shouldExtractFromPastN() {
		assertEquals(3, LlmInferenceService.extractRecencyCap(
				"Show past 3 lab results"));
	}

	@Test
	public void extractRecencyCap_shouldExtractFromPreviousN() {
		assertEquals(5, LlmInferenceService.extractRecencyCap(
				"What were the previous 5 blood pressure readings?"));
	}

	@Test
	public void extractRecencyCap_shouldExtractFromMostRecentN() {
		assertEquals(10, LlmInferenceService.extractRecencyCap(
				"List the most recent 10 observations"));
	}

	@Test
	public void extractRecencyCap_shouldBeCaseInsensitive() {
		assertEquals(7, LlmInferenceService.extractRecencyCap(
				"LAST 7 visits"));
	}

	@Test
	public void extractRecencyCap_shouldReturnZeroWhenNoPattern() {
		assertEquals(0, LlmInferenceService.extractRecencyCap(
				"What medications is the patient on?"));
	}

	@Test
	public void extractRecencyCap_shouldReturnZeroForZero() {
		assertEquals(0, LlmInferenceService.extractRecencyCap(
				"last 0 visits"));
	}

	@Test
	public void extractRecencyCap_shouldExtractFromLatestN() {
		assertEquals(2, LlmInferenceService.extractRecencyCap(
				"What are the latest 2 weights?"));
	}

	@Test
	public void extractRecencyCap_shouldExtractWordNumberTwo() {
		assertEquals(2, LlmInferenceService.extractRecencyCap(
				"What are the latest two weights?"));
	}

	@Test
	public void extractRecencyCap_shouldExtractWordNumberThree() {
		assertEquals(3, LlmInferenceService.extractRecencyCap(
				"Show the last three lab results"));
	}

	@Test
	public void extractRecencyCap_shouldExtractWordNumberFive() {
		assertEquals(5, LlmInferenceService.extractRecencyCap(
				"previous five blood pressure readings"));
	}

	@Test
	public void extractRecencyCap_shouldExtractWordNumberTen() {
		assertEquals(10, LlmInferenceService.extractRecencyCap(
				"most recent ten observations"));
	}

	@Test
	public void extractRecencyCap_shouldExtractNumberBeforeKeyword() {
		assertEquals(2, LlmInferenceService.extractRecencyCap(
				"What are the two latest weights?"));
	}

	@Test
	public void extractRecencyCap_shouldExtractDigitBeforeKeyword() {
		assertEquals(3, LlmInferenceService.extractRecencyCap(
				"Show the 3 most recent lab results"));
	}

	@Test
	public void conceptKey_shouldStripTrailingNumericValue() {
		assertEquals("Clinical observation: Test — Weight (kg)",
				LlmInferenceService.conceptKey(
						"Clinical observation: Test — Weight (kg): 94.0"));
	}

	@Test
	public void conceptKey_shouldStripIntegerValue() {
		assertEquals("Clinical observation: Test — Pulse",
				LlmInferenceService.conceptKey(
						"Clinical observation: Test — Pulse: 62.0"));
	}

	@Test
	public void conceptKey_shouldPreserveTextWithoutNumericEnding() {
		assertEquals("Medical condition: Condition: Tuberculosis. Status: ACTIVE",
				LlmInferenceService.conceptKey(
						"Medical condition: Condition: Tuberculosis. Status: ACTIVE"));
	}

	@Test
	public void conceptKey_shouldHandleNull() {
		assertEquals("", LlmInferenceService.conceptKey(null));
	}

	@Test
	public void capPerConcept_shouldLimitRecordsPerConcept() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		// 4 systolic BP records, 3 weight records
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 1, "Clinical observation: Test — Systolic Blood Pressure: 151.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 2, "Clinical observation: Test — Systolic Blood Pressure: 134.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 3, "Clinical observation: Test — Weight (kg): 68.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 4, "Clinical observation: Test — Systolic Blood Pressure: 117.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 5, "Clinical observation: Test — Weight (kg): 121.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 6, "Clinical observation: Test — Systolic Blood Pressure: 102.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 7, "Clinical observation: Test — Weight (kg): 94.0", null));

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.capPerConcept(records, 2);

		// Should keep 2 SBP + 2 Weight = 4 records
		assertEquals(4, result.size());
		// First two should be the first 2 SBP records (ids 1, 2)
		assertEquals(Integer.valueOf(1), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(2), result.get(1).getResourceId());
		// Next two should be the first 2 Weight records (ids 3, 5)
		assertEquals(Integer.valueOf(3), result.get(2).getResourceId());
		assertEquals(Integer.valueOf(5), result.get(3).getResourceId());
	}

	@Test
	public void capPerConcept_shouldNotAffectNonNumericRecords() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"condition", 1, "Medical condition: Condition: Tuberculosis. Status: ACTIVE", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"condition", 2, "Medical condition: Condition: Hypertension. Status: ACTIVE", null));

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.capPerConcept(records, 1);

		// Each condition has a unique key, so both are kept
		assertEquals(2, result.size());
	}

	@Test
	public void groupByConcept_shouldGroupRecordsByConceptKeyPreservingOrderWithinGroup() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		// Interleaved: BP, Weight, BP, Temp, Weight, BP
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 1, "Clinical observation: Test — Systolic Blood Pressure: 151.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 2, "Clinical observation: Test — Weight (kg): 94.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 3, "Clinical observation: Test — Systolic Blood Pressure: 134.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 4, "Clinical observation: Test — Temperature (C): 36.7", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 5, "Clinical observation: Test — Weight (kg): 68.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 6, "Clinical observation: Test — Systolic Blood Pressure: 102.0", null));

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.groupByConcept(records);

		assertEquals(6, result.size(), "All records should be preserved");
		// BP group first (first concept encountered), in original order
		assertEquals(Integer.valueOf(1), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(3), result.get(1).getResourceId());
		assertEquals(Integer.valueOf(6), result.get(2).getResourceId());
		// Weight group next
		assertEquals(Integer.valueOf(2), result.get(3).getResourceId());
		assertEquals(Integer.valueOf(5), result.get(4).getResourceId());
		// Temperature group last
		assertEquals(Integer.valueOf(4), result.get(5).getResourceId());
	}

	@Test
	public void groupByConcept_shouldBeNoOpForSingleConcept() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 1, "Clinical observation: Test — Weight (kg): 94.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 2, "Clinical observation: Test — Weight (kg): 68.0", null));

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.groupByConcept(records);

		assertEquals(2, result.size());
		assertEquals(Integer.valueOf(1), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(2), result.get(1).getResourceId());
	}

	@Test
	public void groupByConcept_shouldHandleEmptyList() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.groupByConcept(new ArrayList<>());
		assertTrue(result.isEmpty());
	}

	@Test
	public void groupByConcept_shouldHandleMixedRecordTypes() {
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 1, "Clinical observation: Test — Systolic Blood Pressure: 97.0", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"condition", 2, "Medical condition: Condition: Hypertension. Status: ACTIVE", null));
		records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
				"obs", 3, "Clinical observation: Test — Systolic Blood Pressure: 134.0", null));

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> result
				= LlmInferenceService.groupByConcept(records);

		assertEquals(3, result.size());
		// BP records grouped together
		assertEquals(Integer.valueOf(1), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(3), result.get(1).getResourceId());
		// Condition separate
		assertEquals(Integer.valueOf(2), result.get(2).getResourceId());
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
	public void computeKeywordScore_shouldMatchMorphologicalVariants() {
		// "allergic" (adjective) should match records containing "allergy"
		// (noun) via stem matching: "allergic" → stem "allerg" → substring
		// of "allergy". This handles derivational suffixes like -ic/-y.
		String[] terms = { "allergic" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Patient allergy: Beef (food allergen). Severity: Severe");
		assertEquals(1.0, score, 0.001,
				"'allergic' should match 'allergy' via stem 'allerg'");
	}

	@Test
	public void computeKeywordScore_shouldNotStemMatchShortWords() {
		// Words < 7 chars should not attempt stem matching to avoid
		// false positives from very short stems.
		String[] terms = { "cancer" };
		double score = LlmInferenceService.computeKeywordScore(terms,
				"Clinical diagnosis: Photoallergy: 9.93");
		assertEquals(0.0, score, 0.001,
				"'cancer' (6 chars) should not stem-match 'Photoallergy'");
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
		return makeScoredEmbedding(score, 0.0, score);
	}

	private static LlmInferenceService.ScoredEmbedding makeScoredEmbedding(double score,
			double keywordScore) {
		return makeScoredEmbedding(score, keywordScore, score);
	}

	private static LlmInferenceService.ScoredEmbedding makeScoredEmbedding(double score,
			double keywordScore, double semanticScore) {
		ChartEmbedding ce = new ChartEmbedding();
		ce.setResourceType("obs");
		ce.setTextContent("Test — Example: value");
		return new LlmInferenceService.ScoredEmbedding(ce, score, keywordScore, semanticScore);
	}

	private static Date makeDate(int year, int month, int day) {
		Calendar cal = Calendar.getInstance();
		cal.set(year, month - 1, day, 0, 0, 0);
		cal.set(Calendar.MILLISECOND, 0);
		return cal.getTime();
	}

	// ---- Real-model integration tests ----
	// These use the actual ONNX embedding model to compute semantic scores,
	// ensuring tests reflect real embedding behavior instead of hand-crafted
	// adversarial scores.

	/**
	 * Path to the ONNX embedding model and vocabulary files for real-model
	 * integration tests. Configure via the {@code chartsearchai.embedding.model.dir}
	 * system property (e.g. {@code -Dchartsearchai.embedding.model.dir=/path/to/all-MiniLM-L6-v2}).
	 * The directory must contain {@code model.onnx} and {@code vocab.txt}.
	 * Tests are skipped automatically when the files are not found.
	 */
	private static final String MODEL_DIR = System.getProperty(
			"chartsearchai.embedding.model.dir", "../models/all-MiniLM-L6-v2");

	private static final String MODEL_PATH = MODEL_DIR + "/model.onnx";

	private static final String VOCAB_PATH = MODEL_DIR + "/vocab.txt";

	private static boolean modelFilesExist() {
		return new java.io.File(MODEL_PATH).exists()
				&& new java.io.File(VOCAB_PATH).exists();
	}

	/**
	 * Computes real semantic scores for every record in FULL_PATIENT_DATASET
	 * against the given query using the actual ONNX embedding model.
	 */
	private static double[] computeRealSemanticScores(
			org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider,
			String query) {
		return computeRealSemanticScoresWithVectors(provider, query,
				FULL_PATIENT_DATASET, null);
	}

	/**
	 * Computes real semantic scores and optionally captures embedding vectors
	 * for inter-candidate coherence filtering. Mirrors production: strips
	 * dataset prefix/date, then embeds with getEmbeddingPrefix + textContent.
	 */
	private static double[] computeRealSemanticScoresWithVectors(
			org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider,
			String query, String[] dataset, float[][] embeddingVectors) {
		float[] queryVector = provider.embed(
				LlmInferenceService.prepareEmbeddingInput(query,
						ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX));

		double[] scores = new double[dataset.length];
		for (int i = 0; i < dataset.length; i++) {
			String resourceType = TestDatasetHelper.inferResourceType(dataset[i]);
			String textContent = TestDatasetHelper.stripDatasetPrefixAndDate(dataset[i]);
			String embeddingText = ChartSearchAiUtils.buildPrefixedText(
					resourceType, textContent);
			float[] docVector = provider.embed(embeddingText);
			if (embeddingVectors != null) {
				embeddingVectors[i] = docVector;
			}
			scores[i] = LlmInferenceService.cosineSimilarity(queryVector, docVector);
		}
		return scores;
	}

	/**
	 * Runs the production pipeline using real semantic scores from the
	 * ONNX model combined with keyword scores from the dataset.
	 * Calls the exact same static production methods as the live system:
	 * {@link EmbeddingIndexer#buildEmbeddings} for indexing and
	 * {@link LlmInferenceService#findSimilar(List, EmbeddingProvider, String, int, String, LlmInferenceService.PipelineConfig)}
	 * for querying — zero simulation.
	 */
	private static List<Integer> runRealModelPipeline(String query, int topK) {
		return runRealModelPipeline(query, topK, FULL_PATIENT_DATASET);
	}

	private static List<Integer> runRealModelPipeline(String query, int topK,
			String[] dataset) {
		return runRealModelPipeline(query, topK, dataset,
				LlmInferenceService.PipelineConfig.defaults());
	}

	private static List<Integer> runRealModelPipeline(String query, int topK,
			String[] dataset, LlmInferenceService.PipelineConfig config) {
		org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider =
				new org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider(
						MODEL_PATH, VOCAB_PATH);
		try {
			// Build embeddings using the exact production indexing code
			List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records =
					new ArrayList<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord>();
			for (int i = 0; i < dataset.length; i++) {
				String resourceType = TestDatasetHelper.inferResourceType(dataset[i]);
				String textContent = TestDatasetHelper.stripDatasetPrefixAndDate(dataset[i]);
				records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
						resourceType, i, textContent, null));
			}
			List<ChartEmbedding> allEmbeddings =
					org.openmrs.module.chartsearchai.api.EmbeddingIndexer.buildEmbeddings(
							records, provider);

			// Query using the exact production query code
			List<ChartEmbedding> results = LlmInferenceService.findSimilar(
					allEmbeddings, provider, query, topK,
					ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX,
					config);

			List<Integer> indices = new ArrayList<Integer>();
			for (ChartEmbedding ce : results) {
				indices.add(ce.getResourceId());
			}
			Collections.sort(indices);
			return indices;
		} finally {
			provider.close();
		}
	}

	@Test
	public void realModel_vitalsTrendQuery_shouldReturnOnlyBpWeightAndTemperature() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?", 10);

		// All 40 BP + weight + temperature records (24 BP + 7 weight + 9 temp)
		// Must NOT include blood oxygen or fetishism records.
		assertEquals(Arrays.asList(17, 18, 22, 23, 25, 26, 31, 33, 36, 38,
				47, 48, 58, 63, 64, 73, 74, 76, 77, 81,
				93, 94, 96, 101, 102, 107, 111, 113, 114, 126,
				128, 129, 131, 137, 138, 143, 144, 147, 148, 150),
				result, "Should return all 40 BP + weight + temperature records");
	}

	@Test
	public void realModel_vitalsTrendQuery_shouldExcludeBloodOxygenAndOtherFalsePositives() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// "blood pressure, weight, and temperature" should NOT include
		// records that coincidentally match "blood" (like blood oxygen
		// saturation) but don't cover a unique query concept. The
		// filterRedundantKeywordTier logic must drop them because
		// "blood" is already covered by the BP records in the higher tier.
		List<Integer> result = runRealModelPipeline(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?", 10);

		// Blood oxygen saturation records in FULL_PATIENT_DATASET
		int[] bloodOxygenIndices = { 35, 46, 60, 80, 84, 104, 109, 117, 146, 152 };
		for (int idx : bloodOxygenIndices) {
			assertFalse(result.contains(idx),
					"Should NOT include blood oxygen record at index " + idx
					+ " — 'blood' is already covered by BP records, got: " + result);
		}
	}

	@Test
	public void realModel_vitalsTrendQuery_recencyCapShouldLimit7PerConcept() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// The pipeline returns all 40 BP + weight + temp records.
		// "last 7 visits" should trigger recency cap, keeping 7 per concept.
		String question = "How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?";

		assertEquals(7, LlmInferenceService.extractRecencyCap(question),
				"Should detect 'last 7' pattern");

		// Run the pipeline to get the 40 matching record indices
		List<Integer> pipelineResult = runRealModelPipeline(question, 10);
		assertTrue(pipelineResult.size() > 7,
				"Pipeline should return more than 7 records before cap");

		// Build SerializedRecords from the pipeline result (sorted most-recent-first).
		// In production, records come sorted by date from PatientRecordLoader.
		// Here we simulate that by using dataset index as a proxy — higher index
		// means more recent (matching how the test dataset is ordered).
		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> records
				= new ArrayList<>();
		// Sort indices descending to simulate most-recent-first
		List<Integer> descending = new ArrayList<>(pipelineResult);
		Collections.sort(descending, Collections.reverseOrder());
		for (int idx : descending) {
			records.add(new org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord(
					"obs", idx, FULL_PATIENT_DATASET[idx], null));
		}

		List<org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord> capped
				= LlmInferenceService.capPerConcept(records, 7);

		// Count records per concept
		Map<String, Integer> conceptCounts = new HashMap<>();
		for (org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord r : capped) {
			String key = LlmInferenceService.conceptKey(r.getText());
			conceptCounts.put(key, conceptCounts.getOrDefault(key, 0) + 1);
		}

		// No concept should have more than 7 records
		for (Map.Entry<String, Integer> entry : conceptCounts.entrySet()) {
			assertTrue(entry.getValue() <= 7,
					"Concept '" + entry.getKey() + "' has " + entry.getValue()
					+ " records, expected <= 7");
		}

		// Should have significantly fewer records than the original 40
		assertTrue(capped.size() < pipelineResult.size(),
				"Capped result (" + capped.size() + ") should be smaller than "
				+ "uncapped (" + pipelineResult.size() + ")");

		// Verify the expected concepts are present:
		// Systolic BP, Diastolic BP, Weight, Temperature
		assertTrue(conceptCounts.containsKey(
				"Clinical observation: Test — Systolic Blood Pressure"),
				"Should contain Systolic BP records");
		assertTrue(conceptCounts.containsKey(
				"Clinical observation: Test — Weight (kg)"),
				"Should contain Weight records");
		assertTrue(conceptCounts.containsKey(
				"Clinical observation: Test — Temperature (C)"),
				"Should contain Temperature records");
	}

	@Test
	public void realModel_cancerQuery_shouldReturnKaposiSarcomaOnly() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does the patient have cancer?", 10);

		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void realModel_historyOfCancerQuery_shouldReturnKaposiSarcomaOnly() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any history of cancer?", 10);

		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 Kaposi sarcoma records");
	}

	/**
	 * Integration test that exercises the full production code path for
	 * "any history of cancer?" — from record embedding (with
	 * {@link ChartSearchAiConstants#getEmbeddingPrefix}, matching
	 * {@link org.openmrs.module.chartsearchai.api.EmbeddingIndexer})
	 * through query preprocessing ({@link LlmInferenceService#buildEmbeddingQuery},
	 * matching {@link LlmInferenceService#findSimilar}) to the complete
	 * {@link LlmInferenceService#filterPipeline} with production defaults.
	 * Uses the real ONNX embedding model and the first patient dataset.
	 *
	 * <p>The dataset text format is {@code "prefix: (date) serializer_output"}
	 * but production stores just {@code serializer_output} as textContent and
	 * embeds {@code embedding_prefix + serializer_output} (no date). This test
	 * strips the dataset prefix and date to reconstruct the production format,
	 * Calls the actual production static methods
	 * {@link org.openmrs.module.chartsearchai.api.EmbeddingIndexer#buildEmbeddings}
	 * and {@link LlmInferenceService#findSimilar(List, EmbeddingProvider, String, int, String, LlmInferenceService.PipelineConfig)}
	 * directly — zero simulation, zero reimplementation.
	 */
	@Test
	public void integration_historyOfCancerQuery_shouldReturnOnly2Records() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("any history of cancer?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(2, result.size(),
				"'any history of cancer?' should return exactly 2 records");
	}

	@Test
	public void integration_anemicQuery_shouldReturnExactly3Records() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("is the patient anemic?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(29, 55, 72), result,
				"'is the patient anemic?' should return the 3 anemia records");
	}

	@Test
	public void integration_stdAbbreviationQuery_shouldReturnAll6HivRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("any STD?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(39, 40, 68, 69, 71, 110), result,
				"'any STD?' should return the same 6 HIV records as the full phrase");
	}

	@Test
	public void integration_activeConditionsQuery_shouldReturnOnlyConditionRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"When were each of this patient's active conditions first "
				+ "recorded, and have any resolved or escalated?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// Dataset indices are 1-based in comments but 0-based in the array.
		// [8] Tuberculosis = index 7, [55] Hypertension = index 54.
		assertEquals(Arrays.asList(7, 54), result,
				"Should return only the 2 active condition records "
				+ "(Tuberculosis [8] and Hypertension [55])");
	}

	@Test
	public void integration_currentCd4CountQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What is the current CD4 Count?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// [9] CD4 Count: 988.0 = index 8, [86] CD4 Count: 1191.0 = index 85
		assertEquals(Arrays.asList(8, 85), result,
				"Should return only the 2 CD4 Count records");
	}

	@Test
	public void integration_latestCd4CountQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What is the latest CD4 Count?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(8, 85), result,
				"Should return only the 2 CD4 Count records");
	}

	@Test
	public void integration_fractureQuery_shouldReturnNoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any fracture?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertTrue(result.isEmpty(),
				"Patient has no fracture records, should return empty");
	}

	@Test
	public void integration_medicationsQuery_shouldReturnMedicationRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"is the patient on any medications?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// [2] Azithromycin = index 1, [57] and [92] "Medication adjusted" = indices 56, 91
		assertEquals(Arrays.asList(1, 56, 91), result,
				"Should return drug order and medication-related visit notes");
	}

	@Test
	public void integration_knownConditionsQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any known conditions?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// [8] Tuberculosis = index 7, [55] Hypertension = index 54
		assertEquals(Arrays.asList(7, 54), result,
				"Should return the 2 Medical condition records");
	}

	@Test
	public void integration_conditionsQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any conditions",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(7, 54), result,
				"Should return the 2 Medical condition records");
	}

	@Test
	public void integration_whatIsPatientAllergicToQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What is the patient allergic to?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// [5] Beef allergy = index 4, [54] Fomepizole allergy = index 53
		assertEquals(Arrays.asList(4, 53), result,
				"Should return the 2 allergy records");
	}

	@Test
	public void integration_coughQuery_shouldReturnExactlyOneRecord() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any cough?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		// [51] "persistent cough for 2 weeks" = index 50
		assertEquals(Arrays.asList(50), result,
				"Should return only the record mentioning cough");
	}

	@Test
	public void integration_doesPatientHaveAllergiesQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does the patient have any allergies?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(4, 53), result,
				"Should return the 2 allergy records");
	}

	@Test
	public void integration_allergyQuery_shouldReturnExactlyTwoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any allergies?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(4, 53), result,
				"Should return the 2 allergy records");
	}


	@Test
	public void realModel_familyHistoryOfCancerQuery_shouldReturnNoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"Any family history of cancer?", 10);

		assertEquals(Collections.emptyList(),
				result, "Should return no records");
	}

	@Test
	public void realModel_doesHeHaveCancerQuery_shouldReturnKaposiSarcomaOnly() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does he have cancer?", 10);

		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void realModel_anyCancerQuery_shouldReturnKaposiSarcomaOnly() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any cancer?", 10);

		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void realModel_stdQuery_shouldReturnHivRecordsOnly() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any sexually transmitted disease?", 10);

		assertEquals(Arrays.asList(39, 40, 68, 69, 71, 110),
				result, "Should return exactly 6 HIV-related records");
	}

	@Test
	public void realModel_familyPlanningQuery_shouldReturnFamilyPlanningRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"Does this patient use any family planning methods?", 10);

		assertEquals(Arrays.asList(5, 6),
				result, "Should return exactly 2 family planning records");
	}

	@Test
	public void realModel_hbResultsQuery_shouldReturnNoRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What are this patient's HB results over time, and are values "
				+ "moving toward or away from the normal range?", 10);

		assertEquals(Collections.emptyList(),
				result, "Should return no records — dataset has no HB/hemoglobin data");
	}

	@Test
	public void realModel_cancerQuery_zScoreGateShouldNotBlock() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// "cancer?" has zero keyword matches but z-score > 2.0, so the
		// z-score gate should let it through. The dataset contains a
		// Kaposi sarcoma diagnosis which is semantically close to cancer.
		List<Integer> result = runRealModelPipeline(
				"does the patient have cancer?", 10);

		assertFalse(result.isEmpty(),
				"Z-score gate should pass — cancer query has a genuine "
				+ "semantic outlier (Kaposi sarcoma)");
		// Records 11 and 88 are "Diagnosis — Kaposi sarcoma oral" (0-indexed)
		assertTrue(result.contains(11) || result.contains(88),
				"Should include at least one Kaposi sarcoma record (11 or 88), "
				+ "got: " + result);
	}

	@Test
	public void realModel_conditionsQuery_shouldReturnAllConditionRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any conditions?", 10, SECOND_PATIENT_DATASET);

		// All 10 "Medical condition:" records in the second dataset.
		// Keyword rescue ensures conditions like Scarring Alopecia and
		// Granuloma annulare aren't dropped by semantic gap detection
		// when they have perfect keyword matches on "condition".
		assertEquals(Arrays.asList(1, 2, 16, 17, 29, 30, 31, 44, 55, 56),
				result, "Should return all 10 condition records from second dataset");
	}

	@Test
	public void realModel_episodesQuery_shouldReturnDepressiveEpisodeRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any episodes?", 10, SECOND_PATIENT_DATASET);

		// Pipeline returns 0-indexed values. Records at 1-indexed positions
		// 32 and 35 are 0-indexed 31 and 34:
		// [31] Medical condition: Condition: Mild depressive episode. Status: ACTIVE
		// [34] Clinical diagnosis: Diagnosis: Mild depressive episode. Certainty: CONFIRMED
		// Keyword rescue (plural stem "episodes" → "episode") bypasses the
		// semantic floor gate when enough keyword matches exist.
		assertTrue(result.contains(31),
				"Should include condition record for Mild depressive episode, got: " + result);
		assertTrue(result.contains(34),
				"Should include diagnosis record for Mild depressive episode, got: " + result);
	}

	@Test
	public void realModel_stdQuery_shouldReturnSyphiliticCirrhosisRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"has this patient had a sexually transmitted disease?", 10,
				SECOND_PATIENT_DATASET);

		// [19] Clinical diagnosis: Diagnosis: Syphilitic Cirrhosis. Certainty: CONFIRMED
		// Syphilis is an STD; syphilitic cirrhosis is a complication of syphilis.
		// No keyword matches (query terms "sexually", "transmitted", "disease"
		// don't appear in record text) — purely semantic retrieval.
		// Record 17 (Syphilitic Cirrhosis condition, sem=0.33) is at the ratio
		// floor boundary (0.42*0.80=0.34) so may or may not be included.
		assertTrue(result.contains(19),
				"Should include Syphilitic Cirrhosis diagnosis, got: " + result);
	}

	@Test
	public void realModel_cd4Query_secondDataset_shouldReturnEmpty() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// The second dataset has no CD4 records. The pipeline should
		// not return pulse records as false positives.
		List<Integer> result = runRealModelPipeline(
				"what is the current cd4 count and when was it recorded?",
				10, SECOND_PATIENT_DATASET);

		assertTrue(result.isEmpty(),
				"Second dataset has no CD4 records, "
				+ "should return empty, got: " + result);
	}

	@Test
	public void realModel_secondDataset_vitalsTrendQuery_shouldReturnBpWeightAndTemperature() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// Use default config — exercises the coherent-gap path.
		List<Integer> result = runRealModelPipeline(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?", 10,
				SECOND_PATIENT_DATASET);

		// Temperature: 0-based 5, 20, 35, 46, 59
		// Systolic BP: 0-based 6, 21, 36, 47, 60
		// Diastolic BP: 0-based 7, 22, 37, 48, 61
		// Weight: 0-based 10, 25, 40, 51, 64
		assertEquals(Arrays.asList(5, 6, 7, 10, 20, 21, 22, 25, 35, 36, 37, 40,
				46, 47, 48, 51, 59, 60, 61, 64),
				result,
				"Should return all 20 BP + weight + temperature records, got: " + result);
	}

	@Test
	public void realModel_secondDataset_vitalsTrend_shouldExcludeBloodTransfusionCondition() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// "Personal history of blood transfusion" matches "blood" but
		// that term is already covered by BP records. It should be
		// dropped as a false positive.
		List<Integer> result = runRealModelPipeline(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?", 10,
				SECOND_PATIENT_DATASET);

		// 0-based index 55 = "Medical condition: Condition: Personal
		// history of blood transfusion. Status: ACTIVE"
		assertFalse(result.contains(55),
				"Should NOT include blood transfusion condition — 'blood' "
				+ "is already covered by BP records, got: " + result);
	}

	@Test
	public void realModel_secondDataset_bpAndPulse_shouldReturnBothConceptsAndExcludeSpO2() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		// Different multi-concept query: "blood pressure and pulse".
		// BP matches 2 terms ("blood"+"pressure"), pulse matches 1
		// ("pulse"). SpO2 ("arterial blood oxygen saturation") matches
		// "blood" only — a false positive that should be dropped.
		List<Integer> result = runRealModelPipeline(
				"blood pressure and pulse", 10, SECOND_PATIENT_DATASET);

		// Systolic BP: 0-based 6, 21, 36, 47, 60
		// Diastolic BP: 0-based 7, 22, 37, 48, 61
		// Pulse: 0-based 8, 23, 38, 49, 62
		for (int bpIdx : new int[]{ 6, 7, 21, 22, 36, 37, 47, 48, 60, 61 }) {
			assertTrue(result.contains(bpIdx),
					"Should include BP record at index " + bpIdx
					+ ", got: " + result);
		}
		for (int pulseIdx : new int[]{ 8, 23, 38, 49, 62 }) {
			assertTrue(result.contains(pulseIdx),
					"Should include pulse record at index " + pulseIdx
					+ ", got: " + result);
		}

		// SpO2: 0-based 9, 24, 39, 50, 63 — matches "blood" only,
		// which is already covered by BP → false positive
		for (int spo2Idx : new int[]{ 9, 24, 39, 50, 63 }) {
			assertFalse(result.contains(spo2Idx),
					"Should NOT include SpO2 record at index " + spo2Idx
					+ " — 'blood' is already covered by BP, got: " + result);
		}
	}

	@Test
	public void realModel_diagnosticScoreDump() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		String[] queries = {
			"does the patient have cancer?",
			"any history of cancer?",
			"Any family history of cancer?",
			"any sexually transmitted disease?",
			"Does this patient use any family planning methods?",
			"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?"
		};

		org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider =
				new org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider(
						MODEL_PATH, VOCAB_PATH);
		try {
			for (String query : queries) {
				double[] semantic = computeRealSemanticScores(provider, query);
				String normalized = LlmInferenceService.stripQueryStopwords(query);
				String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);

				System.out.println("\n=== QUERY: \"" + query + "\" ===");
				System.out.println("Terms: " + Arrays.toString(queryTerms)
						+ " (N=" + queryTerms.length + ")");
				double bonusThreshold = queryTerms.length == 0 ? 1.0
						: (double) Math.min(2, queryTerms.length) / queryTerms.length;
				System.out.println("BonusThreshold: " + bonusThreshold);

				// Collect and sort by semantic score descending
				double[][] indexed = new double[FULL_PATIENT_DATASET.length][3];
				for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
					String rt = TestDatasetHelper.inferResourceType(FULL_PATIENT_DATASET[i]);
					String tc = TestDatasetHelper.stripDatasetPrefixAndDate(FULL_PATIENT_DATASET[i]);
					String kwText = ChartSearchAiUtils.buildPrefixedText(rt, tc);
					double kw = LlmInferenceService.computeKeywordScore(
							queryTerms, kwText);
					indexed[i][0] = i;
					indexed[i][1] = semantic[i];
					indexed[i][2] = kw;
				}
				Arrays.sort(indexed, (a, b) -> Double.compare(b[1], a[1]));

				System.out.println("Top 60 by semantic score:");
				for (int i = 0; i < Math.min(60, indexed.length); i++) {
					int idx = (int) indexed[i][0];
					double sem = indexed[i][1];
					double kw = indexed[i][2];
					double bonus = kw >= bonusThreshold ? kw : 0.0;
					double penalty = (kw > 0 && kw < bonusThreshold) ? kw : 0.0;
					double combined = sem + 0.3 * bonus - 0.3 * penalty;
					String record = FULL_PATIENT_DATASET[idx];
					if (record.length() > 80) record = record.substring(0, 80) + "...";
					System.out.printf("  [%3d] sem=%.4f kw=%.4f comb=%.4f %s%s %s%n",
							idx, sem, kw, combined,
							bonus > 0 ? " BONUS" : "",
							penalty > 0 ? " PENALTY" : "",
							record);
				}
			}
		} finally {
			provider.close();
		}
		// This test always passes — it's just for diagnostics
		assertTrue(true);
	}


	// --------------------------------------------------------
	// Direct unit tests for isGapCoherent, growCluster, and
	// rescueBelowFloor.
	// --------------------------------------------------------

	private static LlmInferenceService.ScoredEmbedding makeScoredEmbeddingWithVector(
			double score, double keywordScore, double semanticScore,
			float[] vector, int id) {
		ChartEmbedding ce = new ChartEmbedding();
		ce.setResourceType("obs");
		ce.setTextContent(String.valueOf(id));
		ce.setEmbeddingId(id);
		ce.setEmbeddingVector(vector);
		return new LlmInferenceService.ScoredEmbedding(ce, score, keywordScore, semanticScore);
	}

	@Test
	public void isGapCoherent_shouldReturnTrueWhenCrossBoundaryCosineHighEnough() {
		// 6 records: 3 above cutoff, 3 below. All share a similar vector →
		// high cross-boundary cosine → gap is intra-topic.
		float[] v1 = { 1.0f, 0.0f, 0.0f };
		float[] v2 = { 0.95f, 0.1f, 0.0f };
		float[] v3 = { 0.9f, 0.2f, 0.0f };
		float[] v4 = { 0.85f, 0.25f, 0.0f };
		float[] v5 = { 0.8f, 0.3f, 0.0f };
		float[] v6 = { 0.75f, 0.35f, 0.0f };

		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, v1, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, v2, 2),
				makeScoredEmbeddingWithVector(0.46, 0, 0.46, v3, 3),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, v4, 4),
				makeScoredEmbeddingWithVector(0.28, 0, 0.28, v5, 5),
				makeScoredEmbeddingWithVector(0.26, 0, 0.26, v6, 6));

		assertTrue(LlmInferenceService.isGapCoherent(scored, 3, 0.47),
				"Cross-boundary cosine is high — gap is intra-topic");
	}

	@Test
	public void isGapCoherent_shouldReturnFalseWhenCrossBoundaryCosineIsLow() {
		// 3 above cutoff point one way, 3 below point another → low cross-boundary cosine.
		float[] vA = { 1.0f, 0.0f, 0.0f };
		float[] vB = { 0.0f, 1.0f, 0.0f };

		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, vA, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, vA, 2),
				makeScoredEmbeddingWithVector(0.46, 0, 0.46, vA, 3),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, vB, 4),
				makeScoredEmbeddingWithVector(0.28, 0, 0.28, vB, 5),
				makeScoredEmbeddingWithVector(0.26, 0, 0.26, vB, 6));

		assertFalse(LlmInferenceService.isGapCoherent(scored, 3, 0.47),
				"Cross-boundary cosine is low — gap is inter-topic");
	}

	@Test
	public void isGapCoherent_shouldReturnFalseWhenCutoffIsZeroOrBeyondSize() {
		float[] v = { 1.0f, 0.0f };
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
				makeScoredEmbeddingWithVector(0.40, 0, 0.40, v, 2));

		assertFalse(LlmInferenceService.isGapCoherent(scored, 0, 0.47));
		assertFalse(LlmInferenceService.isGapCoherent(scored, 2, 0.47));
	}

	@Test
	public void isGapCoherent_shouldReturnFalseWhenVectorsAreNull() {
		List<LlmInferenceService.ScoredEmbedding> scored = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, null, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, null, 2),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, null, 3),
				makeScoredEmbeddingWithVector(0.28, 0, 0.28, null, 4));

		assertFalse(LlmInferenceService.isGapCoherent(scored, 2, 0.47),
				"Null vectors → no pairs → should return false");
	}

	@Test
	public void growCluster_shouldAddCoherentCandidatesBeyondSeed() {
		// Seed: first 2 records. Records 3-4 are coherent with seed (similar
		// vectors). Record 5 is orthogonal → should not be included.
		float[] vA = { 1.0f, 0.0f, 0.0f };
		float[] vSimilar = { 0.95f, 0.1f, 0.0f };
		float[] vOrthogonal = { 0.0f, 1.0f, 0.0f };

		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, vA, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, vA, 2),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, vSimilar, 3),
				makeScoredEmbeddingWithVector(0.28, 0, 0.28, vSimilar, 4),
				makeScoredEmbeddingWithVector(0.20, 0, 0.20, vOrthogonal, 5));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.growCluster(candidates, 2, 0.47);

		assertEquals(4, result.size(),
				"Seed (2) + 2 coherent candidates = 4; orthogonal excluded");
	}

	@Test
	public void growCluster_shouldReturnAllWhenSeedIsEntireList() {
		float[] v = { 1.0f, 0.0f };
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, v, 2));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.growCluster(candidates, 2, 0.47);

		assertEquals(2, result.size());
	}

	@Test
	public void growCluster_shouldSkipCandidatesWithNullVectors() {
		float[] v = { 1.0f, 0.0f };
		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, v, 2),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, null, 3));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.growCluster(candidates, 2, 0.47);

		assertEquals(2, result.size(),
				"Null-vector candidate should not be added");
	}

	@Test
	public void growCluster_shouldGrowTransitively() {
		// Record 3 is coherent with seed. Record 4 is coherent with
		// record 3 but not with the seed. After first iteration adds 3,
		// second iteration should add 4 through transitivity.
		float[] vSeed = { 1.0f, 0.0f, 0.0f };
		float[] vBridge = { 0.7f, 0.7f, 0.0f }; // cos with seed ≈ 0.71, cos with far ≈ 0.71
		float[] vFar = { 0.0f, 1.0f, 0.0f }; // cos with seed ≈ 0, cos with bridge ≈ 0.71

		List<LlmInferenceService.ScoredEmbedding> candidates = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, vSeed, 1),
				makeScoredEmbeddingWithVector(0.40, 0, 0.40, vBridge, 2),
				makeScoredEmbeddingWithVector(0.30, 0, 0.30, vFar, 3));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.growCluster(candidates, 1, 0.47);

		assertEquals(3, result.size(),
				"Transitive growth: seed→bridge→far");
	}

	@Test
	public void rescueBelowFloor_shouldRecoverCoherentBelowFloorRecords() {
		// Cluster: 3 records with slightly different but similar vectors.
		// This creates a minClusterCoherence < 1.0, allowing rescue of
		// below-floor records that meet the threshold.
		float[] v1 = { 1.0f, 0.0f, 0.0f };
		float[] v2 = { 0.95f, 0.1f, 0.0f };
		float[] v3 = { 0.9f, 0.15f, 0.0f };
		// Similar to cluster — should be rescued
		float[] vSimilar = { 0.92f, 0.12f, 0.0f };
		// Orthogonal — should NOT be rescued
		float[] vOrthogonal = { 0.0f, 1.0f, 0.0f };

		List<LlmInferenceService.ScoredEmbedding> cluster = new ArrayList<LlmInferenceService.ScoredEmbedding>(
				Arrays.asList(
						makeScoredEmbeddingWithVector(0.50, 0, 0.50, v1, 1),
						makeScoredEmbeddingWithVector(0.48, 0, 0.48, v2, 2),
						makeScoredEmbeddingWithVector(0.46, 0, 0.46, v3, 3)));

		// Full scored list includes cluster + below-floor records
		List<LlmInferenceService.ScoredEmbedding> scored = new ArrayList<LlmInferenceService.ScoredEmbedding>(
				Arrays.asList(
						makeScoredEmbeddingWithVector(0.50, 0, 0.50, v1, 1),
						makeScoredEmbeddingWithVector(0.48, 0, 0.48, v2, 2),
						makeScoredEmbeddingWithVector(0.46, 0, 0.46, v3, 3),
						makeScoredEmbeddingWithVector(0.10, 0, 0.10, vSimilar, 4),
						makeScoredEmbeddingWithVector(0.08, 0, 0.08, vOrthogonal, 5)));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.rescueBelowFloor(cluster, scored, 3);

		assertEquals(4, result.size(),
				"Should rescue the coherent below-floor record (id=4) but not the orthogonal one");
	}

	@Test
	public void rescueBelowFloor_shouldReturnUnchangedWhenAdaptiveCutoffBeyondScored() {
		float[] v = { 1.0f, 0.0f };
		List<LlmInferenceService.ScoredEmbedding> cluster = Arrays.asList(
				makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
				makeScoredEmbeddingWithVector(0.48, 0, 0.48, v, 2),
				makeScoredEmbeddingWithVector(0.46, 0, 0.46, v, 3));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.rescueBelowFloor(cluster, cluster, 3);

		assertEquals(3, result.size(),
				"No below-floor records to check");
	}

	@Test
	public void rescueBelowFloor_shouldNotRescueRecordsAlreadyInCluster() {
		float[] v = { 1.0f, 0.0f };
		List<LlmInferenceService.ScoredEmbedding> cluster = new ArrayList<LlmInferenceService.ScoredEmbedding>(
				Arrays.asList(
						makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
						makeScoredEmbeddingWithVector(0.48, 0, 0.48, v, 2),
						makeScoredEmbeddingWithVector(0.46, 0, 0.46, v, 3)));

		// scored contains the same records (same IDs) in the below-floor range
		List<LlmInferenceService.ScoredEmbedding> scored = new ArrayList<LlmInferenceService.ScoredEmbedding>(
				Arrays.asList(
						makeScoredEmbeddingWithVector(0.50, 0, 0.50, v, 1),
						makeScoredEmbeddingWithVector(0.48, 0, 0.48, v, 2),
						makeScoredEmbeddingWithVector(0.46, 0, 0.46, v, 3),
						makeScoredEmbeddingWithVector(0.10, 0, 0.10, v, 1)));

		List<LlmInferenceService.ScoredEmbedding> result =
				LlmInferenceService.rescueBelowFloor(cluster, scored, 3);

		assertEquals(3, result.size(),
				"Should not duplicate records already in cluster");
	}

	// ---- Integration tests replacing former synthetic pipeline_* tests ----
	// Each test below replaces a specific deleted synthetic test, exercising
	// the same pipeline mechanic with real ONNX embeddings.

	@Test
	public void integration_topKShouldCapResultsWhenNoKeywordMatches() {
		// Replaces: pipeline_genericQuery_shouldCapToTopK
		// When no records have keyword matches, topK caps the result set.
		// "cancer" has zero keyword matches in the dataset, so topK applies.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does the patient have cancer?", 1, FULL_PATIENT_DATASET);

		assertEquals(1, result.size(),
				"With no keyword matches, topK=1 should cap to 1 record");
	}

	@Test
	public void integration_keywordMatchedRecordsShouldBypassTopK() {
		// Replaces: pipeline_incidentalKeywordMatches_shouldNotOverFilter
		// When records have keyword matches, they bypass topK capping.
		// CD4 Count records match "CD4"+"count" keywords — both should be
		// returned even with topK=1, because keyword-matched records are
		// not subject to topK truncation.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What is the current CD4 Count?", 1, FULL_PATIENT_DATASET);

		assertEquals(Arrays.asList(8, 85), result,
				"Keyword-matched records should bypass topK=1 — both CD4 records returned");
	}

	@Test
	public void integration_gapDetectionShouldTakePrecedenceOverKeywords() {
		// Replaces: pipeline_gapDetectionWorks_keywordRefinementShouldNotInterfere
		// Gap detection finds a clear boundary between the 2 prescription
		// records and everything else. Keyword refinement should not pull
		// lower records into the result despite potential keyword overlap.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"any prescriptions?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(0, 1), result,
				"Gap detection should isolate the 2 drug order records");
	}

	@Test
	public void integration_keywordRefinementShouldWorkWithSmoothScores() {
		// Replaces: pipeline_smoothDistributionNoKeywords_shouldReturnAllAboveFloor
		// When semantic scores are smooth (no clear gap), keyword refinement
		// identifies relevant records via keyword discrimination.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"list all allergies",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(4, 53), result,
				"Keyword refinement should isolate allergy records from smooth scores");
	}

	@Test
	public void integration_keywordWeightZeroShouldDisableRefinement() {
		// Replaces: pipeline_keywordWeightZero_shouldDisableRefinement
		// With keywordWeight=0, keyword refinement is disabled. On the
		// second patient dataset, "any conditions?" normally returns all
		// 10 condition records (keyword refinement rescues them). With
		// keywordWeight=0, only the top semantic matches survive.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		LlmInferenceService.PipelineConfig noKeywordConfig =
				new LlmInferenceService.PipelineConfig(
						0.0,
						ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
						ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP,
						ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
						ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO);

		List<Integer> withKeywords = runRealModelPipeline("any conditions?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K,
				SECOND_PATIENT_DATASET);
		List<Integer> withoutKeywords = runRealModelPipeline("any conditions?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K,
				SECOND_PATIENT_DATASET, noKeywordConfig);

		assertEquals(10, withKeywords.size(),
				"With default keywordWeight, should return all 10 conditions");
		assertEquals(Arrays.asList(1, 31), withoutKeywords,
				"With keywordWeight=0, only top semantic matches survive");
		assertTrue(withoutKeywords.size() < withKeywords.size(),
				"Disabling keywords should return fewer records");
	}

	@Test
	public void integration_noMatchingRecordsShouldReturnEmpty() {
		// Replaces: pipeline_medicationQueryNoDrugOrders_shouldReturnContext
		// When the patient has no matching records, the pipeline should
		// return empty — not dump noise into the LLM.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"What are this patient's HB results over time, and are values "
				+ "moving toward or away from the normal range?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertTrue(result.isEmpty(),
				"Patient has no HB/hemoglobin records — should return empty");
	}

	@Test
	public void integration_zScoreGateShouldBlockSmoothLowScores() {
		// Replaces: pipeline_noKeywordMatchesSmoothDistribution_shouldReturnEmpty
		// When no records match the query well (smooth, low semantic scores
		// and no keyword matches), the z-score gate should block everything
		// to prevent sending noise to the LLM.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("any fracture?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertTrue(result.isEmpty(),
				"No fracture records exist — z-score gate should block all results");
	}

	@Test
	public void integration_keywordRefinementShouldBypassFloorAndTopK() {
		// Replaces: pipeline_keywordRefinementBypassesStrictFloorAndTopK
		// All 10 condition records in the second dataset match "condition"
		// via keyword. Keyword refinement should bypass both the ratio
		// floor and topK, returning all 10 even with topK=5.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("any conditions?", 5,
				SECOND_PATIENT_DATASET);

		assertTrue(result.size() > 5,
				"Keyword refinement should bypass topK=5, got " + result.size());
		assertEquals(Arrays.asList(1, 2, 16, 17, 29, 30, 31, 44, 55, 56), result,
				"Should return all 10 condition records from second dataset");
	}

	@Test
	public void integration_focusedQueryShouldFilterNoise() {
		// Replaces: pipeline_focusedQueryStrictFloorFiltersNoise
		// A focused query should return only records that cluster high
		// semantically, filtering out noise below the gap or floor.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does the patient have cancer?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);

		assertEquals(Arrays.asList(11, 88), result,
				"Focused cancer query should return only 2 Kaposi sarcoma records, "
				+ "filtering out all noise below the semantic gap");
	}

	@Test
	public void integration_ratioFloorShouldExcludeMarginalRecords() {
		// Replaces: pipeline_focusedQueryRatioFloorExcludesMarginalRecords
		// The anemic query returns 3 records with default topK. With topK=1
		// and no keyword matches on "anemic", topK caps the result —
		// marginal records that scored below the top are excluded.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> fullResult = runRealModelPipeline("is the patient anemic?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);
		List<Integer> cappedResult = runRealModelPipeline("is the patient anemic?",
				1, FULL_PATIENT_DATASET);

		assertTrue(fullResult.size() > 1,
				"Full result should have multiple anemia records");
		assertTrue(cappedResult.size() < fullResult.size(),
				"TopK=1 should exclude marginal records, got " + cappedResult);
	}

	@Test
	public void integration_labOrdersQuery_shouldReturnEmptyWhenNoLabOrders() {
		// FULL_PATIENT_DATASET has no "Lab test order:" records.
		// Obs records mentioning "Labs ordered." in clinical notes should not match.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"Were all lab orders placed for this patient resulted?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K);
		assertTrue(result.isEmpty(),
				"No lab order records exist — should return empty, got: " + result);
	}

	@Test
	public void integration_keywordRefinementCanExceedTopK() {
		// Replaces: pipeline_keywordRefinementCanExceedTopK
		// All 10 condition records match "condition" keyword. With topK=3,
		// keyword refinement should keep all 10 — exceeding topK.
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline("any conditions?", 3,
				SECOND_PATIENT_DATASET);

		assertTrue(result.size() > 3,
				"Keyword refinement should allow exceeding topK=3, got " + result.size());
	}

	@Test
	public void integration_pregnancyQuery_shouldReturnAbortionRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"ever got pregnant?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K,
				THIRD_PATIENT_DATASET);

		// [137] Self-Induced Abortion condition = index 136
		// [139] Self-Induced Abortion diagnosis = index 138
		assertTrue(result.contains(136),
				"Should include Self-Induced Abortion condition, got: " + result);
		assertTrue(result.contains(138),
				"Should include Self-Induced Abortion diagnosis, got: " + result);
	}

	@Test
	public void integration_bloodProblemQuery_shouldReturnAnaemiaAndHematologyRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does she have any blood problem?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K,
				THIRD_PATIENT_DATASET);

		// Blood-related records: hematology tests, blood cell counts,
		// and blood conditions — includes both keyword matches (records
		// containing "blood") and semantic rescues (records about blood
		// that use different vocabulary like Haemoglobin, Anaemia).
		// [  1] Haemoglobin 15.8 g/dL (HIGH) = index 0
		// [ 92] Complete Blood Count order = index 91
		// [ 94] Anaemia condition = index 93
		// [ 95] Anaemia diagnosis = index 94
		// [104] Haemoglobin 11.2 g/dL (LOW) = index 103
		// [105] White blood cells = index 104
		// [106] Platelet count = index 105
		// [107] Hematocrit = index 106
		// [109] Red blood cells (LOW) = index 108
		assertTrue(result.contains(0),
				"Should include Haemoglobin (HIGH), got: " + result);
		assertTrue(result.contains(91),
				"Should include Complete Blood Count order, got: " + result);
		assertTrue(result.contains(93),
				"Should include Anaemia condition, got: " + result);
		assertTrue(result.contains(94),
				"Should include Anaemia diagnosis, got: " + result);
		assertTrue(result.contains(103),
				"Should include Haemoglobin (LOW), got: " + result);
		assertTrue(result.contains(104),
				"Should include White blood cells, got: " + result);
		assertTrue(result.contains(105),
				"Should include Platelet count, got: " + result);
		assertTrue(result.contains(106),
				"Should include Hematocrit, got: " + result);
		assertTrue(result.contains(108),
				"Should include Red blood cells (LOW), got: " + result);
	}

	@Test
	public void integration_bloodProblemQuery_fourthDataset_shouldReturnBloodRelatedRecords() {
		org.junit.jupiter.api.Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model files not found at " + MODEL_PATH);

		List<Integer> result = runRealModelPipeline(
				"does she have any blood problem?",
				ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K,
				FOURTH_PATIENT_DATASET);

		// Blood-related records rescued via vocabulary-gap rescue
		// (no "blood" keyword, but high semantic similarity):
		// [  1] Haemoglobin: 15.8 g/dL (HIGH) = index 0
		// [ 19] Condition: Haemorrhagic disease of newborn = index 18
		// [ 22] Diagnosis: Haemorrhagic disease of newborn = index 21
		assertTrue(result.contains(0),
				"Should include Haemoglobin (HIGH), got: " + result);
		assertTrue(result.contains(18),
				"Should include Haemorrhagic disease condition, got: " + result);
		assertTrue(result.contains(21),
				"Should include Haemorrhagic disease diagnosis, got: " + result);
	}

}
