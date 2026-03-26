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

	// Full 153-record dataset from a real 16-year-old Male patient chart.
	// Used by integration tests that need the complete patient record set.
	private static final String[] FULL_PATIENT_DATASET = {
			/* [  1] */ "Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Every six hours. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: REVISE. Urgency: ROUTINE. Reason: Spectrum",
			/* [  2] */ "Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Thrice daily. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: NEW. Urgency: ROUTINE. Reason: Spectrum. Stopped: 2026-03-18",
			/* [  3] */ "Program: PMTCT. Enrolled: 2026-03-18. Status: Active",
			/* [  4] */ "Finding — Immunization history: Immunizations: Polio vaccination, oral, Oral polio vacc); Vaccination date: 2026-03-18; Immunization sequence number: 1.0",
			/* [  5] */ "Allergy: Beef (food allergen). Severity: Severe. Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
			/* [  6] */ "Assessment — Method of family planning: Condoms",
			/* [  7] */ "Assessment — Method of family planning: Diaphragm. Note: in bathroom",
			/* [  8] */ "Condition: Tuberculosis. Status: ACTIVE",
			/* [  9] */ "Test — CD4 Count: 988.0",
			/* [ 10] */ "Drug — Pyrimethamine / sulfadoxine: 11.58",
			/* [ 11] */ "Units of Measure — Syringe): 65.0",
			/* [ 12] */ "Diagnosis — Kaposi sarcoma oral: 3.91",
			/* [ 13] */ "Assessment — Primary Diagnosis: Tuberculosis",
			/* [ 14] */ "Frequency — Every twenty-four hours: Every four hours",
			/* [ 15] */ "Test — Height (cm): 131.0",
			/* [ 16] */ "Test — Respiratory Rate: 18.0",
			/* [ 17] */ "Test — Pulse: 95.0",
			/* [ 18] */ "Test — Temperature (C): 36.7",
			/* [ 19] */ "Test — Weight (kg): 94.0",
			/* [ 20] */ "Diagnosis — Fetishism: Patient presents with mild symptoms. Advised rest and fluids.",
			/* [ 21] */ "Frequency — Every twenty-four hours: Every eight hours",
			/* [ 22] */ "Diagnosis: Gastroenteritis. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 23] */ "Test — Systolic Blood Pressure: 97.0",
			/* [ 24] */ "Test — Diastolic Blood Pressure: 99.0",
			/* [ 25] */ "Test — Pulse: 62.0",
			/* [ 26] */ "Test — Temperature (C): 37.7",
			/* [ 27] */ "Test — Weight (kg): 107.0",
			/* [ 28] */ "Test — Height (cm): 137.0",
			/* [ 29] */ "Test — Respiratory Rate: 24.0",
			/* [ 30] */ "Assessment — Primary Diagnosis: Anemia",
			/* [ 31] */ "Frequency — Every twenty-four hours: Every five hours",
			/* [ 32] */ "Test — Diastolic Blood Pressure: 92.0",
			/* [ 33] */ "Test — Respiratory Rate: 32.0",
			/* [ 34] */ "Test — Weight (kg): 139.0",
			/* [ 35] */ "Test — Blood Oxygen Saturation: 88.0",
			/* [ 36] */ "Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [ 37] */ "Test — Systolic Blood Pressure: 122.0",
			/* [ 38] */ "Test — Height (cm): 103.0",
			/* [ 39] */ "Test — Temperature (C): 40.3",
			/* [ 40] */ "Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 41] */ "Assessment — Primary Diagnosis: HIV Disease",
			/* [ 42] */ "Diagnosis — Fetishism: Patient stable on current regimen.",
			/* [ 43] */ "Frequency — Every twenty-four hours: Married",
			/* [ 44] */ "Test — Pulse: 62.0",
			/* [ 45] */ "Test — Respiratory Rate: 23.0",
			/* [ 46] */ "Test — Blood Oxygen Saturation: 92.0",
			/* [ 47] */ "Test — Height (cm): 126.0",
			/* [ 48] */ "Test — Systolic Blood Pressure: 101.0",
			/* [ 49] */ "Test — Diastolic Blood Pressure: 99.0",
			/* [ 50] */ "Assessment — Primary Diagnosis: Diabetes Mellitus",
			/* [ 51] */ "Diagnosis — Fetishism: New complaint of persistent cough for 2 weeks.",
			/* [ 52] */ "Diagnosis: Urinary Tract Infection. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 53] */ "Diagnosis: Tuberculosis. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 54] */ "Allergy: Fomepizole (drug allergen)",
			/* [ 55] */ "Condition: Hypertension. Status: ACTIVE",
			/* [ 56] */ "Assessment — Primary Diagnosis: Anemia",
			/* [ 57] */ "Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 58] */ "Test — Height (cm): 101.0",
			/* [ 59] */ "Test — Systolic Blood Pressure: 123.0",
			/* [ 60] */ "Test — Blood Oxygen Saturation: 86.0",
			/* [ 61] */ "Test — Pulse: 51.0",
			/* [ 62] */ "Diagnosis: Skin Infection. Certainty: CONFIRMED. Rank: Primary",
			/* [ 63] */ "Assessment — Primary Diagnosis: Pneumonia",
			/* [ 64] */ "Test — Weight (kg): 38.0",
			/* [ 65] */ "Test — Diastolic Blood Pressure: 71.0",
			/* [ 66] */ "Test — Respiratory Rate: 16.0",
			/* [ 67] */ "Diagnosis: Malaria. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 68] */ "Diagnosis: Diabetes Mellitus. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 69] */ "Assessment — Primary Diagnosis: HIV Disease",
			/* [ 70] */ "Diagnosis: HIV Disease. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 71] */ "Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [ 72] */ "Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 73] */ "Diagnosis: Anemia. Certainty: PROVISIONAL. Rank: Secondary",
			/* [ 74] */ "Test — Systolic Blood Pressure: 137.0",
			/* [ 75] */ "Test — Diastolic Blood Pressure: 67.0",
			/* [ 76] */ "Test — Pulse: 86.0",
			/* [ 77] */ "Test — Temperature (C): 39.3",
			/* [ 78] */ "Test — Weight (kg): 146.0",
			/* [ 79] */ "Test — Height (cm): 107.0",
			/* [ 80] */ "Test — Blood Oxygen Saturation: 86.0",
			/* [ 81] */ "Test — Respiratory Rate: 30.0",
			/* [ 82] */ "Test — Diastolic Blood Pressure: 105.0",
			/* [ 83] */ "Test — Pulse: 70.0",
			/* [ 84] */ "Test — Blood Oxygen Saturation: 100.0",
			/* [ 85] */ "Test — Respiratory Rate: 40.0",
			/* [ 86] */ "Test — CD4 Count: 1191.0",
			/* [ 87] */ "Units of Measure — Syringe): 339.0",
			/* [ 88] */ "Misc — Milligram per meter squared: 47.0",
			/* [ 89] */ "Diagnosis — Kaposi sarcoma oral: 3.5",
			/* [ 90] */ "Diagnosis — Photoallergy: 9.93",
			/* [ 91] */ "Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [ 92] */ "Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 93] */ "Diagnosis: Hypertension. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 94] */ "Test — Systolic Blood Pressure: 147.0",
			/* [ 95] */ "Test — Diastolic Blood Pressure: 58.0",
			/* [ 96] */ "Test — Pulse: 53.0",
			/* [ 97] */ "Test — Temperature (C): 36.4",
			/* [ 98] */ "Test — Height (cm): 163.0",
			/* [ 99] */ "Test — Respiratory Rate: 19.0",
			/* [100] */ "Assessment — Primary Diagnosis: Headache",
			/* [101] */ "Test — Pulse: 83.0",
			/* [102] */ "Test — Weight (kg): 68.0",
			/* [103] */ "Test — Diastolic Blood Pressure: 50.0",
			/* [104] */ "Test — Blood Oxygen Saturation: 94.0",
			/* [105] */ "Test — Respiratory Rate: 40.0",
			/* [106] */ "Assessment — Primary Diagnosis: Malaria",
			/* [107] */ "Diagnosis — Fetishism: Well-child visit. Growth and development normal.",
			/* [108] */ "Test — Diastolic Blood Pressure: 93.0",
			/* [109] */ "Test — Blood Oxygen Saturation: 95.0",
			/* [110] */ "Test — Respiratory Rate: 36.0",
			/* [111] */ "Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [112] */ "Test — Systolic Blood Pressure: 98.0",
			/* [113] */ "Test — Pulse: 66.0",
			/* [114] */ "Test — Temperature (C): 37.8",
			/* [115] */ "Test — Weight (kg): 121.0",
			/* [116] */ "Test — Height (cm): 173.0",
			/* [117] */ "Test — Blood Oxygen Saturation: 94.0",
			/* [118] */ "Test — Respiratory Rate: 29.0",
			/* [119] */ "Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [120] */ "Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [121] */ "Diagnosis — Fetishism: Patient counseled on lifestyle modifications.",
			/* [122] */ "Frequency — Every twenty-four hours: Every four hours",
			/* [123] */ "Assessment — Primary Diagnosis: Skin Infection",
			/* [124] */ "Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [125] */ "Test — Height (cm): 137.0",
			/* [126] */ "Test — Respiratory Rate: 28.0",
			/* [127] */ "Test — Systolic Blood Pressure: 134.0",
			/* [128] */ "Diagnosis: Asthma. Certainty: CONFIRMED. Rank: Primary",
			/* [129] */ "Test — Systolic Blood Pressure: 117.0",
			/* [130] */ "Test — Diastolic Blood Pressure: 70.0",
			/* [131] */ "Test — Pulse: 115.0",
			/* [132] */ "Test — Temperature (C): 40.1",
			/* [133] */ "Test — Height (cm): 186.0",
			/* [134] */ "Test — Respiratory Rate: 22.0",
			/* [135] */ "Assessment — Primary Diagnosis: Tuberculosis",
			/* [136] */ "Assessment — Primary Diagnosis: Tuberculosis",
			/* [137] */ "Test — Respiratory Rate: 28.0",
			/* [138] */ "Test — Diastolic Blood Pressure: 76.0",
			/* [139] */ "Test — Systolic Blood Pressure: 102.0",
			/* [140] */ "Units of Measure — Syringe): 237.0",
			/* [141] */ "Misc — Milligram per meter squared: 17.0",
			/* [142] */ "Diagnosis — Photoallergy: 8.27",
			/* [143] */ "Diagnosis — Fetishism: Routine checkup. No significant findings.",
			/* [144] */ "Test — Temperature (C): 39.3",
			/* [145] */ "Test — Diastolic Blood Pressure: 78.0",
			/* [146] */ "Test — Blood Oxygen Saturation: 88.0",
			/* [147] */ "Diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL. Rank: Primary",
			/* [148] */ "Test — Systolic Blood Pressure: 151.0",
			/* [149] */ "Test — Diastolic Blood Pressure: 53.0",
			/* [150] */ "Test — Pulse: 117.0",
			/* [151] */ "Test — Temperature (C): 39.4",
			/* [152] */ "Test — Blood Oxygen Saturation: 88.0",
			/* [153] */ "Test — Respiratory Rate: 15.0",
	};

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
	public void refineByKeywords_singleTermMatchesInLongQuery_shouldPassWhenMixed() {
		// 2 records match 2/6 terms, 4 records match only 1/6.
		// With minMatchCount=1, threshold = 1/6 = 0.167. All 6 pass.
		// Since 6/6 is NOT a proper subset, refinement doesn't activate —
		// returns all candidates. Gap detection is the primary noise filter.
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

		assertEquals(6, refined.size(),
				"All candidates have keyword matches — refinement should not discriminate");
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
		return simulatePipeline(semanticScores, keywordScores, keywordWeight, queryTermCount, 10);
	}

	private static int simulatePipeline(double[] semanticScores, double[] keywordScores,
			double keywordWeight, int queryTermCount, int topK) {
		double minScore = ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR / 2;
		double gapMultiplier = ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER;
		double minGap = ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP;

		// Gate check on raw semantic score — keywords must not rescue
		// records the embedding model considers irrelevant
		double maxSemanticScore = 0;
		for (double s : semanticScores) {
			if (s > maxSemanticScore) {
				maxSemanticScore = s;
			}
		}
		if (maxSemanticScore < ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR) {
			return 0;
		}

		List<LlmInferenceService.ScoredEmbedding> scored =
				new ArrayList<LlmInferenceService.ScoredEmbedding>();
		double bonusThreshold = (double) Math.min(2, queryTermCount) / queryTermCount;
		for (int i = 0; i < semanticScores.length; i++) {
			double keywordBonus = keywordScores[i] >= bonusThreshold ? keywordScores[i] : 0.0;
			double baseScore = semanticScores[i] + keywordWeight * keywordBonus;
			scored.add(makeScoredEmbedding(baseScore, keywordScores[i], semanticScores[i]));
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

		boolean refinementActivated = false;
		if (keywordWeight > 0) {
			List<LlmInferenceService.ScoredEmbedding> refined =
					LlmInferenceService.refineByKeywords(candidates, queryTermCount);
			refinementActivated = refined.size() < candidates.size();
			candidates = refined;
		}

		if (!refinementActivated) {
			double maxScore = scored.isEmpty() ? 0 : scored.get(0).score;
			double ratioFloor = maxScore * ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO;
			double effectiveFloor = Math.max(ratioFloor,
					ChartSearchAiConstants.ABSOLUTE_SIMILARITY_FLOOR);
			List<LlmInferenceService.ScoredEmbedding> strict =
					new ArrayList<LlmInferenceService.ScoredEmbedding>();
			for (LlmInferenceService.ScoredEmbedding se : candidates) {
				if (se.score >= effectiveFloor) {
					strict.add(se);
				}
			}
			candidates = strict;
			if (candidates.size() > topK) {
				candidates = candidates.subList(0, topK);
			}
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
	public void pipeline_genericQuery_shouldCapToTopK() {
		// Models: "tell me about this patient" — 2 terms after stopwords,
		// no keyword matches. Gap detection finds no gap in the smooth
		// distribution. With no keyword discrimination, the pipeline falls
		// back to the ratio-based floor (topScore*0.80 = 0.40*0.80 = 0.32)
		// + topK cap. Records scoring below 0.32 are filtered out.
		double[] semantic = new double[12];
		double[] keyword = new double[12];
		for (int i = 0; i < 12; i++) {
			semantic[i] = 0.40 - i * 0.01;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 2);

		assertTrue(result >= 8 && result <= 9,
				"Generic query should return records above ratio floor (0.32), got " + result);
	}

	@Test
	public void pipeline_incidentalKeywordMatches_shouldNotOverFilter() {
		// Models: "HB results values moving normal range" — 6 terms.
		// HB records match 2 terms ("hb" + one other) and get keyword bonus
		// (≥2 matches). Some vital signs match only "normal" (1 term = 0.17)
		// and get NO bonus. Gap detection separates the HB cluster (boosted
		// to 0.69-0.75) from the "normal" cluster (0.38-0.48).
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
		// the ratio floor (0.50*0.80=0.40) filters out the 2 lowest
		// records (0.38 and 0.36) that are < 80% of the top score.
		double[] semantic = new double[8];
		double[] keyword = new double[8];
		for (int i = 0; i < 8; i++) {
			semantic[i] = 0.50 - i * 0.02;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 2);

		assertEquals(6, result,
				"Ratio floor (topScore*0.80=0.40) should exclude records below 80% of top score");
	}

	@Test
	public void pipeline_keywordWeightZero_shouldDisableRefinement() {
		// When keywordWeight is 0, keyword refinement should be completely
		// disabled. The pipeline falls back to the ratio floor
		// (0.55*0.80=0.44) which keeps only the records within 80% of top.
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

		assertTrue(result >= 5 && result <= 6,
				"With keywordWeight=0, ratio floor should filter low scores, got " + result);
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
		// falls back to the ratio floor (0.40*0.80=0.32) which still returns
		// enough context for the LLM to say "no medications found."
		double[] semantic = new double[10];
		double[] keyword = new double[10];
		for (int i = 0; i < 10; i++) {
			semantic[i] = 0.40 - i * 0.01;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 3);

		assertTrue(result >= 8 && result <= 9,
				"No keyword matches should still return context via ratio floor, got " + result);
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
	public void refineByKeywords_longQuerySingleMatchShouldPass() {
		// For 6-term queries, threshold = 1/6 = 0.167. A record matching
		// 1/6 terms (0.17) SHOULD pass — any keyword relevance is sufficient
		// for refinement. Gap detection handles noise filtering.
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

		assertEquals(4, refined.size(),
				"6-term query: single-match records (0.17) should pass refinement; " +
				"zero-match records should not");
	}

	@Test
	public void pipeline_noKeywordMatchesSmoothDistribution_shouldFallBackToStrictFloorAndTopK() {
		// Models: "HB results over time" — the patient has NO HB results.
		// All 93 records are vital signs with smooth, similar semantic scores
		// and ZERO keyword matches. Gap detection won't trigger (smooth
		// distribution), keyword refinement won't activate (0 matches < 2).
		// The pipeline should fall back to the strict floor (0.25) + topK.
		// Scores range from 0.42 down to 0.236 (step 0.002).
		// Strict floor 0.25: records 0..85 pass (0.42 - 85*0.002 = 0.25),
		// records 86..92 don't (< 0.25). Then topK=10 caps to 10.
		int recordCount = 93;
		double[] semantic = new double[recordCount];
		double[] keyword = new double[recordCount];
		for (int i = 0; i < recordCount; i++) {
			semantic[i] = 0.42 - i * 0.002;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 7);

		assertEquals(10, result,
				"When no keywords match, should fall back to strict floor + topK cap");
	}

	@Test
	public void pipeline_keywordRefinementBypassesStrictFloorAndTopK() {
		// When keyword refinement successfully narrows to a subset, the strict
		// floor + topK fallback should NOT apply. Keywords identified the
		// relevant records — trust them even if count exceeds topK.
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
				"Keyword refinement should not be clipped by strict floor or topK");
	}

	@Test
	public void pipeline_focusedQueryStrictFloorFiltersNoise() {
		// Models: "does the patient have cancer?" — 1 term "cancer" after
		// stopwords. 2 Kaposi sarcoma records score high semantically (0.45),
		// 18 other records score lower (0.22-0.18). No keyword "cancer" in
		// any text. With strict floor=0.25, only the 2 cancer records pass.
		double[] semantic = new double[20];
		double[] keyword = new double[20];
		// 2 Kaposi sarcoma: high semantic, no keyword match
		semantic[0] = 0.45;
		semantic[1] = 0.43;
		keyword[0] = 0.0;
		keyword[1] = 0.0;
		// 18 unrelated vitals: below strict floor
		for (int i = 0; i < 18; i++) {
			semantic[2 + i] = 0.22 - i * 0.002;
			keyword[2 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 1);

		assertEquals(2, result,
				"Focused query should return only records above strict floor when keywords can't help");
	}

	@Test
	public void pipeline_focusedQueryRatioFloorExcludesMarginalRecords() {
		// Regression test: "does the patient have cancer?" returned 3 records
		// instead of 2. Photoallergy diagnosis scored 0.28 — above the
		// absolute floor (0.25) but below topScore * similarityRatio
		// (0.38 * 0.80 = 0.304). Scores are close enough that gap detection
		// cannot separate them (gap 0.07 < minGap 0.10), so the strict
		// fallback must use the ratio-based floor to exclude marginal records.
		double[] semantic = new double[20];
		double[] keyword = new double[20];
		// 2 Kaposi sarcoma: moderately high semantic (close together)
		semantic[0] = 0.38;
		semantic[1] = 0.35;
		// 1 Photoallergy: above absolute floor (0.25) but below ratio floor
		semantic[2] = 0.28;
		// 17 other records: below absolute floor
		for (int i = 0; i < 17; i++) {
			semantic[3 + i] = 0.22 - i * 0.002;
		}
		// No keyword matches for "cancer" in any record
		Arrays.fill(keyword, 0.0);

		int result = simulatePipeline(semantic, keyword, 0.3, 1);

		assertEquals(2, result,
				"Ratio-based floor (topScore*0.80=0.304) should exclude Photoallergy at 0.28");
	}

	@Test
	public void pipeline_keywordRefinementCanExceedTopK() {
		// When keyword refinement identifies a relevant subset that exceeds
		// topK (10), all keyword-matched records should still be returned.
		// Models: 15 conditions out of 30 total records.
		double[] semantic = new double[30];
		double[] keyword = new double[30];
		for (int i = 0; i < 15; i++) {
			semantic[i] = 0.55 - i * 0.02;
			keyword[i] = 2.0 / 6;
		}
		for (int i = 0; i < 15; i++) {
			semantic[15 + i] = 0.45 - i * 0.02;
			keyword[15 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 6);

		assertEquals(15, result,
				"Keyword refinement should return all 15 conditions even though topK is 10");
	}

	@Test
	public void pipeline_allergyQueryShouldReturnOnlyAllergyRecords() {
		// Regression test: "What is the patient allergic to?" returned 10
		// records instead of 2. The query term "allergic" couldn't match
		// "allergy" in record text, so keyword refinement didn't activate.
		// With stem matching, "allergic" → stem "allerg" → matches "allergy",
		// enabling keyword refinement to filter to just allergy records.
		double[] semantic = new double[10];
		double[] keyword = new double[10];
		// 2 allergy records: "allergic" matches via stem "allerg" → "allergy"
		semantic[0] = 0.42;
		keyword[0] = 1.0; // "allergic" matches (1/1 term)
		semantic[1] = 0.38;
		keyword[1] = 1.0;
		// 8 diagnosis records: no keyword match for "allergic"
		for (int i = 0; i < 8; i++) {
			semantic[2 + i] = 0.36 - i * 0.01;
			keyword[2 + i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 1);

		assertEquals(2, result,
				"Allergy query should return only allergy records via stem-based keyword refinement");
	}

	@Test
	public void pipeline_cd4CountQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "What is the current CD4 Count?" → expected: exactly 2 CD4 records.

		// Step 1: Normalize query and extract terms
		String normalized = LlmInferenceService.stripQueryStopwords("What is the current CD4 Count?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cd4", "count" }, queryTerms,
				"Should have 'cd4' and 'count' after stopword removal");

		// Step 2: Representative records with actual text (prefix + content)
		String[] recordTexts = {
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — CD4 Count: 1191.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Clinical observation: Test — Height (cm): 131.0",
				"Clinical observation: Test — Respiratory Rate: 18.0",
				"Clinical observation: Test — Systolic Blood Pressure: 97.0",
				"Clinical observation: Test — Diastolic Blood Pressure: 99.0",
				"Clinical observation: Test — Blood Oxygen Saturation: 88.0",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
		};

		// Step 3: Compute ACTUAL keyword scores
		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// Verify keyword matching: only CD4 Count records match both terms
		assertEquals(1.0, keyword[0], 0.001, "CD4 Count #1 should match both 'cd4' and 'count'");
		assertEquals(1.0, keyword[1], 0.001, "CD4 Count #2 should match both 'cd4' and 'count'");
		for (int i = 2; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-CD4 record should not match: " + recordTexts[i].substring(0, 30));
		}

		// Step 4: Simulate pipeline with realistic semantic scores
		double[] semantic = {
				0.58, // CD4 Count: 988.0 (high - exact concept match)
				0.54, // CD4 Count: 1191.0
				0.30, // Pulse (generic test observation)
				0.28, // Temperature
				0.27, // Weight
				0.26, // Height
				0.25, // Respiratory Rate
				0.24, // Systolic BP
				0.23, // Diastolic BP
				0.22, // Blood Oxygen Saturation
				0.20, // Kaposi sarcoma
				0.18, // Tuberculosis
				0.17, // Hypertension
				0.15, // Allergy
				0.14, // Drug order
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'What is the current CD4 Count?' should return exactly 2 CD4 records");
	}

	@Test
	public void pipeline_latestCd4CountQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "What is the latest CD4 Count?" → expected: exactly 2 CD4 records.
		// "latest" is a stopword, so this should reduce to same terms as "current CD4 Count".

		String normalized = LlmInferenceService.stripQueryStopwords("What is the latest CD4 Count?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cd4", "count" }, queryTerms,
				"'latest' should be stripped as stopword, leaving 'cd4' and 'count'");

		String[] recordTexts = {
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — CD4 Count: 1191.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Clinical observation: Test — Height (cm): 131.0",
				"Clinical observation: Test — Respiratory Rate: 18.0",
				"Clinical observation: Test — Systolic Blood Pressure: 97.0",
				"Clinical observation: Test — Diastolic Blood Pressure: 99.0",
				"Clinical observation: Test — Blood Oxygen Saturation: 88.0",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "CD4 Count #1 should match");
		assertEquals(1.0, keyword[1], 0.001, "CD4 Count #2 should match");
		for (int i = 2; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-CD4 record should not match: " + recordTexts[i].substring(0, 30));
		}

		double[] semantic = {
				0.58, 0.54, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22,
				0.20, 0.18, 0.17, 0.15, 0.14,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'What is the latest CD4 Count?' should return exactly 2 CD4 records");
	}

	@Test
	public void pipeline_cancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "does the patient have cancer?" → expected: exactly 2 Kaposi sarcoma records.
		// "cancer" doesn't appear literally in any record text (records say "Kaposi sarcoma"),
		// so keyword refinement won't activate. This is a PURELY SEMANTIC query where
		// gap detection + ratio floor must isolate the 2 cancer-related records.

		// Step 1: Normalize query and extract terms
		String normalized = LlmInferenceService.stripQueryStopwords("does the patient have cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms,
				"Only 'cancer' should remain after stopword removal");
		// Full query preserved for embedding since <2 content words
		assertTrue(normalized.contains("does") && normalized.contains("patient"),
				"Full query should be preserved for embedding context");

		// Step 2: Representative records with actual text (prefix + content)
		String[] recordTexts = {
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.5",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		// Step 3: Verify no keyword matches (cancer ≠ Kaposi sarcoma literally)
		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"No literal 'cancer' match expected in: " + recordTexts[i].substring(0, 30));
		}

		// Step 4: Simulate pipeline with realistic semantic scores.
		// The embedding model understands Kaposi sarcoma IS cancer semantically.
		// Gap detection should cut between the 2 cancer records and the rest.
		double[] semantic = {
				0.45, // Kaposi sarcoma oral (IS cancer → high semantic match)
				0.42, // Kaposi sarcoma oral (IS cancer)
				0.28, // Photoallergy (medical but not cancer)
				0.27, // Photoallergy
				0.26, // Tuberculosis (disease, not cancer)
				0.25, // Hypertension
				0.24, // HIV Disease
				0.23, // Skin Infection
				0.22, // Gastroenteritis
				0.21, // Diabetes Mellitus
				0.19, // CD4 Count
				0.17, // Pulse
				0.16, // Allergy
				0.15, // Drug order
				0.18, // Assessment
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'does the patient have cancer?' should return exactly 2 Kaposi sarcoma records "
						+ "via gap detection and ratio floor (no keyword help available)");
	}

	@Test
	public void pipeline_doesHeHaveCancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Variant of cancer query using pronoun "he" instead of "the patient".
		// "does", "he", "have" are all stopwords → only "cancer" remains (<2 content words)
		// → full query preserved for embedding, same pipeline path as "does the patient have cancer?"

		String normalized = LlmInferenceService.stripQueryStopwords("does he have cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms,
				"Only 'cancer' should remain after stopword removal");
		assertTrue(normalized.contains("does") && normalized.contains("he"),
				"Full query should be preserved for embedding context");

		String[] recordTexts = {
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.5",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"No literal 'cancer' match expected");
		}

		double[] semantic = {
				0.45, 0.42, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21,
				0.19, 0.17, 0.16, 0.15, 0.18,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'does he have cancer?' should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void pipeline_anyCancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Variant: "any cancer?" — "any" is a stopword → only "cancer" remains
		// (<2 content words → full query "any cancer" preserved for embedding).
		// Purely semantic query, same path as "does the patient have cancer?"

		String normalized = LlmInferenceService.stripQueryStopwords("any cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms,
				"Only 'cancer' should remain after stopword removal");

		String[] recordTexts = {
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.5",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "No literal 'cancer' match expected");
		}

		double[] semantic = {
				0.45, 0.42, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21,
				0.19, 0.17, 0.16, 0.15, 0.18,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'any cancer?' should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void pipeline_historyOfCancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any history of cancer?" → terms: ["history", "cancer"] (N=2).
		// Only record [4] ("Immunization history") matches "history" (1/2 = 0.5).
		// No record contains literal "cancer". Kaposi sarcoma ([12],[89]) is
		// semantically closest to "cancer" but has zero keyword match.
		// With keyword bonus threshold (≥2 terms for N=2), NO record gets bonus.
		// Gap detection separates the 2 Kaposi sarcoma records from the rest.

		String normalized = LlmInferenceService.stripQueryStopwords("any history of cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "history", "cancer" }, queryTerms,
				"'history' and 'cancer' should remain after stopword removal");

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// Only "Immunization history" matches "history" (1/2 terms = 0.5)
		assertEquals(0.5, keyword[3], 0.001,
				"[4] 'history' should match 'Immunization history'");
		// Kaposi sarcoma records have zero keyword match
		assertEquals(0.0, keyword[11], 0.001,
				"[12] Kaposi sarcoma should not match 'history' or 'cancer'");
		assertEquals(0.0, keyword[88], 0.001,
				"[89] Kaposi sarcoma should not match 'history' or 'cancer'");
		// All other records should also have zero keyword match
		for (int i = 0; i < keyword.length; i++) {
			if (i == 3) continue; // skip immunization history
			assertEquals(0.0, keyword[i], 0.001,
					"Record [" + (i + 1) + "] should not match 'history' or 'cancer'");
		}

		// Semantic scores: Kaposi sarcoma is semantically closest to "cancer".
		// Disease/condition records have moderate similarity. Vitals, meds,
		// frequencies, etc. have low similarity. The key gap is between the
		// 2 Kaposi sarcoma records (0.38-0.40) and all others (≤0.22).
		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.10);

		// Cancer-related — clearly the highest semantic similarity
		semantic[11] = 0.40; // [12] Kaposi sarcoma oral: 3.91
		semantic[88] = 0.38; // [89] Kaposi sarcoma oral: 3.5

		// Disease/condition records — moderate semantic similarity to "cancer"
		semantic[3]   = 0.22; // [4] Immunization history (has "history" concept)
		semantic[39]  = 0.21; // [40] HIV Disease
		semantic[71]  = 0.20; // [72] HIV Disease
		semantic[110] = 0.20; // [111] HIV Disease
		semantic[7]   = 0.19; // [8] Condition: TB
		semantic[52]  = 0.19; // [53] TB
		semantic[69]  = 0.19; // [70] HIV Disease
		semantic[54]  = 0.18; // [55] Condition: Hypertension
		semantic[40]  = 0.18; // [41] Assessment: HIV Disease
		semantic[68]  = 0.18; // [69] Assessment: HIV Disease
		semantic[12]  = 0.17; // [13] Assessment: TB
		semantic[62]  = 0.17; // [63] Assessment: Pneumonia
		semantic[66]  = 0.17; // [67] Malaria
		semantic[92]  = 0.17; // [93] Hypertension
		semantic[134] = 0.17; // [135] Assessment: TB
		semantic[135] = 0.17; // [136] Assessment: TB
		semantic[21]  = 0.16; // [22] Gastroenteritis
		semantic[61]  = 0.16; // [62] Skin Infection
		semantic[72]  = 0.16; // [73] Anemia
		semantic[127] = 0.16; // [128] Asthma
		semantic[146] = 0.16; // [147] Diabetes
		semantic[29]  = 0.16; // [30] Assessment: Anemia
		semantic[55]  = 0.16; // [56] Assessment: Anemia
		semantic[105] = 0.16; // [106] Assessment: Malaria
		semantic[49]  = 0.15; // [50] Assessment: Diabetes
		semantic[90]  = 0.15; // [91] Assessment: UTI
		semantic[118] = 0.15; // [119] Assessment: UTI
		semantic[89]  = 0.14; // [90] Photoallergy
		semantic[141] = 0.14; // [142] Photoallergy
		semantic[8]   = 0.14; // [9] CD4 Count
		semantic[85]  = 0.14; // [86] CD4 Count

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'any history of cancer?' should return exactly 2 Kaposi sarcoma records");
	}

	@Test
	public void pipeline_fractureQuery_realData_shouldReturnNoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "any fracture?" → expected: 0 records. No record in the dataset
		// mentions fracture. All semantic scores should be low enough that the
		// gate check (topScore < ABSOLUTE_SIMILARITY_FLOOR 0.25) returns empty.

		String normalized = LlmInferenceService.stripQueryStopwords("any fracture?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "fracture" }, queryTerms,
				"Only 'fracture' should remain after stopword removal");

		String[] recordTexts = {
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Clinical observation: Test — Height (cm): 131.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "No record should match 'fracture'");
		}

		// All semantic scores below the ABSOLUTE_SIMILARITY_FLOOR (0.25)
		// because nothing in this chart is related to fractures
		double[] semantic = {
				0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09,
				0.08, 0.12,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(0, result,
				"Query 'any fracture?' should return 0 records — nothing in the chart is related");
	}

	@Test
	public void pipeline_familyHistoryOfCancerQuery_realData_shouldReturnNoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "Any family history of cancer?" → expected: 0 records.
		// No record in the dataset is about family history of cancer.
		// Individual keywords ("family", "history") coincidentally match
		// unrelated records ("family planning", "immunization history"),
		// but the gate check on raw SEMANTIC scores (all < 0.25) ensures
		// keyword bonus cannot rescue irrelevant records.

		String normalized = LlmInferenceService.stripQueryStopwords(
				"Any family history of cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "family", "history", "cancer" }, queryTerms,
				"Should have 'family', 'history', 'cancer' after stopword removal");

		String[] recordTexts = {
				"Clinical observation: Finding — Immunization history: Immunizations: "
						+ "Polio vaccination, oral; Vaccination date: 2026-03-18",
				"Clinical observation: Assessment — Method of family planning: Condoms",
				"Clinical observation: Assessment — Method of family planning: Diaphragm",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Medical condition: Hypertension. Status: ACTIVE",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// Verify coincidental keyword matches exist (this is the problem we're guarding against)
		assertTrue(keyword[0] > 0, "Immunization history matches 'history' — coincidental");
		assertTrue(keyword[1] > 0, "Family planning matches 'family' — coincidental");
		assertTrue(keyword[2] > 0, "Family planning matches 'family' — coincidental");
		for (int i = 3; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "Should not match");
		}

		// All semantic scores below ABSOLUTE_SIMILARITY_FLOOR (0.25):
		// no record is about family history of cancer
		double[] semantic = {
				0.22, // Immunization history (model sees "history" but unrelated)
				0.20, // Family planning: Condoms (model sees "family" but unrelated)
				0.19, // Family planning: Diaphragm
				0.24, // Kaposi sarcoma (IS cancer, but question is about family history)
				0.18, // TB
				0.16, // HIV
				0.14, // CD4 Count
				0.12, // Pulse
				0.11, // Allergy
				0.10, // Drug order
				0.09, // Weight
				0.15, // Hypertension
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(0, result,
				"Query 'Any family history of cancer?' should return 0 records — "
						+ "keyword coincidences ('family' in family planning, 'history' in "
						+ "immunization history) must not rescue low semantic scores");
	}

	@Test
	public void pipeline_medicationsQuery_realData_shouldReturnFourRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "is the patient on any medications?" → expected: 4 records.
		// All words except "medications" are stopwords → <2 content words → full query
		// preserved for embedding. "medications" → plural strip → "medication" matches:
		//   - 2 drug orders (prefix "Medication prescription:")
		//   - 2 visit notes containing "Medication adjusted."
		// All 4 are clinically relevant: drug orders show WHAT medications,
		// visit notes show WHEN medication changes happened.

		String normalized = LlmInferenceService.stripQueryStopwords(
				"is the patient on any medications?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "medications" }, queryTerms,
				"Only 'medications' should remain after stopword removal");
		assertTrue(normalized.contains("patient"),
				"Full query should be preserved for embedding context");

		String[] recordTexts = {
				// 2 drug orders (actual medications)
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0 Tablet) "
						+ "Intravenous Every six hours. Duration: 5 Days. Action: REVISE",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0 Tablet) "
						+ "Intravenous Thrice daily. Duration: 5 Days. Action: NEW. Stopped: 2026-03-18",
				// 2 visit notes mentioning medication adjustment
				"Clinical diagnosis: Fetishism: Chronic disease management visit. Medication adjusted.",
				"Clinical diagnosis: Fetishism: Chronic disease management visit. Medication adjusted.",
				// Non-matching records
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical diagnosis: Fetishism: Routine checkup. No significant findings.",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// Drug orders match via prefix "Medication prescription"
		assertEquals(1.0, keyword[0], 0.001, "Drug order #1 should match 'medication'");
		assertEquals(1.0, keyword[1], 0.001, "Drug order #2 should match 'medication'");
		// Visit notes match via "Medication adjusted" in text
		assertEquals(1.0, keyword[2], 0.001, "Visit note #1 should match 'medication'");
		assertEquals(1.0, keyword[3], 0.001, "Visit note #2 should match 'medication'");
		for (int i = 4; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-medication record should not match: " + recordTexts[i].substring(0, 30));
		}

		double[] semantic = {
				0.55, // Drug order: Azithromycin REVISE (highest - direct medication)
				0.45, // Drug order: Azithromycin NEW
				0.40, // Visit note: Medication adjusted
				0.39, // Visit note: Medication adjusted
				0.25, // TB condition
				0.23, // Hypertension
				0.22, // HIV
				0.21, // Diabetes
				0.18, // CD4 Count
				0.16, // Pulse
				0.15, // Temperature
				0.14, // Weight
				0.13, // Allergy
				0.17, // Assessment
				0.20, // Routine checkup
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(4, result,
				"Query 'is the patient on any medications?' should return 4 records: "
						+ "2 drug orders + 2 visit notes mentioning medication adjustment");
	}

	@Test
	public void pipeline_knownConditionsQuery_realData_shouldReturnExactlyTwoRecords() {
		// Variant: "any known conditions?" — "any" and "known" are stopwords →
		// only "conditions" remains, same as "any conditions" query.

		String normalized = LlmInferenceService.stripQueryStopwords("any known conditions?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "conditions" }, queryTerms,
				"'known' should be stripped as stopword, leaving only 'conditions'");

		String[] recordTexts = {
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical diagnosis: Fetishism: Routine checkup. No significant findings.",
				"Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "TB condition should match");
		assertEquals(1.0, keyword[1], 0.001, "Hypertension condition should match");
		for (int i = 2; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "Non-condition record should not match");
		}

		double[] semantic = {
				0.50, 0.48, 0.35, 0.33, 0.32, 0.28, 0.30, 0.20, 0.18, 0.17,
				0.16, 0.15, 0.14, 0.22, 0.29,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'any known conditions?' should return exactly 2 condition records");
	}

	@Test
	public void pipeline_anemicQuery_realData_shouldReturnExactlyThreeRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "is the patient anemic?" → expected: exactly 3 anemia records.
		// "is", "the", "patient" are stopwords → only "anemic" remains (<2 content words
		// → full query preserved for embedding). "anemic" is 6 chars (<7, no stem matching)
		// and isn't a substring of "anemia", so keyword matching cannot help.
		// This is a PURELY SEMANTIC query — gap detection + ratio floor must isolate
		// the 3 anemia-related records.

		String normalized = LlmInferenceService.stripQueryStopwords("is the patient anemic?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "anemic" }, queryTerms,
				"Only 'anemic' should remain after stopword removal");
		assertTrue(normalized.contains("patient"),
				"Full query should be preserved for embedding context");

		String[] recordTexts = {
				// 3 anemia records (expected results)
				"Clinical observation: Assessment — Primary Diagnosis: Anemia",
				"Clinical observation: Assessment — Primary Diagnosis: Anemia",
				"Clinical diagnosis: Anemia. Certainty: PROVISIONAL. Rank: Secondary",
				// Non-matching records
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Clinical observation: Test — Height (cm): 131.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		// Verify no keyword matches ("anemic" ≠ "anemia" for keyword matching)
		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"No keyword match expected — 'anemic' can't match 'anemia' at 6 chars");
		}

		// Semantic scores: anemia records score high, clear gap to other records
		double[] semantic = {
				0.48, // Assessment: Anemia (semantically close to "anemic")
				0.46, // Assessment: Anemia
				0.43, // Diagnosis: Anemia
				0.28, // HIV Disease
				0.26, // Gastroenteritis
				0.25, // Tuberculosis
				0.24, // Hypertension
				0.19, // CD4 Count
				0.17, // Pulse
				0.16, // Weight
				0.15, // Height
				0.14, // Allergy
				0.13, // Drug order
				0.20, // Kaposi sarcoma
				0.22, // Assessment: TB
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(3, result,
				"Query 'is the patient anemic?' should return exactly 3 anemia records "
						+ "via gap detection and ratio floor (no keyword help — 'anemic' ≠ 'anemia')");
	}

	@Test
	public void pipeline_conditionsQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "any conditions" → expected: exactly 2 condition records.
		// "any" is a stopword → "conditions" remains (<2 content words → full query preserved).
		// Plural strip "conditions" → "condition" matches the "Medical condition:" prefix.

		String normalized = LlmInferenceService.stripQueryStopwords("any conditions");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "conditions" }, queryTerms,
				"Only 'conditions' should remain after stopword removal");

		String[] recordTexts = {
				// 2 condition records (expected results)
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				// Non-matching records
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical diagnosis: Fetishism: Routine checkup. No significant findings.",
				"Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "TB condition should match via plural strip");
		assertEquals(1.0, keyword[1], 0.001, "Hypertension condition should match via plural strip");
		for (int i = 2; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-condition record should not match: " + recordTexts[i].substring(0, 30));
		}

		double[] semantic = {
				0.50, // TB condition
				0.48, // Hypertension condition
				0.35, // HIV diagnosis
				0.33, // Gastroenteritis
				0.32, // Skin Infection
				0.28, // Kaposi sarcoma
				0.30, // Assessment: TB
				0.20, // CD4 Count
				0.18, // Pulse
				0.17, // Temperature
				0.16, // Weight
				0.15, // Allergy
				0.14, // Drug order
				0.22, // Routine checkup
				0.29, // Assessment: HIV
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'any conditions' should return exactly 2 condition records");
	}

	@Test
	public void pipeline_stdQuery_realData_shouldReturnExactlySixRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "any sexually transmitted disease?" → expected: 6 HIV Disease records.
		// None of the query terms are stopwords → ["sexually", "transmitted", "disease"].
		// Only "disease" matches literally in "HIV Disease" records → keyword score 1/3.
		// Threshold for 3 terms: min(2, max(1, 3/3)) / 3 = 1/3 → passes.
		// Keyword refinement narrows to the 6 HIV Disease records.

		String normalized = LlmInferenceService.stripQueryStopwords("any sexually transmitted disease?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "sexually", "transmitted", "disease" }, queryTerms,
				"All three clinical terms should remain after stopword removal");

		String[] recordTexts = {
				// 6 HIV Disease records (expected results)
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				"Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
				"Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
				"Clinical diagnosis: HIV Disease. Certainty: PROVISIONAL. Rank: Primary",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
				// Non-matching records
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Clinical diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Patient allergy: Beef (food allergen). Severity: Severe",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// HIV Disease records match "disease" → 1/3
		double expectedKw = 1.0 / 3.0;
		for (int i = 0; i < 6; i++) {
			assertEquals(expectedKw, keyword[i], 0.001,
					"HIV Disease record should match 'disease' (1/3 terms)");
		}
		for (int i = 6; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-HIV record should not match: " + recordTexts[i].substring(0, 30));
		}

		// Semantic scores: HIV Disease records high (STD is semantically related),
		// other diseases moderate, vitals/allergies low
		double[] semantic = {
				0.50, // HIV Disease diagnosis
				0.47, // Assessment: HIV Disease
				0.46, // Assessment: HIV Disease
				0.48, // HIV Disease provisional
				0.49, // HIV Disease confirmed
				0.45, // HIV Disease confirmed
				0.30, // TB (infectious disease but not STD)
				0.22, // Hypertension
				0.21, // Diabetes
				0.20, // Gastroenteritis
				0.19, // Skin Infection
				0.18, // Kaposi sarcoma
				0.17, // CD4 Count
				0.15, // Pulse
				0.14, // Allergy
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(6, result,
				"Query 'any sexually transmitted disease?' should return exactly 6 HIV Disease records");
	}

	@Test
	public void pipeline_whatIsPatientAllergicToQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "What is the patient allergic to?" → expected: exactly 2 allergy records.
		// "what", "is", "the", "patient", "to" are stopwords → only "allergic" remains
		// (<2 content words → full query preserved for embedding).
		// "allergic" (8 chars ≥ 7) → stem "allerg" → word-prefix matches "allergy"/"allergen"
		// but NOT "Photoallergy" (starts with "photo").

		String normalized = LlmInferenceService.stripQueryStopwords("What is the patient allergic to?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "allergic" }, queryTerms,
				"Only 'allergic' should remain after stopword removal");
		assertTrue(normalized.contains("patient") && normalized.contains("allergic"),
				"Full query should be preserved for embedding context");

		String[] recordTexts = {
				"Patient allergy: Beef (food allergen). Severity: Severe. "
						+ "Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
				"Patient allergy: Fomepizole (drug allergen)",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "Beef allergy should match 'allergic' via stem 'allerg'");
		assertEquals(1.0, keyword[1], 0.001, "Fomepizole allergy should match via stem 'allerg'");
		assertEquals(0.0, keyword[2], 0.001, "Photoallergy must NOT match (compound word)");
		assertEquals(0.0, keyword[3], 0.001, "Photoallergy must NOT match (compound word)");
		for (int i = 4; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "Non-allergy record should not match");
		}

		double[] semantic = {
				0.55, 0.50, 0.35, 0.33, 0.25, 0.27, 0.24, 0.30, 0.29, 0.26,
				0.20, 0.18, 0.25, 0.17, 0.16,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'What is the patient allergic to?' should return exactly 2 allergy records");
	}

	@Test
	public void pipeline_coughQuery_realData_shouldReturnExactlyOneRecord() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "any cough?" → expected: exactly 1 record mentioning cough.
		// "any" is a stopword, "cough" is 5 chars (<7, no stem matching).
		// Only record #51 mentions "cough": "New complaint of persistent cough for 2 weeks."

		String normalized = LlmInferenceService.stripQueryStopwords("any cough?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cough" }, queryTerms,
				"Only 'cough' should remain after stopword removal");

		String[] recordTexts = {
				"Clinical diagnosis: Fetishism: New complaint of persistent cough for 2 weeks.",
				"Clinical diagnosis: Fetishism: Annual physical examination. Labs ordered.",
				"Clinical diagnosis: Fetishism: Patient stable on current regimen.",
				"Clinical diagnosis: Fetishism: Presenting with fever and body aches for 3 days.",
				"Clinical diagnosis: Fetishism: Patient presents with mild symptoms. Advised rest and fluids.",
				"Clinical diagnosis: Fetishism: Routine checkup. No significant findings.",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Clinical diagnosis: Gastroenteritis. Certainty: PROVISIONAL",
				"Clinical observation: Test — Respiratory Rate: 18.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "Cough record should match 'cough' exactly");
		for (int i = 1; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Non-cough record should not match: " + recordTexts[i].substring(0, 30));
		}

		double[] semantic = {
				0.48, // cough complaint (direct match)
				0.30, // annual physical
				0.28, // stable on regimen
				0.32, // fever and body aches (respiratory-adjacent)
				0.29, // mild symptoms
				0.25, // routine checkup
				0.33, // TB (associated with cough)
				0.24, // HIV
				0.22, // Gastroenteritis
				0.26, // Respiratory Rate (respiratory-adjacent)
				0.18, // Pulse
				0.17, // Temperature
				0.15, // Allergy
				0.14, // Drug order
				0.20, // Assessment
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(1, result,
				"Query 'any cough?' should return exactly 1 cough record");
	}

	@Test
	public void pipeline_doesPatientHaveAllergiesQuery_realData_shouldReturnExactlyTwoRecords() {
		// Variant: "does the patient have any allergies?" — all words except "allergies"
		// are stopwords → <2 content words → full query preserved for embedding.
		// Keyword matching uses ["allergies"] via stem "allerg" → matches allergy records,
		// excludes Photoallergy (compound word, word-prefix check).

		String normalized = LlmInferenceService.stripQueryStopwords("does the patient have any allergies?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "allergies" }, queryTerms,
				"Only 'allergies' should remain after stopword removal");
		assertTrue(normalized.contains("does") && normalized.contains("patient"),
				"Full query should be preserved for embedding context");

		String[] recordTexts = {
				"Patient allergy: Beef (food allergen). Severity: Severe. "
						+ "Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
				"Patient allergy: Fomepizole (drug allergen)",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		assertEquals(1.0, keyword[0], 0.001, "Beef allergy should match via stem 'allerg'");
		assertEquals(1.0, keyword[1], 0.001, "Fomepizole allergy should match via stem 'allerg'");
		assertEquals(0.0, keyword[2], 0.001, "Photoallergy must NOT match (compound word)");
		assertEquals(0.0, keyword[3], 0.001, "Photoallergy must NOT match (compound word)");
		for (int i = 4; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001, "Non-allergy record should not match");
		}

		double[] semantic = {
				0.55, 0.50, 0.35, 0.33, 0.25, 0.27, 0.24, 0.30, 0.29, 0.26,
				0.20, 0.18, 0.25, 0.17, 0.16,
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'does the patient have any allergies?' should return exactly 2 allergy records");
	}

	@Test
	public void pipeline_allergyQuery_realData_shouldReturnExactlyTwoRecords() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "any allergies?" → expected: exactly 2 allergy records.
		// Records #5 and #54 from the full chart. Must NOT include the
		// 2 Photoallergy diagnoses (#90 and #142) even though they contain
		// "allerg" as a substring inside the compound word "Photoallergy".

		// Step 1: Normalize query and extract terms
		String normalized = LlmInferenceService.stripQueryStopwords("any allergies?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertEquals(1, queryTerms.length, "Should have 1 query term after stopword removal");
		assertEquals("allergies", queryTerms[0]);

		// Step 2: Representative records with actual text (prefix + content)
		// as used for keyword matching
		String[] recordTexts = {
				"Patient allergy: Beef (food allergen). Severity: Severe. "
						+ "Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
				"Patient allergy: Fomepizole (drug allergen)",
				"Clinical diagnosis: Photoallergy: 9.93",
				"Clinical diagnosis: Photoallergy: 8.27",
				"Clinical diagnosis: Kaposi sarcoma oral: 3.91",
				"Clinical diagnosis: Skin Infection. Certainty: CONFIRMED",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Medical condition: Tuberculosis. Status: ACTIVE",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Weight (kg): 94.0",
		};

		// Step 3: Compute ACTUAL keyword scores using our matching logic
		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// Verify keyword matching correctness
		assertEquals(1.0, keyword[0], 0.001,
				"Beef allergy should match 'allergies' via stem 'allerg'");
		assertEquals(1.0, keyword[1], 0.001,
				"Fomepizole allergy should match 'allergies' via stem 'allerg'");
		assertEquals(0.0, keyword[2], 0.001,
				"Photoallergy diagnosis must NOT match — 'allerg' is inside compound word");
		assertEquals(0.0, keyword[3], 0.001,
				"Photoallergy diagnosis must NOT match — 'allerg' is inside compound word");
		for (int i = 4; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Record should not match: " + recordTexts[i].substring(0, 30));
		}

		// Step 4: Simulate pipeline with realistic semantic scores
		double[] semantic = {
				0.55,  // Allergy: Beef (highest for allergy query)
				0.50,  // Allergy: Fomepizole
				0.35,  // Photoallergy (semantic proximity to "allergy")
				0.33,  // Photoallergy
				0.25,  // Kaposi sarcoma
				0.27,  // Skin Infection
				0.24,  // HIV
				0.30,  // TB condition
				0.29,  // Hypertension
				0.26,  // Drug order
				0.20,  // CD4
				0.18,  // Pulse
				0.25,  // Assessment
				0.17,  // Temperature
				0.16,  // Weight
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(2, result,
				"Query 'any allergies?' should return exactly 2 allergy records from real patient data");
	}

	@Test
	public void pipeline_vitalsTrendQuery_realData_shouldReturnBpWeightAndTemperature() {
		// Real patient dataset: 16-year-old Male with 153 records.
		// Query: "How have this patient's blood pressure, weight, and
		// temperature trended across their last 7 visits?"
		// Multi-concept query: should return blood pressure, weight, AND
		// temperature records — not just blood pressure (which gets 2/N
		// keyword matches vs 1/N for the others).
		// Before fix: only 4 BP records returned (keyword bonus created
		// artificial gap, refinement threshold rejected 1-term matches).

		String normalized = LlmInferenceService.stripQueryStopwords(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		// "patient's" → possessive stripped → "patient" → stopword → removed
		// "how", "have", "this", "and", "across", "their" → stopwords
		// "7" → single char → filtered by extractQueryTerms
		assertTrue(queryTerms.length >= 4,
				"Should have at least blood/pressure/weight/temperature as terms");
		List<String> termList = Arrays.asList(queryTerms);
		assertTrue(termList.contains("blood"), "Should contain 'blood'");
		assertTrue(termList.contains("pressure"), "Should contain 'pressure'");
		assertTrue(termList.contains("weight"), "Should contain 'weight'");
		assertTrue(termList.contains("temperature"), "Should contain 'temperature'");

		String[] recordTexts = {
				// 4 Blood Pressure records (systolic + diastolic)
				"Clinical observation: Test — Systolic Blood Pressure: 97.0",
				"Clinical observation: Test — Diastolic Blood Pressure: 99.0",
				"Clinical observation: Test — Systolic Blood Pressure: 122.0",
				"Clinical observation: Test — Diastolic Blood Pressure: 71.0",
				// 2 Weight records
				"Clinical observation: Test — Weight (kg): 94.0",
				"Clinical observation: Test — Weight (kg): 88.0",
				// 2 Temperature records
				"Clinical observation: Test — Temperature (C): 36.7",
				"Clinical observation: Test — Temperature (C): 37.2",
				// 7 distractors — other vitals and non-vitals
				"Clinical observation: Test — Pulse: 95.0",
				"Clinical observation: Test — CD4 Count: 988.0",
				"Clinical observation: Test — Height (cm): 131.0",
				"Clinical diagnosis: HIV Disease. Certainty: CONFIRMED",
				"Medical condition: Hypertension. Status: ACTIVE",
				"Patient allergy: Beef (food allergen). Severity: Severe",
				"Medication prescription: Drug order: Azithromycin. Dose: 2.0",
		};

		double[] keyword = new double[recordTexts.length];
		for (int i = 0; i < recordTexts.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, recordTexts[i]);
		}

		// Blood pressure matches "blood" + "pressure" (2 terms)
		double bpScore = keyword[0];
		assertTrue(bpScore > 0, "BP should match 'blood' + 'pressure'");
		assertEquals(bpScore, keyword[1], 0.001, "All BP records should have same keyword score");
		assertEquals(bpScore, keyword[2], 0.001);
		assertEquals(bpScore, keyword[3], 0.001);
		// Weight matches "weight" (1 term)
		double weightScore = keyword[4];
		assertTrue(weightScore > 0, "Weight should match 'weight'");
		assertTrue(bpScore > weightScore, "BP should have higher keyword score than weight");
		assertEquals(weightScore, keyword[5], 0.001);
		// Temperature matches "temperature" (1 term)
		double tempScore = keyword[6];
		assertTrue(tempScore > 0, "Temperature should match 'temperature'");
		assertEquals(tempScore, keyword[7], 0.001);
		// Distractors: no content-term matches
		for (int i = 8; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"Distractor should not match: " + recordTexts[i].substring(0, 30));
		}

		// Semantic scores: all vitals-related records score well for a
		// trending-vitals query. BP, weight, temperature cluster together.
		// Pulse and Height are related vitals but NOT asked for.
		double[] semantic = {
				0.55, 0.54, 0.53, 0.52,  // BP records (4)
				0.50, 0.49,               // Weight records (2)
				0.48, 0.47,               // Temperature records (2)
				0.40,                     // Pulse (related but not asked)
				0.30,                     // CD4
				0.38,                     // Height (related but not asked)
				0.20, 0.19, 0.15, 0.14,  // Non-vitals
		};

		int result = simulatePipeline(semantic, keyword, 0.3, queryTerms.length);

		assertEquals(8, result,
				"Query should return exactly 8 records: 4 BP + 2 weight + 2 temperature");
	}

	private static Date makeDate(int year, int month, int day) {
		Calendar cal = Calendar.getInstance();
		cal.set(year, month - 1, day, 0, 0, 0);
		cal.set(Calendar.MILLISECOND, 0);
		return cal.getTime();
	}
}
