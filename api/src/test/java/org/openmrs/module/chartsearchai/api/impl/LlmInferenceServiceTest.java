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
			/* [  1] */ "Medication prescription: Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Every six hours. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: REVISE. Urgency: ROUTINE. Reason: Spectrum",
			/* [  2] */ "Medication prescription: Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Thrice daily. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: NEW. Urgency: ROUTINE. Reason: Spectrum. Stopped: 2026-03-18",
			/* [  3] */ "Program enrollment: Program: PMTCT. Enrolled: 2026-03-18. Status: Active",
			/* [  4] */ "Clinical observation: Finding — Immunization history: Immunizations: Polio vaccination, oral, Oral polio vacc); Vaccination date: 2026-03-18; Immunization sequence number: 1.0",
			/* [  5] */ "Patient allergy: Allergy: Beef (food allergen). Severity: Severe. Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
			/* [  6] */ "Clinical observation: Assessment — Method of family planning: Condoms",
			/* [  7] */ "Clinical observation: Assessment — Method of family planning: Diaphragm. Note: in bathroom",
			/* [  8] */ "Medical condition: Condition: Tuberculosis. Status: ACTIVE",
			/* [  9] */ "Clinical observation: Test — CD4 Count: 988.0",
			/* [ 10] */ "Clinical observation: Drug — Pyrimethamine / sulfadoxine: 11.58",
			/* [ 11] */ "Clinical observation: Units of Measure — Syringe): 65.0",
			/* [ 12] */ "Clinical observation: Diagnosis — Kaposi sarcoma oral: 3.91",
			/* [ 13] */ "Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
			/* [ 14] */ "Clinical observation: Frequency — Every twenty-four hours: Every four hours",
			/* [ 15] */ "Clinical observation: Test — Height (cm): 131.0",
			/* [ 16] */ "Clinical observation: Test — Respiratory Rate: 18.0",
			/* [ 17] */ "Clinical observation: Test — Pulse: 95.0",
			/* [ 18] */ "Clinical observation: Test — Temperature (C): 36.7",
			/* [ 19] */ "Clinical observation: Test — Weight (kg): 94.0",
			/* [ 20] */ "Clinical observation: Diagnosis — Fetishism: Patient presents with mild symptoms. Advised rest and fluids.",
			/* [ 21] */ "Clinical observation: Frequency — Every twenty-four hours: Every eight hours",
			/* [ 22] */ "Clinical diagnosis: Diagnosis: Gastroenteritis. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 23] */ "Clinical observation: Test — Systolic Blood Pressure: 97.0",
			/* [ 24] */ "Clinical observation: Test — Diastolic Blood Pressure: 99.0",
			/* [ 25] */ "Clinical observation: Test — Pulse: 62.0",
			/* [ 26] */ "Clinical observation: Test — Temperature (C): 37.7",
			/* [ 27] */ "Clinical observation: Test — Weight (kg): 107.0",
			/* [ 28] */ "Clinical observation: Test — Height (cm): 137.0",
			/* [ 29] */ "Clinical observation: Test — Respiratory Rate: 24.0",
			/* [ 30] */ "Clinical observation: Assessment — Primary Diagnosis: Anemia",
			/* [ 31] */ "Clinical observation: Frequency — Every twenty-four hours: Every five hours",
			/* [ 32] */ "Clinical observation: Test — Diastolic Blood Pressure: 92.0",
			/* [ 33] */ "Clinical observation: Test — Respiratory Rate: 32.0",
			/* [ 34] */ "Clinical observation: Test — Weight (kg): 139.0",
			/* [ 35] */ "Clinical observation: Test — Blood Oxygen Saturation: 88.0",
			/* [ 36] */ "Clinical observation: Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [ 37] */ "Clinical observation: Test — Systolic Blood Pressure: 122.0",
			/* [ 38] */ "Clinical observation: Test — Height (cm): 103.0",
			/* [ 39] */ "Clinical observation: Test — Temperature (C): 40.3",
			/* [ 40] */ "Clinical diagnosis: Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 41] */ "Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
			/* [ 42] */ "Clinical observation: Diagnosis — Fetishism: Patient stable on current regimen.",
			/* [ 43] */ "Clinical observation: Frequency — Every twenty-four hours: Married",
			/* [ 44] */ "Clinical observation: Test — Pulse: 62.0",
			/* [ 45] */ "Clinical observation: Test — Respiratory Rate: 23.0",
			/* [ 46] */ "Clinical observation: Test — Blood Oxygen Saturation: 92.0",
			/* [ 47] */ "Clinical observation: Test — Height (cm): 126.0",
			/* [ 48] */ "Clinical observation: Test — Systolic Blood Pressure: 101.0",
			/* [ 49] */ "Clinical observation: Test — Diastolic Blood Pressure: 99.0",
			/* [ 50] */ "Clinical observation: Assessment — Primary Diagnosis: Diabetes Mellitus",
			/* [ 51] */ "Clinical observation: Diagnosis — Fetishism: New complaint of persistent cough for 2 weeks.",
			/* [ 52] */ "Clinical diagnosis: Diagnosis: Urinary Tract Infection. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 53] */ "Clinical diagnosis: Diagnosis: Tuberculosis. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 54] */ "Patient allergy: Allergy: Fomepizole (drug allergen)",
			/* [ 55] */ "Medical condition: Condition: Hypertension. Status: ACTIVE",
			/* [ 56] */ "Clinical observation: Assessment — Primary Diagnosis: Anemia",
			/* [ 57] */ "Clinical observation: Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 58] */ "Clinical observation: Test — Height (cm): 101.0",
			/* [ 59] */ "Clinical observation: Test — Systolic Blood Pressure: 123.0",
			/* [ 60] */ "Clinical observation: Test — Blood Oxygen Saturation: 86.0",
			/* [ 61] */ "Clinical observation: Test — Pulse: 51.0",
			/* [ 62] */ "Clinical diagnosis: Diagnosis: Skin Infection. Certainty: CONFIRMED. Rank: Primary",
			/* [ 63] */ "Clinical observation: Assessment — Primary Diagnosis: Pneumonia",
			/* [ 64] */ "Clinical observation: Test — Weight (kg): 38.0",
			/* [ 65] */ "Clinical observation: Test — Diastolic Blood Pressure: 71.0",
			/* [ 66] */ "Clinical observation: Test — Respiratory Rate: 16.0",
			/* [ 67] */ "Clinical diagnosis: Diagnosis: Malaria. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 68] */ "Clinical diagnosis: Diagnosis: Diabetes Mellitus. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 69] */ "Clinical observation: Assessment — Primary Diagnosis: HIV Disease",
			/* [ 70] */ "Clinical diagnosis: Diagnosis: HIV Disease. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 71] */ "Clinical observation: Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [ 72] */ "Clinical diagnosis: Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 73] */ "Clinical diagnosis: Diagnosis: Anemia. Certainty: PROVISIONAL. Rank: Secondary",
			/* [ 74] */ "Clinical observation: Test — Systolic Blood Pressure: 137.0",
			/* [ 75] */ "Clinical observation: Test — Diastolic Blood Pressure: 67.0",
			/* [ 76] */ "Clinical observation: Test — Pulse: 86.0",
			/* [ 77] */ "Clinical observation: Test — Temperature (C): 39.3",
			/* [ 78] */ "Clinical observation: Test — Weight (kg): 146.0",
			/* [ 79] */ "Clinical observation: Test — Height (cm): 107.0",
			/* [ 80] */ "Clinical observation: Test — Blood Oxygen Saturation: 86.0",
			/* [ 81] */ "Clinical observation: Test — Respiratory Rate: 30.0",
			/* [ 82] */ "Clinical observation: Test — Diastolic Blood Pressure: 105.0",
			/* [ 83] */ "Clinical observation: Test — Pulse: 70.0",
			/* [ 84] */ "Clinical observation: Test — Blood Oxygen Saturation: 100.0",
			/* [ 85] */ "Clinical observation: Test — Respiratory Rate: 40.0",
			/* [ 86] */ "Clinical observation: Test — CD4 Count: 1191.0",
			/* [ 87] */ "Clinical observation: Units of Measure — Syringe): 339.0",
			/* [ 88] */ "Clinical observation: Misc — Milligram per meter squared: 47.0",
			/* [ 89] */ "Clinical observation: Diagnosis — Kaposi sarcoma oral: 3.5",
			/* [ 90] */ "Clinical observation: Diagnosis — Photoallergy: 9.93",
			/* [ 91] */ "Clinical observation: Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [ 92] */ "Clinical observation: Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 93] */ "Clinical diagnosis: Diagnosis: Hypertension. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 94] */ "Clinical observation: Test — Systolic Blood Pressure: 147.0",
			/* [ 95] */ "Clinical observation: Test — Diastolic Blood Pressure: 58.0",
			/* [ 96] */ "Clinical observation: Test — Pulse: 53.0",
			/* [ 97] */ "Clinical observation: Test — Temperature (C): 36.4",
			/* [ 98] */ "Clinical observation: Test — Height (cm): 163.0",
			/* [ 99] */ "Clinical observation: Test — Respiratory Rate: 19.0",
			/* [100] */ "Clinical observation: Assessment — Primary Diagnosis: Headache",
			/* [101] */ "Clinical observation: Test — Pulse: 83.0",
			/* [102] */ "Clinical observation: Test — Weight (kg): 68.0",
			/* [103] */ "Clinical observation: Test — Diastolic Blood Pressure: 50.0",
			/* [104] */ "Clinical observation: Test — Blood Oxygen Saturation: 94.0",
			/* [105] */ "Clinical observation: Test — Respiratory Rate: 40.0",
			/* [106] */ "Clinical observation: Assessment — Primary Diagnosis: Malaria",
			/* [107] */ "Clinical observation: Diagnosis — Fetishism: Well-child visit. Growth and development normal.",
			/* [108] */ "Clinical observation: Test — Diastolic Blood Pressure: 93.0",
			/* [109] */ "Clinical observation: Test — Blood Oxygen Saturation: 95.0",
			/* [110] */ "Clinical observation: Test — Respiratory Rate: 36.0",
			/* [111] */ "Clinical diagnosis: Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [112] */ "Clinical observation: Test — Systolic Blood Pressure: 98.0",
			/* [113] */ "Clinical observation: Test — Pulse: 66.0",
			/* [114] */ "Clinical observation: Test — Temperature (C): 37.8",
			/* [115] */ "Clinical observation: Test — Weight (kg): 121.0",
			/* [116] */ "Clinical observation: Test — Height (cm): 173.0",
			/* [117] */ "Clinical observation: Test — Blood Oxygen Saturation: 94.0",
			/* [118] */ "Clinical observation: Test — Respiratory Rate: 29.0",
			/* [119] */ "Clinical observation: Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [120] */ "Clinical observation: Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [121] */ "Clinical observation: Diagnosis — Fetishism: Patient counseled on lifestyle modifications.",
			/* [122] */ "Clinical observation: Frequency — Every twenty-four hours: Every four hours",
			/* [123] */ "Clinical observation: Assessment — Primary Diagnosis: Skin Infection",
			/* [124] */ "Clinical observation: Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [125] */ "Clinical observation: Test — Height (cm): 137.0",
			/* [126] */ "Clinical observation: Test — Respiratory Rate: 28.0",
			/* [127] */ "Clinical observation: Test — Systolic Blood Pressure: 134.0",
			/* [128] */ "Clinical diagnosis: Diagnosis: Asthma. Certainty: CONFIRMED. Rank: Primary",
			/* [129] */ "Clinical observation: Test — Systolic Blood Pressure: 117.0",
			/* [130] */ "Clinical observation: Test — Diastolic Blood Pressure: 70.0",
			/* [131] */ "Clinical observation: Test — Pulse: 115.0",
			/* [132] */ "Clinical observation: Test — Temperature (C): 40.1",
			/* [133] */ "Clinical observation: Test — Height (cm): 186.0",
			/* [134] */ "Clinical observation: Test — Respiratory Rate: 22.0",
			/* [135] */ "Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
			/* [136] */ "Clinical observation: Assessment — Primary Diagnosis: Tuberculosis",
			/* [137] */ "Clinical observation: Test — Respiratory Rate: 28.0",
			/* [138] */ "Clinical observation: Test — Diastolic Blood Pressure: 76.0",
			/* [139] */ "Clinical observation: Test — Systolic Blood Pressure: 102.0",
			/* [140] */ "Clinical observation: Units of Measure — Syringe): 237.0",
			/* [141] */ "Clinical observation: Misc — Milligram per meter squared: 17.0",
			/* [142] */ "Clinical observation: Diagnosis — Photoallergy: 8.27",
			/* [143] */ "Clinical observation: Diagnosis — Fetishism: Routine checkup. No significant findings.",
			/* [144] */ "Clinical observation: Test — Temperature (C): 39.3",
			/* [145] */ "Clinical observation: Test — Diastolic Blood Pressure: 78.0",
			/* [146] */ "Clinical observation: Test — Blood Oxygen Saturation: 88.0",
			/* [147] */ "Clinical diagnosis: Diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL. Rank: Primary",
			/* [148] */ "Clinical observation: Test — Systolic Blood Pressure: 151.0",
			/* [149] */ "Clinical observation: Test — Diastolic Blood Pressure: 53.0",
			/* [150] */ "Clinical observation: Test — Pulse: 117.0",
			/* [151] */ "Clinical observation: Test — Temperature (C): 39.4",
			/* [152] */ "Clinical observation: Test — Blood Oxygen Saturation: 88.0",
			/* [153] */ "Clinical observation: Test — Respiratory Rate: 15.0",
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
		return simulatePipelineIndices(semanticScores, keywordScores, keywordWeight,
				queryTermCount, 10).size();
	}

	private static int simulatePipeline(double[] semanticScores, double[] keywordScores,
			double keywordWeight, int queryTermCount, int topK) {
		return simulatePipelineIndices(semanticScores, keywordScores, keywordWeight,
				queryTermCount, topK).size();
	}

	private static List<Integer> simulatePipelineIndices(double[] semanticScores,
			double[] keywordScores, double keywordWeight, int queryTermCount, int topK) {
		return simulatePipelineIndices(semanticScores, keywordScores,
				keywordWeight, queryTermCount, topK, null);
	}

	private static List<Integer> simulatePipelineIndices(double[] semanticScores,
			double[] keywordScores, double keywordWeight, int queryTermCount,
			int topK, float[][] embeddingVectors) {
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
			return new ArrayList<Integer>();
		}

		// Zero-keyword gate: z-score check when too few records match keywords.
		// Only meaningful with enough data points for statistics.
		if (queryTermCount > 0
				&& semanticScores.length
				>= ChartSearchAiConstants.MIN_RECORDS_FOR_Z_SCORE) {
			int keywordMatchCount = 0;
			for (double kw : keywordScores) {
				if (kw > 0) {
					keywordMatchCount++;
				}
			}
			if (keywordMatchCount < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				double sumSem = 0;
				for (double s : semanticScores) {
					sumSem += s;
				}
				double meanSem = sumSem / semanticScores.length;
				double sumSqDiff = 0;
				for (double s : semanticScores) {
					double diff = s - meanSem;
					sumSqDiff += diff * diff;
				}
				double stddev = Math.sqrt(sumSqDiff / semanticScores.length);
				double zScore = stddev == 0 ? 0
						: (maxSemanticScore - meanSem) / stddev;
				if (zScore < ChartSearchAiConstants.ZERO_KEYWORD_MIN_Z_SCORE) {
					return new ArrayList<Integer>();
				}
			}
		}

		List<LlmInferenceService.ScoredEmbedding> scored =
				new ArrayList<LlmInferenceService.ScoredEmbedding>();
		double bonusThreshold = queryTermCount >= 4
				? 1.0 / queryTermCount
				: queryTermCount == 0 ? 1.0
				: (double) Math.min(2, queryTermCount) / queryTermCount;
		for (int i = 0; i < semanticScores.length; i++) {
			double keywordBonus = keywordScores[i] >= bonusThreshold ? keywordScores[i] : 0.0;
			double keywordPenalty = 0.0;
			if (queryTermCount <= 2 && keywordScores[i] > 0
					&& keywordScores[i] < bonusThreshold) {
				keywordPenalty = keywordScores[i];
			}
			double baseScore = semanticScores[i] + keywordWeight * keywordBonus
					- keywordWeight * keywordPenalty;
			LlmInferenceService.ScoredEmbedding se =
					makeScoredEmbedding(baseScore, keywordScores[i], semanticScores[i]);
			se.embedding.setTextContent(String.valueOf(i));
			if (embeddingVectors != null && embeddingVectors[i] != null) {
				se.embedding.setEmbeddingVector(embeddingVectors[i]);
			}
			scored.add(se);
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

		if (refinementActivated) {
			double maxSemanticRefined = 0;
			for (LlmInferenceService.ScoredEmbedding se : candidates) {
				if (se.semanticScore > maxSemanticRefined) {
					maxSemanticRefined = se.semanticScore;
				}
			}
			double adaptiveMinGap = Math.max(
					maxSemanticRefined
							* ChartSearchAiConstants.REFINEMENT_ADAPTIVE_GAP_RATIO,
					ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
			int refinedCutoff = LlmInferenceService.findAdaptiveCutoff(
					candidates, candidates.size(), minScore, gapMultiplier,
					adaptiveMinGap);
			candidates = new ArrayList<LlmInferenceService.ScoredEmbedding>(
					candidates.subList(0, refinedCutoff));
		} else {
			int secondCutoff = LlmInferenceService.findAdaptiveCutoff(
					candidates, candidates.size(), minScore,
					ChartSearchAiConstants.SECOND_PASS_GAP_MULTIPLIER,
					ChartSearchAiConstants.SECOND_PASS_MIN_GAP);
			if (secondCutoff < candidates.size()) {
				candidates = new ArrayList<LlmInferenceService.ScoredEmbedding>(
						candidates.subList(0, secondCutoff));
			}
			double maxScore = scored.isEmpty() ? 0 : scored.get(0).score;
			double ratioFloor = maxScore
					* ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO;
			List<LlmInferenceService.ScoredEmbedding> strict =
					new ArrayList<LlmInferenceService.ScoredEmbedding>();
			for (LlmInferenceService.ScoredEmbedding se : candidates) {
				if (se.score >= ratioFloor) {
					strict.add(se);
				}
			}
			candidates = strict;
			boolean allHaveKeywords = true;
			for (LlmInferenceService.ScoredEmbedding se : candidates) {
				if (se.keywordScore == 0) {
					allHaveKeywords = false;
					break;
				}
			}
			if (!allHaveKeywords && candidates.size() > topK) {
				candidates = candidates.subList(0, topK);
			}
		}

		// Inter-candidate coherence filter (mirrors production pipeline).
		// filterByCoherence validates embedding vectors internally.
		if (candidates.size() >= 3) {
			candidates = LlmInferenceService.filterByCoherence(candidates);
		}

		List<Integer> indices = new ArrayList<Integer>();
		for (LlmInferenceService.ScoredEmbedding se : candidates) {
			indices.add(Integer.parseInt(se.embedding.getTextContent()));
		}
		Collections.sort(indices);
		return indices;
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
	public void pipeline_noKeywordMatchesSmoothDistribution_shouldReturnEmpty() {
		// Models: "HB results over time" — the patient has NO HB results.
		// All 93 records are vital signs with smooth, similar semantic scores
		// and ZERO keyword matches. The z-score gate detects that the top
		// score (0.42) is not a statistical outlier — it's just part of
		// the noise floor (z ≈ 1.73 < 2.0). Without keyword corroboration,
		// these results are unreliable and the pipeline returns empty.
		int recordCount = 93;
		double[] semantic = new double[recordCount];
		double[] keyword = new double[recordCount];
		for (int i = 0; i < recordCount; i++) {
			semantic[i] = 0.42 - i * 0.002;
			keyword[i] = 0.0;
		}

		int result = simulatePipeline(semantic, keyword, 0.3, 7);

		assertEquals(0, result,
				"Z-score gate should reject smooth distribution with no keyword matches");
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
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "What is the current CD4 Count?" → expected: exactly 2 CD4 records.
		// "current" is a stopword → terms: ["cd4", "count"]. Both CD4 Count records
		// match both terms (kw=1.0). Gap detection separates them from all others.

		String normalized = LlmInferenceService.stripQueryStopwords("What is the current CD4 Count?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cd4", "count" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(1.0, keyword[8], 0.001, "[9] CD4 Count should match both terms");
		assertEquals(1.0, keyword[85], 0.001, "[86] CD4 Count should match both terms");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  8] = 0.58; // [  9]
		semantic[ 85] = 0.54; // [ 86]

		semantic[ 22] = 0.30; // [ 23]
		semantic[ 17] = 0.28; // [ 18]
		semantic[ 18] = 0.27; // [ 19]
		semantic[ 19] = 0.26; // [ 20]
		semantic[ 20] = 0.25; // [ 21]
		semantic[ 23] = 0.24; // [ 24]
		semantic[ 11] = 0.20; // [ 12]
		semantic[  7] = 0.18; // [  8]
		semantic[ 54] = 0.17; // [ 55]
		semantic[  4] = 0.15; // [  5]
		semantic[  0] = 0.14; // [  1]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [9] CD4 Count: 988, [86] CD4 Count: 1191
		assertEquals(Arrays.asList(8, 85),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_latestCd4CountQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "What is the latest CD4 Count?" → expected: exactly 2 CD4 records.
		// "latest" is a stopword → terms: ["cd4", "count"]. Same targets as
		// the "current CD4 Count" query.

		String normalized = LlmInferenceService.stripQueryStopwords("What is the latest CD4 Count?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cd4", "count" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(1.0, keyword[8], 0.001, "[9] CD4 Count should match both terms");
		assertEquals(1.0, keyword[85], 0.001, "[86] CD4 Count should match both terms");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  8] = 0.56; // [  9]
		semantic[ 85] = 0.52; // [ 86]

		semantic[ 22] = 0.28; // [ 23]
		semantic[ 17] = 0.26; // [ 18]
		semantic[ 18] = 0.25; // [ 19]
		semantic[ 23] = 0.23; // [ 24]
		semantic[ 11] = 0.19; // [ 12]
		semantic[  7] = 0.17; // [  8]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [9] CD4 Count: 988, [86] CD4 Count: 1191
		assertEquals(Arrays.asList(8, 85),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_cancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "does the patient have cancer?" → terms: ["cancer"]. No record
		// contains literal "cancer". Kaposi sarcoma records ([12],[89]) are
		// semantically closest. Gap detection separates them from disease records.

		String normalized = LlmInferenceService.stripQueryStopwords("does the patient have cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(0.0, keyword[11], 0.001, "[12] Kaposi sarcoma has no literal cancer match");
		assertEquals(0.0, keyword[88], 0.001, "[89] Kaposi sarcoma has no literal cancer match");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 11] = 0.40; // [ 12]
		semantic[ 88] = 0.38; // [ 89]

		semantic[  7] = 0.22; // [  8]
		semantic[ 39] = 0.21; // [ 40]
		semantic[ 71] = 0.20; // [ 72]
		semantic[110] = 0.20; // [111]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 69] = 0.19; // [ 70]
		semantic[ 40] = 0.18; // [ 41]
		semantic[ 68] = 0.18; // [ 69]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]
		semantic[ 66] = 0.17; // [ 67]
		semantic[ 92] = 0.17; // [ 93]
		semantic[ 72] = 0.16; // [ 73]
		semantic[ 29] = 0.16; // [ 30]
		semantic[ 55] = 0.16; // [ 56]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [12] Kaposi sarcoma: 3.91, [89] Kaposi sarcoma: 3.5
		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_doesHeHaveCancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "does he have cancer?" → terms: ["cancer"]. Same semantic behavior
		// as other cancer queries. Kaposi sarcoma records are the targets.

		String normalized = LlmInferenceService.stripQueryStopwords("does he have cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(0.0, keyword[11], 0.001, "[12] Kaposi sarcoma has no literal cancer match");
		assertEquals(0.0, keyword[88], 0.001, "[89] Kaposi sarcoma has no literal cancer match");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 11] = 0.40; // [ 12]
		semantic[ 88] = 0.38; // [ 89]

		semantic[  7] = 0.22; // [  8]
		semantic[ 39] = 0.21; // [ 40]
		semantic[ 71] = 0.20; // [ 72]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 40] = 0.18; // [ 41]
		semantic[ 68] = 0.18; // [ 69]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]
		semantic[ 72] = 0.16; // [ 73]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [12] Kaposi sarcoma: 3.91, [89] Kaposi sarcoma: 3.5
		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_anyCancerQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any cancer?" → terms: ["cancer"]. Same targets as other cancer queries.

		String normalized = LlmInferenceService.stripQueryStopwords("any cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cancer" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(0.0, keyword[11], 0.001, "[12] Kaposi sarcoma has no literal cancer match");
		assertEquals(0.0, keyword[88], 0.001, "[89] Kaposi sarcoma has no literal cancer match");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 11] = 0.40; // [ 12]
		semantic[ 88] = 0.38; // [ 89]

		semantic[  7] = 0.22; // [  8]
		semantic[ 39] = 0.21; // [ 40]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [12] Kaposi sarcoma: 3.91, [89] Kaposi sarcoma: 3.5
		assertEquals(Arrays.asList(11, 88),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_fractureQuery_realData_shouldReturnNoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any fracture?" → expected: 0 records. No record in the dataset
		// mentions fracture. All semantic scores below the gate threshold (0.25),
		// so the pipeline returns empty.

		String normalized = LlmInferenceService.stripQueryStopwords("any fracture?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "fracture" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// No record mentions "fracture" — verify no keyword matches
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001);
		}

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.15);

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		assertEquals(Collections.emptyList(), result,
				"Should return no record(s)");
	}

	@Test
	public void pipeline_familyHistoryOfCancerQuery_realData_shouldReturnNoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "Any family history of cancer?" → expected: 0 records.
		// Although some records match individual terms ("family", "history"),
		// no record is semantically related to family cancer history.
		// All semantic scores are below gate threshold → empty result.

		String normalized = LlmInferenceService.stripQueryStopwords("Any family history of cancer?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "family", "history", "cancer" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// Some records match individual terms but none match "cancer"
		assertEquals(0.3333, keyword[3], 0.01, "[4] matches history");
		assertEquals(0.3333, keyword[5], 0.01, "[6] matches family");
		assertEquals(0.3333, keyword[6], 0.01, "[7] matches family");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.10);

		// Real model gives all records below the absolute similarity floor
		// (0.25) for this query. "Family history of cancer" is a specific
		// concept that no record in the dataset represents. The model gives
		// Kaposi sarcoma only ~0.25 (cancer concept) and Family planning
		// ~0.22 (word overlap). All below gate threshold.
		semantic[3]  = 0.19; // [4] Immunization history — "history" word overlap
		semantic[5]  = 0.23; // [6] Family planning: Condoms — "family" word overlap
		semantic[6]  = 0.22; // [7] Family planning: Diaphragm — "family" word overlap
		semantic[11] = 0.24; // [12] Kaposi sarcoma — cancer concept (below gate)
		semantic[88] = 0.23; // [89] Kaposi sarcoma — cancer concept (below gate)

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		assertEquals(Collections.emptyList(), result,
				"Should return no record(s)");
	}

	@Test
	public void pipeline_medicationsQuery_realData_shouldReturnFourRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "is the patient on any medications?" → expected: 4 records.
		// 2 drug orders match via "Medication prescription:" prefix,
		// 2 visit notes match via "Medication adjusted" in text.
		// Keyword bonus (kw=1.0) + semantic scores create clear gap.

		String normalized = LlmInferenceService.stripQueryStopwords("is the patient on any medications?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "medications" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// Drug orders match via "Medication prescription:" prefix
		assertEquals(1.0, keyword[0], 0.001, "[1] Drug order matches medications");
		assertEquals(1.0, keyword[1], 0.001, "[2] Drug order matches medications");
		// Visit notes with "Medication adjusted" also match
		assertEquals(1.0, keyword[56], 0.001, "[57] Visit note matches medications");
		assertEquals(1.0, keyword[91], 0.001, "[92] Visit note matches medications");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  0] = 0.40; // [  1]
		semantic[  1] = 0.39; // [  2]
		semantic[ 56] = 0.38; // [ 57]
		semantic[ 91] = 0.37; // [ 92]

		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[  4] = 0.18; // [  5]
		semantic[ 53] = 0.17; // [ 54]
		semantic[ 11] = 0.15; // [ 12]
		semantic[ 88] = 0.14; // [ 89]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [1] Drug order: Azithromycin (REVISE), [2] Drug order: Azithromycin (NEW), [57] Visit note: Medication adjusted, [92] Visit note: Medication adjusted
		assertEquals(Arrays.asList(0, 1, 56, 91),
				result, "Should return exactly 4 record(s)");
	}

	@Test
	public void pipeline_knownConditionsQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any known conditions?" → expected: 2 records.
		// "known" is a stopword → terms: ["conditions"]. Both Condition records
		// match via "Medical condition: Condition:" prefix + content.

		String normalized = LlmInferenceService.stripQueryStopwords("any known conditions?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "conditions" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// "conditions" matches Condition records via "Medical condition:" prefix
		assertEquals(1.0, keyword[7], 0.001, "[8] TB condition matches");
		assertEquals(1.0, keyword[54], 0.001, "[55] Hypertension condition matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  7] = 0.40; // [  8]
		semantic[ 54] = 0.38; // [ 55]

		semantic[ 39] = 0.21; // [ 40]
		semantic[ 71] = 0.20; // [ 72]
		semantic[110] = 0.20; // [111]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 69] = 0.19; // [ 70]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [8] Condition: TB, [55] Condition: Hypertension
		assertEquals(Arrays.asList(7, 54),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_anemicQuery_realData_shouldReturnExactlyThreeRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "is the patient anemic?" → expected: 3 records.
		// "anemic" (6 chars) is below the 7-char minimum for stem matching,
		// so no keyword matches. Purely semantic: 2 "Assessment: Anemia"
		// records ([30],[56]) + 1 "Diagnosis: Anemia" record ([73]).

		String normalized = LlmInferenceService.stripQueryStopwords("is the patient anemic?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "anemic" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// "anemic" is 6 chars < 7 minimum for stem matching → no keyword matches
		for (int i = 0; i < keyword.length; i++) {
			assertEquals(0.0, keyword[i], 0.001,
					"No record should match anemic (too short for stemming)");
		}

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 29] = 0.40; // [ 30]
		semantic[ 55] = 0.38; // [ 56]
		semantic[ 72] = 0.36; // [ 73]

		semantic[  7] = 0.22; // [  8]
		semantic[ 54] = 0.21; // [ 55]
		semantic[ 39] = 0.20; // [ 40]
		semantic[ 71] = 0.20; // [ 72]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]
		semantic[ 66] = 0.17; // [ 67]
		semantic[  8] = 0.14; // [  9]
		semantic[ 85] = 0.14; // [ 86]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [30] Assessment: Anemia, [56] Assessment: Anemia, [73] Dx: Anemia (PROVISIONAL)
		assertEquals(Arrays.asList(29, 55, 72),
				result, "Should return exactly 3 record(s)");
	}

	@Test
	public void pipeline_conditionsQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any conditions" → expected: 2 records.
		// terms: ["conditions"]. Same targets as "known conditions" query.

		String normalized = LlmInferenceService.stripQueryStopwords("any conditions");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "conditions" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(1.0, keyword[7], 0.001, "[8] TB condition matches");
		assertEquals(1.0, keyword[54], 0.001, "[55] Hypertension condition matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  7] = 0.40; // [  8]
		semantic[ 54] = 0.38; // [ 55]

		semantic[ 39] = 0.21; // [ 40]
		semantic[ 71] = 0.20; // [ 72]
		semantic[ 52] = 0.19; // [ 53]
		semantic[ 69] = 0.19; // [ 70]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.17; // [ 63]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [8] Condition: TB, [55] Condition: Hypertension
		assertEquals(Arrays.asList(7, 54),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_stdQuery_realData_shouldReturnExactlySixRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any sexually transmitted disease?" → expected: 6 HIV Disease records.
		// terms: ["sexually", "transmitted", "disease"]. "disease" matches 8 records
		// (6 HIV + 2 Fetishism visit notes). Bonus threshold is 2/3 so no bonus.
		// Gap detection on semantic scores separates 6 HIV records (high semantic)
		// from the 2 Fetishism notes (low semantic).

		String normalized = LlmInferenceService.stripQueryStopwords("any sexually transmitted disease?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "sexually", "transmitted", "disease" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// "disease" matches HIV Disease records (1/3 terms = 0.33)
		assertEquals(0.3333, keyword[39], 0.01, "[40] HIV Disease matches disease");
		assertEquals(0.3333, keyword[40], 0.01, "[41] HIV Disease matches disease");
		assertEquals(0.3333, keyword[68], 0.01, "[69] HIV Disease matches disease");
		assertEquals(0.3333, keyword[69], 0.01, "[70] HIV Disease matches disease");
		assertEquals(0.3333, keyword[71], 0.01, "[72] HIV Disease matches disease");
		assertEquals(0.3333, keyword[110], 0.01, "[111] HIV Disease matches disease");
		// Non-HIV records also match "disease" but have low semantic scores
		assertEquals(0.3333, keyword[56], 0.01, "[57] Fetishism note matches disease");
		assertEquals(0.3333, keyword[91], 0.01, "[92] Fetishism note matches disease");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 39] = 0.40; // [ 40]
		semantic[ 40] = 0.39; // [ 41]
		semantic[ 68] = 0.38; // [ 69]
		semantic[ 69] = 0.37; // [ 70]
		semantic[ 71] = 0.36; // [ 72]
		semantic[110] = 0.35; // [111]

		// Fetishism notes share "disease" → model inflates their scores
		semantic[ 56] = 0.33; // [ 57] Fetishism "Chronic disease management" — "disease" word overlap
		semantic[ 91] = 0.33; // [ 92] Fetishism "Chronic disease management" — "disease" word overlap
		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 52] = 0.18; // [ 53]
		semantic[ 12] = 0.17; // [ 13]
		semantic[ 62] = 0.16; // [ 63]
		semantic[ 66] = 0.16; // [ 67]
		semantic[ 11] = 0.14; // [ 12]
		semantic[ 88] = 0.14; // [ 89]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [40] Dx: HIV Disease (CONFIRMED), [41] Assessment: HIV Disease, [69] Assessment: HIV Disease, [70] Dx: HIV Disease (PROVISIONAL), [72] Dx: HIV Disease (CONFIRMED), [111] Dx: HIV Disease (CONFIRMED)
		assertEquals(Arrays.asList(39, 40, 68, 69, 71, 110),
				result, "Should return exactly 6 record(s)");
	}

	@Test
	public void pipeline_whatIsPatientAllergicToQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "What is the patient allergic to?" → expected: 2 allergy records.
		// terms: ["allergic"]. Stem "allerg" matches "Allergy" in both
		// allergy records ([5],[54]).

		String normalized = LlmInferenceService.stripQueryStopwords("What is the patient allergic to?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "allergic" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// "allergic" stem-matches "allergy" in allergy records
		assertEquals(1.0, keyword[4], 0.001, "[5] Beef allergy matches");
		assertEquals(1.0, keyword[53], 0.001, "[54] Fomepizole allergy matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  4] = 0.40; // [  5]
		semantic[ 53] = 0.38; // [ 54]

		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 39] = 0.18; // [ 40]
		semantic[ 11] = 0.15; // [ 12]
		semantic[ 88] = 0.14; // [ 89]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [5] Allergy: Beef, [54] Allergy: Fomepizole
		assertEquals(Arrays.asList(4, 53),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_coughQuery_realData_shouldReturnExactlyOneRecord() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any cough?" → expected: 1 record.
		// terms: ["cough"]. Only record [51] contains "cough" in
		// "persistent cough for the last 3 weeks".
		// Gap detection cannot fire before position 2, but keyword
		// refinement selects the single matching record.

		String normalized = LlmInferenceService.stripQueryStopwords("any cough?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "cough" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(1.0, keyword[50], 0.001, "[51] Cough visit note matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[ 50] = 0.40; // [ 51]

		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 39] = 0.18; // [ 40]
		semantic[ 11] = 0.17; // [ 12]
		semantic[ 52] = 0.16; // [ 53]
		semantic[ 12] = 0.15; // [ 13]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [51] Cough visit note
		assertEquals(Arrays.asList(50),
				result, "Should return exactly 1 record(s)");
	}

	@Test
	public void pipeline_doesPatientHaveAllergiesQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "does the patient have any allergies?" → expected: 2 allergy records.
		// terms: ["allergies"]. Stem "allerg" matches both allergy records.

		String normalized = LlmInferenceService.stripQueryStopwords("does the patient have any allergies?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "allergies" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// "allergies" stem-matches "Allergy" in allergy records
		assertEquals(1.0, keyword[4], 0.001, "[5] Beef allergy matches");
		assertEquals(1.0, keyword[53], 0.001, "[54] Fomepizole allergy matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  4] = 0.40; // [  5]
		semantic[ 53] = 0.38; // [ 54]

		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 39] = 0.18; // [ 40]
		semantic[ 11] = 0.15; // [ 12]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [5] Allergy: Beef, [54] Allergy: Fomepizole
		assertEquals(Arrays.asList(4, 53),
				result, "Should return exactly 2 record(s)");
	}

	@Test
	public void pipeline_allergyQuery_realData_shouldReturnExactlyTwoRecords() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "any allergies?" → expected: 2 allergy records.
		// terms: ["allergies"]. Same allergy targets as other allergy queries.

		String normalized = LlmInferenceService.stripQueryStopwords("any allergies?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertArrayEquals(new String[] { "allergies" }, queryTerms);

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		assertEquals(1.0, keyword[4], 0.001, "[5] Beef allergy matches");
		assertEquals(1.0, keyword[53], 0.001, "[54] Fomepizole allergy matches");

		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.1);

		semantic[  4] = 0.40; // [  5]
		semantic[ 53] = 0.38; // [ 54]

		semantic[  7] = 0.20; // [  8]
		semantic[ 54] = 0.19; // [ 55]
		semantic[ 39] = 0.18; // [ 40]

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [5] Allergy: Beef, [54] Allergy: Fomepizole
		assertEquals(Arrays.asList(4, 53),
				result, "Should return exactly 2 record(s)");
	}


	@Test
	public void pipeline_vitalsTrendQuery_realData_shouldReturnBpWeightAndTemperature() {
		// Full 153-record patient dataset: 16-year-old Male.
		// Query: "How have this patient's blood pressure, weight, and
		// temperature trended across their last 7 visits?"
		// Multi-concept query with 7 terms. BP matches "blood"+"pressure"
		// (2/7=0.29 → at bonus threshold), weight/temp match 1 term each
		// (1/7=0.14 → below bonus threshold). Gap detection on semantic
		// scores separates the 8 target vitals from all other records.

		String normalized = LlmInferenceService.stripQueryStopwords(
				"How have this patient's blood pressure, weight, and temperature "
				+ "trended across their last 7 visits?");
		String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);
		assertTrue(queryTerms.length >= 4,
				"Should have at least blood/pressure/weight/temperature as terms");
		List<String> termList = Arrays.asList(queryTerms);
		assertTrue(termList.contains("blood"), "Should contain 'blood'");
		assertTrue(termList.contains("pressure"), "Should contain 'pressure'");
		assertTrue(termList.contains("weight"), "Should contain 'weight'");
		assertTrue(termList.contains("temperature"), "Should contain 'temperature'");

		double[] keyword = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			keyword[i] = LlmInferenceService.computeKeywordScore(queryTerms, FULL_PATIENT_DATASET[i]);
		}

		// BP records match "blood"+"pressure" (2 terms)
		double bpKw = keyword[22]; // Systolic BP
		assertTrue(bpKw > 0, "BP should match blood+pressure");
		// Weight matches "weight" (1 term)
		double weightKw = keyword[18]; // Weight
		assertTrue(weightKw > 0, "Weight should match weight");
		assertTrue(bpKw > weightKw, "BP should have higher keyword score than weight");
		// Temperature matches "temperature" (1 term)
		double tempKw = keyword[17]; // Temperature
		assertTrue(tempKw > 0, "Temperature should match temperature");

		// Realistic scores: the embedding model scores ALL vitals of the same
		// type similarly — it cannot distinguish "recent" from "old" visits.
		// ALL Blood Pressure, Weight, and Temperature records score high.
		double[] semantic = new double[FULL_PATIENT_DATASET.length];
		Arrays.fill(semantic, 0.10);

		// ALL Systolic/Diastolic Blood Pressure records (~24 total)
		semantic[ 22] = 0.50; // [ 23] Systolic BP: 97.0
		semantic[ 23] = 0.50; // [ 24] Diastolic BP: 99.0
		semantic[ 31] = 0.50; // [ 32] Diastolic BP: 92.0
		semantic[ 36] = 0.50; // [ 37] Systolic BP: 122.0
		semantic[ 47] = 0.50; // [ 48] Systolic BP: 101.0
		semantic[ 48] = 0.50; // [ 49] Diastolic BP: 99.0
		semantic[ 58] = 0.50; // [ 59] Systolic BP: 123.0
		semantic[ 64] = 0.50; // [ 65] Diastolic BP: 71.0
		semantic[ 73] = 0.50; // [ 74] Systolic BP: 137.0
		semantic[ 74] = 0.50; // [ 75] Diastolic BP: 67.0
		semantic[ 81] = 0.50; // [ 82] Diastolic BP: 105.0
		semantic[ 93] = 0.50; // [ 94] Systolic BP: 147.0
		semantic[ 94] = 0.50; // [ 95] Diastolic BP: 58.0
		semantic[102] = 0.50; // [103] Diastolic BP: 50.0
		semantic[107] = 0.50; // [108] Diastolic BP: 93.0
		semantic[111] = 0.50; // [112] Systolic BP: 98.0
		semantic[126] = 0.50; // [127] Systolic BP: 134.0
		semantic[128] = 0.50; // [129] Systolic BP: 117.0
		semantic[129] = 0.50; // [130] Diastolic BP: 70.0
		semantic[137] = 0.50; // [138] Diastolic BP: 76.0
		semantic[138] = 0.50; // [139] Systolic BP: 102.0
		semantic[144] = 0.50; // [145] Diastolic BP: 78.0
		semantic[147] = 0.50; // [148] Systolic BP: 151.0
		semantic[148] = 0.50; // [149] Diastolic BP: 53.0
		// ALL Weight records (~7 total)
		semantic[ 18] = 0.50; // [ 19] Weight: 94.0
		semantic[ 26] = 0.50; // [ 27] Weight: 107.0
		semantic[ 33] = 0.50; // [ 34] Weight: 139.0
		semantic[ 63] = 0.50; // [ 64] Weight: 38.0
		semantic[ 77] = 0.50; // [ 78] Weight: 146.0
		semantic[101] = 0.50; // [102] Weight: 68.0
		semantic[114] = 0.50; // [115] Weight: 121.0
		// ALL Temperature records (~9 total)
		semantic[ 17] = 0.50; // [ 18] Temperature: 36.7
		semantic[ 25] = 0.50; // [ 26] Temperature: 37.7
		semantic[ 38] = 0.50; // [ 39] Temperature: 40.3
		semantic[ 76] = 0.50; // [ 77] Temperature: 39.3
		semantic[ 96] = 0.50; // [ 97] Temperature: 36.4
		semantic[113] = 0.50; // [114] Temperature: 37.8
		semantic[131] = 0.50; // [132] Temperature: 40.1
		semantic[143] = 0.50; // [144] Temperature: 39.3
		semantic[150] = 0.50; // [151] Temperature: 39.4
		// Blood Oxygen also matches "blood" keyword
		semantic[ 34] = 0.40; // [ 35] Blood O2: 88.0
		semantic[ 45] = 0.40; // [ 46] Blood O2: 92.0
		semantic[ 59] = 0.40; // [ 60] Blood O2: 86.0
		semantic[ 79] = 0.40; // [ 80] Blood O2: 86.0
		semantic[ 83] = 0.40; // [ 84] Blood O2: 100.0
		semantic[103] = 0.40; // [104] Blood O2: 94.0
		semantic[108] = 0.40; // [109] Blood O2: 95.0
		semantic[116] = 0.40; // [117] Blood O2: 94.0
		semantic[145] = 0.40; // [146] Blood O2: 88.0
		semantic[151] = 0.40; // [152] Blood O2: 88.0

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// All 40 BP + weight + temperature records (24 BP + 7 weight + 9 temp)
		assertEquals(Arrays.asList(17, 18, 22, 23, 25, 26, 31, 33, 36, 38,
				47, 48, 58, 63, 64, 73, 74, 76, 77, 81,
				93, 94, 96, 101, 102, 107, 111, 113, 114, 126,
				128, 129, 131, 137, 138, 143, 144, 147, 148, 150),
				result, "Should return all 40 BP + weight + temperature records");
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
		semantic[3]   = 0.39; // [4] Immunization history — model inflates due to "history" word overlap
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

		List<Integer> result = simulatePipelineIndices(semantic, keyword, 0.3, queryTerms.length, 10);

		// [12] Kaposi sarcoma: 3.91, [89] Kaposi sarcoma: 3.5
		assertEquals(Arrays.asList(11, 88),
				result, "Query 'any history of cancer?' should return exactly 2 Kaposi sarcoma records");
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
		return computeRealSemanticScoresWithVectors(provider, query, null);
	}

	/**
	 * Computes real semantic scores and optionally captures embedding vectors
	 * for inter-candidate coherence filtering.
	 */
	private static double[] computeRealSemanticScoresWithVectors(
			org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider,
			String query, float[][] embeddingVectors) {
		String normalizedQuery = LlmInferenceService.stripQueryStopwords(query);
		float[] queryVector = provider.embed(normalizedQuery);

		double[] scores = new double[FULL_PATIENT_DATASET.length];
		for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
			String text = FULL_PATIENT_DATASET[i];
			float[] docVector = provider.embed(text);
			if (embeddingVectors != null) {
				embeddingVectors[i] = docVector;
			}
			double dot = 0, normA = 0, normB = 0;
			for (int j = 0; j < queryVector.length; j++) {
				dot += queryVector[j] * docVector[j];
				normA += queryVector[j] * queryVector[j];
				normB += docVector[j] * docVector[j];
			}
			double denom = Math.sqrt(normA) * Math.sqrt(normB);
			scores[i] = denom == 0 ? 0 : dot / denom;
		}
		return scores;
	}

	/**
	 * Runs the full pipeline simulation using real semantic scores from the
	 * ONNX model combined with keyword scores from the dataset.
	 */
	private static List<Integer> runRealModelPipeline(String query, int topK) {
		org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider provider =
				new org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider(
						MODEL_PATH, VOCAB_PATH);
		try {
			float[][] embeddingVectors = new float[FULL_PATIENT_DATASET.length][];
			double[] semantic = computeRealSemanticScoresWithVectors(
					provider, query, embeddingVectors);

			String normalized = LlmInferenceService.stripQueryStopwords(query);
			String[] queryTerms = LlmInferenceService.extractQueryTerms(normalized);

			double[] keyword = new double[FULL_PATIENT_DATASET.length];
			for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
				keyword[i] = LlmInferenceService.computeKeywordScore(
						queryTerms, FULL_PATIENT_DATASET[i]);
			}

			return simulatePipelineIndices(semantic, keyword, 0.3,
					queryTerms.length, topK, embeddingVectors);
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
				double bonusThreshold = queryTerms.length >= 4
						? 1.0 / queryTerms.length
						: (double) Math.min(2, queryTerms.length) / queryTerms.length;
				System.out.println("BonusThreshold: " + bonusThreshold);

				// Collect and sort by semantic score descending
				double[][] indexed = new double[FULL_PATIENT_DATASET.length][3];
				for (int i = 0; i < FULL_PATIENT_DATASET.length; i++) {
					double kw = LlmInferenceService.computeKeywordScore(
							queryTerms, FULL_PATIENT_DATASET[i]);
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

}
