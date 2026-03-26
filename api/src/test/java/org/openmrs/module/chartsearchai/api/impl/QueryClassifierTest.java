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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.impl.QueryClassifier.QueryIntent;

public class QueryClassifierTest {

	// =================================================================
	// Type detection tests
	// =================================================================

	@Test
	public void classify_shouldDetectMedicationQuery() {
		QueryIntent intent = QueryClassifier.classify("medications");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
		assertTrue(intent.getTargetTypes().contains(
				ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE));
	}

	@Test
	public void classify_shouldDetectDrugQuery() {
		QueryIntent intent = QueryClassifier.classify("drugs prescribed");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldDetectAllergyQuery() {
		QueryIntent intent = QueryClassifier.classify("allergies");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY));
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldDetectLabQuery() {
		QueryIntent intent = QueryClassifier.classify("lab results");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldDetectConditionQuery() {
		QueryIntent intent = QueryClassifier.classify("conditions");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
	}

	@Test
	public void classify_shouldDetectDiagnosisQuery() {
		QueryIntent intent = QueryClassifier.classify("any diagnoses");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldDetectProgramQuery() {
		QueryIntent intent = QueryClassifier.classify("enrolled programs");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM));
	}

	@Test
	public void classify_shouldDetectVitalsAsObs() {
		QueryIntent intent = QueryClassifier.classify("blood pressure");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldReturnEmptyForUnrelatedQuery() {
		QueryIntent intent = QueryClassifier.classify("teacher schedule");
		assertTrue(intent.getTargetTypes().isEmpty());
	}

	@Test
	public void classify_shouldBeCaseInsensitive() {
		QueryIntent intent = QueryClassifier.classify("MEDICATIONS");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldMatchClinicalAbbreviations() {
		QueryIntent intent = QueryClassifier.classify("cd4 count");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldMatchArvMedications() {
		QueryIntent intent = QueryClassifier.classify("arv regimen");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldDetectDispenseQuery() {
		QueryIntent intent = QueryClassifier.classify("dispensed pharmacy");
		assertTrue(intent.getTargetTypes().contains(
				ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE));
	}

	@Test
	public void classify_shouldHandleMultiTypeQuery() {
		QueryIntent intent = QueryClassifier.classify("medications allergies");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY));
	}

	@Test
	public void classify_shouldDetectDiabetesAsCondition() {
		QueryIntent intent = QueryClassifier.classify("diabetes");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldDetectHivAsCondition() {
		QueryIntent intent = QueryClassifier.classify("hiv status");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldDetectIllnessesAsCondition() {
		QueryIntent intent = QueryClassifier.classify("illnesses");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldDetectSpecificDrugAsMedication() {
		QueryIntent intent = QueryClassifier.classify("metformin dose");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldDetectSpecificDiseaseAsCondition() {
		QueryIntent intent = QueryClassifier.classify("does the patient have diabetes?");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldDetectSpecificVitalAsObs() {
		QueryIntent intent = QueryClassifier.classify("what is the blood pressure?");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldDetectHbAsLab() {
		QueryIntent intent = QueryClassifier.classify("HB results over time");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldDetectCd4AsLab() {
		QueryIntent intent = QueryClassifier.classify("what is the latest CD4 count?");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	// =================================================================
	// Category detection is always false — handled by retrieval layer
	// =================================================================

	@Test
	public void classify_shouldNeverReturnCategoryQuery() {
		// Category vs focused is now determined by the retrieval layer's
		// gap detection within type-matched records, not by query wording.
		assertFalse(QueryClassifier.classify("any medications?").isCategoryQuery());
		assertFalse(QueryClassifier.classify("list all conditions").isCategoryQuery());
		assertFalse(QueryClassifier.classify("show allergies").isCategoryQuery());
		assertFalse(QueryClassifier.classify("every condition listed").isCategoryQuery());
		assertFalse(QueryClassifier.classify("each medication in the regimen").isCategoryQuery());
		assertFalse(QueryClassifier.classify("what are the latest labs?").isCategoryQuery());
		assertFalse(QueryClassifier.classify("HB results over time").isCategoryQuery());
		assertFalse(QueryClassifier.classify("what is the latest CD4 count?").isCategoryQuery());
		assertFalse(QueryClassifier.classify("teacher schedule").isCategoryQuery());
	}
}
