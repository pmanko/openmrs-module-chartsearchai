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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.impl.QueryClassifier.QueryIntent;

public class QueryClassifierTest {

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
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
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
	public void classify_shouldDetectCategoryQuery() {
		QueryIntent intent = QueryClassifier.classify("any medications");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldDetectListAllAsCategory() {
		QueryIntent intent = QueryClassifier.classify("list all conditions");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldNotBeCategoryWithoutTypeMatch() {
		// "tell me about the patient" has a category indicator but no type match
		QueryIntent intent = QueryClassifier.classify("tell about patient");
		assertFalse(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().isEmpty());
	}

	@Test
	public void classify_shouldNotBeCategoryForSpecificQueries() {
		// "Metformin dose" is specific — no category indicator
		QueryIntent intent = QueryClassifier.classify("metformin dose");
		assertFalse(intent.isCategoryQuery());
		// Still matches medication type for boosting purposes
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldReturnEmptyForUnrelatedQuery() {
		QueryIntent intent = QueryClassifier.classify("teacher schedule");
		assertTrue(intent.getTargetTypes().isEmpty());
		assertFalse(intent.isCategoryQuery());
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
		// "medications and allergies" should match both types
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
	public void classify_shouldDetectShowAsCategory() {
		QueryIntent intent = QueryClassifier.classify("show allergies");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY));
	}
}
