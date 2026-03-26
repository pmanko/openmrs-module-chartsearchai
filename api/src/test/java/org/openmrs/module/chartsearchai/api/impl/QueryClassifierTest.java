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
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
	}

	@Test
	public void classify_shouldDetectDiagnosisQuery() {
		QueryIntent intent = QueryClassifier.classify("any diagnoses");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
		assertTrue(intent.isCategoryQuery());
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

	@Test
	public void classify_shouldDetectAnyConditionsAsCategory() {
		QueryIntent intent = QueryClassifier.classify("any conditions");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
		assertFalse(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS));
	}

	@Test
	public void classify_shouldNotBeCategoryWhenIndicatorFarFromType() {
		// "what" is 4+ words from "count", and "count"/"cd4" are specific
		// concepts, not generic type keywords — should be focused
		QueryIntent intent = QueryClassifier.classify("what is the latest CD4 count?");
		assertFalse(intent.isCategoryQuery());
		// Still detects obs type for boosting
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldBeCategoryWhenIndicatorAdjacentToType() {
		// "what" directly before "medications" → category
		QueryIntent intent = QueryClassifier.classify("what medications does the patient take?");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldNotBeCategoryForSpecificConcept() {
		// "does the patient have diabetes?" — specific concept, no indicator
		// near a generic type word
		QueryIntent intent = QueryClassifier.classify("does the patient have diabetes?");
		assertFalse(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldNotBeCategoryForSpecificVital() {
		// "what is the blood pressure?" — specific vital, not generic "vitals"
		QueryIntent intent = QueryClassifier.classify("what is the blood pressure?");
		assertFalse(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldBeCategoryForShowWithProximity() {
		// "show me the patient's allergies" — 3 words between indicator and type
		QueryIntent intent = QueryClassifier.classify("show me the patient's allergies");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY));
	}

	@Test
	public void classify_shouldBeCategoryForWhatAreTheLabs() {
		QueryIntent intent = QueryClassifier.classify("what are the latest labs?");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldNotBeCategoryForSingularTypeWord() {
		// "what test" uses singular "test" — asking about a specific test,
		// not listing all tests. Only plural forms trigger category.
		QueryIntent intent = QueryClassifier.classify("what test did they run?");
		assertFalse(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_OBS));
	}

	@Test
	public void classify_shouldNotBeCategoryForSingularDrug() {
		QueryIntent intent = QueryClassifier.classify("which drug caused the reaction?");
		assertFalse(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldBeCategoryForEveryWithSingular() {
		// "every" requires singular in English but is inherently exhaustive
		QueryIntent intent = QueryClassifier.classify("every condition listed");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldBeCategoryForEachWithSingular() {
		QueryIntent intent = QueryClassifier.classify("each medication in the regimen");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_ORDER));
	}

	@Test
	public void classify_shouldNotBeCategoryForEveryWithNonType() {
		// "every time" — "time" is not a type word
		QueryIntent intent = QueryClassifier.classify("every time I visit the clinic");
		assertFalse(intent.isCategoryQuery());
	}

	@Test
	public void classify_shouldDetectIllnessesAsCondition() {
		// "illnesses" (plural) must match CONDITION_PATTERN for type detection
		QueryIntent intent = QueryClassifier.classify("illnesses");
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}

	@Test
	public void classify_shouldBeCategoryForAnyIllnesses() {
		// "any illnesses?" must trigger BOTH type detection AND category expansion
		QueryIntent intent = QueryClassifier.classify("any illnesses?");
		assertTrue(intent.isCategoryQuery());
		assertTrue(intent.getTargetTypes().contains(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION));
	}
}
