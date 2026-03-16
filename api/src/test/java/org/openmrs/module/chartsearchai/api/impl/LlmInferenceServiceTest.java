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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;

public class LlmInferenceServiceTest {

	@Test
	public void extractCitedReferences_shouldExtractReferencesFromCitations() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"The patient is on Metformin [Obs #456] and has hypertension [Order #201].");

		assertEquals(2, result.size());
		assertEquals("obs", result.get(0).getResourceType());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
		assertEquals("order", result.get(1).getResourceType());
		assertEquals(Integer.valueOf(201), result.get(1).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldReturnEmptyWhenNoCitations() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"I could not find relevant information in the records.");

		assertTrue(result.isEmpty());
	}

	@Test
	public void extractCitedReferences_shouldDeduplicateRepeatedCitations() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"Blood pressure was 120/80 [Obs #456]. This is within normal range [Obs #456].");

		assertEquals(1, result.size());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldHandleCommaSeparatedCitations() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"Blood pressure is 133/57 mmHg [Obs #101, Obs #102, Obs #103].");

		assertEquals(3, result.size());
		assertEquals(Integer.valueOf(101), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(102), result.get(1).getResourceId());
		assertEquals(Integer.valueOf(103), result.get(2).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldHandleEmptyAnswer() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences("");

		assertTrue(result.isEmpty());
	}

	@Test
	public void extractCitedReferences_shouldPreserveOrder() {
		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"The patient has [Condition #50] and takes [Order #30] with labs [Obs #999].");

		assertEquals(3, result.size());
		assertEquals("condition", result.get(0).getResourceType());
		assertEquals(Integer.valueOf(50), result.get(0).getResourceId());
		assertEquals("order", result.get(1).getResourceType());
		assertEquals(Integer.valueOf(30), result.get(1).getResourceId());
		assertEquals("obs", result.get(2).getResourceType());
		assertEquals(Integer.valueOf(999), result.get(2).getResourceId());
	}
}
