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

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;

public class LlmInferenceServiceTest {

	@Test
	public void extractCitedReferences_shouldExtractReferencesFromCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456),
				new RecordMapping(2, "order", 201));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"The patient is on Metformin [1] and has hypertension [2].", mappings);

		assertEquals(2, result.size());
		assertEquals(1, result.get(0).getIndex());
		assertEquals("obs", result.get(0).getResourceType());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
		assertEquals(2, result.get(1).getIndex());
		assertEquals("order", result.get(1).getResourceType());
		assertEquals(Integer.valueOf(201), result.get(1).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldReturnEmptyWhenNoCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"I could not find relevant information in the records.", mappings);

		assertTrue(result.isEmpty());
	}

	@Test
	public void extractCitedReferences_shouldDeduplicateRepeatedCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"Blood pressure was 120/80 [1]. This is within normal range [1].", mappings);

		assertEquals(1, result.size());
		assertEquals(Integer.valueOf(456), result.get(0).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldHandleCommaSeparatedCitations() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 101),
				new RecordMapping(2, "obs", 102),
				new RecordMapping(3, "obs", 103));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"Blood pressure is 133/57 mmHg [1, 2, 3].", mappings);

		assertEquals(3, result.size());
		assertEquals(Integer.valueOf(101), result.get(0).getResourceId());
		assertEquals(Integer.valueOf(102), result.get(1).getResourceId());
		assertEquals(Integer.valueOf(103), result.get(2).getResourceId());
	}

	@Test
	public void extractCitedReferences_shouldHandleEmptyAnswer() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "obs", 456));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences("", mappings);

		assertTrue(result.isEmpty());
	}

	@Test
	public void extractCitedReferences_shouldPreserveOrder() {
		List<RecordMapping> mappings = Arrays.asList(
				new RecordMapping(1, "condition", 50),
				new RecordMapping(2, "order", 30),
				new RecordMapping(3, "obs", 999));

		List<RecordReference> result = LlmInferenceService.extractCitedReferences(
				"The patient has [1] and takes [2] with labs [3].", mappings);

		assertEquals(3, result.size());
		assertEquals("condition", result.get(0).getResourceType());
		assertEquals(Integer.valueOf(50), result.get(0).getResourceId());
		assertEquals("order", result.get(1).getResourceType());
		assertEquals(Integer.valueOf(30), result.get(1).getResourceId());
		assertEquals("obs", result.get(2).getResourceType());
		assertEquals(Integer.valueOf(999), result.get(2).getResourceId());
	}
}
