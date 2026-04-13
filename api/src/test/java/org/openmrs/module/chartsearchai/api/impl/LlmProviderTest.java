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

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Pure unit tests for {@link LlmProvider} response parsing and configuration logic.
 */
public class LlmProviderTest {

	@Test
	public void defaultSystemPrompt_shouldMentionClinicalAssistant() {
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("clinical assistant"));
	}

	@Test
	public void defaultSystemPrompt_shouldRequireCitations() {
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("[1]") || LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("[2]"));
	}

	@Test
	public void defaultSystemPrompt_shouldConstrainToQuestionAsked() {
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("ONLY the specific query"));
	}

	@Test
	public void defaultSystemPrompt_shouldRequireAllRelevantRecords() {
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("ALL relevant records"),
				"System prompt must instruct LLM to include ALL relevant records");
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("never omit"),
				"System prompt must explicitly tell LLM not to omit records for brevity");
	}

	@Test
	public void extractResponse_shouldParseJsonWithCitations() {
		String response = "{\"answer\": \"The patient has Hypertension [48] and Diabetes [49].\", \"citations\": [48, 49]}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("The patient has Hypertension [48] and Diabetes [49].", result.getAnswer());
		assertEquals(Arrays.asList(48, 49), result.getCitations());
	}

	@Test
	public void extractResponse_shouldHandleEmptyCitations() {
		String response = "{\"answer\": \"There are no records about diabetes in this patient's chart.\", \"citations\": []}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("There are no records about diabetes in this patient's chart.", result.getAnswer());
		assertTrue(result.getCitations().isEmpty());
	}

	@Test
	public void extractResponse_shouldHandleEscapedQuotes() {
		String response = "{\"answer\": \"The patient said \\\"I feel fine\\\" during the visit.\", \"citations\": []}";
		assertEquals("The patient said \"I feel fine\" during the visit.",
				LlmProvider.extractResponse(response).getAnswer());
	}

	@Test
	public void extractResponse_shouldHandleEscapedBackslashes() {
		String response = "{\"answer\": \"Path: C:\\\\Users\\\\data\", \"citations\": []}";
		assertEquals("Path: C:\\Users\\data",
				LlmProvider.extractResponse(response).getAnswer());
	}

	@Test
	public void extractResponse_shouldHandleEscapedNewlines() {
		String response = "{\"answer\": \"Line 1\\nLine 2\", \"citations\": []}";
		assertEquals("Line 1\nLine 2",
				LlmProvider.extractResponse(response).getAnswer());
	}

	@Test
	public void extractResponse_shouldHandleEmptyString() {
		LlmProvider.LlmResponse result = LlmProvider.extractResponse("");
		assertEquals("", result.getAnswer());
		assertTrue(result.getCitations().isEmpty());
	}

	@Test
	public void extractResponse_shouldFallBackToRawResponseWhenNotJson() {
		String response = "The patient has Diabetes [1].";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals(response, result.getAnswer());
		assertTrue(result.getCitations().isEmpty());
	}

	@Test
	public void extractResponse_shouldDecodeUnicodeEscapes() {
		String response = "{\"answer\": \"Temperature is 38.9\\u00b0C\", \"citations\": [1]}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("Temperature is 38.9\u00b0C", result.getAnswer());
		assertEquals(Arrays.asList(1), result.getCitations());
	}

	@Test
	public void extractResponse_shouldHandleWhitespaceInJson() {
		String response = "{ \"answer\" : \"There are no records about diabetes in this patient's chart.\" , \"citations\" : [] }";
		assertEquals("There are no records about diabetes in this patient's chart.",
				LlmProvider.extractResponse(response).getAnswer());
	}

	@Test
	public void extractResponse_shouldHandleMissingCitationsField() {
		String response = "{\"answer\": \"Some answer.\"}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("Some answer.", result.getAnswer());
		assertTrue(result.getCitations().isEmpty());
	}

	@Test
	public void normalizeSlashCitations_shouldConvertSlashesToSeparateBrackets() {
		assertEquals("Tuberculosis [1], [2] and Malaria [3], [4]",
				LlmProvider.normalizeSlashCitations("Tuberculosis [1/2] and Malaria [3/4]"));
	}

	@Test
	public void normalizeSlashCitations_shouldHandleTripleSlash() {
		assertEquals("Infections [5], [12], [15]",
				LlmProvider.normalizeSlashCitations("Infections [5/12/15]"));
	}

	@Test
	public void normalizeSlashCitations_shouldLeaveNormalCitationsUnchanged() {
		assertEquals("Has condition [1], [2].",
				LlmProvider.normalizeSlashCitations("Has condition [1], [2]."));
	}

	@Test
	public void normalizeSlashCitations_shouldLeaveTextWithNoCitationsUnchanged() {
		assertEquals("No citations here.",
				LlmProvider.normalizeSlashCitations("No citations here."));
	}

	@Test
	public void extractResponse_shouldNormalizeSlashCitationsInAnswer() {
		String response = "{\"answer\": \"Infections [7/13] and HIV [4/11].\", \"citations\": [7, 13, 4, 11]}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("Infections [7], [13] and HIV [4], [11].", result.getAnswer());
		assertEquals(Arrays.asList(7, 13, 4, 11), result.getCitations());
	}

	@Test
	public void extractResponse_shouldPreserveClinicalSlashTerms() {
		String response = "{\"answer\": \"The patient has HIV/AIDS [1] and nausea/vomiting [2].\", \"citations\": [1, 2]}";
		LlmProvider.LlmResponse result = LlmProvider.extractResponse(response);
		assertEquals("The patient has HIV/AIDS [1] and nausea/vomiting [2].", result.getAnswer());
	}

	@Test
	public void defaultRepeatPenalty_shouldBeDisabledToAllowFullEnumeration() {
		assertEquals(1.0f, ChartSearchAiConstants.DEFAULT_REPEAT_PENALTY,
				"Repeat penalty must be 1.0 (disabled) to allow complete enumeration");
	}

}
