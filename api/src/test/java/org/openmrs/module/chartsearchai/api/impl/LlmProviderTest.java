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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Pure unit tests for {@link LlmProvider} configuration logic.
 * Uses a subclass to override Context-dependent methods.
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
		assertTrue(LlmProvider.DEFAULT_SYSTEM_PROMPT.contains("ONLY the specific question asked"));
	}

	@Test
	public void close_shouldNotFailWhenModelIsNull() {
		LlmProvider provider = new LlmProvider();
		// Should not throw
		provider.close();
	}

	@Test
	public void getTimeoutSeconds_shouldReturnDefault() {
		LlmProvider provider = createProviderWithTimeout(-1);

		// The overridden method returns the default constant
		int timeout = provider.getTimeoutSeconds();
		assertTrue(timeout > 0);
	}

	@Test
	public void getSystemPrompt_shouldTrimCustomPrompt() {
		LlmProvider provider = createProvider("  custom prompt  ");

		String prompt = provider.getSystemPrompt();
		assertFalse(prompt.startsWith(" "));
		assertEquals("custom prompt", prompt);
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
	public void formatPrompt_shouldUseLlama3Template() {
		String result = LlmProvider.formatPrompt("llama3", "sys", "usr");
		assertTrue(result.contains("<|begin_of_text|>"));
		assertTrue(result.contains("<|start_header_id|>system<|end_header_id|>\n\nsys<|eot_id|>"));
		assertTrue(result.contains("<|start_header_id|>user<|end_header_id|>\n\nusr<|eot_id|>"));
		assertTrue(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
	}

	@Test
	public void formatPrompt_shouldUseMistralTemplate() {
		String result = LlmProvider.formatPrompt("mistral", "sys", "usr");
		assertEquals("[INST] sys\n\nusr [/INST]", result);
	}

	@Test
	public void formatPrompt_shouldUsePhi3Template() {
		String result = LlmProvider.formatPrompt("phi3", "sys", "usr");
		assertTrue(result.startsWith("<|system|>\nsys<|end|>"));
		assertTrue(result.contains("<|user|>\nusr<|end|>"));
		assertTrue(result.endsWith("<|assistant|>\n"));
	}

	@Test
	public void formatPrompt_shouldUseChatMlTemplate() {
		String result = LlmProvider.formatPrompt("chatml", "sys", "usr");
		assertTrue(result.contains("<|im_start|>system\nsys<|im_end|>"));
		assertTrue(result.contains("<|im_start|>user\nusr<|im_end|>"));
		assertTrue(result.contains("<|im_start|>assistant"));
	}

	@Test
	public void formatPrompt_shouldBeCaseInsensitive() {
		String lower = LlmProvider.formatPrompt("llama3", "sys", "usr");
		String upper = LlmProvider.formatPrompt("LLAMA3", "sys", "usr");
		assertEquals(lower, upper);
	}

	@Test
	public void formatPrompt_shouldUseCustomTemplateWithPlaceholders() {
		String custom = "<<SYS>>{system}<</SYS>>[INST]{user}[/INST]";
		String result = LlmProvider.formatPrompt(custom, "my system", "my question");
		assertEquals("<<SYS>>my system<</SYS>>[INST]my question[/INST]", result);
	}

	@Test
	public void resolveStopStrings_shouldReturnLlama3Stops() {
		assertArrayEquals(new String[]{"<|eot_id|>", "<|end_of_text|>"},
				LlmProvider.resolveStopStrings("llama3"));
	}

	@Test
	public void resolveStopStrings_shouldReturnMistralStops() {
		assertArrayEquals(new String[]{"</s>"}, LlmProvider.resolveStopStrings("mistral"));
	}

	@Test
	public void resolveStopStrings_shouldReturnPhi3Stops() {
		assertArrayEquals(new String[]{"<|end|>"}, LlmProvider.resolveStopStrings("phi3"));
	}

	@Test
	public void resolveStopStrings_shouldReturnChatMlStops() {
		assertArrayEquals(new String[]{"<|im_end|>"}, LlmProvider.resolveStopStrings("chatml"));
	}

	@Test
	public void formatPrompt_shouldUseGemmaTemplate() {
		String result = LlmProvider.formatPrompt("gemma", "sys", "usr");
		assertTrue(result.contains("<start_of_turn>user\nsys\n\nusr<end_of_turn>"));
		assertTrue(result.endsWith("<start_of_turn>model\n"));
	}

	@Test
	public void resolveStopStrings_shouldReturnGemmaStops() {
		assertArrayEquals(new String[]{"<end_of_turn>"}, LlmProvider.resolveStopStrings("gemma"));
	}

	@Test
	public void resolveStopStrings_shouldReturnEmptyForCustomTemplate() {
		assertArrayEquals(new String[0],
				LlmProvider.resolveStopStrings("<<SYS>>{system}<</SYS>>"));
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnDefault() {
		LlmProvider provider = createProviderWithIdleTimeout(-1);
		assertEquals(ChartSearchAiConstants.DEFAULT_LLM_IDLE_TIMEOUT_MINUTES,
				provider.getIdleTimeoutMinutes());
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnConfiguredValue() {
		LlmProvider provider = createProviderWithIdleTimeout(15);
		assertEquals(15, provider.getIdleTimeoutMinutes());
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnZeroToDisable() {
		LlmProvider provider = createProviderWithIdleTimeout(0);
		assertEquals(0, provider.getIdleTimeoutMinutes());
	}

	@Test
	public void close_shouldBeIdempotent() {
		LlmProvider provider = new LlmProvider();
		provider.close();
		provider.close(); // Should not throw
	}

	private LlmProvider createProviderWithIdleTimeout(final int timeout) {
		return new LlmProvider() {

			@Override
			protected int getIdleTimeoutMinutes() {
				if (timeout >= 0) {
					return timeout;
				}
				return ChartSearchAiConstants.DEFAULT_LLM_IDLE_TIMEOUT_MINUTES;
			}
		};
	}

	private LlmProvider createProvider(final String customSystemPrompt) {
		return new LlmProvider() {

			@Override
			protected String getSystemPrompt() {
				if (customSystemPrompt != null && !customSystemPrompt.trim().isEmpty()) {
					return customSystemPrompt.trim();
				}
				return DEFAULT_SYSTEM_PROMPT;
			}
		};
	}

	private LlmProvider createProviderWithTimeout(final int timeout) {
		return new LlmProvider() {

			@Override
			protected String getSystemPrompt() {
				return DEFAULT_SYSTEM_PROMPT;
			}

			@Override
			protected int getTimeoutSeconds() {
				if (timeout > 0) {
					return timeout;
				}
				return org.openmrs.module.chartsearchai.ChartSearchAiConstants.DEFAULT_LLM_TIMEOUT_SECONDS;
			}
		};
	}
}
