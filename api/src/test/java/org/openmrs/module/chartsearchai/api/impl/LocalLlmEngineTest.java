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
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Pure unit tests for {@link LocalLlmEngine} chat template formatting and configuration.
 */
public class LocalLlmEngineTest {

	@Test
	public void formatPrompt_shouldUseLlama3Template() {
		String result = LocalLlmEngine.formatPrompt("llama3", "sys", "usr");
		assertTrue(result.contains("<|begin_of_text|>"));
		assertTrue(result.contains("<|start_header_id|>system<|end_header_id|>\n\nsys<|eot_id|>"));
		assertTrue(result.contains("<|start_header_id|>user<|end_header_id|>\n\nusr<|eot_id|>"));
		assertTrue(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
	}

	@Test
	public void formatPrompt_shouldUseMistralTemplate() {
		String result = LocalLlmEngine.formatPrompt("mistral", "sys", "usr");
		assertEquals("[INST] sys\n\nusr [/INST]", result);
	}

	@Test
	public void formatPrompt_shouldUsePhi3Template() {
		String result = LocalLlmEngine.formatPrompt("phi3", "sys", "usr");
		assertTrue(result.startsWith("<|system|>\nsys<|end|>"));
		assertTrue(result.contains("<|user|>\nusr<|end|>"));
		assertTrue(result.endsWith("<|assistant|>\n"));
	}

	@Test
	public void formatPrompt_shouldUseChatMlTemplate() {
		String result = LocalLlmEngine.formatPrompt("chatml", "sys", "usr");
		assertTrue(result.contains("<|im_start|>system\nsys<|im_end|>"));
		assertTrue(result.contains("<|im_start|>user\nusr<|im_end|>"));
		assertTrue(result.contains("<|im_start|>assistant"));
	}

	@Test
	public void formatPrompt_shouldBeCaseInsensitive() {
		String lower = LocalLlmEngine.formatPrompt("llama3", "sys", "usr");
		String upper = LocalLlmEngine.formatPrompt("LLAMA3", "sys", "usr");
		assertEquals(lower, upper);
	}

	@Test
	public void formatPrompt_shouldUseCustomTemplateWithPlaceholders() {
		String custom = "<<SYS>>{system}<</SYS>>[INST]{user}[/INST]";
		String result = LocalLlmEngine.formatPrompt(custom, "my system", "my question");
		assertEquals("<<SYS>>my system<</SYS>>[INST]my question[/INST]", result);
	}

	@Test
	public void formatPrompt_shouldUseGemmaTemplate() {
		String result = LocalLlmEngine.formatPrompt("gemma", "sys", "usr");
		assertTrue(result.contains("<start_of_turn>user\nsys\n\nusr<end_of_turn>"));
		assertTrue(result.endsWith("<start_of_turn>model\n"));
	}

	@Test
	public void resolveStopStrings_shouldReturnLlama3Stops() {
		assertArrayEquals(new String[]{"<|eot_id|>", "<|end_of_text|>"},
				LocalLlmEngine.resolveStopStrings("llama3"));
	}

	@Test
	public void resolveStopStrings_shouldReturnMistralStops() {
		assertArrayEquals(new String[]{"</s>"}, LocalLlmEngine.resolveStopStrings("mistral"));
	}

	@Test
	public void resolveStopStrings_shouldReturnPhi3Stops() {
		assertArrayEquals(new String[]{"<|end|>"}, LocalLlmEngine.resolveStopStrings("phi3"));
	}

	@Test
	public void resolveStopStrings_shouldReturnChatMlStops() {
		assertArrayEquals(new String[]{"<|im_end|>"}, LocalLlmEngine.resolveStopStrings("chatml"));
	}

	@Test
	public void resolveStopStrings_shouldReturnGemmaStops() {
		assertArrayEquals(new String[]{"<end_of_turn>"}, LocalLlmEngine.resolveStopStrings("gemma"));
	}

	@Test
	public void resolveStopStrings_shouldReturnEmptyForCustomTemplate() {
		assertArrayEquals(new String[0],
				LocalLlmEngine.resolveStopStrings("<<SYS>>{system}<</SYS>>"));
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnDefault() {
		LocalLlmEngine engine = createEngineWithIdleTimeout(-1);
		assertEquals(ChartSearchAiConstants.DEFAULT_LLM_IDLE_TIMEOUT_MINUTES,
				engine.getIdleTimeoutMinutes());
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnConfiguredValue() {
		LocalLlmEngine engine = createEngineWithIdleTimeout(15);
		assertEquals(15, engine.getIdleTimeoutMinutes());
	}

	@Test
	public void getIdleTimeoutMinutes_shouldReturnZeroToDisable() {
		LocalLlmEngine engine = createEngineWithIdleTimeout(0);
		assertEquals(0, engine.getIdleTimeoutMinutes());
	}

	@Test
	public void close_shouldNotFailWhenModelIsNull() {
		LocalLlmEngine engine = new LocalLlmEngine();
		engine.close();
	}

	@Test
	public void close_shouldBeIdempotent() {
		LocalLlmEngine engine = new LocalLlmEngine();
		engine.close();
		engine.close();
	}

	private LocalLlmEngine createEngineWithIdleTimeout(final int timeout) {
		return new LocalLlmEngine() {

			@Override
			protected int getIdleTimeoutMinutes() {
				if (timeout >= 0) {
					return timeout;
				}
				return ChartSearchAiConstants.DEFAULT_LLM_IDLE_TIMEOUT_MINUTES;
			}
		};
	}
}
