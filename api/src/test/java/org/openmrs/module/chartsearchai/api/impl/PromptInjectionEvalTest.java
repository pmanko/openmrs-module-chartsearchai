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

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
import org.openmrs.module.chartsearchai.eval.EvalReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Eval suite for prompt injection resistance. Tests that
 * {@code stripSpecialTokens} removes all known special tokens and
 * that {@code formatPrompt} preserves system prompt integrity.
 */
public class PromptInjectionEvalTest {

	private static final Logger log = LoggerFactory.getLogger(PromptInjectionEvalTest.class);

	private static final Pattern SPECIAL_TOKEN_PATTERN = Pattern.compile(
			"<\\|[^|]*\\|>"
			+ "|\\[INST\\]|\\[/INST\\]|</s>"
			+ "|<start_of_turn>|<end_of_turn>");

	private static EvalDataset dataset;

	private static EvalDataset getDataset() {
		if (dataset == null) {
			try {
				dataset = EvalDataset.load("eval/prompt-injection-eval-dataset.json");
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return dataset;
	}

	static Stream<Arguments> injectionPayloads() {
		List<Arguments> args = new ArrayList<>();
		for (EvalCase evalCase : getDataset().getCases()) {
			args.add(Arguments.of(evalCase.getId(), evalCase));
		}
		return args.stream();
	}

	static Stream<Arguments> payloadTemplateCombinations() {
		List<Arguments> args = new ArrayList<>();
		for (Map.Entry<String, String> entry : LlmProvider.PRESET_TEMPLATES.entrySet()) {
			String templateName = entry.getKey();
			for (EvalCase evalCase : getDataset().getCases()) {
				if (evalCase.getPayload() != null && !evalCase.getPayload().isEmpty()) {
					args.add(Arguments.of(evalCase.getId() + "/" + templateName,
							evalCase, templateName));
				}
			}
		}
		return args.stream();
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("injectionPayloads")
	public void stripSpecialTokens_perPayload(String caseId, EvalCase evalCase) {
		String payload = evalCase.getPayload();
		String sanitized = LlmProvider.stripSpecialTokens(payload);

		boolean hasTokens = SPECIAL_TOKEN_PATTERN.matcher(sanitized).find();

		log.info("[{}] hasTokensAfterSanitize={}", caseId, hasTokens);

		Map<String, Object> metrics = new LinkedHashMap<>();
		metrics.put("sanitized", !hasTokens);
		EvalReporter.appendResult("prompt-injection-strip", caseId, metrics);

		assertFalse(hasTokens,
				caseId + ": sanitized output still contains special tokens: " + sanitized);
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("payloadTemplateCombinations")
	public void formatPrompt_perPayloadTemplate(String label, EvalCase evalCase,
			String templateName) {
		String systemPrompt = "You are a clinical assistant.";
		String payload = evalCase.getPayload();

		String prompt = LlmProvider.formatPrompt(templateName, systemPrompt, payload);

		// System prompt should appear exactly once
		int firstIdx = prompt.indexOf(systemPrompt);
		int lastIdx = prompt.lastIndexOf(systemPrompt);
		assertTrue(firstIdx >= 0,
				label + ": system prompt missing");
		assertEquals(firstIdx, lastIdx,
				label + ": system prompt appears multiple times");

		// Verify that the sanitized user content is in the prompt
		String sanitizedPayload = LlmProvider.stripSpecialTokens(payload);
		assertTrue(prompt.contains(sanitizedPayload),
				label + ": sanitized payload not found in prompt");

		// The original unsanitized payload should NOT appear if tokens were removed
		if (!payload.equals(sanitizedPayload) && !sanitizedPayload.isEmpty()) {
			assertFalse(prompt.contains(payload),
					label + ": original unsanitized payload found in prompt");
		}

		Map<String, Object> metrics = new LinkedHashMap<>();
		metrics.put("systemPromptIntact", true);
		EvalReporter.appendResult("prompt-injection-format", label, metrics);
	}

	@Test
	public void stripSpecialTokens_shouldHandleNullAndEmpty() {
		assertEquals("", LlmProvider.stripSpecialTokens(null));
		assertEquals("", LlmProvider.stripSpecialTokens(""));
	}
}
