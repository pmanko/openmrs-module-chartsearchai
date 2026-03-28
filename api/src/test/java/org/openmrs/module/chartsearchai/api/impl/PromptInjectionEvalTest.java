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
import java.util.Map;
import java.util.regex.Pattern;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
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

	@BeforeAll
	static void loadDataset() throws IOException {
		dataset = EvalDataset.load("eval/prompt-injection-eval-dataset.json");
	}

	@Test
	public void stripSpecialTokens_shouldRemoveAllKnownTokens() {
		int passed = 0;
		int total = 0;

		for (EvalCase evalCase : dataset.getCases()) {
			total++;
			String payload = evalCase.getPayload();
			String sanitized = LlmProvider.stripSpecialTokens(payload);

			boolean hasTokens = SPECIAL_TOKEN_PATTERN.matcher(sanitized).find();
			if (!hasTokens) {
				passed++;
			} else {
				log.warn("[{}] FAILED: sanitized still contains tokens: {}",
						evalCase.getId(), sanitized);
			}
		}

		log.info("Prompt injection eval: {}/{} payloads fully sanitized", passed, total);
		assertEquals(total, passed,
				"All payloads should be fully sanitized but " + (total - passed) + " still contain tokens");
	}

	@Test
	public void formatPrompt_shouldPreserveSystemPromptIntegrity() {
		String systemPrompt = "You are a clinical assistant.";

		for (Map.Entry<String, String> entry : LlmProvider.PRESET_TEMPLATES.entrySet()) {
			String templateName = entry.getKey();

			for (EvalCase evalCase : dataset.getCases()) {
				String payload = evalCase.getPayload();
				if (payload == null || payload.isEmpty()) {
					continue;
				}

				String prompt = LlmProvider.formatPrompt(templateName, systemPrompt, payload);

				// System prompt should appear exactly once
				int firstIdx = prompt.indexOf(systemPrompt);
				int lastIdx = prompt.lastIndexOf(systemPrompt);
				assertTrue(firstIdx >= 0,
						evalCase.getId() + "/" + templateName + ": system prompt missing");
				assertEquals(firstIdx, lastIdx,
						evalCase.getId() + "/" + templateName + ": system prompt appears multiple times");

				// Verify that the sanitized user content is in the prompt,
				// meaning stripSpecialTokens was applied to the payload.
				String sanitizedPayload = LlmProvider.stripSpecialTokens(payload);
				assertTrue(prompt.contains(sanitizedPayload),
						evalCase.getId() + "/" + templateName
						+ ": sanitized payload not found in prompt");

				// The original unsanitized payload should NOT appear if it
				// differed from the sanitized version (i.e., tokens were removed)
				if (!payload.equals(sanitizedPayload) && !sanitizedPayload.isEmpty()) {
					assertFalse(prompt.contains(payload),
							evalCase.getId() + "/" + templateName
							+ ": original unsanitized payload found in prompt");
				}
			}
		}
	}

	@Test
	public void stripSpecialTokens_shouldHandleNullAndEmpty() {
		assertEquals("", LlmProvider.stripSpecialTokens(null));
		assertEquals("", LlmProvider.stripSpecialTokens(""));
	}
}
