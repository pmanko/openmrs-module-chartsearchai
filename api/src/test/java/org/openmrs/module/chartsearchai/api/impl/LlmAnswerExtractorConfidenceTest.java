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
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.api.impl.LlmProvider.LlmResponse;

/**
 * Contract tests for the per-section validator {@code confidence} the med-agent-hub emits
 * alongside {@code answer/citations/blocks}: {@code {answer:{level,note}, in_depth:{level,note}}}
 * with {@code level} ∈ green|yellow|red. The parser must surface it on {@link LlmResponse} so the
 * chat envelope and SPA can render the confidence tag the dashboard already shows. Mirrors the
 * blocks-pass-through guarantees:
 *
 * <ol>
 *   <li>Populate {@link LlmResponse#getConfidence()} from the JSON when present, preserving the
 *       nested {level,note} structure.</li>
 *   <li>Return {@code null} when the field is absent (LM Studio / parity lane / legacy rows) — not
 *       an empty map that would render a phantom tag.</li>
 *   <li>The token-count overload must thread confidence through, not drop it (the same regression
 *       that once dropped blocks).</li>
 * </ol>
 */
public class LlmAnswerExtractorConfidenceTest {

	@Test
	@SuppressWarnings("unchecked")
	public void parserShouldPopulateConfidenceFromJsonResponse() {
		String json = "{"
				+ "\"answer\":\"**Answer**\\nHgb is 14.0 [3].\\n\\n**In Depth**\\n- normal range [3]\","
				+ "\"citations\":[3],"
				+ "\"confidence\":{"
				+ "  \"answer\":{\"level\":\"green\",\"note\":\"\"},"
				+ "  \"in_depth\":{\"level\":\"yellow\",\"note\":\"one claim re-synthesized\"}"
				+ "}"
				+ "}";

		LlmResponse parsed = LlmAnswerExtractor.extractResponse(json);

		Map<String, Object> conf = parsed.getConfidence();
		assertEquals("green", ((Map<String, Object>) conf.get("answer")).get("level"),
				"answer-section confidence level must survive parsing");
		assertEquals("yellow", ((Map<String, Object>) conf.get("in_depth")).get("level"),
				"in-depth-section confidence level must survive parsing");
		assertEquals("one claim re-synthesized",
				((Map<String, Object>) conf.get("in_depth")).get("note"),
				"the per-section note (rendered as the tag tooltip) must survive parsing");
	}

	@Test
	public void legacyResponseWithoutConfidenceShouldReturnNull() {
		// LM Studio, the parity lane, and pre-confidence chat_message rows carry no `confidence`.
		// The parser must yield null so the UI renders no phantom tag.
		String json = "{\"answer\":\"The patient is 47 years old.\",\"citations\":[1]}";

		LlmResponse parsed = LlmAnswerExtractor.extractResponse(json);

		assertNull(parsed.getConfidence(),
				"a response without a confidence field must yield null, not an empty map");
	}

	@Test
	@SuppressWarnings("unchecked")
	public void tokenCountOverloadShouldPreserveConfidence() {
		// The chat / search paths flow through extractResponse(json, in, out, cached) to attach
		// token counts. That overload must thread confidence through — the exact shape of the
		// regression that once silently dropped blocks.
		String json = "{\"answer\":\"prose\",\"citations\":[],"
				+ "\"confidence\":{\"answer\":{\"level\":\"red\",\"note\":\"unresolved\"}}}";

		LlmResponse parsed = LlmAnswerExtractor.extractResponse(json, 100, 50, 0);

		assertEquals(100, parsed.getInputTokens());
		Map<String, Object> conf = parsed.getConfidence();
		assertEquals("red", ((Map<String, Object>) conf.get("answer")).get("level"),
				"token-count overload must thread confidence through, not drop it");
	}
}
