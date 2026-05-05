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

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Verifies the warmup contract: the user-message prefix sent during {@code warmup}
 * MUST be a byte-prefix of the user-message sent during a real query on the same
 * chart, otherwise llama-server's KV-cache prefix match breaks and the warmup
 * is wasted work.
 *
 * <p>Both production paths (search, searchStreaming, warmup) call
 * {@link LlmProvider#buildUserMessage} — testing through that helper exercises
 * the same code production uses, per the project's "no reimplementation in tests"
 * rule.
 */
public class LlmProviderUserMessageTest {

	private static final String CHART = "[1] (2025-01-15) Clinical observation: BP 120/80\n"
			+ "[2] (2025-01-10) Diagnosis: Hypertension";

	@Test
	public void warmupUserMessageShouldBePrefixOfRealQuery() {
		String warmup = LlmProvider.buildUserMessage(CHART, "");
		String realQuery = LlmProvider.buildUserMessage(CHART, "What is the patient's BP?");

		assertTrue(realQuery.startsWith(warmup),
				"warmup must produce a byte-prefix of the real query so llama-server's "
				+ "--cache-reuse 256 can match the cached prefix tokens. If this ever "
				+ "fails, the warmup is silently wasted: the prefix diverges before the "
				+ "question and llama-server reprocesses the whole chart from scratch.\n"
				+ "  warmup:    " + warmup + "\n"
				+ "  realQuery: " + realQuery);
	}

	@Test
	public void buildUserMessageShouldVaryByQuestion() {
		String a = LlmProvider.buildUserMessage(CHART, "What is the BP?");
		String b = LlmProvider.buildUserMessage(CHART, "Any allergies?");
		assertNotEquals(a, b,
				"sanity check: changing the question must change the user message, "
				+ "otherwise the prefix-match assertion above would pass trivially");
	}

	@Test
	public void warmupUserMessageShouldEndWithEmptyQueryMarker() {
		String warmup = LlmProvider.buildUserMessage(CHART, "");
		// The trailing "Clinician's query: " is what llama-server caches up to;
		// when the real query arrives, only the question text after it is new.
		assertTrue(warmup.endsWith("Clinician's query: "),
				"warmup must end with the 'Clinician's query: ' marker so the cache "
				+ "reuses everything up to that point and only the question text is new "
				+ "tokens to process. Got: '" + warmup + "'");
	}

	@Test
	public void buildUserMessageShouldIncludeRecordsHeader() {
		String msg = LlmProvider.buildUserMessage(CHART, "Q?");
		assertTrue(msg.startsWith("Patient records (most recent first):\n"),
				"the records-header is part of the cached prefix; if it changes, every "
				+ "previously-cached patient's KV cache is silently invalidated");
	}

	@Test
	public void buildUserMessageShouldHandleEmptyChart() {
		// When the embedding pre-filter narrows to no records, normalizeRecords
		// substitutes a placeholder. Both warmup and search must agree on it,
		// otherwise switching pre-filter on/off mid-session would break caching.
		String warmupEmpty = LlmProvider.buildUserMessage("", "");
		String realEmpty = LlmProvider.buildUserMessage("", "Q?");
		assertTrue(realEmpty.startsWith(warmupEmpty),
				"empty-chart prefix must still match between warmup and real query");
	}
}
