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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.model.ChatMessage;

/**
 * Contract tests for the multi-turn message-array builder. Each test exercises
 * {@link ChatMessages#fromTurns} directly — that is the actual production code
 * path used by {@link LlmProvider} when assembling the LLM request body.
 *
 * <p>The token-budget trim is the load-bearing behavior here. If these tests
 * pass while the implementation has regressed to single-shot {@code [system,
 * user]}, the multi-turn feature is broken in production.
 */
public class ChatMessagesFromTurnsTest {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private static final String SYSTEM = "You are a clinical assistant.";

	private static final int LARGE_BUDGET = 100_000;

	@Test
	public void emptyPriorTurnsShouldMatchSystemAndUserShape() {
		ArrayNode out = ChatMessages.fromTurns(
				MAPPER, SYSTEM, Collections.emptyList(), "What meds is she on?", LARGE_BUDGET);

		ArrayNode reference = ChatMessages.systemAndUser(MAPPER, SYSTEM, "What meds is she on?");
		assertEquals(reference.toString(), out.toString(),
				"empty prior turns must produce the same shape as systemAndUser");
	}

	@Test
	public void oneFullTurnShouldProduceFourMessagesInRoleOrder() {
		List<ChatMessage> prior = Arrays.asList(
				turn(ChatMessage.ROLE_USER, "What meds is she on?"),
				turn(ChatMessage.ROLE_ASSISTANT, "Lisinopril 10mg and Metformin 500mg."));

		ArrayNode out = ChatMessages.fromTurns(
				MAPPER, SYSTEM, prior, "What are typical dose ranges?", LARGE_BUDGET);

		assertEquals(4, out.size(),
				"expected [system, user1, assistant1, user2]; got " + out);

		assertRole(out, 0, "system", SYSTEM);
		assertRole(out, 1, "user", "What meds is she on?");
		assertRole(out, 2, "assistant", "Lisinopril 10mg and Metformin 500mg.");
		assertRole(out, 3, "user", "What are typical dose ranges?");
	}

	@Test
	public void systemPromptMustNeverBeDropped() {
		// Many large prior turns, very small budget — system must still survive.
		List<ChatMessage> prior = new ArrayList<>();
		for (int i = 0; i < 20; i++) {
			prior.add(turn(ChatMessage.ROLE_USER, repeat("a", 400)));
			prior.add(turn(ChatMessage.ROLE_ASSISTANT, repeat("b", 400)));
		}

		ArrayNode out = ChatMessages.fromTurns(
				MAPPER, SYSTEM, prior, "current question", 50);

		assertTrue(out.size() >= 2, "must contain at least system + current user");
		assertRole(out, 0, "system", SYSTEM);
		assertRole(out, out.size() - 1, "user", "current question");
	}

	@Test
	public void overBudgetShouldDropOldestPairsKeepingNewest() {
		// 4 pairs, each ~50 tokens (200 chars / 4). System is small, current is small.
		// Budget = system + 2 pairs + current ≈ 5 + 200 + 5 ≈ 210 tokens worth.
		String filler200 = repeat("x", 200);
		List<ChatMessage> prior = Arrays.asList(
				turn(ChatMessage.ROLE_USER, "OLDEST-user-" + filler200),
				turn(ChatMessage.ROLE_ASSISTANT, "OLDEST-assist-" + filler200),
				turn(ChatMessage.ROLE_USER, "MID1-user-" + filler200),
				turn(ChatMessage.ROLE_ASSISTANT, "MID1-assist-" + filler200),
				turn(ChatMessage.ROLE_USER, "MID2-user-" + filler200),
				turn(ChatMessage.ROLE_ASSISTANT, "MID2-assist-" + filler200),
				turn(ChatMessage.ROLE_USER, "NEWEST-user-" + filler200),
				turn(ChatMessage.ROLE_ASSISTANT, "NEWEST-assist-" + filler200));

		int budget = 250; // tokens — fits system + ~2 pairs + current

		ArrayNode out = ChatMessages.fromTurns(MAPPER, SYSTEM, prior, "current", budget);

		// MUST keep the NEWEST pair; MUST have dropped the OLDEST.
		String body = out.toString();
		assertTrue(body.contains("NEWEST-user-") && body.contains("NEWEST-assist-"),
				"newest turn must be retained; got " + body);
		assertFalse(body.contains("OLDEST-user-") || body.contains("OLDEST-assist-"),
				"oldest turn must be dropped under budget pressure; got " + body);

		// Ordering invariants: system first, current last, alternation preserved.
		assertEquals("system", out.get(0).get("role").asText());
		assertEquals("user", out.get(out.size() - 1).get("role").asText());
		assertEquals("current", out.get(out.size() - 1).get("content").asText());

		// Adjacent priors must remain in user→assistant pairs (no orphaned assistant
		// without a preceding user). Walk the middle; every assistant must be
		// preceded by a user.
		for (int i = 1; i < out.size() - 1; i++) {
			String role = out.get(i).get("role").asText();
			if ("assistant".equals(role)) {
				String previousRole = out.get(i - 1).get("role").asText();
				assertEquals("user", previousRole,
						"assistant at " + i + " must follow a user turn; got " + previousRole);
			}
		}
	}

	@Test
	public void zeroBudgetMustStillEmitSystemAndCurrentUser() {
		// Even with no prior budget, the new question must reach the LLM —
		// otherwise the call is meaningless.
		List<ChatMessage> prior = Arrays.asList(
				turn(ChatMessage.ROLE_USER, "ignored prior"),
				turn(ChatMessage.ROLE_ASSISTANT, "ignored response"));

		ArrayNode out = ChatMessages.fromTurns(MAPPER, SYSTEM, prior, "the new question", 0);

		assertEquals(2, out.size(),
				"budget=0 must still produce [system, current user]; got " + out);
		assertRole(out, 0, "system", SYSTEM);
		assertRole(out, 1, "user", "the new question");
	}

	@Test
	public void singleShotCallSiteMustNotSurviveRegression() {
		// Guard against accidental regression to systemAndUser-only:
		// if fromTurns silently drops prior turns even when they fit the budget,
		// this test catches it. Two pairs, generous budget — output MUST be 6 nodes.
		List<ChatMessage> prior = Arrays.asList(
				turn(ChatMessage.ROLE_USER, "a"),
				turn(ChatMessage.ROLE_ASSISTANT, "b"),
				turn(ChatMessage.ROLE_USER, "c"),
				turn(ChatMessage.ROLE_ASSISTANT, "d"));

		ArrayNode out = ChatMessages.fromTurns(MAPPER, SYSTEM, prior, "e", LARGE_BUDGET);

		assertEquals(6, out.size(),
				"with budget headroom, every prior turn must be carried through. "
						+ "If this fails with size=2, fromTurns has regressed to single-shot. "
						+ "got " + out);
	}

	private static ChatMessage turn(String role, String content) {
		ChatMessage m = new ChatMessage();
		m.setRole(role);
		m.setContent(content);
		return m;
	}

	private static void assertRole(ArrayNode arr, int index, String role, String content) {
		JsonNode node = arr.get(index);
		assertEquals(role, node.get("role").asText(),
				"role mismatch at index " + index + " in " + arr);
		assertEquals(content, node.get("content").asText(),
				"content mismatch at index " + index + " in " + arr);
	}

	private static String repeat(String s, int n) {
		StringBuilder sb = new StringBuilder(s.length() * n);
		for (int i = 0; i < n; i++) {
			sb.append(s);
		}
		return sb.toString();
	}
}
