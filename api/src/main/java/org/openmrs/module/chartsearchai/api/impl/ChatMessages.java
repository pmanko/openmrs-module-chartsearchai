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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.openmrs.module.chartsearchai.model.ChatMessage;

final class ChatMessages {

	private ChatMessages() {
	}

	static ArrayNode systemAndUser(ObjectMapper mapper, String system, String user) {
		ArrayNode messages = mapper.createArrayNode();
		appendMessage(messages, mapper, "system", system);
		appendMessage(messages, mapper, "user", user);
		return messages;
	}

	/**
	 * Build a multi-turn OpenAI-compatible {@code messages[]} array with
	 * drop-oldest token-budget trimming.
	 *
	 * <p>The system prompt and the current user message are always emitted.
	 * Prior turns are walked from newest backward and added in user/assistant
	 * pairs until either the prior list is exhausted or the next pair would
	 * exceed {@code maxTokens}. Pairs are kept together so the LLM never sees
	 * a dangling assistant turn without its preceding user.
	 *
	 * <p>Token counts are estimated as {@code ceil(length / 4)} characters per
	 * token — sufficient for budget trimming heuristics without taking a hard
	 * dependency on a model-specific tokenizer.
	 */
	static ArrayNode fromTurns(ObjectMapper mapper, String system, List<ChatMessage> prior,
			String currentUser, int maxTokens) {
		ArrayNode messages = mapper.createArrayNode();
		appendMessage(messages, mapper, "system", system);

		int remaining = maxTokens - estimateTokens(system) - estimateTokens(currentUser);

		List<ChatMessage> kept = new ArrayList<>();
		int i = prior == null ? -1 : prior.size() - 1;
		while (i >= 0 && remaining > 0) {
			ChatMessage last = prior.get(i);
			ChatMessage secondLast = i >= 1 ? prior.get(i - 1) : null;

			int pairCost;
			int step;
			if (secondLast != null
					&& ChatMessage.ROLE_ASSISTANT.equals(last.getRole())
					&& ChatMessage.ROLE_USER.equals(secondLast.getRole())) {
				pairCost = estimateTokens(last.getContent()) + estimateTokens(secondLast.getContent());
				step = 2;
			} else {
				pairCost = estimateTokens(last.getContent());
				step = 1;
			}

			if (pairCost > remaining) {
				break;
			}

			kept.add(last);
			if (step == 2) {
				kept.add(secondLast);
			}
			i -= step;
			remaining -= pairCost;
		}

		Collections.reverse(kept);
		for (ChatMessage m : kept) {
			appendMessage(messages, mapper, m.getRole(), m.getContent());
		}
		appendMessage(messages, mapper, "user", currentUser);
		return messages;
	}

	private static void appendMessage(ArrayNode messages, ObjectMapper mapper, String role, String content) {
		ObjectNode node = mapper.createObjectNode();
		node.put("role", role);
		node.put("content", content);
		messages.add(node);
	}

	private static int estimateTokens(String s) {
		if (s == null || s.isEmpty()) {
			return 0;
		}
		return (int) Math.ceil(s.length() / 4.0);
	}
}
