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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * Orchestrates LLM inference by constructing prompts, delegating to the active
 * {@link LlmEngine}, and parsing responses. The engine is selected via the
 * {@code chartsearchai.llm.engine} global property ({@code local} or {@code remote}).
 */
@Component
public class LlmProvider {

	private static final Logger log = LoggerFactory.getLogger(LlmProvider.class);

	static final String DEFAULT_SYSTEM_PROMPT = "You are a clinical assistant helping a clinician "
			+ "review a patient's chart. Answer ONLY the specific question asked. "
			+ "Use only the patient records below (grouped by type, most recent first within each group). "
			+ "Never infer, assume, or add information not explicitly stated in the records. "
			+ "Include ALL relevant records in your answer — never omit any for brevity. "
			+ "Cite EVERY record you reference by its number in brackets (e.g. [1], [3]). "
			+ "Respond with ONLY a JSON object with an \"answer\" string and a \"citations\" array "
			+ "listing every record number you cited.\n\n"
			+ "If no records are relevant, name what is missing.\n"
			+ "Your answer must not vary based on the punctuation or phrasing of the query "
			+ "— focus only on its semantic meaning.\n\n"
			+ "The following is a FORMAT DEMONSTRATION ONLY using fake non-medical data. "
			+ "Do NOT use any of this data in your answer.\n\n"
			+ "Records:\n"
			+ "[1] (2024-03-10) Fruit delivery: 12 apples\n"
			+ "[2] (2024-02-15) Fruit delivery: 8 oranges\n"
			+ "[3] (2024-01-20) Fruit delivery: 5 apples\n\n"
			+ "Clinician's query: How many apples were delivered?\n"
			+ "{\"answer\": \"12 apples on 2024-03-10 [1] and 5 apples on 2024-01-20 [3].\","
			+ " \"citations\": [1, 3]}\n\n"
			+ "Clinician's query: Were any bananas delivered?\n"
			+ "{\"answer\": \"There are no records of banana deliveries.\", \"citations\": []}\n\n"
			+ "END OF FORMAT DEMONSTRATION. Now answer using ONLY the actual patient records below.";

	@Autowired
	@Qualifier("chartSearchAi.localLlmEngine")
	private LocalLlmEngine localEngine;

	@Autowired
	@Qualifier("chartSearchAi.remoteLlmEngine")
	private RemoteLlmEngine remoteEngine;

	/**
	 * Send numbered patient records and a question to the LLM for synthesis.
	 *
	 * @param numberedRecords the numbered patient records text
	 * @param question the clinician's natural language question
	 * @return the LLM's response with answer text and structured citation indices
	 */
	public LlmResponse search(String numberedRecords, String question) {
		String systemPrompt = getSystemPrompt();
		String userMessage = "Patient records (most recent first):\n" + normalizeRecords(numberedRecords)
				+ "\n\nClinician's query: " + question;
		int timeoutSeconds = getTimeoutSeconds();

		LlmEngine.InferenceResult result = getActiveEngine().infer(
				systemPrompt, userMessage, timeoutSeconds);

		return extractResponse(result.getText(), result.getInputTokens(), result.getOutputTokens());
	}

	/**
	 * Streaming variant of {@link #search}. Calls the tokenConsumer for each token as it is
	 * generated, and returns the full response when complete.
	 *
	 * @param numberedRecords the numbered patient records text
	 * @param question the clinician's natural language question
	 * @param tokenConsumer called with each token fragment as it is generated
	 * @return the complete LLM response with answer text and structured citation indices
	 */
	public LlmResponse searchStreaming(String numberedRecords, String question,
			Consumer<String> tokenConsumer) {
		String systemPrompt = getSystemPrompt();
		String userMessage = "Patient records (most recent first):\n" + normalizeRecords(numberedRecords)
				+ "\n\nClinician's query: " + question;
		int timeoutSeconds = getTimeoutSeconds();

		LlmEngine.InferenceResult result = getActiveEngine().inferStreaming(
				systemPrompt, userMessage, timeoutSeconds, tokenConsumer);

		return extractResponse(result.getText(), result.getInputTokens(), result.getOutputTokens());
	}

	private static String normalizeRecords(String numberedRecords) {
		return (numberedRecords == null || numberedRecords.trim().isEmpty())
				? "This patient has no records matching this query."
				: numberedRecords.stripTrailing();
	}

	static LlmResponse extractResponse(String response, int inputTokens, int outputTokens) {
		LlmResponse parsed = extractResponse(response);
		return new LlmResponse(parsed.getAnswer(), parsed.getCitations(), inputTokens, outputTokens);
	}

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private static final Pattern SLASH_CITATION = Pattern.compile("\\[(\\d+(?:/\\d+)+)\\]");

	static LlmResponse extractResponse(String response) {
		String trimmed = response.trim();
		if (trimmed.isEmpty()) {
			return new LlmResponse(trimmed, Collections.emptyList());
		}

		try {
			JsonNode root = MAPPER.readTree(trimmed);
			JsonNode answerNode = root.get("answer");
			if (answerNode != null && answerNode.isTextual()) {
				String answer = normalizeSlashCitations(answerNode.asText().trim());
				List<Integer> citations = new ArrayList<>();
				JsonNode citationsNode = root.get("citations");
				if (citationsNode != null && citationsNode.isArray()) {
					for (JsonNode n : citationsNode) {
						if (n.isInt()) {
							citations.add(n.asInt());
						}
					}
				}
				return new LlmResponse(answer, citations);
			}
		}
		catch (IOException e) {
			log.warn("LLM response did not contain expected JSON format, returning raw response");
		}
		return new LlmResponse(trimmed, Collections.emptyList());
	}

	static String normalizeSlashCitations(String text) {
		Matcher matcher = SLASH_CITATION.matcher(text);
		if (!matcher.find()) {
			return text;
		}
		StringBuffer sb = new StringBuffer();
		matcher.reset();
		while (matcher.find()) {
			String[] parts = matcher.group(1).split("/");
			StringBuilder replacement = new StringBuilder();
			for (int i = 0; i < parts.length; i++) {
				if (i > 0) {
					replacement.append(", ");
				}
				replacement.append("[").append(parts[i]).append("]");
			}
			matcher.appendReplacement(sb, Matcher.quoteReplacement(replacement.toString()));
		}
		matcher.appendTail(sb);
		return sb.toString();
	}

	static class LlmResponse {

		private final String answer;

		private final List<Integer> citations;

		private final int inputTokens;

		private final int outputTokens;

		LlmResponse(String answer, List<Integer> citations) {
			this(answer, citations, 0, 0);
		}

		LlmResponse(String answer, List<Integer> citations, int inputTokens, int outputTokens) {
			this.answer = answer;
			this.citations = Collections.unmodifiableList(new ArrayList<>(citations));
			this.inputTokens = inputTokens;
			this.outputTokens = outputTokens;
		}

		String getAnswer() {
			return answer;
		}

		List<Integer> getCitations() {
			return citations;
		}

		int getInputTokens() {
			return inputTokens;
		}

		int getOutputTokens() {
			return outputTokens;
		}
	}

	public void close() {
		localEngine.close();
		remoteEngine.close();
	}

	public void shutdown() {
		localEngine.shutdown();
		remoteEngine.shutdown();
	}

	protected String getSystemPrompt() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_SYSTEM_PROMPT);
		if (value != null && !value.trim().isEmpty()) {
			return value.trim();
		}
		return DEFAULT_SYSTEM_PROMPT;
	}

	protected int getTimeoutSeconds() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_TIMEOUT_SECONDS);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed > 0) {
					return parsed;
				}
				log.warn("Timeout must be positive, got '{}', using default", parsed);
			}
			catch (NumberFormatException e) {
				log.warn("Invalid timeout value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_LLM_TIMEOUT_SECONDS;
	}

	LlmEngine getActiveEngine() {
		String engineType = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE);
		if (engineType != null
				&& ChartSearchAiConstants.LLM_ENGINE_REMOTE.equalsIgnoreCase(engineType.trim())) {
			return remoteEngine;
		}
		return localEngine;
	}
}
