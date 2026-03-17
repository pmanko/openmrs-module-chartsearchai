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
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;

import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Manages the local LLM model lifecycle and provides inference. The model is loaded
 * once on first use and kept in memory for subsequent calls. Shared by both the direct
 * LLM inference and embedding-based RAG search services.
 */
@Component
public class LlmProvider {

	private static final Logger log = LoggerFactory.getLogger(LlmProvider.class);

	static final String DEFAULT_SYSTEM_PROMPT = "You are a clinical assistant helping a clinician "
			+ "review a patient's chart. Answer ONLY the specific question asked. "
			+ "Use only the patient records below (sorted most recent first). "
			+ "Include the relevant details from the records in your answer "
			+ "and cite EVERY record you use by its number in separate brackets (e.g. [1], [3]). "
			+ "NEVER combine record numbers with slashes like [1/2] — always use [1], [2] instead. "
			+ "Do not list records that are irrelevant to the question. "
			+ "If the records do not contain enough information to answer, respond with: "
			+ "\"No relevant information was found in the patient's records.\" "
			+ "Keep your answer concise — one to three sentences. "
			+ "Respond with ONLY a JSON object with an \"answer\" string and a \"citations\" array "
			+ "listing every record number you cited.\n\n"
			+ "Examples:\n\n"
			+ "Records:\n[1] Condition: Zorblitis\n"
			+ "[2] Medication: Xanthuril 50mg daily\n"
			+ "[3] Flobnar level: 12.4\n"
			+ "[4] Flobnar level: 9.8\n\n"
			+ "Question: What are the Flobnar levels?\n"
			+ "{\"answer\": \"The Flobnar levels are 12.4 [3] and 9.8 [4].\", \"citations\": [3, 4]}\n\n"
			+ "Question: Does the patient have diabetes?\n"
			+ "{\"answer\": \"No relevant information was found in the patient's records.\", \"citations\": []}";

	private static final String JSON_ANSWER_GRAMMAR = loadGrammar("json-answer.gbnf");

	private static String loadGrammar(String resourceName) {
		try (InputStream is = LlmProvider.class.getClassLoader().getResourceAsStream(resourceName)) {
			if (is == null) {
				throw new IllegalStateException("Grammar resource not found: " + resourceName);
			}
			return new String(is.readAllBytes(), StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load grammar: " + resourceName, e);
		}
	}

	static final Map<String, String> PRESET_TEMPLATES;

	static final Map<String, List<String>> PRESET_STOP_STRINGS;

	static {
		Map<String, String> templates = new LinkedHashMap<>();
		templates.put("llama3",
				"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
				+ "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
				+ "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
		templates.put("mistral", "[INST] {system}\n\n{user} [/INST]");
		templates.put("phi3",
				"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n");
		templates.put("chatml",
				"<|im_start|>system\n{system}<|im_end|>\n"
				+ "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
		templates.put("gemma",
				"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
				+ "<start_of_turn>model\n");
		PRESET_TEMPLATES = Collections.unmodifiableMap(templates);

		Map<String, List<String>> stops = new LinkedHashMap<>();
		stops.put("llama3", List.of("<|eot_id|>", "<|end_of_text|>"));
		stops.put("mistral", List.of("</s>"));
		stops.put("phi3", List.of("<|end|>"));
		stops.put("chatml", List.of("<|im_end|>"));
		stops.put("gemma", List.of("<end_of_turn>"));
		PRESET_STOP_STRINGS = Collections.unmodifiableMap(stops);
	}

	private LlamaModel model;

	private String loadedModelPath;

	/**
	 * Send numbered patient records and a question to the LLM for synthesis.
	 * Uses streaming generation with a wall-clock timeout to prevent indefinite blocking.
	 *
	 * @param numberedRecords the numbered patient records text
	 * @param question the clinician's natural language question
	 * @return the LLM's response with answer text and structured citation indices
	 * @throws APIException if the request exceeds the configured timeout
	 */
	public synchronized LlmResponse search(String numberedRecords, String question) {
		LlamaModel llm = getModel();

		int timeoutSeconds = getTimeoutSeconds();
		InferenceParameters params = createInferenceParameters(numberedRecords, question);

		long deadline = System.currentTimeMillis() + (timeoutSeconds * 1000L);
		StringBuilder result = new StringBuilder();

		for (LlamaOutput output : llm.generate(params)) {
			if (System.currentTimeMillis() > deadline) {
				log.warn("LLM inference timed out after {} seconds", timeoutSeconds);
				throw new APIException("LLM inference timed out after " + timeoutSeconds
						+ " seconds. Try a more specific question or increase the timeout via "
						+ ChartSearchAiConstants.GP_LLM_TIMEOUT_SECONDS);
			}
			result.append(output);
		}

		return extractResponse(result.toString());
	}

	/**
	 * Streaming variant of {@link #search}. Calls the tokenConsumer for each token as it is
	 * generated, and returns the full response when complete.
	 *
	 * @param numberedRecords the numbered patient records text
	 * @param question the clinician's natural language question
	 * @param tokenConsumer called with each token fragment as it is generated
	 * @return the complete LLM response with answer text and structured citation indices
	 * @throws APIException if the request exceeds the configured timeout
	 */
	public synchronized LlmResponse searchStreaming(String numberedRecords, String question,
			Consumer<String> tokenConsumer) {
		LlamaModel llm = getModel();

		int timeoutSeconds = getTimeoutSeconds();
		InferenceParameters params = createInferenceParameters(numberedRecords, question);

		long deadline = System.currentTimeMillis() + (timeoutSeconds * 1000L);
		StringBuilder result = new StringBuilder();

		for (LlamaOutput output : llm.generate(params)) {
			if (System.currentTimeMillis() > deadline) {
				log.warn("LLM inference timed out after {} seconds", timeoutSeconds);
				throw new APIException("LLM inference timed out after " + timeoutSeconds
						+ " seconds. Try a more specific question or increase the timeout via "
						+ ChartSearchAiConstants.GP_LLM_TIMEOUT_SECONDS);
			}
			String token = output.toString();
			result.append(token);
			tokenConsumer.accept(token);
		}

		return extractResponse(result.toString());
	}

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private static final Pattern SLASH_CITATION = Pattern.compile("\\[(\\d+(?:/\\d+)+)\\]");

	/**
	 * Extracts the answer and citations from the JSON response produced by the
	 * grammar-constrained LLM. Expected format: {"answer": "...", "citations": [1, 2]}
	 * Falls back to returning the raw response with empty citations if JSON parsing fails.
	 */
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

	/**
	 * Normalizes slash-separated citations like [7/13] to [7], [13] in the answer text.
	 * Small LLMs sometimes produce this format despite prompt instructions.
	 */
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

	/**
	 * Parsed LLM response containing the answer text and structured citation indices.
	 */
	static class LlmResponse {

		private final String answer;

		private final List<Integer> citations;

		LlmResponse(String answer, List<Integer> citations) {
			this.answer = answer;
			this.citations = citations;
		}

		String getAnswer() {
			return answer;
		}

		List<Integer> getCitations() {
			return citations;
		}
	}

	public synchronized void close() {
		if (model != null) {
			log.info("Closing LLM model");
			model.close();
			model = null;
		}
		loadedModelPath = null;
	}

	private InferenceParameters createInferenceParameters(String numberedRecords, String question) {
		String systemPrompt = getSystemPrompt();
		String userMessage = "Patient records (most recent first):\n" + numberedRecords + "\n\nQuestion: " + question;
		String templateValue = getChatTemplate();

		String prompt = formatPrompt(templateValue, systemPrompt, userMessage);
		log.warn("LLM prompt size: {} tokens", model.encode(prompt).length);
		String[] stopStrings = resolveStopStrings(templateValue);

		return new InferenceParameters(prompt)
				.setTemperature(0.0f)
				.setSeed(42)
				.setNPredict(ChartSearchAiConstants.DEFAULT_MAX_TOKENS)
				.setGrammar(JSON_ANSWER_GRAMMAR)
				.setRepeatPenalty(1.1f)
				.setRepeatLastN(256)
				.setFrequencyPenalty(0.1f)
				.setStopStrings(stopStrings);
	}

	static String formatPrompt(String templateValue, String systemPrompt, String userMessage) {
		String template = PRESET_TEMPLATES.getOrDefault(templateValue.toLowerCase(), templateValue);
		return template.replace("{system}", systemPrompt).replace("{user}", userMessage);
	}

	static String[] resolveStopStrings(String templateValue) {
		List<String> stops = PRESET_STOP_STRINGS.get(templateValue.toLowerCase());
		if (stops != null) {
			return stops.toArray(new String[0]);
		}
		return new String[0];
	}

	protected String getChatTemplate() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_CHAT_TEMPLATE);
		if (value != null && !value.trim().isEmpty()) {
			return value.trim();
		}
		return "llama3";
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

	private synchronized LlamaModel getModel() {
		String configuredPath = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_MODEL_FILE_PATH);
		if (configuredPath == null || configuredPath.trim().isEmpty()) {
			throw new IllegalStateException(
					"LLM model path not configured. Set the global property: "
							+ ChartSearchAiConstants.GP_LLM_MODEL_FILE_PATH);
		}
		String modelPath = ChartSearchAiConstants.resolveModelPath(
				configuredPath.trim(), ChartSearchAiConstants.GP_LLM_MODEL_FILE_PATH);

		if (model != null && !modelPath.equals(loadedModelPath)) {
			log.info("LLM model path changed from {} to {}, reloading", loadedModelPath, modelPath);
			close();
		}

		if (model == null) {
			log.info("Loading LLM from {}", modelPath);
			ModelParameters modelParams = new ModelParameters()
					.setModel(modelPath)
					.setGpuLayers(0);
			model = new LlamaModel(modelParams);
			loadedModelPath = modelPath;
			log.info("LLM loaded successfully");
		}
		return model;
	}
}
