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
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import de.kherud.llama.Pair;
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
			+ "Use only the patient records below (grouped by type, most recent first within each group). "
			+ "Never infer, assume, or add information not explicitly stated in the records. "
			+ "Include ALL relevant records in your answer — never omit any for brevity. "
			+ "Cite EVERY record you reference by its number in brackets (e.g. [1], [3]). "
			+ "Respond with ONLY a JSON object with an \"answer\" string and a \"citations\" array "
			+ "listing every record number you cited.\n\n"
			+ "If no records are relevant, name what is missing. For example:\n"
			+ "{\"answer\": \"There are no records about diabetes in this patient's chart.\", \"citations\": []}\n\n"
			+ "Examples:\n\n"
			+ "Patient records (grouped by type, most recent first within each group):\n"
			+ "Patient: 52-year-old Male\n\n"
			+ "[1] (2024-11-05) Condition: Zarplexia. Status: ACTIVE. Verification: CONFIRMED\n"
			+ "[2] (2024-09-20) Drug order: Venoflax 500mg. Dose: 1.0 Tablet(s) Oral twice daily."
			+ " Action: NEW. Urgency: ROUTINE\n"
			+ "[3] (2024-08-15) Test — Flobnar level: 12.4 mg/dL (HIGH)\n"
			+ "[4] (2024-06-10) Test — Flobnar level: 9.8 mg/dL\n"
			+ "[5] (2024-05-01) Allergy: Zyphenicol (drug allergen). Severity: Severe."
			+ " Reactions: Rash\n\n"
			+ "Question: What are the Flobnar levels?\n"
			+ "{\"answer\": \"The Flobnar levels are 12.4 mg/dL (HIGH) on 2024-08-15 [3]"
			+ " and 9.8 mg/dL on 2024-06-10 [4].\","
			+ " \"citations\": [3, 4]}\n\n"
			+ "Question: What medications is the patient on?\n"
			+ "{\"answer\": \"The patient is on Venoflax 500mg, 1 tablet orally twice daily [2].\","
			+ " \"citations\": [2]}\n\n"
			+ "Question: Does the patient have any allergies?\n"
			+ "{\"answer\": \"Yes, the patient has a severe allergy to Zyphenicol with a reaction"
			+ " of Rash [5].\", \"citations\": [5]}\n\n"
			+ "Question: How has the Flobnar level trended?\n"
			+ "{\"answer\": \"The Flobnar level increased from 9.8 mg/dL on 2024-06-10 [4]"
			+ " to 12.4 mg/dL (HIGH) on 2024-08-15 [3].\", \"citations\": [4, 3]}\n\n"
			+ "Question: Does the patient have diabetes?\n"
			+ "{\"answer\": \"There are no records about diabetes in this patient's chart.\","
			+ " \"citations\": []}\n\n"
			+ "Question: What lab orders were placed?\n"
			+ "{\"answer\": \"There are no lab order records in this patient's chart.\","
			+ " \"citations\": []}";

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

	private final ScheduledExecutorService idleTimer = Executors.newSingleThreadScheduledExecutor(
			new java.util.concurrent.ThreadFactory() {
				@Override
				public Thread newThread(Runnable r) {
					Thread t = new Thread(r, "chartsearchai-llm-idle-timer");
					t.setDaemon(true);
					return t;
				}
			});

	private ScheduledFuture<?> idleUnloadFuture;

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
		PreparedInference prepared = createInferenceParameters(llm, numberedRecords, question);

		long deadline = System.currentTimeMillis() + (timeoutSeconds * 1000L);
		StringBuilder result = new StringBuilder();
		int completionTokens = 0;

		for (LlamaOutput output : llm.generate(prepared.params)) {
			if (System.currentTimeMillis() > deadline) {
				log.warn("LLM inference timed out after {} seconds", timeoutSeconds);
				throw new APIException("LLM inference timed out after " + timeoutSeconds
						+ " seconds. Try a more specific question or increase the timeout via "
						+ ChartSearchAiConstants.GP_LLM_TIMEOUT_SECONDS);
			}
			result.append(output);
			completionTokens++;
		}

		int tokenCount = prepared.promptTokens + completionTokens;
		log.info("LLM token usage: {} total ({} prompt + {} completion)", tokenCount,
				prepared.promptTokens, completionTokens);

		LlmResponse llmResponse = extractResponse(result.toString(), tokenCount);
		resetIdleTimer();
		return llmResponse;
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
		PreparedInference prepared = createInferenceParameters(llm, numberedRecords, question);

		long deadline = System.currentTimeMillis() + (timeoutSeconds * 1000L);
		StringBuilder result = new StringBuilder();
		int completionTokens = 0;

		for (LlamaOutput output : llm.generate(prepared.params)) {
			if (System.currentTimeMillis() > deadline) {
				log.warn("LLM inference timed out after {} seconds", timeoutSeconds);
				throw new APIException("LLM inference timed out after " + timeoutSeconds
						+ " seconds. Try a more specific question or increase the timeout via "
						+ ChartSearchAiConstants.GP_LLM_TIMEOUT_SECONDS);
			}
			String token = output.toString();
			result.append(token);
			tokenConsumer.accept(token);
			completionTokens++;
		}

		int tokenCount = prepared.promptTokens + completionTokens;
		log.info("LLM token usage: {} total ({} prompt + {} completion)", tokenCount,
				prepared.promptTokens, completionTokens);

		LlmResponse llmResponse = extractResponse(result.toString(), tokenCount);
		resetIdleTimer();
		return llmResponse;
	}

	static LlmResponse extractResponse(String response, int tokenCount) {
		LlmResponse parsed = extractResponse(response);
		return new LlmResponse(parsed.getAnswer(), parsed.getCitations(), tokenCount);
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

		private final int tokenCount;

		LlmResponse(String answer, List<Integer> citations) {
			this(answer, citations, 0);
		}

		LlmResponse(String answer, List<Integer> citations, int tokenCount) {
			this.answer = answer;
			this.citations = Collections.unmodifiableList(new ArrayList<>(citations));
			this.tokenCount = tokenCount;
		}

		String getAnswer() {
			return answer;
		}

		List<Integer> getCitations() {
			return citations;
		}

		int getTokenCount() {
			return tokenCount;
		}
	}

	public synchronized void close() {
		if (idleUnloadFuture != null) {
			idleUnloadFuture.cancel(false);
			idleUnloadFuture = null;
		}
		if (model != null) {
			log.info("Closing LLM model");
			model.close();
			model = null;
		}
		loadedModelPath = null;
	}

	/**
	 * Shuts down the idle timer executor. Called on module shutdown to
	 * release the daemon thread. After this call, idle unloading is
	 * permanently disabled for this instance.
	 */
	public void shutdown() {
		idleTimer.shutdownNow();
		close();
	}

	private synchronized void resetIdleTimer() {
		if (idleUnloadFuture != null) {
			idleUnloadFuture.cancel(false);
			idleUnloadFuture = null;
		}
		int idleMinutes = getIdleTimeoutMinutes();
		if (idleMinutes > 0) {
			try {
				idleUnloadFuture = idleTimer.schedule(new Runnable() {
					@Override
					public void run() {
						log.info("LLM idle for {} minutes, unloading model to free memory", idleMinutes);
						close();
					}
				}, idleMinutes, TimeUnit.MINUTES);
			}
			catch (java.util.concurrent.RejectedExecutionException e) {
				log.debug("Idle timer already shut down, skipping schedule");
			}
		}
	}

	protected int getIdleTimeoutMinutes() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_IDLE_TIMEOUT_MINUTES);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed >= 0) {
					return parsed;
				}
				log.warn("Idle timeout must be non-negative, got '{}', using default", parsed);
			}
			catch (NumberFormatException e) {
				log.warn("Invalid idle timeout value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_LLM_IDLE_TIMEOUT_MINUTES;
	}

	private PreparedInference createInferenceParameters(LlamaModel llm, String numberedRecords,
			String question) {
		String systemPrompt = getSystemPrompt();
		String userMessage = "Patient records (most recent first):\n" + numberedRecords.stripTrailing()
				+ "\n\nQuestion: " + question;
		String templateValue = getChatTemplate();

		InferenceParameters params;
		int promptTokens;
		if ("auto".equalsIgnoreCase(templateValue)) {
			// Use the model's built-in GGUF chat template for correct formatting
			List<Pair<String, String>> messages = new ArrayList<Pair<String, String>>();
			messages.add(new Pair<String, String>("user", userMessage));
			params = new InferenceParameters("")
					.setUseChatTemplate(true)
					.setMessages(systemPrompt, messages);
			// Approximate: tokenize the raw content since the formatted prompt
			// is not accessible when using the model's built-in template
			promptTokens = llm.encode(systemPrompt + "\n\n" + userMessage).length;
			log.debug("Using model's built-in chat template");
		} else {
			String prompt = formatPrompt(templateValue, systemPrompt, userMessage);
			promptTokens = llm.encode(prompt).length;
			log.debug("LLM prompt size: {} tokens", promptTokens);
			params = new InferenceParameters(prompt)
					.setStopStrings(resolveStopStrings(templateValue));
		}

		InferenceParameters finalParams = params
				.setTemperature(0.0f)
				.setSeed(42)
				.setCachePrompt(false)
				.setNPredict(ChartSearchAiConstants.DEFAULT_MAX_TOKENS)
				.setRepeatPenalty(ChartSearchAiConstants.DEFAULT_REPEAT_PENALTY)
				.setGrammar(JSON_ANSWER_GRAMMAR);

		return new PreparedInference(finalParams, promptTokens);
	}

	static class PreparedInference {

		final InferenceParameters params;

		final int promptTokens;

		PreparedInference(InferenceParameters params, int promptTokens) {
			this.params = params;
			this.promptTokens = promptTokens;
		}
	}

	static String formatPrompt(String templateValue, String systemPrompt, String userMessage) {
		String template = PRESET_TEMPLATES.getOrDefault(templateValue.toLowerCase(), templateValue);
		// Sanitize user-controlled content to prevent prompt injection via
		// special tokens. Strip tokens that could break out of the user role
		// in any of the supported chat templates.
		String safeUser = stripSpecialTokens(userMessage);
		return template.replace("{system}", systemPrompt).replace("{user}", safeUser);
	}

	/**
	 * Strips chat template special tokens from user-controlled text to prevent
	 * prompt injection. Covers all preset template formats plus any tokens found
	 * in custom templates (strings matching {@code <|...|>} or {@code <..._of_...>}).
	 */
	static String stripSpecialTokens(String text) {
		if (text == null) {
			return "";
		}
		return text
				// llama3 tokens
				.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
				.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
				.replace("<|eot_id|>", "")
				// mistral tokens
				.replace("[INST]", "").replace("[/INST]", "")
				.replace("</s>", "")
				// phi3 tokens
				.replace("<|system|>", "").replace("<|user|>", "")
				.replace("<|assistant|>", "").replace("<|end|>", "")
				// chatml tokens
				.replace("<|im_start|>", "").replace("<|im_end|>", "")
				// gemma tokens
				.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
				// Catch-all: strip any remaining <|...|> tokens that custom
				// templates may use, preventing injection via unknown formats
				.replaceAll("<\\|[^|]*\\|>", "");
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
		return "auto";
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
					.setGpuLayers(-1)
					.setCtxSize(4096);
			model = new LlamaModel(modelParams);
			loadedModelPath = modelPath;
			log.info("LLM loaded successfully");
		}
		return model;
	}
}
