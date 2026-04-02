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
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.Pair;

import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Local LLM engine using java-llama.cpp for in-process GGUF model inference.
 * The model is loaded lazily on first use and automatically unloaded after an
 * idle period to free memory.
 */
@Component("chartSearchAi.localLlmEngine")
public class LocalLlmEngine implements LlmEngine {

	private static final Logger log = LoggerFactory.getLogger(LocalLlmEngine.class);

	private static final String JSON_ANSWER_GRAMMAR = loadGrammar("json-answer.gbnf");

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

	@Override
	public synchronized InferenceResult infer(String systemPrompt, String userMessage,
			int timeoutSeconds) {
		LlamaModel llm = getModel();

		PreparedInference prepared = createInferenceParameters(llm, systemPrompt, userMessage);

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

		resetIdleTimer();
		return new InferenceResult(result.toString(), tokenCount);
	}

	@Override
	public synchronized InferenceResult inferStreaming(String systemPrompt, String userMessage,
			int timeoutSeconds, Consumer<String> tokenConsumer) {
		LlamaModel llm = getModel();

		PreparedInference prepared = createInferenceParameters(llm, systemPrompt, userMessage);

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

		resetIdleTimer();
		return new InferenceResult(result.toString(), tokenCount);
	}

	@Override
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

	@Override
	public void shutdown() {
		idleTimer.shutdownNow();
		close();
	}

	private PreparedInference createInferenceParameters(LlamaModel llm, String systemPrompt,
			String userMessage) {
		String templateValue = getChatTemplate();

		InferenceParameters params;
		int promptTokens;
		if ("auto".equalsIgnoreCase(templateValue)) {
			List<Pair<String, String>> messages = new ArrayList<Pair<String, String>>();
			messages.add(new Pair<String, String>("user", userMessage));
			params = new InferenceParameters("")
					.setUseChatTemplate(true)
					.setMessages(systemPrompt, messages);
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
		String safeUser = stripSpecialTokens(userMessage);
		return template.replace("{system}", systemPrompt).replace("{user}", safeUser);
	}

	static String stripSpecialTokens(String text) {
		if (text == null) {
			return "";
		}
		return text
				.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
				.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
				.replace("<|eot_id|>", "")
				.replace("[INST]", "").replace("[/INST]", "")
				.replace("</s>", "")
				.replace("<|system|>", "").replace("<|user|>", "")
				.replace("<|assistant|>", "").replace("<|end|>", "")
				.replace("<|im_start|>", "").replace("<|im_end|>", "")
				.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
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
						log.info("LLM idle for {} minutes, unloading model to free memory",
								idleMinutes);
						close();
					}
				}, idleMinutes, TimeUnit.MINUTES);
			}
			catch (java.util.concurrent.RejectedExecutionException e) {
				log.debug("Idle timer already shut down, skipping schedule");
			}
		}
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
			log.info("LLM model path changed from {} to {}, reloading", loadedModelPath,
					modelPath);
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

	private static String loadGrammar(String resourceName) {
		try (InputStream is = LocalLlmEngine.class.getClassLoader()
				.getResourceAsStream(resourceName)) {
			if (is == null) {
				throw new IllegalStateException("Grammar resource not found: " + resourceName);
			}
			return new String(is.readAllBytes(), StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load grammar: " + resourceName, e);
		}
	}
}
