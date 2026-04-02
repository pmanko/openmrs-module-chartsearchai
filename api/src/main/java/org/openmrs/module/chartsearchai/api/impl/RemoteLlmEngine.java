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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.function.Consumer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Remote LLM engine that calls an OpenAI-compatible chat completions API.
 * Supports any provider that implements the OpenAI API format, including
 * OpenAI, Azure OpenAI, Anthropic (via proxy), Google AI, vLLM, Ollama,
 * and other self-hosted inference servers.
 */
@Component("chartSearchAi.remoteLlmEngine")
public class RemoteLlmEngine implements LlmEngine {

	private static final Logger log = LoggerFactory.getLogger(RemoteLlmEngine.class);

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private HttpClient httpClient;

	@Override
	public InferenceResult infer(String systemPrompt, String userMessage, int timeoutSeconds) {
		String endpointUrl = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_ENDPOINT_URL);
		String apiKey = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_API_KEY);
		String modelName = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_MODEL_NAME);

		String requestBody = buildRequestBody(systemPrompt, userMessage, modelName, false);

		HttpRequest request = HttpRequest.newBuilder()
				.uri(URI.create(endpointUrl))
				.timeout(Duration.ofSeconds(timeoutSeconds))
				.header("Content-Type", "application/json")
				.header("Authorization", "Bearer " + apiKey)
				.POST(HttpRequest.BodyPublishers.ofString(requestBody, StandardCharsets.UTF_8))
				.build();

		try {
			HttpResponse<String> response = getHttpClient().send(request,
					HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

			if (response.statusCode() < 200 || response.statusCode() >= 300) {
				log.error("Remote LLM API returned HTTP {}: {}", response.statusCode(),
						truncateForLog(response.body()));
				throw new APIException("Remote LLM API returned HTTP " + response.statusCode()
						+ ". Check the endpoint URL, API key, and model name in the "
						+ "chartsearchai.llm.remote.* global properties.");
			}

			return parseResponse(response.body());
		}
		catch (IOException e) {
			throw new APIException("Failed to call remote LLM API: " + e.getMessage(), e);
		}
		catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new APIException("Remote LLM API call was interrupted", e);
		}
	}

	@Override
	public InferenceResult inferStreaming(String systemPrompt, String userMessage,
			int timeoutSeconds, Consumer<String> tokenConsumer) {
		String endpointUrl = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_ENDPOINT_URL);
		String apiKey = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_API_KEY);
		String modelName = getRequiredProperty(ChartSearchAiConstants.GP_LLM_REMOTE_MODEL_NAME);

		String requestBody = buildRequestBody(systemPrompt, userMessage, modelName, true);

		HttpRequest request = HttpRequest.newBuilder()
				.uri(URI.create(endpointUrl))
				.timeout(Duration.ofSeconds(timeoutSeconds))
				.header("Content-Type", "application/json")
				.header("Authorization", "Bearer " + apiKey)
				.POST(HttpRequest.BodyPublishers.ofString(requestBody, StandardCharsets.UTF_8))
				.build();

		try {
			HttpResponse<InputStream> response = getHttpClient().send(request,
					HttpResponse.BodyHandlers.ofInputStream());

			if (response.statusCode() < 200 || response.statusCode() >= 300) {
				String body = new String(response.body().readAllBytes(), StandardCharsets.UTF_8);
				log.error("Remote LLM API returned HTTP {}: {}", response.statusCode(),
						truncateForLog(body));
				throw new APIException("Remote LLM API returned HTTP " + response.statusCode());
			}

			return parseStreamingResponse(response.body(), tokenConsumer);
		}
		catch (IOException e) {
			throw new APIException("Failed to call remote LLM API: " + e.getMessage(), e);
		}
		catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new APIException("Remote LLM API call was interrupted", e);
		}
	}

	@Override
	public void close() {
		// HttpClient does not require explicit cleanup
	}

	@Override
	public void shutdown() {
		close();
	}

	String buildRequestBody(String systemPrompt, String userMessage, String modelName,
			boolean stream) {
		ObjectNode root = MAPPER.createObjectNode();
		root.put("model", modelName);
		root.put("temperature", 0.0);
		root.put("max_tokens", ChartSearchAiConstants.DEFAULT_MAX_TOKENS);
		root.put("stream", stream);

		ObjectNode responseFormat = MAPPER.createObjectNode();
		responseFormat.put("type", "json_object");
		root.set("response_format", responseFormat);

		ArrayNode messages = MAPPER.createArrayNode();

		ObjectNode systemMsg = MAPPER.createObjectNode();
		systemMsg.put("role", "system");
		systemMsg.put("content", systemPrompt);
		messages.add(systemMsg);

		ObjectNode userMsg = MAPPER.createObjectNode();
		userMsg.put("role", "user");
		userMsg.put("content", userMessage);
		messages.add(userMsg);

		root.set("messages", messages);

		try {
			return MAPPER.writeValueAsString(root);
		}
		catch (IOException e) {
			throw new APIException("Failed to build request body", e);
		}
	}

	InferenceResult parseResponse(String responseBody) throws IOException {
		JsonNode root = MAPPER.readTree(responseBody);

		String text = "";
		JsonNode choices = root.get("choices");
		if (choices != null && choices.isArray() && !choices.isEmpty()) {
			JsonNode message = choices.get(0).get("message");
			if (message != null && message.has("content")) {
				text = message.get("content").asText("");
			}
		}

		int tokenCount = 0;
		JsonNode usage = root.get("usage");
		if (usage != null) {
			int promptTokens = usage.has("prompt_tokens") ? usage.get("prompt_tokens").asInt(0) : 0;
			int completionTokens = usage.has("completion_tokens")
					? usage.get("completion_tokens").asInt(0) : 0;
			tokenCount = promptTokens + completionTokens;
			log.info("Remote LLM token usage: {} total ({} prompt + {} completion)", tokenCount,
					promptTokens, completionTokens);
		}

		return new InferenceResult(text, tokenCount);
	}

	InferenceResult parseStreamingResponse(InputStream inputStream,
			Consumer<String> tokenConsumer) throws IOException {
		StringBuilder fullText = new StringBuilder();
		int tokenCount = 0;

		try (BufferedReader reader = new BufferedReader(
				new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
			String line;
			while ((line = reader.readLine()) != null) {
				if (!line.startsWith("data: ")) {
					continue;
				}
				String data = line.substring(6).trim();
				if ("[DONE]".equals(data)) {
					break;
				}
				try {
					JsonNode chunk = MAPPER.readTree(data);
					JsonNode choices = chunk.get("choices");
					if (choices == null || !choices.isArray() || choices.isEmpty()) {
						continue;
					}
					JsonNode delta = choices.get(0).get("delta");
					if (delta != null && delta.has("content")) {
						String content = delta.get("content").asText("");
						if (!content.isEmpty()) {
							fullText.append(content);
							tokenConsumer.accept(content);
						}
					}

					// Check for usage in the final chunk
					JsonNode usage = chunk.get("usage");
					if (usage != null) {
						int prompt = usage.has("prompt_tokens")
								? usage.get("prompt_tokens").asInt(0) : 0;
						int completion = usage.has("completion_tokens")
								? usage.get("completion_tokens").asInt(0) : 0;
						tokenCount = prompt + completion;
					}
				}
				catch (IOException e) {
					log.debug("Skipping unparseable SSE chunk: {}", data);
				}
			}
		}

		if (tokenCount > 0) {
			log.info("Remote LLM token usage: {} total", tokenCount);
		}
		return new InferenceResult(fullText.toString(), tokenCount);
	}

	private String getRequiredProperty(String propertyName) {
		String value = Context.getAdministrationService().getGlobalProperty(propertyName);
		if (value == null || value.trim().isEmpty()) {
			throw new IllegalStateException(
					"Required property not configured: " + propertyName);
		}
		return value.trim();
	}

	private HttpClient getHttpClient() {
		if (httpClient == null) {
			httpClient = HttpClient.newBuilder()
					.connectTimeout(Duration.ofSeconds(30))
					.build();
		}
		return httpClient;
	}

	private static String truncateForLog(String text) {
		if (text == null) {
			return "";
		}
		return text.length() > 500 ? text.substring(0, 500) + "..." : text;
	}
}
