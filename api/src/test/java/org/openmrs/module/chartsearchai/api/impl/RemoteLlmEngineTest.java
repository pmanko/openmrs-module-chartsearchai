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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link RemoteLlmEngine} request building and response parsing.
 */
public class RemoteLlmEngineTest {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final RemoteLlmEngine engine = new RemoteLlmEngine();

	@Test
	public void buildRequestBody_shouldContainModelAndMessages() throws IOException {
		String body = engine.buildRequestBody("system prompt", "user message", "gpt-4o", false);
		JsonNode root = MAPPER.readTree(body);

		assertEquals("gpt-4o", root.get("model").asText());
		assertEquals(0.0, root.get("temperature").asDouble());
		assertEquals(false, root.get("stream").asBoolean());
		assertEquals("json_object", root.get("response_format").get("type").asText());

		JsonNode messages = root.get("messages");
		assertEquals(2, messages.size());
		assertEquals("system", messages.get(0).get("role").asText());
		assertEquals("system prompt", messages.get(0).get("content").asText());
		assertEquals("user", messages.get(1).get("role").asText());
		assertEquals("user message", messages.get(1).get("content").asText());
	}

	@Test
	public void buildRequestBody_shouldSetStreamTrue() throws IOException {
		String body = engine.buildRequestBody("sys", "usr", "model", true);
		JsonNode root = MAPPER.readTree(body);

		assertTrue(root.get("stream").asBoolean());
	}

	@Test
	public void parseResponse_shouldExtractContentAndTokenCount() throws IOException {
		String responseBody = "{"
				+ "\"choices\": [{\"message\": {\"content\": \"{\\\"answer\\\": \\\"test\\\", \\\"citations\\\": []}\"}}],"
				+ "\"usage\": {\"prompt_tokens\": 100, \"completion_tokens\": 50}"
				+ "}";

		LlmEngine.InferenceResult result = engine.parseResponse(responseBody);
		assertEquals("{\"answer\": \"test\", \"citations\": []}", result.getText());
		assertEquals(100, result.getInputTokens());
		assertEquals(50, result.getOutputTokens());
	}

	@Test
	public void parseResponse_shouldHandleMissingUsage() throws IOException {
		String responseBody = "{"
				+ "\"choices\": [{\"message\": {\"content\": \"hello\"}}]"
				+ "}";

		LlmEngine.InferenceResult result = engine.parseResponse(responseBody);
		assertEquals("hello", result.getText());
		assertEquals(0, result.getInputTokens());
		assertEquals(0, result.getOutputTokens());
	}

	@Test
	public void parseResponse_shouldHandleEmptyChoices() throws IOException {
		String responseBody = "{\"choices\": []}";

		LlmEngine.InferenceResult result = engine.parseResponse(responseBody);
		assertEquals("", result.getText());
	}

	@Test
	public void parseStreamingResponse_shouldAssembleTokens() throws IOException {
		String sseData = "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"answer\\\"\"}}]}\n\n"
				+ "data: {\"choices\":[{\"delta\":{\"content\":\": \\\"test\\\"\"}}]}\n\n"
				+ "data: {\"choices\":[{\"delta\":{\"content\":\", \\\"citations\\\": []}\"}}]}\n\n"
				+ "data: [DONE]\n\n";

		ByteArrayInputStream inputStream = new ByteArrayInputStream(
				sseData.getBytes(StandardCharsets.UTF_8));

		List<String> tokens = new ArrayList<>();
		LlmEngine.InferenceResult result = engine.parseStreamingResponse(inputStream,
				tokens::add);

		assertEquals("{\"answer\": \"test\", \"citations\": []}", result.getText());
		assertEquals(3, tokens.size());
	}

	@Test
	public void parseStreamingResponse_shouldSkipBlankLines() throws IOException {
		String sseData = "\n\n"
				+ "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"
				+ "\n"
				+ "data: [DONE]\n\n";

		ByteArrayInputStream inputStream = new ByteArrayInputStream(
				sseData.getBytes(StandardCharsets.UTF_8));

		List<String> tokens = new ArrayList<>();
		LlmEngine.InferenceResult result = engine.parseStreamingResponse(inputStream,
				tokens::add);

		assertEquals("hello", result.getText());
		assertEquals(1, tokens.size());
	}

	@Test
	public void parseStreamingResponse_shouldExtractUsageFromFinalChunk() throws IOException {
		String sseData = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
				+ "data: {\"choices\":[{\"delta\":{}}], \"usage\":{\"prompt_tokens\":80,\"completion_tokens\":20}}\n\n"
				+ "data: [DONE]\n\n";

		ByteArrayInputStream inputStream = new ByteArrayInputStream(
				sseData.getBytes(StandardCharsets.UTF_8));

		LlmEngine.InferenceResult result = engine.parseStreamingResponse(inputStream,
				token -> {});

		assertEquals("hi", result.getText());
		assertEquals(80, result.getInputTokens());
		assertEquals(20, result.getOutputTokens());
	}
}
