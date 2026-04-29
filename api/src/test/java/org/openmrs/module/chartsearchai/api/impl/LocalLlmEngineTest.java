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

import java.io.IOException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link LocalLlmEngine} request building.
 */
public class LocalLlmEngineTest {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final LocalLlmEngine engine = new LocalLlmEngine();

	@Test
	public void buildRequestBody_shouldUseJsonSchemaStrictMode() throws IOException {
		String body = engine.buildRequestBody("system prompt", "user message", false);
		JsonNode root = MAPPER.readTree(body);

		JsonNode responseFormat = root.get("response_format");
		assertEquals("json_schema", responseFormat.get("type").asText(),
				"Local engine must use json_schema mode so the model stops at the closing brace "
				+ "instead of looping past a complete answer until max_tokens is hit.");

		JsonNode jsonSchema = responseFormat.get("json_schema");
		assertTrue(jsonSchema.get("strict").asBoolean(),
				"Schema must be strict so the model cannot generate content after the closing brace");

		JsonNode schema = jsonSchema.get("schema");
		assertEquals("object", schema.get("type").asText());
		assertFalse(schema.get("additionalProperties").asBoolean(),
				"additionalProperties must be false to prevent post-answer rambling");

		JsonNode properties = schema.get("properties");
		assertEquals("string", properties.get("answer").get("type").asText());
		assertEquals("array", properties.get("citations").get("type").asText());
		assertEquals("integer", properties.get("citations").get("items").get("type").asText());

		JsonNode required = schema.get("required");
		assertEquals(2, required.size());
	}

	@Test
	public void buildRequestBody_shouldSetTemperatureZeroAndMaxTokens() throws IOException {
		String body = engine.buildRequestBody("sys", "usr", false);
		JsonNode root = MAPPER.readTree(body);

		assertEquals(0.0, root.get("temperature").asDouble());
		assertTrue(root.get("max_tokens").asInt() > 0);
	}

	@Test
	public void buildRequestBody_shouldIncludeSystemAndUserMessages() throws IOException {
		String body = engine.buildRequestBody("system prompt", "user message", false);
		JsonNode root = MAPPER.readTree(body);

		JsonNode messages = root.get("messages");
		assertEquals(2, messages.size());
		assertEquals("system", messages.get(0).get("role").asText());
		assertEquals("system prompt", messages.get(0).get("content").asText());
		assertEquals("user", messages.get(1).get("role").asText());
		assertEquals("user message", messages.get(1).get("content").asText());
	}

	@Test
	public void buildRequestBody_shouldEnableUsageWhenStreaming() throws IOException {
		String body = engine.buildRequestBody("sys", "usr", true);
		JsonNode root = MAPPER.readTree(body);

		assertTrue(root.get("stream").asBoolean());
		assertTrue(root.get("stream_options").get("include_usage").asBoolean());
	}

	@Test
	public void buildRequestBody_shouldOmitStreamOptionsWhenNotStreaming() throws IOException {
		String body = engine.buildRequestBody("sys", "usr", false);
		JsonNode root = MAPPER.readTree(body);

		assertFalse(root.get("stream").asBoolean());
		assertTrue(root.path("stream_options").isMissingNode());
	}

	@Test
	public void isContextOverflowError_shouldDetectLlamaServerOverflow() {
		// Actual response body llama-server returns when the prompt exceeds -c
		String body = "{\"error\":{\"code\":400,\"message\":\"request (21953 tokens) "
				+ "exceeds the available context size (16384 tokens), try increasing it\","
				+ "\"type\":\"exceed_context_size_error\","
				+ "\"n_prompt_tokens\":21953,\"n_ctx\":16384}}";
		assertTrue(LocalLlmEngine.isContextOverflowError(body),
				"should detect llama-server's exceed_context_size_error");
	}

	@Test
	public void isContextOverflowError_shouldReturnFalseForOtherErrors() {
		assertFalse(LocalLlmEngine.isContextOverflowError(
				"{\"error\":{\"code\":400,\"message\":\"some other 400\"}}"));
		assertFalse(LocalLlmEngine.isContextOverflowError("not json at all"));
		assertFalse(LocalLlmEngine.isContextOverflowError(""));
		assertFalse(LocalLlmEngine.isContextOverflowError(null));
	}

	@Test
	public void shouldRestartServer_modelPathChanged() {
		assertTrue(LocalLlmEngine.shouldRestartServer(
				"old/model.gguf", 16384, "new/model.gguf", 16384),
				"should restart when model path changes");
	}

	@Test
	public void shouldRestartServer_contextSizeChanged() {
		assertTrue(LocalLlmEngine.shouldRestartServer(
				"same/model.gguf", 16384, "same/model.gguf", 32768),
				"should restart when context size changes — admins editing the GP "
				+ "expect the new value to take effect on the next query");
	}

	@Test
	public void shouldRestartServer_nothingChanged() {
		assertFalse(LocalLlmEngine.shouldRestartServer(
				"same/model.gguf", 16384, "same/model.gguf", 16384),
				"no restart needed when both model and context size are unchanged");
	}

	@Test
	public void shouldRestartServer_neverLoaded() {
		assertFalse(LocalLlmEngine.shouldRestartServer(
				null, -1, "any/model.gguf", 16384),
				"if no server has loaded yet, the caller handles the initial start; "
				+ "this helper only decides whether to restart an already-running one");
	}
}
