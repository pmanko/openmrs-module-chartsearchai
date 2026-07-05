/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.web.rest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.function.Consumer;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.openmrs.Patient;
import org.openmrs.User;
import org.openmrs.api.AdministrationService;
import org.openmrs.api.PatientService;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.AuditLogService;
import org.openmrs.module.chartsearchai.api.ChartSearchService.ChartAnswer;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.ChatService.ChatTurnResult;
import org.openmrs.module.chartsearchai.api.PatientAccessCheck;
import org.openmrs.module.chartsearchai.api.impl.ModelSwitchService;
import org.openmrs.module.chartsearchai.api.impl.RequestLlmOverride;
import org.openmrs.module.chartsearchai.model.ChatMessage;
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.test.util.ReflectionTestUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;

/**
 * Behavioral tests for the SSE chat-streaming endpoint
 * ({@link ChartSearchAiRestController#chatStream}). These drive the real
 * controller method against a {@link MockHttpServletResponse} and assert on the
 * actual Server-Sent-Events bytes written to the response — the {@code token}
 * stream, the terminal {@code done} envelope, and that authorization is enforced
 * before any streaming begins.
 *
 * <p>Only the injected collaborators (ChatService, PatientAccessCheck, ...) and
 * the static {@link Context} are mocked; the controller's own SSE serialization
 * and ordering are exercised for real.</p>
 */
public class ChartSearchAiStreamingTest {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	@AfterEach
	public void clearOverride() {
		RequestLlmOverride.clear();
	}

	/**
	 * Wires the four collaborators the controller autowires, plus the
	 * common {@link Context} stubs, and returns a ready-to-call controller.
	 */
	private static Fixture newFixture(boolean accessGranted) {
		Fixture f = new Fixture();
		f.patient = mock(Patient.class);
		f.user = mock(User.class);
		f.patientService = mock(PatientService.class);
		f.adminService = mock(AdministrationService.class);
		f.patientAccessCheck = mock(PatientAccessCheck.class);
		f.auditLogService = mock(AuditLogService.class);
		f.chatService = mock(ChatService.class);
		f.modelSwitchService = mock(ModelSwitchService.class);
		f.session = mock(ChatSession.class);

		when(f.patientService.getPatientByUuid("patient-uuid")).thenReturn(f.patient);
		lenient().when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_RATE_LIMIT_PER_MINUTE))
				.thenReturn(null);
		lenient().when(f.adminService.getGlobalProperty(
				eq(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER), any()))
				.thenReturn("false");
		when(f.patientAccessCheck.canAccess(any(), eq(f.patient))).thenReturn(accessGranted);
		lenient().when(f.auditLogService.getQueryCountByUserSince(any(), any())).thenReturn(0L);
		lenient().when(f.chatService.openOrLoadActiveSession(f.patient)).thenReturn(f.session);
		lenient().when(f.session.getUuid()).thenReturn("session-uuid");
		lenient().when(f.chatService.priorTurnsForRelay(any())).thenReturn(Collections.emptyList());
		lenient().when(f.modelSwitchService.isStagedModel(any(), any())).thenReturn(false);

		f.controller = new ChartSearchAiRestController();
		ReflectionTestUtils.setField(f.controller, "patientAccessCheck", f.patientAccessCheck);
		ReflectionTestUtils.setField(f.controller, "auditLogService", f.auditLogService);
		ReflectionTestUtils.setField(f.controller, "chatService", f.chatService);
		ReflectionTestUtils.setField(f.controller, "modelSwitchService", f.modelSwitchService);
		return f;
	}

	private static Map<String, String> chatBody() {
		Map<String, String> body = new HashMap<String, String>();
		body.put("patient", "patient-uuid");
		body.put("question", "What medications is this patient taking?");
		return body;
	}

	/**
	 * Happy path: a token streams, then a terminal {@code done} event carries the
	 * answer, the session uuid, and the answering model. The session uuid is also
	 * surfaced as a header before the stream opens.
	 */
	@Test
	public void chatStream_shouldStreamTokenThenDoneEnvelope() throws Exception {
		Fixture f = newFixture(true);

		ChartAnswer answer = new ChartAnswer(
				"Lisinopril 10mg daily.",
				Collections.singletonList(
						new RecordReference(1, "MedicationRequest", "med-uuid", null)));
		// The controller writes a `token` event for each consumed token, so the
		// stub must actually invoke the consumer (thenReturn alone emits nothing).
		when(f.chatService.chatStreaming(eq(f.session), eq("What medications is this patient taking?"), any()))
				.thenAnswer(inv -> {
					inv.<Consumer<String>>getArgument(2).accept("Lisinopril ");
					inv.<Consumer<String>>getArgument(2).accept("10mg daily.");
					return new ChatTurnResult(answer, "session-uuid", "assistant-msg-uuid");
				});

		MockHttpServletResponse response = new MockHttpServletResponse();

		try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
			ctx.when(() -> Context.requirePrivilege(
					ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
			ctx.when(Context::getPatientService).thenReturn(f.patientService);
			ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
			ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);

			f.controller.chatStream(chatBody(), response);
		}

		String sse = response.getContentAsString();

		int tokenIdx = sse.indexOf("event: token");
		int doneIdx = sse.indexOf("event: done");
		assertTrue(tokenIdx >= 0, "Expected a token event in the SSE stream, got:\n" + sse);
		assertTrue(doneIdx >= 0, "Expected a done event in the SSE stream, got:\n" + sse);
		assertTrue(tokenIdx < doneIdx, "token event must precede the done event");
		assertTrue(sse.contains("data: Lisinopril "),
				"Streamed token text missing from SSE, got:\n" + sse);

		assertTrue(response.getContentType().startsWith("text/event-stream"),
				"Content-Type must be an event-stream, got " + response.getContentType());
		assertEquals("session-uuid", response.getHeader("X-ChartSearchAi-Session"),
				"Session uuid must be surfaced as a header before the stream opens");

		// Parse the JSON payload of the done event and assert the envelope fields.
		JsonNode done = parseDoneEvent(sse);
		assertEquals("Lisinopril 10mg daily.", done.get("answer").asText());
		assertEquals("session-uuid", done.get("session").asText());
		// No override + unstubbed engine GP (null => local) => local model name.
		assertEquals("local", done.get("model").asText());
		assertEquals(1, done.get("references").size());
		assertEquals("MedicationRequest",
				done.get("references").get(0).get("resourceType").asText());
	}

	/**
	 * Authorization is enforced BEFORE streaming: when the user cannot access the
	 * patient's chart the endpoint returns 403 with a JSON error and never calls
	 * the streaming service — no SSE bytes are written.
	 */
	@Test
	public void chatStream_shouldReturn403AndNotStream_whenAccessDenied() throws Exception {
		Fixture f = newFixture(false);

		MockHttpServletResponse response = new MockHttpServletResponse();

		try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
			ctx.when(() -> Context.requirePrivilege(
					ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
			ctx.when(Context::getPatientService).thenReturn(f.patientService);
			ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
			ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);

			f.controller.chatStream(chatBody(), response);
		}

		assertEquals(403, response.getStatus(), "Access-denied must return 403");
		// The genuine "auth before streaming" guarantee: the streaming service is
		// never invoked once access is denied.
		verify(f.chatService, never()).chatStreaming(any(), any(), any());

		String body = response.getContentAsString();
		assertTrue(body.contains("\"error\""),
				"403 response must carry a JSON error body, got:\n" + body);
		assertTrue(!body.contains("event: token") && !body.contains("event: done"),
				"No SSE events may be written after a 403, got:\n" + body);
		assertTrue(response.getContentType() != null
						&& response.getContentType().startsWith("application/json"),
				"403 error must be JSON, not an event-stream; got " + response.getContentType());
	}

	/**
	 * A failure while RESOLVING/BUILDING the session — e.g. the chart-snapshot build
	 * throwing a dangling-FK {@code FetchNotFoundException} — happens BEFORE the SSE
	 * stream opens. It must be handled as a clean 500 JSON error, NOT propagate
	 * uncaught to the servlet container (which renders an OpenMRS HTML error page the
	 * SPA can't parse — the "blank 500"). The streaming service must never be reached
	 * and the raw exception must not leak to the client.
	 */
	@Test
	public void chatStream_shouldReturnCleanError_whenSessionOrChartBuildFails() throws Exception {
		Fixture f = newFixture(true);
		// resolveOrOpenSession(patient, null) opens a new session, which builds the
		// chart snapshot. Simulate that build hitting a dangling encounter FK.
		when(f.chatService.openOrLoadActiveSession(f.patient))
				.thenThrow(new RuntimeException(
						"org.hibernate.FetchNotFoundException: Entity Encounter id 958 does not exist"));

		MockHttpServletResponse response = new MockHttpServletResponse();

		try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
			ctx.when(() -> Context.requirePrivilege(
					ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
			ctx.when(Context::getPatientService).thenReturn(f.patientService);
			ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
			ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);

			// Must NOT throw — the controller has to catch the pre-stream failure.
			f.controller.chatStream(chatBody(), response);
		}

		assertEquals(500, response.getStatus(),
				"a pre-stream build failure must be a handled 500, not a propagated exception");
		verify(f.chatService, never()).chatStreaming(any(), any(), any());

		String body = response.getContentAsString();
		assertTrue(body.contains("\"error\""),
				"500 must carry a JSON error body, got:\n" + body);
		assertTrue(response.getContentType() != null
						&& response.getContentType().startsWith("application/json"),
				"pre-stream failure must be JSON, not an event-stream; got " + response.getContentType());
		assertTrue(!body.contains("FetchNotFoundException") && !body.contains("event: token"),
				"must not leak the raw exception or open the SSE stream, got:\n" + body);
	}

	/**
	 * Gate 13: a staged TEAM profile (capability-advertised by the hub, not a "single-" id) must
	 * route through the SAME one-hub-call relay as a single-writer profile — never the legacy
	 * Java-side 3-call decomposition. Routing comes from ModelSwitchService#isStagedModel (a
	 * capability lookup), not from matching the model-id string.
	 */
	@Test
	public void chatStream_stagedTeamModel_relaysOneHubCallNotTheLegacyThreeCallDecomposition()
			throws Exception {
		Fixture f = newFixture(true);
		AtomicReference<String> hubRequestBody = new AtomicReference<String>();
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			hubRequestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
			String sse = ""
					+ "event: answer_done\n"
					+ "data: {\"answer\":\"Direct answer [1].\",\"references\":[],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"validating\"},"
					+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n"
					+ "event: answer_validation\n"
					+ "data: {\"answer\":\"Direct answer [1].\",\"references\":[],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"checked\"},"
					+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n"
					+ "event: indepth_pending\ndata: {\"status\":\"pending\",\"answer\":\"\"}\n\n"
					+ "event: indepth_done\ndata: {\"status\":\"complete\",\"answer\":\"Background claim.\"}\n\n"
					+ "event: done\n"
					+ "data: {\"answer\":\"Direct answer [1].\",\"references\":[],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"checked\"},"
					+ "\"inDepth\":{\"status\":\"complete\",\"answer\":\"Background claim.\"}}\n\n";
			byte[] bytes = sse.getBytes(StandardCharsets.UTF_8);
			exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
			exchange.sendResponseHeaders(200, bytes.length);
			try (OutputStream body = exchange.getResponseBody()) {
				body.write(bytes);
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "med-agent-team-high-validated");
			body.put("staged", "true");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(hubUrl, "med-agent-team-high-validated"))
					.thenReturn(new String[] { hubUrl, "med-agent-team-high-validated" });
			when(f.modelSwitchService.isStagedModel(hubUrl, "med-agent-team-high-validated"))
					.thenReturn(true);
			when(f.chatService.persistHubStagedAnswer(eq(f.session), any(), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Direct answer [1].", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));
			when(f.chatService.updateHubStagedMessage(eq(f.session), eq("assistant-msg-uuid"), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Direct answer [1].", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));

			MockHttpServletResponse response = new MockHttpServletResponse();

			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				f.controller.chatStream(body, response);
			}

			String sse = response.getContentAsString();
			JsonNode done = parseDoneEvent(sse);
			assertEquals("Direct answer [1].", done.get("answer").asText());
			assertEquals("checked", done.get("answerValidation").get("status").asText());
			assertEquals("complete", done.get("inDepth").get("status").asText());

			JsonNode hubRequest = MAPPER.readTree(hubRequestBody.get());
			assertEquals("med-agent-team-high-validated", hubRequest.get("model").asText());
			verify(f.chatService, never()).chatStreaming(any(), any(), any());
			verify(f.chatService, times(1)).persistHubStagedAnswer(eq(f.session), any(), any());
		}
		finally {
			hub.stop(0);
		}
	}

	/**
	 * Gate 2/11: a NON-staged remote model (e.g. bare parity, no phased decomposition) must still
	 * relay through the hub — the hub retrieves the chart itself (patient ref, not a locally
	 * assembled snapshot) and resolves references — instead of the local chatService.chatStreaming
	 * orchestration. Only the fully-local bundled engine (no remote endpoint at all) keeps using
	 * chatService.chatStreaming.
	 */
	@Test
	public void chatStream_nonStagedRemoteModel_relaysToHubInsteadOfLocalOrchestration() throws Exception {
		Fixture f = newFixture(true);
		AtomicReference<String> hubRequestBody = new AtomicReference<String>();
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			hubRequestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
			String completion = "{\"choices\":[{\"message\":{\"content\":"
					+ "\"{\\\"answer\\\":\\\"Bare answer [1].\\\",\\\"references\\\":"
					+ "[{\\\"index\\\":1,\\\"resourceType\\\":\\\"Observation\\\",\\\"resourceUuid\\\":\\\"obs-1\\\"}],"
					+ "\\\"blocks\\\":[]}\"}}]}";
			byte[] bytes = completion.getBytes(StandardCharsets.UTF_8);
			exchange.getResponseHeaders().add("Content-Type", "application/json");
			exchange.sendResponseHeaders(200, bytes.length);
			try (OutputStream body = exchange.getResponseBody()) {
				body.write(bytes);
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "med-agent-team-parity");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(hubUrl, "med-agent-team-parity"))
					.thenReturn(new String[] { hubUrl, "med-agent-team-parity" });
			when(f.chatService.persistHubStagedAnswer(eq(f.session), any(), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Bare answer [1].", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));

			MockHttpServletResponse response = new MockHttpServletResponse();

			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				f.controller.chatStream(body, response);
			}

			String sse = response.getContentAsString();
			JsonNode done = parseDoneEvent(sse);
			assertEquals("Bare answer [1].", done.get("answer").asText());
			assertEquals("Observation", done.get("references").get(0).get("resourceType").asText());
			assertEquals("med-agent-team-parity", done.get("model").asText());

			JsonNode hubRequest = MAPPER.readTree(hubRequestBody.get());
			assertEquals("med-agent-team-parity", hubRequest.get("model").asText());
			assertEquals("patient-uuid", hubRequest.get("patient").asText());
			assertFalse(hubRequest.get("stream").asBoolean(), "non-staged relay must not request the SSE contract");
			verify(f.chatService, never()).chatStreaming(any(), any(), any());
			verify(f.chatService, times(1)).persistHubStagedAnswer(eq(f.session), any(), any());
		}
		finally {
			hub.stop(0);
		}
	}

	/**
	 * Gate 11: the harness's own synchronous research client drives chartsearchai via
	 * {@code POST /chat}. A model resolved to a remote endpoint (any hub level id, including the
	 * low-level `answer:`/`answer-review:`/`indepth-only:` legs the harness uses for arm
	 * comparisons) must relay to the SAME hub engine the product streams against — not run
	 * chatService.chat's local chart-build + inference orchestration.
	 */
	@Test
	public void chat_remoteModel_relaysToHubInsteadOfLocalChat() throws Exception {
		Fixture f = newFixture(true);
		AtomicReference<String> hubRequestBody = new AtomicReference<String>();
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			hubRequestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
			String completion = "{\"choices\":[{\"message\":{\"content\":"
					+ "\"{\\\"answer\\\":\\\"Sync answer [1].\\\",\\\"references\\\":"
					+ "[{\\\"index\\\":1,\\\"resourceType\\\":\\\"Observation\\\",\\\"resourceUuid\\\":\\\"obs-1\\\"}],"
					+ "\\\"blocks\\\":[]}\"}}]}";
			byte[] bytes = completion.getBytes(StandardCharsets.UTF_8);
			exchange.getResponseHeaders().add("Content-Type", "application/json");
			exchange.sendResponseHeaders(200, bytes.length);
			try (OutputStream body = exchange.getResponseBody()) {
				body.write(bytes);
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "answer:gemma-4-12b@synthesis-answer~enforce~temp0");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(
					hubUrl, "answer:gemma-4-12b@synthesis-answer~enforce~temp0"))
					.thenReturn(new String[] { hubUrl, "answer:gemma-4-12b@synthesis-answer~enforce~temp0" });
			when(f.chatService.persistHubStagedAnswer(eq(f.session), any(), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Sync answer [1].", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));

			ResponseEntity<Object> response;
			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				response = f.controller.chat(body);
			}

			JsonNode result = MAPPER.valueToTree(response.getBody());
			assertEquals("Sync answer [1].", result.get("answer").asText());
			assertEquals("answer:gemma-4-12b@synthesis-answer~enforce~temp0", result.get("model").asText());

			JsonNode hubRequest = MAPPER.readTree(hubRequestBody.get());
			assertEquals("answer:gemma-4-12b@synthesis-answer~enforce~temp0", hubRequest.get("model").asText());
			assertEquals("patient-uuid", hubRequest.get("patient").asText());
			assertFalse(hubRequest.get("stream").asBoolean());
			verify(f.chatService, never()).chat(any(), any());

			// The PERSISTED wire (not the mocked ChatTurnResult) is what proves the hub's completion
			// body was actually parsed correctly — the response body above only reflects the stub.
			@SuppressWarnings("unchecked")
			ArgumentCaptor<Map<String, Object>> wireCaptor = ArgumentCaptor.forClass(Map.class);
			verify(f.chatService, times(1)).persistHubStagedAnswer(eq(f.session), any(), wireCaptor.capture());
			JsonNode persistedWire = MAPPER.valueToTree(wireCaptor.getValue());
			assertEquals("Sync answer [1].", persistedWire.get("answer").asText());
			assertEquals("Observation", persistedWire.get("references").get(0).get("resourceType").asText());
		}
		finally {
			hub.stop(0);
		}
	}

	@Test
	public void chatStream_hubNativeSingleProfile_relaysOneHubStreamAndUpdatesSameMessage()
			throws Exception {
		Fixture f = newFixture(true);
		AtomicReference<String> hubRequestBody = new AtomicReference<String>();
		AtomicReference<String> hubRequestProtocol = new AtomicReference<String>();
		AtomicReference<String> hubRequestAccept = new AtomicReference<String>();
		AtomicReference<String> hubRequestContentType = new AtomicReference<String>();
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			hubRequestProtocol.set(exchange.getProtocol());
			hubRequestAccept.set(exchange.getRequestHeaders().getFirst("Accept"));
			hubRequestContentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
			hubRequestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
			String sse = ""
					+ "event: answer_done\n"
					+ "data: {\"answer\":\"Initial answer [1].\",\"references\":[{\"index\":1,"
					+ "\"resourceType\":\"Observation\",\"resourceUuid\":\"obs-1\","
					+ "\"groundingStatus\":\"checking\",\"grounded\":null}],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"validating\"},"
					+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n"
					+ "event: answer_validation\n"
					+ "data: {\"answer\":\"Edited answer [2].\",\"references\":[{\"index\":2,"
					+ "\"resourceType\":\"Order\",\"resourceUuid\":\"ord-2\","
					+ "\"groundingStatus\":\"checking\",\"grounded\":null}],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"edited\",\"originalAnswer\":\"Initial answer [1].\"},"
					+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n"
					+ "event: indepth_pending\n"
					+ "data: {\"status\":\"pending\",\"answer\":\"\"}\n\n"
					+ "event: indepth_done\n"
					+ "data: {\"status\":\"complete\",\"answer\":\"Deep details.\"}\n\n"
					+ "event: done\n"
					+ "data: {\"answer\":\"Edited answer [2].\",\"references\":[{\"index\":2,"
					+ "\"resourceType\":\"Order\",\"resourceUuid\":\"ord-2\","
					+ "\"groundingStatus\":\"verified\",\"grounded\":true}],\"blocks\":[],"
					+ "\"answerValidation\":{\"status\":\"edited\",\"originalAnswer\":\"Initial answer [1].\"},"
					+ "\"inDepth\":{\"status\":\"complete\",\"answer\":\"Deep details.\"}}\n\n";
			byte[] bytes = sse.getBytes(StandardCharsets.UTF_8);
			exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
			exchange.sendResponseHeaders(200, bytes.length);
			try (OutputStream body = exchange.getResponseBody()) {
				body.write(bytes);
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "single-12b-checked");
			body.put("staged", "true");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(hubUrl, "single-12b-checked"))
					.thenReturn(new String[] { hubUrl, "single-12b-checked" });
			when(f.modelSwitchService.isStagedModel(hubUrl, "single-12b-checked")).thenReturn(true);
			when(f.chatService.persistHubStagedAnswer(
					eq(f.session), eq("What medications is this patient taking?"), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Initial answer [1].",
							Collections.emptyList()), "session-uuid", "assistant-msg-uuid"));
			when(f.chatService.updateHubStagedMessage(eq(f.session), eq("assistant-msg-uuid"), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Edited answer [2].",
							Collections.emptyList()), "session-uuid", "assistant-msg-uuid"));

			MockHttpServletResponse response = new MockHttpServletResponse();

			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				f.controller.chatStream(body, response);
			}

			String sse = response.getContentAsString();
			assertTrue(sse.indexOf("event: answer_done") >= 0, "answer_done missing:\n" + sse);
			assertTrue(sse.indexOf("event: answer_validation") > sse.indexOf("event: answer_done"),
					"answer_validation must follow answer_done:\n" + sse);
			assertTrue(sse.indexOf("event: indepth_done") > sse.indexOf("event: indepth_pending"),
					"indepth_done must follow pending:\n" + sse);
			assertTrue(sse.indexOf("event: done") > sse.indexOf("event: indepth_done"),
					"done must follow indepth_done:\n" + sse);

			JsonNode answerDone = parseEvent(sse, "answer_done");
			assertEquals("assistant-msg-uuid", answerDone.get("messageId").asText());
			assertEquals("checking", answerDone.get("references").get(0).get("groundingStatus").asText());
			JsonNode done = parseDoneEvent(sse);
			assertEquals("Edited answer [2].", done.get("answer").asText());
			assertEquals("verified", done.get("references").get(0).get("groundingStatus").asText());
			assertEquals("single-12b-checked", done.get("model").asText());

			JsonNode hubRequest = MAPPER.readTree(hubRequestBody.get());
			assertEquals("HTTP/1.1", hubRequestProtocol.get());
			assertEquals("text/event-stream", hubRequestAccept.get());
			assertTrue(hubRequestContentType.get().startsWith("application/json"));
			assertTrue(hubRequestBody.get().length() > 0, "hub request JSON body must not be empty");
			assertEquals("single-12b-checked", hubRequest.get("model").asText());
			assertEquals("patient-uuid", hubRequest.get("patient").asText());
			assertEquals("What medications is this patient taking?",
					hubRequest.get("messages").get(0).get("content").asText());
			verify(f.modelSwitchService, times(2)).validateEndpointAndModel(hubUrl, "single-12b-checked");
			verify(f.modelSwitchService, never()).validateEndpointAndModel(
					eq(hubUrl), eq("answer:gemma-4-12b@synthesis-answer~enforce~temp0"));
			verify(f.modelSwitchService, never()).validateEndpointAndModel(
					eq(hubUrl), eq("answer-review:qwen2.5-14b"));
			verify(f.modelSwitchService, never()).validateEndpointAndModel(
					eq(hubUrl), eq("indepth-only:single-12b-checked"));
			verify(f.chatService, times(1)).persistHubStagedAnswer(
					eq(f.session), eq("What medications is this patient taking?"), any());
			verify(f.chatService, times(3)).updateHubStagedMessage(
					eq(f.session), eq("assistant-msg-uuid"), any());
			verify(f.chatService, never()).chatStreaming(any(), any(), any());
		}
		finally {
			hub.stop(0);
		}
	}

	/**
	 * Gate 5: the hub relay must thread prior conversation turns (prose-only, never the raw stored
	 * JSON envelope) into the hub request, with the CURRENT question last — a follow-up question
	 * ("what was the ISO date in your last answer?") is unanswerable from the chart alone, so
	 * without this the hub-native default path silently loses multi-turn context.
	 */
	@Test
	public void chatStream_hubNativeSingleProfile_threadsPriorConversationTurnsBeforeTheQuestion()
			throws Exception {
		Fixture f = newFixture(true);
		ChatMessage priorUser = new ChatMessage();
		priorUser.setRole(ChatMessage.ROLE_USER);
		priorUser.setContent("What was the most recent visit date?");
		ChatMessage priorAssistant = new ChatMessage();
		priorAssistant.setRole(ChatMessage.ROLE_ASSISTANT);
		priorAssistant.setContent("2026-01-26"); // priorTurnsForRelay's contract: prose, not JSON
		when(f.chatService.priorTurnsForRelay(f.session))
				.thenReturn(java.util.Arrays.asList(priorUser, priorAssistant));

		AtomicReference<String> hubRequestBody = new AtomicReference<String>();
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			hubRequestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
			String sse = ""
					+ "event: answer_done\n"
					+ "data: {\"answer\":\"2026-01-26.\",\"references\":[],\"blocks\":[],"
					+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n"
					+ "event: indepth_pending\ndata: {\"status\":\"pending\",\"answer\":\"\"}\n\n"
					+ "event: indepth_done\ndata: {\"status\":\"complete\",\"answer\":\"\"}\n\n"
					+ "event: done\n"
					+ "data: {\"answer\":\"2026-01-26.\",\"references\":[],\"blocks\":[],"
					+ "\"inDepth\":{\"status\":\"complete\",\"answer\":\"\"}}\n\n";
			byte[] bytes = sse.getBytes(StandardCharsets.UTF_8);
			exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
			exchange.sendResponseHeaders(200, bytes.length);
			try (OutputStream body = exchange.getResponseBody()) {
				body.write(bytes);
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("question", "Repeat just the ISO date from your previous answer.");
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "single-12b-checked");
			body.put("staged", "true");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(hubUrl, "single-12b-checked"))
					.thenReturn(new String[] { hubUrl, "single-12b-checked" });
			when(f.modelSwitchService.isStagedModel(hubUrl, "single-12b-checked")).thenReturn(true);
			when(f.chatService.persistHubStagedAnswer(eq(f.session), any(), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("2026-01-26.", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));
			when(f.chatService.updateHubStagedMessage(eq(f.session), eq("assistant-msg-uuid"), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("2026-01-26.", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));

			MockHttpServletResponse response = new MockHttpServletResponse();

			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				f.controller.chatStream(body, response);
			}

			JsonNode hubRequest = MAPPER.readTree(hubRequestBody.get());
			JsonNode messages = hubRequest.get("messages");
			assertEquals(3, messages.size(), "expected 2 prior turns + the current question");
			assertEquals("user", messages.get(0).get("role").asText());
			assertEquals("What was the most recent visit date?", messages.get(0).get("content").asText());
			assertEquals("assistant", messages.get(1).get("role").asText());
			assertEquals("2026-01-26", messages.get(1).get("content").asText());
			assertEquals("user", messages.get(2).get("role").asText());
			assertEquals("Repeat just the ISO date from your previous answer.",
					messages.get(2).get("content").asText());
		}
		finally {
			hub.stop(0);
		}
	}

	/**
	 * Gate 6: a mid-leg browser disconnect must be detected via a heartbeat-triggered write, not
	 * only discovered on the NEXT real event. The hub answers fast, then goes quiet (as it would
	 * mid in-depth generation) sending only heartbeat comment lines for well over a second before
	 * ever emitting {@code done}; the browser "disconnects" after the first successful write. If
	 * the relay only writes on real events, it blocks reading heartbeats for the whole stall and
	 * this test takes 1200ms+; if it writes (and so notices the disconnect) on each heartbeat, it
	 * must return promptly.
	 */
	@Test
	public void chatStream_hubNativeSingleProfile_abortsPromptlyOnDisconnectDuringHeartbeats()
			throws Exception {
		Fixture f = newFixture(true);
		HttpServer hub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		hub.createContext("/v1/chat/completions", exchange -> {
			exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
			exchange.sendResponseHeaders(200, 0); // chunked — arbitrary length, flushed incrementally
			try (OutputStream body = exchange.getResponseBody()) {
				body.write((""
						+ "event: answer_done\n"
						+ "data: {\"answer\":\"Ans.\",\"references\":[],\"blocks\":[],"
						+ "\"inDepth\":{\"status\":\"pending\",\"answer\":\"\"}}\n\n")
						.getBytes(StandardCharsets.UTF_8));
				body.flush();
				for (int i = 0; i < 8; i++) {
					body.write(": hb\n\n".getBytes(StandardCharsets.UTF_8));
					body.flush();
					try {
						Thread.sleep(150);
					}
					catch (InterruptedException interrupted) {
						Thread.currentThread().interrupt();
						return;
					}
				}
				body.write((""
						+ "event: done\n"
						+ "data: {\"answer\":\"Ans.\",\"references\":[],\"blocks\":[],"
						+ "\"inDepth\":{\"status\":\"complete\",\"answer\":\"\"}}\n\n")
						.getBytes(StandardCharsets.UTF_8));
			}
			catch (IOException expectedOnceRelayCloses) {
				// the relay closed its connection to us mid-stream once it noticed the "browser"
				// disconnect — exactly the behavior under test.
			}
		});
		hub.start();
		String hubUrl = "http://127.0.0.1:" + hub.getAddress().getPort() + "/v1/chat/completions";
		try {
			Map<String, String> body = chatBody();
			body.put("endpointUrl", hubUrl);
			body.put("modelName", "single-12b-checked");
			body.put("staged", "true");
			when(f.adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
					.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
			when(f.modelSwitchService.validateEndpointAndModel(hubUrl, "single-12b-checked"))
					.thenReturn(new String[] { hubUrl, "single-12b-checked" });
			when(f.modelSwitchService.isStagedModel(hubUrl, "single-12b-checked")).thenReturn(true);
			when(f.chatService.persistHubStagedAnswer(eq(f.session), any(), any()))
					.thenReturn(new ChatTurnResult(new ChartAnswer("Ans.", Collections.emptyList()),
							"session-uuid", "assistant-msg-uuid"));

			// "Browser" that accepts the FIRST write (answer_done) then disconnects.
			DisconnectAfterNWritesResponse response = new DisconnectAfterNWritesResponse(1);

			long startNanos = System.nanoTime();
			try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
				ctx.when(() -> Context.requirePrivilege(
						ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
				ctx.when(Context::getPatientService).thenReturn(f.patientService);
				ctx.when(Context::getAdministrationService).thenReturn(f.adminService);
				ctx.when(Context::getAuthenticatedUser).thenReturn(f.user);
				ctx.when(Context::getRuntimeProperties).thenReturn(new Properties());

				f.controller.chatStream(body, response);
			}
			long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;

			assertTrue(elapsedMs < 700, "relay must abort promptly on a mid-leg disconnect — took "
					+ elapsedMs + "ms; 8 heartbeats * 150ms = 1200ms of missed opportunities to notice "
					+ "means it only writes (and so only notices disconnects) on real events");
		}
		finally {
			hub.stop(0);
		}
	}

	/**
	 * A response whose output stream accepts the first {@code failAfterWrites} write calls, then
	 * throws {@link IOException} on every subsequent write — simulating a browser that received
	 * the fast answer, then closed the connection. NOT an {@link javax.servlet.http.HttpServletResponseWrapper}:
	 * the controller unwraps those (to bypass buffering wrappers), which would strip this behavior.
	 */
	private static final class DisconnectAfterNWritesResponse extends MockHttpServletResponse {

		private final int failAfterWrites;

		private int writeCalls;

		DisconnectAfterNWritesResponse(int failAfterWrites) {
			this.failAfterWrites = failAfterWrites;
		}

		@Override
		public javax.servlet.ServletOutputStream getOutputStream() {
			return new javax.servlet.ServletOutputStream() {

				@Override
				public void write(int b) throws IOException {
					write(new byte[] { (byte) b }, 0, 1);
				}

				@Override
				public void write(byte[] b, int off, int len) throws IOException {
					writeCalls++;
					if (writeCalls > failAfterWrites) {
						throw new IOException("simulated browser disconnect");
					}
				}
			};
		}
	}

	/**
	 * Extracts and parses the JSON object carried by the terminal {@code done}
	 * SSE event. The controller emits {@code data: <json>} lines after
	 * {@code event: done}; reassemble them and parse.
	 */
	private static JsonNode parseDoneEvent(String sse) throws Exception {
		int doneIdx = sse.indexOf("event: done");
		assertTrue(doneIdx >= 0, "no done event to parse");
		String afterDone = sse.substring(doneIdx);
		StringBuilder json = new StringBuilder();
		for (String line : afterDone.split("\n")) {
			if (line.startsWith("data: ")) {
				json.append(line.substring("data: ".length()));
			} else if (line.startsWith("event: ") && json.length() > 0) {
				break;
			}
		}
		return MAPPER.readTree(json.toString());
	}

	private static JsonNode parseEvent(String sse, String eventName) throws Exception {
		int eventIdx = sse.indexOf("event: " + eventName);
		assertTrue(eventIdx >= 0, "no " + eventName + " event to parse");
		String afterEvent = sse.substring(eventIdx);
		StringBuilder json = new StringBuilder();
		for (String line : afterEvent.split("\n")) {
			if (line.startsWith("data: ")) {
				json.append(line.substring("data: ".length()));
			} else if (line.startsWith("event: ") && json.length() > 0) {
				break;
			}
		}
		return MAPPER.readTree(json.toString());
	}

	private static final class Fixture {

		Patient patient;

		User user;

		PatientService patientService;

		AdministrationService adminService;

		PatientAccessCheck patientAccessCheck;

		AuditLogService auditLogService;

		ChatService chatService;

		ModelSwitchService modelSwitchService;

		ChatSession session;

		ChartSearchAiRestController controller;
	}
}
