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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
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
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.test.util.ReflectionTestUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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
