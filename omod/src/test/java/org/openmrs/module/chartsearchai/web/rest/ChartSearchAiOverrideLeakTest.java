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

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.servlet.http.HttpServletResponse;

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
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.PatientAccessCheck;
import org.openmrs.module.chartsearchai.api.impl.ModelSwitchService;
import org.openmrs.module.chartsearchai.api.impl.RequestLlmOverride;
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * Drives {@link ChartSearchAiRestController#chatStream} on the remote engine with
 * a valid per-request backend override, against a response whose
 * {@code getOutputStream()} throws — the unprotected window between
 * {@code resolveOverride} (which sets the {@link RequestLlmOverride} thread-local)
 * and the streaming {@code try} that clears it in {@code finally}.
 *
 * <p>If the throw escapes that window without clearing, the override leaks onto
 * the (pooled) request thread and bleeds into the next request. The test asserts
 * the thread-local is null after the call regardless of the IOException.</p>
 */
public class ChartSearchAiOverrideLeakTest {

	@AfterEach
	public void clearOverride() {
		// A failing run can leave the override set; clear it so it can't poison
		// sibling tests sharing this thread.
		RequestLlmOverride.clear();
	}

	@Test
	public void chatStream_shouldClearRequestOverride_whenStreamOpenThrows() throws Exception {
		Patient patient = mock(Patient.class);
		User user = mock(User.class);
		PatientService patientService = mock(PatientService.class);
		AdministrationService adminService = mock(AdministrationService.class);
		PatientAccessCheck patientAccessCheck = mock(PatientAccessCheck.class);
		AuditLogService auditLogService = mock(AuditLogService.class);
		ChatService chatService = mock(ChatService.class);
		ModelSwitchService modelSwitchService = mock(ModelSwitchService.class);
		ChatSession session = mock(ChatSession.class);

		when(patientService.getPatientByUuid("patient-uuid")).thenReturn(patient);
		// Remote engine — otherwise resolveOverride rejects the override and the
		// thread-local is never set (no leak to test).
		when(adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE))
				.thenReturn(ChartSearchAiConstants.LLM_ENGINE_REMOTE);
		lenient().when(adminService.getGlobalProperty(ChartSearchAiConstants.GP_LLM_REMOTE_MODEL_NAME))
				.thenReturn("configured-model");
		lenient().when(adminService.getGlobalProperty(ChartSearchAiConstants.GP_RATE_LIMIT_PER_MINUTE))
				.thenReturn(null);

		when(patientAccessCheck.canAccess(any(), eq(patient))).thenReturn(true);
		lenient().when(auditLogService.getQueryCountByUserSince(any(), any())).thenReturn(0L);
		when(chatService.openOrLoadActiveSession(patient)).thenReturn(session);
		lenient().when(session.getUuid()).thenReturn("session-uuid");
		// The override is valid: registry/probe pass, returns trimmed [url, model].
		when(modelSwitchService.validateEndpointAndModel(
				"http://endpoint:1234/v1/chat/completions", "override-model"))
				.thenReturn(new String[] { "http://endpoint:1234/v1/chat/completions", "override-model" });

		ChartSearchAiRestController controller = new ChartSearchAiRestController();
		ReflectionTestUtils.setField(controller, "patientAccessCheck", patientAccessCheck);
		ReflectionTestUtils.setField(controller, "auditLogService", auditLogService);
		ReflectionTestUtils.setField(controller, "chatService", chatService);
		ReflectionTestUtils.setField(controller, "modelSwitchService", modelSwitchService);

		// Response whose stream cannot be opened — simulates the container failing
		// to hand back an OutputStream after headers are committed. A real mock of
		// the HttpServletResponse interface lets getOutputStream() throw the checked
		// IOException that MockHttpServletResponse's signature forbids.
		HttpServletResponse response = mock(HttpServletResponse.class);
		when(response.getOutputStream()).thenThrow(new IOException("stream broken"));

		Map<String, String> body = new HashMap<String, String>();
		body.put("patient", "patient-uuid");
		body.put("question", "What medications is this patient taking?");
		body.put("endpointUrl", "http://endpoint:1234/v1/chat/completions");
		body.put("modelName", "override-model");

		try (MockedStatic<Context> ctx = mockStatic(Context.class)) {
			ctx.when(() -> Context.requirePrivilege(
					ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)).then(inv -> null);
			ctx.when(Context::getPatientService).thenReturn(patientService);
			ctx.when(Context::getAdministrationService).thenReturn(adminService);
			ctx.when(Context::getAuthenticatedUser).thenReturn(user);

			// getOutputStream() is OUTSIDE the streaming try, so the IOException
			// propagates out of chatStream (declared throws IOException).
			assertThrows(IOException.class, () -> controller.chatStream(body, response));
		}

		assertNull(RequestLlmOverride.endpointUrl(),
				"Per-request override endpointUrl leaked after the stream-open IOException");
		assertNull(RequestLlmOverride.modelName(),
				"Per-request override modelName leaked after the stream-open IOException");
	}
}
