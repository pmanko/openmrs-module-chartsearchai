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

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;

import org.junit.jupiter.api.Test;

/**
 * Tests for the streaming search endpoint controller structure.
 * Verifies that the controller writes SSE events directly to the response
 * without background threads or shared auth state.
 */
public class ChartSearchAiStreamingTest {

	@Test
	public void controller_shouldNotStoreUserContextAsField() {
		for (Field field : ChartSearchAiRestController.class.getDeclaredFields()) {
			assertTrue(
					!field.getType().getName().contains("UserContext"),
					"Controller must not store UserContext as a field");
		}
	}

	@Test
	public void searchStreamMethod_shouldExist() throws NoSuchMethodException {
		assertNotNull(
				ChartSearchAiRestController.class.getMethod("searchStream",
						java.util.Map.class, javax.servlet.http.HttpServletResponse.class),
				"searchStream method should exist");
	}

	@Test
	public void streamingEndpoint_shouldNotUseBackgroundThreads() throws Exception {
		String sourceFile = "omod/src/main/java/org/openmrs/module/chartsearchai"
				+ "/web/rest/ChartSearchAiRestController.java";
		java.io.File file = new java.io.File(sourceFile);
		if (!file.exists()) {
			file = new java.io.File("../" + sourceFile);
		}
		if (file.exists()) {
			String source = new String(java.nio.file.Files.readAllBytes(file.toPath()));
			assertTrue(!source.contains("new Thread("),
					"Streaming must not create background threads");
			assertTrue(!source.contains("import org.springframework.web.servlet.mvc.method.annotation.SseEmitter"),
					"Streaming must not import SseEmitter");
			assertTrue(!source.contains("addProxyPrivilege"),
					"Streaming must not use proxy privileges");
			assertTrue(!source.contains("setUserContext"),
					"Must not share UserContext across threads");
		}
	}

	@Test
	public void authorizationCheck_shouldHappenBeforeStreaming() throws Exception {
		String sourceFile = "omod/src/main/java/org/openmrs/module/chartsearchai"
				+ "/web/rest/ChartSearchAiRestController.java";
		java.io.File file = new java.io.File(sourceFile);
		if (!file.exists()) {
			file = new java.io.File("../" + sourceFile);
		}
		if (file.exists()) {
			String source = new String(java.nio.file.Files.readAllBytes(file.toPath()));

			int streamMethodIdx = source.indexOf("public void searchStream");
			assertTrue(streamMethodIdx >= 0, "searchStream method must exist");

			int requirePriv = source.indexOf(
					"Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)",
					streamMethodIdx);
			int canAccess = source.indexOf("patientAccessCheck.canAccess(",
					streamMethodIdx);
			int searchStreaming = source.indexOf("searchStreaming(", streamMethodIdx);

			assertTrue(requirePriv >= 0, "Must check PRIV_QUERY_PATIENT_DATA");
			assertTrue(canAccess >= 0, "Must check patient access");
			assertTrue(searchStreaming >= 0, "Must call searchStreaming");

			assertTrue(requirePriv < searchStreaming,
					"Privilege check must happen before streaming");
			assertTrue(canAccess < searchStreaming,
					"Patient access check must happen before streaming");
		}
	}
}
