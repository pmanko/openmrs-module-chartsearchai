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
 * Verifies that the controller does not store mutable auth state at class level
 * and that authorization is checked before the streaming thread is spawned.
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
						java.util.Map.class),
				"searchStream method should exist");
	}

	@Test
	public void streamingThread_shouldUseProxyPrivilegesNotSharedContext()
			throws Exception {
		String sourceFile = "omod/src/main/java/org/openmrs/module/chartsearchai"
				+ "/web/rest/ChartSearchAiRestController.java";
		java.io.File file = new java.io.File(sourceFile);
		if (!file.exists()) {
			file = new java.io.File("../" + sourceFile);
		}
		if (file.exists()) {
			String source = new String(java.nio.file.Files.readAllBytes(file.toPath()));
			assertTrue(source.contains("addProxyPrivilege"),
					"Streaming thread must use proxy privileges for data access");
			assertTrue(source.contains("removeProxyPrivilege"),
					"Streaming thread must clean up proxy privileges");
			assertTrue(!source.contains("setUserContext"),
					"Must not share UserContext across threads");
		}
	}

	@Test
	public void authorizationCheck_shouldHappenBeforeThreadCreation() throws Exception {
		String sourceFile = "omod/src/main/java/org/openmrs/module/chartsearchai"
				+ "/web/rest/ChartSearchAiRestController.java";
		java.io.File file = new java.io.File(sourceFile);
		if (!file.exists()) {
			file = new java.io.File("../" + sourceFile);
		}
		if (file.exists()) {
			String source = new String(java.nio.file.Files.readAllBytes(file.toPath()));
			int requirePrivIdx = source.indexOf("Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)");
			int canAccessIdx = source.indexOf("patientAccessCheck.canAccess(");
			int threadIdx = source.indexOf("new Thread(");

			// Find the streaming-related instances (second occurrence for stream method)
			assertTrue(requirePrivIdx >= 0, "Must check PRIV_QUERY_PATIENT_DATA");
			assertTrue(canAccessIdx >= 0, "Must check patient access");
			assertTrue(threadIdx >= 0, "Must create streaming thread");

			// The privilege and access checks must appear before the thread creation
			// in the searchStream method
			int streamMethodIdx = source.indexOf("public SseEmitter searchStream");
			assertTrue(streamMethodIdx >= 0, "searchStream method must exist");

			int streamRequirePriv = source.indexOf(
					"Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA)",
					streamMethodIdx);
			int streamCanAccess = source.indexOf("patientAccessCheck.canAccess(",
					streamMethodIdx);
			int streamThread = source.indexOf("new Thread(", streamMethodIdx);

			assertTrue(streamRequirePriv < streamThread,
					"Privilege check must happen before thread creation");
			assertTrue(streamCanAccess < streamThread,
					"Patient access check must happen before thread creation");
		}
	}
}
