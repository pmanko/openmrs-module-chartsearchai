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
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * The request-scoped LLM backend override that lets a single request select a
 * backend without mutating the config-controlled global default.
 */
public class RequestLlmOverrideTest {

	@AfterEach
	public void clearOverride() {
		RequestLlmOverride.clear();
	}

	@Test
	public void unset_returnsNullSoTheEngineFallsBackToGlobals() {
		assertNull(RequestLlmOverride.endpointUrl());
		assertNull(RequestLlmOverride.modelName());
	}

	@Test
	public void set_isReadBackForThisRequest() {
		RequestLlmOverride.set("http://medhub:8080/v1/chat/completions", "med-agent-team");
		assertEquals("http://medhub:8080/v1/chat/completions", RequestLlmOverride.endpointUrl());
		assertEquals("med-agent-team", RequestLlmOverride.modelName());
	}

	@Test
	public void clear_removesTheOverride() {
		RequestLlmOverride.set("http://medhub:8080/v1/chat/completions", "med-agent-team");
		RequestLlmOverride.clear();
		assertNull(RequestLlmOverride.endpointUrl());
		assertNull(RequestLlmOverride.modelName());
	}

	@Test
	public void overrideIsScopedToTheRequestThread_notVisibleToAnotherThread()
			throws InterruptedException {
		// The whole point: one request's backend choice must never leak to another
		// request served by a different (pooled) thread.
		RequestLlmOverride.set("http://this-thread:8080/v1/chat/completions", "this-model");

		final String[] seenByOther = new String[2];
		Thread other = new Thread(() -> {
			seenByOther[0] = RequestLlmOverride.endpointUrl();
			seenByOther[1] = RequestLlmOverride.modelName();
		});
		other.start();
		other.join();

		assertNull(seenByOther[0], "another thread must not see this thread's override");
		assertNull(seenByOther[1]);
		assertEquals("http://this-thread:8080/v1/chat/completions",
				RequestLlmOverride.endpointUrl(), "this thread still sees its own override");
	}
}
