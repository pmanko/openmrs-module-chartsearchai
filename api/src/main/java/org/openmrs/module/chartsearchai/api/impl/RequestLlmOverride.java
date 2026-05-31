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

/**
 * Request-scoped override of the active LLM backend (endpoint + model).
 *
 * <p>The active backend normally comes from the config-controlled global
 * properties ({@code chartsearchai.llm.remote.endpointUrl} / {@code modelName}).
 * A single request may override it <em>for that request only</em> — set here by
 * the REST layer (after the override has been validated against the endpoint
 * registry) and read by {@link RemoteLlmEngine}, which falls back to the globals
 * when nothing is set. This lets a caller — the validation harness, or a chat
 * thread that picked a specific model — select a backend per request without
 * mutating the shared, config-controlled global default.</p>
 *
 * <p>The chat/search request path runs entirely on the request thread (no async
 * hop to an executor), so a thread-local is safe. Whoever calls {@link #set} MUST
 * call {@link #clear()} in a {@code finally} block so the value can never leak
 * into a pooled thread serving a later request.</p>
 */
public final class RequestLlmOverride {

	private static final ThreadLocal<String[]> OVERRIDE = new ThreadLocal<String[]>();

	private RequestLlmOverride() {
	}

	/** Set the override for the current request thread (endpoint + model). */
	public static void set(String endpointUrl, String modelName) {
		OVERRIDE.set(new String[] { endpointUrl, modelName });
	}

	/** The override endpoint URL for this request, or {@code null} if none. */
	public static String endpointUrl() {
		String[] o = OVERRIDE.get();
		return o == null ? null : o[0];
	}

	/** The override model name for this request, or {@code null} if none. */
	public static String modelName() {
		String[] o = OVERRIDE.get();
		return o == null ? null : o[1];
	}

	/** Clear any override. Safe to call when none is set; MUST run in a finally. */
	public static void clear() {
		OVERRIDE.remove();
	}
}
