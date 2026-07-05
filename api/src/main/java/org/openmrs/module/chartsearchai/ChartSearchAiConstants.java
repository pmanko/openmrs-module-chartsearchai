/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai;

public class ChartSearchAiConstants {

	public static final String PRIV_QUERY_PATIENT_DATA = "AI Query Patient Data";

	public static final String PRIV_VIEW_AUDIT_LOGS = "View AI Audit Logs";

	/**
	 * When {@code true}, the chart serializer run-length de-dups the obs-group membership label: a member
	 * renders {@code " (part of: <group>)"} only when its group differs from the immediately-preceding
	 * record's group (mirrors the date-run compression). Applies to ALL obs groups (lab panels,
	 * vital-signs sets, exam findings, ...), not only lab panels. VERIFIED 2026-06-18: real saving is only
	 * ~2% of prompt tokens (a chars/4 estimate had overstated it ~3x), and it is SAFE ONLY ON E4B+ — on
	 * the small E2B model it causes a clustering failure (a false "no results" for a thinned-label group).
	 * So only enable on E4B-or-larger deployments. Default {@code false} (legacy every-member labelling).
	 */
	public static final String GP_SERIALIZER_DEDUP_GROUP_LABELS = "chartsearchai.serializer.dedupGroupLabels";

	public static final String GP_AUDIT_LOG_RETENTION_DAYS = "chartsearchai.auditLogRetentionDays";

	public static final int DEFAULT_AUDIT_LOG_RETENTION_DAYS = 90;

	public static final String GP_CHAT_RETENTION_DAYS = "chartsearchai.chat.retentionDays";

	public static final int DEFAULT_CHAT_RETENTION_DAYS = 90;

	public static final String GP_LLM_ENGINE = "chartsearchai.llm.engine";

	public static final String LLM_ENGINE_REMOTE = "remote";

	public static final String GP_LLM_REMOTE_ENDPOINT_URL = "chartsearchai.llm.remote.endpointUrl";

	public static final String RP_LLM_REMOTE_API_KEY = "chartsearchai.llm.remote.apikey";

	public static final String GP_LLM_REMOTE_MODEL_NAME = "chartsearchai.llm.remote.modelName";

	/**
	 * Optional JSON registry of selectable endpoints for the picker's
	 * per-endpoint sections, e.g.
	 * {@code [{"label":"LM Studio","url":".../v1/chat/completions"},
	 * {"label":"Med Agent Hub","url":"http://med-agent-hub:8080/v1/chat/completions"}]}.
	 * When unset/blank the picker falls back to a single section built from
	 * {@link #GP_LLM_REMOTE_ENDPOINT_URL}.
	 */
	public static final String GP_LLM_REMOTE_ENDPOINTS = "chartsearchai.llm.remote.endpoints";

	public static final String GP_RATE_LIMIT_PER_MINUTE = "chartsearchai.rateLimitPerMinute";

	public static final int DEFAULT_RATE_LIMIT_PER_MINUTE = 10;

	// Resource type identifiers used in embeddings and citations
	public static final String RESOURCE_TYPE_OBS = "obs";

	public static final String RESOURCE_TYPE_CONDITION = "condition";

	public static final String RESOURCE_TYPE_ALLERGY = "allergy";

	public static final String RESOURCE_TYPE_DIAGNOSIS = "diagnosis";

	public static final String RESOURCE_TYPE_ORDER = "order";

	public static final String RESOURCE_TYPE_PROGRAM = "program";

	public static final String RESOURCE_TYPE_MEDICATION_DISPENSE = "medication_dispense";

	private ChartSearchAiConstants() {
	}
}
