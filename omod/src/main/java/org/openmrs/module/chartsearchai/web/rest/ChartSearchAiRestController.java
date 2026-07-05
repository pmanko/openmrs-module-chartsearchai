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

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpServletResponseWrapper;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.openmrs.Patient;
import org.openmrs.User;
import org.openmrs.api.APIAuthenticationException;
import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.api.context.ContextAuthenticationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.api.ChartTooLargeException;
import org.openmrs.module.chartsearchai.api.ChartSearchService.ChartAnswer;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.api.AuditLogService;
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.ChatService.ChatTurnResult;
import org.openmrs.module.chartsearchai.api.PatientAccessCheck;
import org.openmrs.module.chartsearchai.api.impl.RequestLlmOverride;
import org.openmrs.module.chartsearchai.api.impl.ResponseBlock;
import org.openmrs.module.chartsearchai.api.impl.WarmupExecutor;
import org.openmrs.module.chartsearchai.model.ChartSearchAuditLog;
import org.openmrs.module.chartsearchai.reference.SafetyWarning;
import org.openmrs.module.chartsearchai.model.ChatMessage;
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.openmrs.module.webservices.rest.web.RestConstants;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * REST endpoint for AI-powered chart search.
 *
 * <pre>
 * POST /ws/rest/v1/chartsearchai/search
 * {
 *   "patient": "patient-uuid-here",
 *   "question": "What medications is this patient on?"
 * }
 * </pre>
 */
@Controller
@RequestMapping("/rest/" + RestConstants.VERSION_1 + "/chartsearchai")
public class ChartSearchAiRestController {

	private static final Logger log = LoggerFactory.getLogger(ChartSearchAiRestController.class);

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private static final int MAX_QUESTION_LENGTH = 1000;

	private static final Pattern CONTROL_CHARS = Pattern.compile("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]");

	private static String formatDate(Date date) {
		return date != null ? DateFormatUtil.formatDate(date) : null;
	}

	/**
	 * Serialize a list of {@link ResponseBlock} into the JSON-Map shape used
	 * by both the {@code /chat} sync response and the SSE {@code done} event.
	 * Keeps wire format identical across the two surfaces so the SPA only
	 * implements one parser.
	 */
	private static List<Map<String, Object>> blocksToJson(List<ResponseBlock> blocks) {
		List<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
		if (blocks == null) {
			return out;
		}
		for (ResponseBlock block : blocks) {
			Map<String, Object> blockMap = new LinkedHashMap<String, Object>();
			blockMap.put("kind", block.getKind());
			if (block.getTitle() != null) {
				blockMap.put("title", block.getTitle());
			}
			List<Map<String, Object>> columns = new ArrayList<Map<String, Object>>();
			for (ResponseBlock.Column c : block.getColumns()) {
				Map<String, Object> col = new LinkedHashMap<String, Object>();
				col.put("key", c.getKey());
				col.put("label", c.getLabel());
				columns.add(col);
			}
			blockMap.put("columns", columns);
			List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
			for (ResponseBlock.Row row : block.getRows()) {
				Map<String, Object> cellsMap = new LinkedHashMap<String, Object>();
				for (Map.Entry<String, ResponseBlock.Cell> entry : row.getCells().entrySet()) {
					Map<String, Object> cellMap = new LinkedHashMap<String, Object>();
					cellMap.put("text", entry.getValue().getText());
					cellMap.put("refs", entry.getValue().getRefs());
					cellsMap.put(entry.getKey(), cellMap);
				}
				Map<String, Object> rowMap = new LinkedHashMap<String, Object>();
				rowMap.put("cells", cellsMap);
				rows.add(rowMap);
			}
			blockMap.put("rows", rows);
			out.add(blockMap);
		}
		return out;
	}

	// Defense-in-depth: catches common prompt injection phrases. This is a blocklist
	// and can be bypassed with paraphrasing. The primary defense is the structured-output
	// constraint (response_format: json_schema, used by both engines) which forces LLM
	// output into a fixed {reasoning, answer, citations} shape regardless of prompt content.
	private static final Pattern PROMPT_INJECTION = Pattern.compile(
			"(?i)(ignore\\s+(previous|above|all)\\s+(instructions|prompts|rules)"
			+ "|disregard\\s+(your|the|all)\\s+(instructions|rules|prompt)"
			+ "|override\\s+(your|the|all)\\s+(instructions|rules|prompt)"
			+ "|bypass\\s+(your|the|all)\\s+(instructions|rules|prompt)"
			+ "|you\\s+are\\s+now|new\\s+instructions:|system\\s+prompt:"
			+ "|forget\\s+(your|the|all|previous)\\s+(instructions|rules|prompt))");

	private static final String DISCLAIMER = "This response is AI-generated and may not be "
			+ "accurate. It is not a substitute for clinical judgment. Always verify against "
			+ "the patient's medical records.";

	@Autowired
	@Qualifier("chartSearchAi.chartSearchServiceRouter")
	private ChartSearchService chartSearchService;

	@Autowired
	@Qualifier("chartSearchAi.patientAccessCheck")
	private PatientAccessCheck patientAccessCheck;

	@Autowired
	@Qualifier("chartSearchAi.auditLogService")
	private AuditLogService auditLogService;

	@Autowired
	@Qualifier("chartSearchAi.warmupExecutor")
	private WarmupExecutor warmupExecutor;

	@Autowired
	@Qualifier("chartSearchAi.chatService")
	private ChatService chatService;

	@Autowired
	@Qualifier("chartSearchAi.modelSwitchService")
	private org.openmrs.module.chartsearchai.api.impl.ModelSwitchService modelSwitchService;

	@RequestMapping(value = "/search", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> search(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		String patientUuid = body.get("patient");
		String question = body.get("question");

		PatientResolution resolved = resolvePatient(patientUuid);
		if (resolved.hasError()) {
			return new ResponseEntity<Object>(
					errorResponse(resolved.errorMessage), resolved.errorStatus);
		}
		Patient patient = resolved.patient;

		if (question == null || question.trim().isEmpty()) {
			return new ResponseEntity<Object>(
					errorResponse("question is required"), HttpStatus.BAD_REQUEST);
		}
		if (question.length() > MAX_QUESTION_LENGTH) {
			return new ResponseEntity<Object>(
					errorResponse("question exceeds maximum length of "
							+ MAX_QUESTION_LENGTH + " characters"),
					HttpStatus.BAD_REQUEST);
		}

		question = CONTROL_CHARS.matcher(question).replaceAll("");
		if (question.trim().isEmpty()) {
			return new ResponseEntity<Object>(
					errorResponse("question is required"), HttpStatus.BAD_REQUEST);
		}

		String sanitizationError = validateQuestion(question);
		if (sanitizationError != null) {
			return new ResponseEntity<Object>(
					errorResponse(sanitizationError), HttpStatus.BAD_REQUEST);
		}

		User user = Context.getAuthenticatedUser();

		ResponseEntity<Object> rateLimitError = checkRateLimit(user);
		if (rateLimitError != null) {
			return rateLimitError;
		}

		String preFilter = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER, "false");
		boolean preFilterEnabled = !"false".equalsIgnoreCase(preFilter.trim());

		ChartAnswer chartAnswer;
		long responseTimeMs;
		try {
			long startTime = System.currentTimeMillis();
			chartAnswer = chartSearchService.search(patient, question);
			responseTimeMs = System.currentTimeMillis() - startTime;
		}
		catch (ChartTooLargeException e) {
			log.warn("Chart too large for LLM context for patient [id={}]: {}",
					patient.getPatientId(), e.getMessage());
			return new ResponseEntity<Object>(
					errorResponse("This patient's chart is too large to process. "
							+ "Contact your administrator to increase the LLM context size."),
					HttpStatus.PAYLOAD_TOO_LARGE);
		}
		catch (IllegalStateException e) {
			log.error("Chart search configuration error", e);
			return new ResponseEntity<Object>(
					errorResponse("Chart search is not properly configured. Contact your administrator."),
					HttpStatus.SERVICE_UNAVAILABLE);
		}
		catch (Exception e) {
			log.error("Chart search failed for patient [id={}]", patient.getPatientId(), e);
			return new ResponseEntity<Object>(
					errorResponse("Chart search failed. Please try again or contact your administrator."),
					HttpStatus.INTERNAL_SERVER_ERROR);
		}

		ChartSearchAuditLog auditLog = new ChartSearchAuditLog();
		auditLog.setUser(user);
		auditLog.setPatient(patient);
		auditLog.setQuestion(question);
		auditLog.setAnswer(chartAnswer.getAnswer());
		auditLog.setReferenceCount(chartAnswer.getReferences().size());
		auditLog.setSearchMode(preFilterEnabled ? "pre-filter" : "full-chart");
		auditLog.setResponseTimeMs(responseTimeMs);
		auditLog.setInputTokens(chartAnswer.getInputTokens() > 0 ? chartAnswer.getInputTokens() : null);
		auditLog.setOutputTokens(chartAnswer.getOutputTokens() > 0 ? chartAnswer.getOutputTokens() : null);
		auditLog.setDateCreated(new Date());
		try {
			auditLogService.saveAuditLog(auditLog);
		}
		catch (Exception e) {
			log.warn("Failed to save audit log for search query", e);
		}

		Map<String, Object> response = new HashMap<String, Object>();
		response.put("answer", chartAnswer.getAnswer());
		response.put("disclaimer", DISCLAIMER);

		List<Map<String, Object>> refs = new ArrayList<Map<String, Object>>();
		for (RecordReference ref : chartAnswer.getReferences()) {
			Map<String, Object> refMap = new LinkedHashMap<String, Object>();
			refMap.put("index", ref.getIndex());
			refMap.put("resourceType", ref.getResourceType());
			refMap.put("resourceUuid", ref.getResourceUuid());
			refMap.put("date", formatDate(ref.getDate()));
			// null when grounding is disabled or could not run — clients must
			// render null as "unverified", never as "verified".
			refMap.put("grounded", ref.getGrounded());
			refs.add(refMap);
		}
		response.put("references", refs);
		response.put("safetyWarnings", serializeSafetyWarnings(chartAnswer.getSafetyWarnings()));
		if (auditLog.getAuditLogId() != null) {
			response.put("questionId", String.valueOf(auditLog.getAuditLogId()));
		}

		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	/**
	 * Pre-warm the LLM prompt cache for a patient's chart. Called by the frontend
	 * when a patient chart is opened, so the first AI query on that patient does
	 * not pay full prefill cost. Returns 202 Accepted immediately; the warmup runs
	 * on a background daemon thread.
	 */
	@RequestMapping(value = "/warmup", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> warmup(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		PatientResolution resolved = resolvePatient(body.get("patient"));
		if (resolved.hasError()) {
			return new ResponseEntity<Object>(
					errorResponse(resolved.errorMessage), resolved.errorStatus);
		}

		warmupExecutor.submit(resolved.patient);
		return new ResponseEntity<Object>(HttpStatus.ACCEPTED);
	}

	/**
	 * List the models the active remote endpoint reports via {@code /v1/models},
	 * alongside the currently-selected model name and active engine.
	 *
	 * <p>Returns {@code 503 Service Unavailable} when {@code engine=local}: model
	 * switching is only meaningful against the remote engine.
	 *
	 * <p>Gated by {@code AI Query Patient Data} — same gate as the chat endpoints
	 * so the picker visibility matches the chat panel visibility.
	 */
	@RequestMapping(value = "/models", method = RequestMethod.GET)
	@ResponseBody
	public ResponseEntity<Object> listModels() {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);
		org.openmrs.module.chartsearchai.api.impl.ModelSwitchService.ModelListResponse snapshot;
		try {
			snapshot = modelSwitchService.listAvailable();
		}
		catch (Exception e) {
			log.warn("Failed to list models: {}", e.getMessage());
			return new ResponseEntity<Object>(
					errorResponse("Failed to list models: " + e.getMessage()),
					HttpStatus.INTERNAL_SERVER_ERROR);
		}
		if (!ChartSearchAiConstants.LLM_ENGINE_REMOTE.equalsIgnoreCase(snapshot.getEngine())) {
			Map<String, Object> body = new LinkedHashMap<String, Object>();
			body.put("engine", snapshot.getEngine());
			body.put("error", "Model listing requires the remote engine; active engine is '"
					+ snapshot.getEngine() + "'.");
			return new ResponseEntity<Object>(body, HttpStatus.SERVICE_UNAVAILABLE);
		}
		Map<String, Object> response = new LinkedHashMap<String, Object>();
		response.put("engine", snapshot.getEngine());
		response.put("current", snapshot.getCurrent());
		response.put("available", snapshot.getAvailable());
		response.put("endpointUrl", snapshot.getEndpointUrl());
		// Enriched fields — populated when LM Studio /api/v1/models probe
		// succeeded; null/empty otherwise. The picker reads `provider` for
		// sub-category grouping and `entries[].loaded` for the per-entry
		// state affix. Older SPA builds ignore these fields.
		if (snapshot.getProvider() != null) {
			response.put("provider", snapshot.getProvider());
		}
		if (snapshot.getEntries() != null && !snapshot.getEntries().isEmpty()) {
			List<Map<String, Object>> entries = new ArrayList<Map<String, Object>>();
			for (org.openmrs.module.chartsearchai.api.impl.ModelSwitchService.ModelEntry e
					: snapshot.getEntries()) {
				Map<String, Object> row = new LinkedHashMap<String, Object>();
				row.put("id", e.getId());
				row.put("displayName", e.getDisplayName());
				row.put("type", e.getType());
				row.put("loaded", e.isLoaded());
				if (e.getMaxContextLength() != null) {
					row.put("maxContextLength", e.getMaxContextLength());
				}
				entries.add(row);
			}
			response.put("entries", entries);
		}
		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	/**
	 * Request that the upstream LM Studio server pre-load the named model
	 * into memory. Returns 200 on success (model loaded; subsequent chat
	 * completion will be fast), 400 for blank input, 503 when the active
	 * engine is local or no LM Studio v1 endpoint is reachable.
	 *
	 * <p>Body: {@code {"modelName": "<id>"}}. Same gate as {@code /model}.
	 */
	@RequestMapping(value = "/model/load", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> loadModel(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);
		String requested = body == null ? null : body.get("modelName");
		try {
			modelSwitchService.requestModelLoad(requested);
			Map<String, Object> response = new LinkedHashMap<String, Object>();
			response.put("loaded", requested == null ? "" : requested.trim());
			return new ResponseEntity<Object>(response, HttpStatus.OK);
		}
		catch (IllegalArgumentException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.BAD_REQUEST);
		}
		catch (org.openmrs.api.APIException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.SERVICE_UNAVAILABLE);
		}
	}

	/**
	 * Switch the active remote model. Validates the requested name is in the
	 * live {@code /v1/models} list before writing the GP, so the next chat
	 * request can't route to a model the endpoint doesn't actually serve.
	 *
	 * <p>Body: {@code {"modelName": "<id>"}}. Returns {@code {current}} on
	 * success, {@code 400} for invalid input, {@code 503} for local engine.
	 */
	@RequestMapping(value = "/model", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> setModel(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);
		String requested = body == null ? null : body.get("modelName");
		try {
			String now = modelSwitchService.setCurrent(requested);
			Map<String, Object> response = new LinkedHashMap<String, Object>();
			response.put("current", now);
			return new ResponseEntity<Object>(response, HttpStatus.OK);
		}
		catch (IllegalArgumentException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.BAD_REQUEST);
		}
		catch (org.openmrs.api.APIException e) {
			// listAvailable throws APIException for local-engine or misconfig.
			// Surface as 503 so the SPA picker can hide itself cleanly.
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.SERVICE_UNAVAILABLE);
		}
	}

	/**
	 * List the configured endpoints as picker sections — each endpoint with its
	 * live model list, reachability, and whether it's currently selected. Drives
	 * the picker's per-endpoint sections (LM Studio, Med Agent Hub, ...).
	 */
	@RequestMapping(value = "/endpoints", method = RequestMethod.GET)
	@ResponseBody
	public ResponseEntity<Object> listEndpoints() {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);
		try {
			List<org.openmrs.module.chartsearchai.api.impl.ModelSwitchService.EndpointSection> sections
					= modelSwitchService.listEndpoints();
			List<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
			for (org.openmrs.module.chartsearchai.api.impl.ModelSwitchService.EndpointSection s : sections) {
				Map<String, Object> row = new LinkedHashMap<String, Object>();
				row.put("label", s.getLabel());
				row.put("url", s.getUrl());
				row.put("provider", s.getProvider());
				row.put("reachable", s.isReachable());
				row.put("current", s.isCurrent());
				List<Map<String, Object>> models = new ArrayList<Map<String, Object>>();
				for (org.openmrs.module.chartsearchai.api.impl.ModelSwitchService.ModelEntry e
						: s.getEntries()) {
					Map<String, Object> m = new LinkedHashMap<String, Object>();
					m.put("id", e.getId());
					m.put("displayName", e.getDisplayName());
					m.put("loaded", e.isLoaded());
					models.add(m);
				}
				row.put("models", models);
				out.add(row);
			}
			Map<String, Object> response = new LinkedHashMap<String, Object>();
			response.put("endpoints", out);
			Map<String, Object> current = new LinkedHashMap<String, Object>();
			current.put("endpointUrl", Context.getAdministrationService()
					.getGlobalProperty(ChartSearchAiConstants.GP_LLM_REMOTE_ENDPOINT_URL));
			current.put("modelName", Context.getAdministrationService()
					.getGlobalProperty(ChartSearchAiConstants.GP_LLM_REMOTE_MODEL_NAME));
			response.put("current", current);
			return new ResponseEntity<Object>(response, HttpStatus.OK);
		}
		catch (org.openmrs.api.APIException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.SERVICE_UNAVAILABLE);
		}
	}

	/**
	 * Switch the active endpoint AND model in one step. Body:
	 * {@code {"endpointUrl": "<url>", "modelName": "<id>"}}. Validates the URL is
	 * a registered endpoint and the model is served there before writing both
	 * GPs. {@code 400} on invalid input, {@code 503} for the local engine.
	 */
	@RequestMapping(value = "/endpoint", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> setEndpoint(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);
		String url = body == null ? null : body.get("endpointUrl");
		String model = body == null ? null : body.get("modelName");
		try {
			modelSwitchService.setEndpointAndModel(url, model);
			Map<String, Object> response = new LinkedHashMap<String, Object>();
			response.put("endpointUrl", url == null ? "" : url.trim());
			response.put("current", model == null ? "" : model.trim());
			return new ResponseEntity<Object>(response, HttpStatus.OK);
		}
		catch (IllegalArgumentException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.BAD_REQUEST);
		}
		catch (org.openmrs.api.APIException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()),
					HttpStatus.SERVICE_UNAVAILABLE);
		}
	}

	/**
	 * Streaming search endpoint using Server-Sent Events. Streams tokens as they are
	 * generated by the LLM, then sends references and disclaimer as a final "done" event.
	 *
	 * <p>Writes SSE events directly to the response output stream in the request
	 * thread. This avoids the need for {@code SseEmitter}, background threads,
	 * async servlet support, and proxy privileges — the authenticated user's
	 * session is naturally available throughout the request.</p>
	 *
	 * <p>SSE event types:</p>
	 * <ul>
	 *   <li>{@code thinking} — a chunk of the model's reasoning (chain-of-thought), emitted
	 *       before the answer; render distinctly (e.g. a collapsible panel), not as the answer</li>
	 *   <li>{@code token} — a chunk of the answer text</li>
	 *   <li>{@code references} — the answer's citations the moment the answer is complete,
	 *       before grounding verdicts exist; render as unverified until verdicts arrive</li>
	 *   <li>{@code done} — final JSON with answer, references, questionId, and disclaimer.
	 *       With async grounding off (the default) the references carry their grounding
	 *       verdicts; with {@code chartsearchai.grounding.async=true} they do not yet</li>
	 *   <li>{@code grounded} — only with async grounding: the references re-sent with their
	 *       grounding verdicts attached, after the Tier-2 verification tail completes; carries
	 *       the same {@code questionId} as {@code done}. Clients must keep consuming the stream
	 *       after {@code done} to receive it</li>
	 *   <li>{@code error} — an error message if something goes wrong</li>
	 * </ul>
	 */
	@RequestMapping(value = "/search/stream", method = RequestMethod.POST)
	public void searchStream(@RequestBody Map<String, String> body,
			HttpServletResponse response) throws IOException {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		String patientUuid = body.get("patient");
		String question = body.get("question");

		if (patientUuid == null || patientUuid.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "patient is required");
			return;
		}
		if (question == null || question.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "question is required");
			return;
		}
		if (question.length() > MAX_QUESTION_LENGTH) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST,
					"question exceeds maximum length of " + MAX_QUESTION_LENGTH + " characters");
			return;
		}

		String sanitizedQuestion = CONTROL_CHARS.matcher(question).replaceAll("");
		if (sanitizedQuestion.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "question is required");
			return;
		}

		String sanitizationError = validateQuestion(sanitizedQuestion);
		if (sanitizationError != null) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, sanitizationError);
			return;
		}

		Patient patient = Context.getPatientService().getPatientByUuid(patientUuid);
		if (patient == null) {
			writeJsonError(response, HttpServletResponse.SC_NOT_FOUND, "Patient not found");
			return;
		}

		User user = Context.getAuthenticatedUser();

		if (!patientAccessCheck.canAccess(user, patient)) {
			writeJsonError(response, HttpServletResponse.SC_FORBIDDEN,
					"You do not have access to this patient's chart");
			return;
		}

		ResponseEntity<Object> rateLimitError = checkRateLimit(user);
		if (rateLimitError != null) {
			writeJsonError(response, 429, "Rate limit exceeded");
			return;
		}

		// All validation passed — start SSE streaming.
		// Unwrap any response wrappers (e.g. Spring's ContentCachingResponseWrapper
		// from ShallowEtagHeaderFilter) that buffer the entire body, which would
		// prevent SSE tokens from streaming to the client in real time.
		HttpServletResponse unwrapped = response;
		while (unwrapped instanceof HttpServletResponseWrapper) {
			javax.servlet.ServletResponse inner =
					((HttpServletResponseWrapper) unwrapped).getResponse();
			if (inner instanceof HttpServletResponse) {
				unwrapped = (HttpServletResponse) inner;
			} else {
				break;
			}
		}

		response.setContentType("text/event-stream");
		response.setCharacterEncoding("UTF-8");
		response.setHeader("Cache-Control", "no-cache");
		response.setHeader("X-Accel-Buffering", "no");
		response.setHeader("Connection", "keep-alive");
		// Disable Tomcat's response buffer so tokens stream immediately
		// instead of accumulating in the default 8KB buffer.
		unwrapped.setBufferSize(0);

		final OutputStream out = unwrapped.getOutputStream();
		// Commit the response headers now so chunked transfer starts
		unwrapped.flushBuffer();

		String preFilterProp = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER, "false");
		String searchMode = !"false".equalsIgnoreCase(preFilterProp.trim())
				? "pre-filter" : "full-chart";

		streamAnswer(out, patient, sanitizedQuestion, user, searchMode, isAsyncGroundingActive());
	}

	/**
	 * Whether the streaming endpoint should emit {@code done} before the grounding pass and
	 * deliver verdicts in a trailing {@code grounded} event. Only meaningful when grounding
	 * itself is on — with grounding disabled there is no tail to move off the response path.
	 * Resolved here (not in {@link #streamAnswer}) so the orchestration stays free of
	 * {@code Context} reads and unit-testable.
	 */
	private static boolean isAsyncGroundingActive() {
		return ChartSearchAiUtils.isGroundingAsyncEnabled() && ChartSearchAiUtils.isGroundingEnabled();
	}

	/**
	 * The SSE orchestration for one streaming search: runs the service call with the token /
	 * thinking / references channels wired to the output stream, then emits the terminal events.
	 *
	 * <p>With {@code asyncGrounding} off, the classic shape: one {@code done} event after the
	 * service returns, carrying the grounded references. With it on, {@code done} is emitted the
	 * moment the answer is complete (references without verdicts, audit row already saved so
	 * {@code questionId} is present) and a trailing {@code grounded} event delivers the
	 * verdict-annotated references once verification finishes — the user's perceived completion
	 * no longer waits out the grounding tail. If the service returns without ever surfacing an
	 * ungrounded answer (a cache hit returns an already-final answer), the classic {@code done}
	 * is emitted instead and no {@code grounded} event follows.</p>
	 *
	 * <p>Package-private and free of {@code Context} reads so event-order behavior is unit-tested
	 * directly (see {@code ChartSearchAiStreamEventOrderTest}); {@code searchStream} resolves all
	 * configuration before delegating here.</p>
	 */
	void streamAnswer(final OutputStream out, Patient patient, String sanitizedQuestion, User user,
			String searchMode, boolean asyncGrounding) {
		try {
			long startTime = System.currentTimeMillis();

			// Carries the early-done state from the consumer (fired mid-call) to the post-return
			// code: [0] = the saved questionId (null if audit failed), and whether done was sent
			// is tracked by earlyDoneSent. Single-element arrays because the consumer lambda needs
			// effectively-final capture.
			final String[] earlyQuestionId = new String[1];
			final boolean[] earlyDoneSent = new boolean[1];

			// Async grounding: the moment the (not yet grounding-verified) answer exists, persist
			// the audit row and emit "done" — the user's perceived completion no longer waits out
			// the grounding tail. The audit's responseTimeMs deliberately measures to THIS point
			// (what the user experienced); the [timing] service log still carries groundMs.
			// Serialization + write failures unwind like any mid-stream disconnect, via the same
			// RuntimeException(IOException) shape writeSseEventOrThrow uses.
			Consumer<ChartAnswer> ungroundedConsumer = !asyncGrounding ? ungrounded -> { }
					: ungrounded -> {
						if (earlyDoneSent[0]) {
							// Interface contract is at-most-once; stay idempotent anyway — a
							// duplicate done would corrupt every client's completion handling.
							log.warn("Ungrounded-answer consumer fired more than once; ignoring");
							return;
						}
						earlyQuestionId[0] = saveAuditLog(user, patient, sanitizedQuestion,
								ungrounded, searchMode, System.currentTimeMillis() - startTime);
						try {
							writeSseEvent(out, "done",
									doneEventJson(ungrounded, earlyQuestionId[0]));
						}
						catch (IOException e) {
							log.debug("Client disconnected during streaming (done)");
							throw new RuntimeException("Client disconnected", e);
						}
						earlyDoneSent[0] = true;
					};

			// Four channels: "token" carries the answer; "thinking" carries the model's reasoning
			// (chain-of-thought), emitted first so the UI can show live progress and the rationale
			// instead of a dead spinner; "references" carries the answer's citations the moment the
			// answer is done — BEFORE the grounding pass — so the UI can render clickable citations
			// immediately and not wait on Tier-2 verification. The terminal events re-send the
			// references with grounding verdicts attached: in the classic shape on "done", or — when
			// async grounding is active — on a trailing "grounded" event after an early "done". The
			// frontend must render "thinking" distinctly (e.g. a collapsible panel), never as the
			// answer; citations must show as unverified until verdicts arrive. All unwind on client
			// disconnect via writeSseEventOrThrow.
			ChartAnswer chartAnswer = chartSearchService.searchStreaming(
					patient, sanitizedQuestion,
					token -> writeSseEventOrThrow(out, "token", token),
					reasoning -> writeSseEventOrThrow(out, "thinking", reasoning),
					citations -> sendReferencesEvent(out, citations),
					ungroundedConsumer);

			if (!earlyDoneSent[0]) {
				// Classic shape: async off, or the service returned an already-final answer (cache
				// hit) without surfacing an ungrounded stage — audit and emit the single done.
				String questionId = saveAuditLog(user, patient, sanitizedQuestion, chartAnswer,
						searchMode, System.currentTimeMillis() - startTime);
				writeSseEvent(out, "done", doneEventJson(chartAnswer, questionId));
			} else {
				// done already went out before grounding; deliver the verdicts in the trailing
				// "grounded" event. Same reference serialization as done, so the client can
				// replace its reference list wholesale; questionId correlates the two events.
				Map<String, Object> groundedData = new HashMap<String, Object>();
				groundedData.put("references", serializeReferences(chartAnswer.getReferences()));
				groundedData.put("safetyWarnings", serializeSafetyWarnings(chartAnswer.getSafetyWarnings()));
				if (earlyQuestionId[0] != null) {
					groundedData.put("questionId", earlyQuestionId[0]);
				}
				writeSseEvent(out, "grounded", new ObjectMapper().writeValueAsString(groundedData));
			}
		}
		catch (ChartTooLargeException e) {
			log.warn("Chart too large for LLM context during streaming for patient [id={}]: {}",
					patient.getPatientId(), e.getMessage());
			try {
				writeSseEvent(out, "error",
						"This patient's chart is too large to process. "
								+ "Contact your administrator to increase the LLM context size.");
			}
			catch (IOException ioe) {
				log.debug("Could not send too-large error event, client likely disconnected");
			}
		}
		catch (IllegalStateException e) {
			log.error("Chart search configuration error during streaming", e);
			try {
				writeSseEvent(out, "error",
						"Chart search is not properly configured. Contact your administrator.");
			}
			catch (IOException ioe) {
				log.debug("Could not send config error event, client likely disconnected");
			}
		}
		catch (Exception e) {
			if (e.getCause() instanceof IOException) {
				log.debug("Streaming ended due to client disconnect");
			} else {
				log.error("Chart search streaming failed for patient [id={}]",
						patient.getPatientId(), e);
				try {
					writeSseEvent(out, "error",
							"Chart search failed. Please try again or contact your administrator.");
				}
				catch (IOException ioe) {
					log.debug("Could not send error event, client likely disconnected");
				}
			}
		}

		try {
			out.flush();
		}
		catch (IOException e) {
			log.debug("Could not flush SSE stream, client likely disconnected");
		}
	}

	/**
	 * Persists the audit row for a streaming answer and returns its id as the client-facing
	 * {@code questionId}, or {@code null} when the save failed — audit failures are logged and
	 * never break the response, exactly as before the async-grounding split. Shared by the
	 * classic post-return path and the async early-{@code done} path so both emit identical
	 * audit rows and {@code done} payloads.
	 */
	private String saveAuditLog(User user, Patient patient, String question, ChartAnswer answer,
			String searchMode, long responseTimeMs) {
		ChartSearchAuditLog auditLog = new ChartSearchAuditLog();
		auditLog.setUser(user);
		auditLog.setPatient(patient);
		auditLog.setQuestion(question);
		auditLog.setAnswer(answer.getAnswer());
		auditLog.setReferenceCount(answer.getReferences().size());
		auditLog.setSearchMode(searchMode);
		auditLog.setResponseTimeMs(responseTimeMs);
		auditLog.setInputTokens(answer.getInputTokens() > 0 ? answer.getInputTokens() : null);
		auditLog.setOutputTokens(answer.getOutputTokens() > 0 ? answer.getOutputTokens() : null);
		auditLog.setDateCreated(new Date());
		try {
			auditLogService.saveAuditLog(auditLog);
		}
		catch (Exception e) {
			log.warn("Failed to save audit log for streaming query", e);
		}
		return auditLog.getAuditLogId() != null ? String.valueOf(auditLog.getAuditLogId()) : null;
	}

	/** Serializes the {@code done} event payload: answer, disclaimer, references, questionId. */
	private String doneEventJson(ChartAnswer answer, String questionId) throws IOException {
		Map<String, Object> doneData = new HashMap<String, Object>();
		doneData.put("answer", answer.getAnswer());
		doneData.put("disclaimer", DISCLAIMER);
		doneData.put("references", serializeReferences(answer.getReferences()));
		doneData.put("safetyWarnings", serializeSafetyWarnings(answer.getSafetyWarnings()));
		if (questionId != null) {
			doneData.put("questionId", questionId);
		}
		return new ObjectMapper().writeValueAsString(doneData);
	}

	/** Test seam: production wires {@link ChartSearchService} via {@code Autowired}. */
	void setChartSearchService(ChartSearchService chartSearchService) {
		this.chartSearchService = chartSearchService;
	}

	/** Test seam: production wires {@link AuditLogService} via {@code Autowired}. */
	void setAuditLogService(AuditLogService auditLogService) {
		this.auditLogService = auditLogService;
	}

	@RequestMapping(value = "/auditlog", method = RequestMethod.GET)
	@ResponseBody
	public ResponseEntity<Object> getAuditLogs(
			@RequestParam(value = "patient", required = false) String patientUuid,
			@RequestParam(value = "user", required = false) String userUuid,
			@RequestParam(value = "fromDate", required = false) Long fromDateMs,
			@RequestParam(value = "toDate", required = false) Long toDateMs,
			@RequestParam(value = "startIndex", required = false) Integer startIndex,
			@RequestParam(value = "limit", required = false) Integer limit) {

		Context.requirePrivilege(ChartSearchAiConstants.PRIV_VIEW_AUDIT_LOGS);

		Patient patient = null;
		if (patientUuid != null && !patientUuid.trim().isEmpty()) {
			patient = Context.getPatientService().getPatientByUuid(patientUuid);
			if (patient == null) {
				return new ResponseEntity<Object>(
						errorResponse("Patient not found"), HttpStatus.NOT_FOUND);
			}
		}

		User user = null;
		if (userUuid != null && !userUuid.trim().isEmpty()) {
			user = Context.getUserService().getUserByUuid(userUuid);
			if (user == null) {
				return new ResponseEntity<Object>(
						errorResponse("User not found"), HttpStatus.NOT_FOUND);
			}
		}

		Date fromDate = fromDateMs != null ? new Date(fromDateMs) : null;
		Date toDate = toDateMs != null ? new Date(toDateMs) : null;

		List<ChartSearchAuditLog> logs = auditLogService.getAuditLogs(patient, user, fromDate, toDate,
				startIndex, limit);
		Long totalCount = auditLogService.getAuditLogCount(patient, user, fromDate, toDate);

		List<Map<String, Object>> results = new ArrayList<Map<String, Object>>();
		for (ChartSearchAuditLog auditLog : logs) {
			Map<String, Object> entry = new HashMap<String, Object>();
			entry.put("auditLogId", auditLog.getAuditLogId());
			entry.put("user", auditLog.getUser() != null ? auditLog.getUser().getUuid() : null);
			entry.put("username", auditLog.getUser() != null ? auditLog.getUser().getUsername() : null);
			entry.put("patient", auditLog.getPatient() != null ? auditLog.getPatient().getUuid() : null);
			entry.put("question", auditLog.getQuestion());
			entry.put("answer", auditLog.getAnswer());
			entry.put("referenceCount", auditLog.getReferenceCount());
			entry.put("searchMode", auditLog.getSearchMode());
			entry.put("responseTimeMs", auditLog.getResponseTimeMs());
			entry.put("inputTokens", auditLog.getInputTokens());
			entry.put("outputTokens", auditLog.getOutputTokens());
			entry.put("dateCreated", auditLog.getDateCreated() != null
					? auditLog.getDateCreated().getTime() : null);
			entry.put("rating", auditLog.getRating());
			entry.put("feedbackComment", auditLog.getFeedbackComment());
			results.add(entry);
		}

		Map<String, Object> response = new HashMap<String, Object>();
		response.put("results", results);
		response.put("totalCount", totalCount);

		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	@RequestMapping(value = "/feedback", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> submitFeedback(@RequestBody Map<String, Object> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		String validationError = validateFeedbackInput(body);
		if (validationError != null) {
			return new ResponseEntity<Object>(
					errorResponse(validationError), HttpStatus.BAD_REQUEST);
		}

		Integer auditLogId = Integer.valueOf(body.get("questionId").toString());
		String rating = body.get("rating").toString();
		String comment = sanitizeFeedbackComment(
				body.get("comment") != null ? body.get("comment").toString() : null);

		ChartSearchAuditLog auditLog = auditLogService.getAuditLog(auditLogId);
		User user = Context.getAuthenticatedUser();
		if (auditLog == null || auditLog.getUser() == null || !auditLog.getUser().equals(user)) {
			return new ResponseEntity<Object>(
					errorResponse("Question not found"), HttpStatus.NOT_FOUND);
		}

		auditLog.setRating(rating);
		auditLog.setFeedbackComment(comment);
		auditLogService.saveAuditLog(auditLog);

		Map<String, Object> response = new HashMap<String, Object>();
		response.put("success", true);
		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	// ============================================================================
	// Multi-turn chat endpoints. The {@code /search} family above is single-shot;
	// these maintain a per-(patient, user) ChatSession that carries prior turns
	// into each LLM call. Persistence is handled by {@link ChatService}.
	// ============================================================================

	/**
	 * Streaming chat turn. Reuses the (patient, user) session if {@code session}
	 * is provided AND resolves to an active session; otherwise opens-or-loads the
	 * latest active session for the user. Surfaces the session uuid via the
	 * {@code X-ChartSearchAi-Session} response header before the SSE stream opens
	 * so the client can pin subsequent posts to the same conversation.
	 *
	 * <pre>
	 * POST /ws/rest/v1/chartsearchai/chat/stream
	 * { "patient": "uuid", "session": "uuid?" , "question": "..." }
	 * </pre>
	 */
	@RequestMapping(value = "/chat/stream", method = RequestMethod.POST)
	public void chatStream(@RequestBody Map<String, String> body,
			HttpServletResponse response) throws IOException {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		String patientUuid = body.get("patient");
		String sessionUuid = body.get("session");
		String question = body.get("question");

		if (patientUuid == null || patientUuid.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "patient is required");
			return;
		}
		if (question == null || question.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "question is required");
			return;
		}
		if (question.length() > MAX_QUESTION_LENGTH) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST,
					"question exceeds maximum length of " + MAX_QUESTION_LENGTH + " characters");
			return;
		}

		String sanitizedQuestion = CONTROL_CHARS.matcher(question).replaceAll("");
		if (sanitizedQuestion.trim().isEmpty()) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, "question is required");
			return;
		}
		String sanitizationError = validateQuestion(sanitizedQuestion);
		if (sanitizationError != null) {
			writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, sanitizationError);
			return;
		}

		Patient patient = Context.getPatientService().getPatientByUuid(patientUuid);
		if (patient == null) {
			writeJsonError(response, HttpServletResponse.SC_NOT_FOUND, "Patient not found");
			return;
		}

		User user = Context.getAuthenticatedUser();
		if (!patientAccessCheck.canAccess(user, patient)) {
			writeJsonError(response, HttpServletResponse.SC_FORBIDDEN,
					"You do not have access to this patient's chart");
			return;
		}

		ResponseEntity<Object> rateLimitError = checkRateLimit(user);
		if (rateLimitError != null) {
			writeJsonError(response, 429, "Rate limit exceeded");
			return;
		}

		// Per-request backend override (see /chat): resolved engine-aware, validated
		// before the stream opens; cleared in finally after streaming so it can't
		// leak to a pooled thread.
			boolean overridden = false;
			OverrideResolution overrideRes;
			String answeredModel;
			try {
				overrideRes = resolveOverride(body);
				answeredModel = overrideRes.answeredModel;
				overridden = overrideRes.overridden;
			}
			catch (IllegalArgumentException e) {
				writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
				return;
			}
			boolean staged = wantsStaged(body) && canStage(overrideRes);
			if (staged) {
				if (overridden) {
					RequestLlmOverride.clear();
					overridden = false;
				}
				try {
					validateStagedModels(overrideRes);
				}
				catch (IllegalArgumentException e) {
					writeJsonError(response, HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
					return;
				}
			}

		// Everything from session resolution through the final flush runs under an
		// outer finally that clears any per-request override — opening the stream
		// (getOutputStream/flushBuffer) can throw before the inner streaming try,
		// and that window must not leak the thread-local onto a pooled thread.
		try {
		ChatSession session = resolveOrOpenSession(patient, sessionUuid);

		// Unwrap any response wrappers that buffer the body (kills SSE liveness).
		HttpServletResponse unwrapped = response;
		while (unwrapped instanceof HttpServletResponseWrapper) {
			javax.servlet.ServletResponse inner =
					((HttpServletResponseWrapper) unwrapped).getResponse();
			if (inner instanceof HttpServletResponse) {
				unwrapped = (HttpServletResponse) inner;
			} else {
				break;
			}
		}

		response.setContentType("text/event-stream");
		response.setCharacterEncoding("UTF-8");
		response.setHeader("Cache-Control", "no-cache");
		response.setHeader("X-Accel-Buffering", "no");
		response.setHeader("Connection", "keep-alive");
		// Surface the session uuid BEFORE the stream opens — the client uses
		// it to pin all subsequent posts in this conversation.
		response.setHeader("X-ChartSearchAi-Session", session.getUuid());
		unwrapped.setBufferSize(0);

		final OutputStream out = unwrapped.getOutputStream();
			unwrapped.flushBuffer();

			try {
				if (staged) {
					streamHubStagedChat(out, session, patientUuid, sanitizedQuestion, overrideRes);
					return;
				}
				// Every chat turn relays through the hub's single-completion contract — chat requires
				// the remote engine (resolveOverride rejects anything else), so there is no local
				// chart-build + inference + citation orchestration left to fall back to.
				streamHubNonStagedChat(out, session, patientUuid, sanitizedQuestion, overrideRes);
		}
		catch (ChartTooLargeException e) {
			log.warn("Chart too large for chat streaming for patient [id={}]: {}",
					patient.getPatientId(), e.getMessage());
			try {
				writeSseEvent(out, "error",
						"This patient's chart is too large to process. "
								+ "Contact your administrator to increase the LLM context size.");
			}
			catch (IOException ioe) {
				log.debug("Could not send too-large error event, client likely disconnected");
			}
		}
		catch (IllegalStateException e) {
			log.error("Chat configuration error during streaming", e);
			try {
				writeSseEvent(out, "error",
						"Chart search is not properly configured. Contact your administrator.");
			}
			catch (IOException ioe) {
				log.debug("Could not send config error event, client likely disconnected");
			}
		}
		catch (Exception e) {
			if (e.getCause() instanceof IOException) {
				log.debug("Chat streaming ended due to client disconnect");
			} else {
				log.error("Chat streaming failed for patient [id={}]", patient.getPatientId(), e);
				try {
					writeSseEvent(out, "error",
							"Chart search failed. Please try again or contact your administrator.");
				}
				catch (IOException ioe) {
					log.debug("Could not send error event, client likely disconnected");
				}
			}
		}

		try {
			out.flush();
		}
		catch (IOException e) {
			log.debug("Could not flush chat SSE stream, client likely disconnected");
		}
		}
		catch (Exception e) {
			// Failure BEFORE the SSE stream opened (session resolution / chart build /
			// opening the stream). Return a clean JSON 500 instead of letting it propagate
			// to the servlet container's HTML error page, which the SPA cannot parse.
			// Post-commit failures are already handled as SSE error events above.
			if (!response.isCommitted()) {
				log.error("Chat stream setup failed for patient [id={}]", patient.getPatientId(), e);
				writeJsonError(response, HttpServletResponse.SC_INTERNAL_SERVER_ERROR,
						"Chart search failed. Please try again or contact your administrator.");
			}
			else {
				log.error("Chat stream failed after response commit for patient [id={}]", patient.getPatientId(), e);
			}
		}
		finally {
			if (overridden) {
				RequestLlmOverride.clear();
			}
		}
	}

	/**
	 * Synchronous chat (non-streaming) — convenience for callers that don't need
	 * SSE. Same persistence semantics as {@link #chatStream}; the session uuid is
	 * returned in the JSON body (no SSE header).
	 */
	@RequestMapping(value = "/chat", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> chat(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		String patientUuid = body.get("patient");
		String sessionUuid = body.get("session");
		String question = body.get("question");

		PatientResolution resolved = resolvePatient(patientUuid);
		if (resolved.hasError()) {
			return new ResponseEntity<Object>(
					errorResponse(resolved.errorMessage), resolved.errorStatus);
		}
		Patient patient = resolved.patient;

		if (question == null || question.trim().isEmpty()) {
			return new ResponseEntity<Object>(
					errorResponse("question is required"), HttpStatus.BAD_REQUEST);
		}
		if (question.length() > MAX_QUESTION_LENGTH) {
			return new ResponseEntity<Object>(
					errorResponse("question exceeds maximum length of "
							+ MAX_QUESTION_LENGTH + " characters"),
					HttpStatus.BAD_REQUEST);
		}
		question = CONTROL_CHARS.matcher(question).replaceAll("");
		if (question.trim().isEmpty()) {
			return new ResponseEntity<Object>(
					errorResponse("question is required"), HttpStatus.BAD_REQUEST);
		}
		String sanitizationError = validateQuestion(question);
		if (sanitizationError != null) {
			return new ResponseEntity<Object>(
					errorResponse(sanitizationError), HttpStatus.BAD_REQUEST);
		}

		User user = Context.getAuthenticatedUser();
		ResponseEntity<Object> rateLimitError = checkRateLimit(user);
		if (rateLimitError != null) {
			return rateLimitError;
		}

		ChatSession session;
		try {
			session = resolveOrOpenSession(patient, sessionUuid);
		}
		catch (Exception e) {
			log.error("Chat session/chart resolution failed for patient [id={}]", patient.getPatientId(), e);
			return new ResponseEntity<Object>(
					errorResponse("Chart search failed. Please try again or contact your administrator."),
					HttpStatus.INTERNAL_SERVER_ERROR);
		}

		// Per-request backend override: if the caller names a backend, use it for
		// THIS request only (remote engine; validated against the registry; the
		// config-controlled global default is untouched). Resolved engine-aware so
		// the answered model is accurate under the local (bundled) engine too.
		// answeredModel is captured HERE because the override is cleared in finally
		// before the response is built.
		boolean overridden = false;
		String answeredModel;
		OverrideResolution overrideRes;
		try {
			overrideRes = resolveOverride(body);
			answeredModel = overrideRes.answeredModel;
			overridden = overrideRes.overridden;
		}
		catch (IllegalArgumentException e) {
			return new ResponseEntity<Object>(errorResponse(e.getMessage()), HttpStatus.BAD_REQUEST);
		}

		ChatTurnResult result;
		try {
			// Every chat turn relays through the hub's single-completion contract -- this is how the
			// harness's synchronous research client drains the SAME engine the product streams
			// against, on any hub level id (including the low-level answer:/answer-review:/
			// indepth-only: legs used for arm comparisons). resolveOverride already rejected a
			// non-remote engine, so there is no local orchestration to fall back to.
			Map<String, Object> wire = hubRelayCompletionWire(session, patientUuid, question, overrideRes);
			result = chatService.persistHubStagedAnswer(session, question, wire);
		}
		catch (ChartTooLargeException e) {
			log.warn("Chart too large for chat for patient [id={}]: {}",
					patient.getPatientId(), e.getMessage());
			return new ResponseEntity<Object>(
					errorResponse("This patient's chart is too large to process. "
							+ "Contact your administrator to increase the LLM context size."),
					HttpStatus.PAYLOAD_TOO_LARGE);
		}
		catch (Exception e) {
			log.error("Chat failed for patient [id={}]", patient.getPatientId(), e);
			return new ResponseEntity<Object>(
					errorResponse("Chart search failed. Please try again or contact your administrator."),
					HttpStatus.INTERNAL_SERVER_ERROR);
		}
		finally {
			if (overridden) {
				RequestLlmOverride.clear();
			}
		}

		ChartAnswer answer = result.getAnswer();
		Map<String, Object> response = new LinkedHashMap<String, Object>();
		response.put("answer", answer.getAnswer());
		response.put("disclaimer", DISCLAIMER);

		List<Map<String, Object>> refs = new ArrayList<Map<String, Object>>();
		for (RecordReference ref : answer.getReferences()) {
			Map<String, Object> refMap = new LinkedHashMap<String, Object>();
			refMap.put("index", ref.getIndex());
			refMap.put("resourceType", ref.getResourceType());
			refMap.put("resourceUuid", ref.getResourceUuid());
			refMap.put("date", formatDate(ref.getDate()));
			refs.add(refMap);
		}
		response.put("references", refs);
		response.put("blocks", blocksToJson(answer.getBlocks()));
		if (answer.getConfidence() != null) {
			response.put("confidence", answer.getConfidence());
		}
		if (answer.getAnswerValidation() != null) {
			response.put("answerValidation", answer.getAnswerValidation());
		}
		response.put("session", result.getSessionUuid());
		response.put("messageId", result.getAssistantMessageUuid());
		response.put("model", answeredModel);

		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	/**
	 * Close the current active session for the (patient, user) pair and open
	 * a fresh one. Returns the new session uuid + empty messages.
	 */
	@RequestMapping(value = "/chat/new", method = RequestMethod.POST)
	@ResponseBody
	public ResponseEntity<Object> chatNew(@RequestBody Map<String, String> body) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		PatientResolution resolved = resolvePatient(body.get("patient"));
		if (resolved.hasError()) {
			return new ResponseEntity<Object>(
					errorResponse(resolved.errorMessage), resolved.errorStatus);
		}

		ChatSession session = chatService.closeAndStartNew(resolved.patient);
		Map<String, Object> response = new LinkedHashMap<String, Object>();
		response.put("session", session.getUuid());
		response.put("messages", new ArrayList<Map<String, Object>>());
		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	/**
	 * Hydrate the SPA on mount: returns the current (patient, user) session
	 * (creating one if none exists) and its prior messages in chronological
	 * order. Empty {@code messages[]} on a freshly-created session.
	 */
	@RequestMapping(value = "/chat", method = RequestMethod.GET)
	@ResponseBody
	public ResponseEntity<Object> chatHistory(
			@RequestParam(value = "patient") String patientUuid) {
		Context.requirePrivilege(ChartSearchAiConstants.PRIV_QUERY_PATIENT_DATA);

		PatientResolution resolved = resolvePatient(patientUuid);
		if (resolved.hasError()) {
			return new ResponseEntity<Object>(
					errorResponse(resolved.errorMessage), resolved.errorStatus);
		}

		ChatSession session = chatService.openOrLoadActiveSession(resolved.patient);
		List<ChatMessage> messages = chatService.getMessages(session);

		List<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
		ObjectMapper hydrateMapper = new ObjectMapper();
		for (ChatMessage m : messages) {
			Map<String, Object> entry = new LinkedHashMap<String, Object>();
			entry.put("messageId", m.getUuid());
			entry.put("role", m.getRole());

			// Assistant rows persist a JSON envelope ({answer, blocks}); user
			// rows are plaintext. Parse JSON when present, surface prose +
			// blocks separately so the SPA can rehydrate the same view it
			// had during streaming. Legacy plaintext rows fall through
			// with blocks=[].
			if (ChatMessage.ROLE_ASSISTANT.equals(m.getRole())) {
					String stored = m.getContent();
					String prose = stored;
					List<Object> blocks = new ArrayList<Object>();
					List<Object> references = new ArrayList<Object>();
					Map<String, Object> confidence = null;
					Map<String, Object> answerValidation = null;
					Map<String, Object> inDepth = null;
					if (stored != null && stored.trim().startsWith("{")) {
						try {
							com.fasterxml.jackson.databind.JsonNode root =
									hydrateMapper.readTree(stored);
						com.fasterxml.jackson.databind.JsonNode answerNode = root.get("answer");
						if (answerNode != null && answerNode.isTextual()) {
							prose = answerNode.asText();
						}
						com.fasterxml.jackson.databind.JsonNode blocksNode = root.get("blocks");
						if (blocksNode != null && blocksNode.isArray()) {
							blocks = hydrateMapper.convertValue(blocksNode, List.class);
						}
						com.fasterxml.jackson.databind.JsonNode refsNode = root.get("references");
						if (refsNode != null && refsNode.isArray()) {
							references = hydrateMapper.convertValue(refsNode, List.class);
						}
						com.fasterxml.jackson.databind.JsonNode confNode = root.get("confidence");
							if (confNode != null && confNode.isObject()) {
								confidence = hydrateMapper.convertValue(confNode, Map.class);
							}
							com.fasterxml.jackson.databind.JsonNode answerValidationNode = root.get("answerValidation");
							if (answerValidationNode != null && answerValidationNode.isObject()) {
								answerValidation = hydrateMapper.convertValue(answerValidationNode, Map.class);
							}
							com.fasterxml.jackson.databind.JsonNode inDepthNode = root.get("inDepth");
							if (inDepthNode != null && inDepthNode.isObject()) {
								inDepth = hydrateMapper.convertValue(inDepthNode, Map.class);
							}
						}
						catch (IOException ignored) {
							// Treat as plaintext.
					}
				}
				entry.put("content", prose);
				entry.put("blocks", blocks);
				entry.put("references", references);
					if (confidence != null) {
						entry.put("confidence", confidence);
					}
					if (answerValidation != null) {
						entry.put("answerValidation", answerValidation);
					}
					if (inDepth != null) {
						entry.put("inDepth", inDepth);
					}
				} else {
				entry.put("content", m.getContent());
			}

			entry.put("createdAt", m.getCreatedAt() != null ? m.getCreatedAt().getTime() : null);
			out.add(entry);
		}

		Map<String, Object> response = new LinkedHashMap<String, Object>();
		response.put("session", session.getUuid());
		response.put("chartBuiltAt",
				session.getChartBuiltAt() != null ? session.getChartBuiltAt().getTime() : null);
		response.put("messages", out);
		return new ResponseEntity<Object>(response, HttpStatus.OK);
	}

	/**
	 * Look up an existing session by uuid (loadByUuid); fall back to
	 * openOrLoadActive when the uuid is missing or stale (e.g. expired).
	 * Always returns a non-null session.
	 */
	private ChatSession resolveOrOpenSession(Patient patient, String sessionUuid) {
		if (sessionUuid != null && !sessionUuid.trim().isEmpty()) {
			ChatSession existing = chatService.loadByUuid(sessionUuid.trim());
			if (existing != null && patient.equals(existing.getPatient())
					&& ChatSession.STATUS_ACTIVE.equals(existing.getStatus())) {
				return existing;
			}
		}
		return chatService.openOrLoadActiveSession(patient);
	}

	@ExceptionHandler(HttpMessageNotReadableException.class)
	@ResponseBody
	public ResponseEntity<Object> handleBadRequest(HttpMessageNotReadableException ex) {
		return new ResponseEntity<Object>(
				errorResponse("Invalid request body. Expected JSON with 'patient' and 'question' fields."),
				HttpStatus.BAD_REQUEST);
	}

	/**
	 * Maps an authorization failure to a proper status with a clean body. Without this, the framework
	 * serializes the thrown exception as an HTTP 200 carrying a full stack trace — both a misleading
	 * status and an information leak. 401 when the caller is unauthenticated; 403 when authenticated
	 * but lacking the privilege.
	 *
	 * <p>Catches both auth-failure types because they are siblings under {@code APIException}, not one
	 * hierarchy: every endpoint's up-front {@link Context#requirePrivilege} gate throws
	 * {@link ContextAuthenticationException} (the active path here, since this controller authorizes
	 * programmatically rather than with {@code @Authorized}), while the {@code @Authorized} AOP throws
	 * {@link APIAuthenticationException}. The second arm is defense-in-depth so an authorization failure
	 * raised by any downstream {@code @Authorized} service call still surfaces as 401/403 rather than
	 * falling to the catch-all as a 500.
	 */
	@ExceptionHandler({ ContextAuthenticationException.class, APIAuthenticationException.class })
	@ResponseBody
	public ResponseEntity<Object> handleAuthFailure(APIException ex) {
		HttpStatus status = Context.isAuthenticated() ? HttpStatus.FORBIDDEN : HttpStatus.UNAUTHORIZED;
		return new ResponseEntity<Object>(
				errorResponse(status == HttpStatus.FORBIDDEN ? "Insufficient privileges" : "Authentication required"),
				status);
	}

	/**
	 * Last-resort handler so an otherwise-unhandled exception surfaces as a clean 500 instead of the
	 * framework's default HTTP 200 + stack trace. The more specific handlers above (auth, malformed
	 * body) take precedence. The streaming endpoint handles its own post-commit errors internally, so
	 * this only sees pre-stream failures there.
	 */
	@ExceptionHandler(Exception.class)
	@ResponseBody
	public ResponseEntity<Object> handleUnexpected(Exception ex) {
		log.error("Unhandled exception in chartsearchai REST controller", ex);
		return new ResponseEntity<Object>(errorResponse("Internal error"),
				HttpStatus.INTERNAL_SERVER_ERROR);
	}

	/**
	 * Result of {@link #resolvePatient}. Either {@code patient} is non-null (success),
	 * or {@code errorStatus} + {@code errorMessage} describe the failure.
	 */
	private static final class PatientResolution {

		final Patient patient;

		final HttpStatus errorStatus;

		final String errorMessage;

		private PatientResolution(Patient patient, HttpStatus errorStatus, String errorMessage) {
			this.patient = patient;
			this.errorStatus = errorStatus;
			this.errorMessage = errorMessage;
		}

		static PatientResolution ok(Patient patient) {
			return new PatientResolution(patient, null, null);
		}

		static PatientResolution error(HttpStatus status, String message) {
			return new PatientResolution(null, status, message);
		}

		boolean hasError() {
			return patient == null;
		}
	}

	private PatientResolution resolvePatient(String patientUuid) {
		if (patientUuid == null || patientUuid.trim().isEmpty()) {
			return PatientResolution.error(HttpStatus.BAD_REQUEST, "patient is required");
		}
		Patient patient = Context.getPatientService().getPatientByUuid(patientUuid);
		if (patient == null) {
			return PatientResolution.error(HttpStatus.NOT_FOUND, "Patient not found");
		}
		User user = Context.getAuthenticatedUser();
		if (!patientAccessCheck.canAccess(user, patient)) {
			return PatientResolution.error(HttpStatus.FORBIDDEN,
					"You do not have access to this patient's chart");
		}
		return PatientResolution.ok(patient);
	}

	private ResponseEntity<Object> checkRateLimit(User user) {
		int maxPerMinute = ChartSearchAiConstants.DEFAULT_RATE_LIMIT_PER_MINUTE;
		String configured = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RATE_LIMIT_PER_MINUTE);
		if (configured != null && !configured.trim().isEmpty()) {
			try {
				maxPerMinute = Integer.parseInt(configured.trim());
			}
			catch (NumberFormatException e) {
				log.warn("Invalid rate limit value '{}', using default", configured);
			}
		}

		if (maxPerMinute <= 0) {
			return null; // rate limiting disabled
		}

		Date oneMinuteAgo = new Date(System.currentTimeMillis() - 60000);
		long recentCount = auditLogService.getQueryCountByUserSince(user, oneMinuteAgo);

		if (recentCount >= maxPerMinute) {
			log.warn("Rate limit exceeded for user {} ({} queries in last minute)",
					user.getUserId(), recentCount);
			return new ResponseEntity<Object>(
					errorResponse("Rate limit exceeded. Maximum " + maxPerMinute
							+ " queries per minute."),
					HttpStatus.TOO_MANY_REQUESTS);
		}
		return null;
	}

	/**
	 * Validates feedback input fields (questionId and rating). Returns an error
	 * message if validation fails, or null if the input is valid.
	 */
	static String validateFeedbackInput(Map<String, Object> body) {
		Object questionIdObj = body.get("questionId");
		if (questionIdObj == null) {
			return "questionId is required";
		}
		try {
			Integer.valueOf(questionIdObj.toString());
		}
		catch (NumberFormatException e) {
			return "Invalid questionId";
		}
		String rating = body.get("rating") != null ? body.get("rating").toString() : null;
		if (rating == null || (!"positive".equals(rating) && !"negative".equals(rating))) {
			return "rating must be 'positive' or 'negative'";
		}
		return null;
	}

	/**
	 * Sanitizes a feedback comment: strips control characters and truncates
	 * to 500 characters. Returns null if the input is null.
	 */
	static String sanitizeFeedbackComment(String comment) {
		if (comment == null) {
			return null;
		}
		comment = CONTROL_CHARS.matcher(comment).replaceAll("");
		if (comment.length() > 500) {
			comment = comment.substring(0, 500);
		}
		return comment;
	}

	/**
	 * Validates and sanitizes the question input. Returns an error message if the question
	 * is rejected, or null if it passes validation.
	 */
	static String validateQuestion(String question) {
		if (PROMPT_INJECTION.matcher(question).find()) {
			log.warn("Rejected question containing prompt injection pattern");
			return "Question contains disallowed content";
		}
		return null;
	}

	/**
	 * Serializes references to the SSE wire shape shared by the early {@code references} event
	 * (grounding verdicts not yet attached) and the final {@code done} event (grounded).
	 * {@code grounded} is null when grounding is disabled or could not run — clients must render
	 * null as "unverified", never as "verified".
	 */
	private List<Map<String, Object>> serializeReferences(List<RecordReference> references) {
		List<Map<String, Object>> refs = new ArrayList<Map<String, Object>>();
		for (RecordReference ref : references) {
			Map<String, Object> refMap = new LinkedHashMap<String, Object>();
			refMap.put("index", ref.getIndex());
			refMap.put("resourceType", ref.getResourceType());
			refMap.put("resourceUuid", ref.getResourceUuid());
			refMap.put("date", formatDate(ref.getDate()));
			refMap.put("grounded", ref.getGrounded());
			refs.add(refMap);
		}
		return refs;
	}

	/**
	 * Emits the {@code references} SSE event carrying the answer's citations before the grounding
	 * pass completes, so the UI can render clickable citations without waiting on Tier-2
	 * verification. A serialization failure is non-fatal — the final {@code done} event re-sends the
	 * references with verdicts — but a client disconnect during the write unwinds the stream like the
	 * other channels (via {@link #writeSseEventOrThrow}).
	 */
	/**
	 * Serializes the post-answer drug-safety advisories to the wire shape rendered as chips below the
	 * answer. Empty list when the drug-reference feature is off or nothing was flagged. The key is
	 * always present (possibly empty) so the frontend can branch on length without a null check.
	 */
	private List<Map<String, Object>> serializeSafetyWarnings(List<SafetyWarning> warnings) {
		List<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
		if (warnings == null) {
			return out;
		}
		for (SafetyWarning warning : warnings) {
			Map<String, Object> map = new LinkedHashMap<String, Object>();
			map.put("type", warning.getType());
			map.put("drug", warning.getDrug());
			map.put("detail", warning.getDetail());
			out.add(map);
		}
		return out;
	}

	private void sendReferencesEvent(OutputStream out, List<RecordReference> references) {
		String json;
		try {
			Map<String, Object> data = new HashMap<String, Object>();
			data.put("references", serializeReferences(references));
			json = new ObjectMapper().writeValueAsString(data);
		}
		catch (IOException e) {
			log.warn("Could not serialize early references event; the final done event still carries them", e);
			return;
		}
		writeSseEventOrThrow(out, "references", json);
	}

	private void streamHubStagedChat(OutputStream out, ChatSession session, String patientUuid,
			String question, OverrideResolution overrideRes) throws IOException {
		List<ChatMessage> priorTurns = chatService.priorTurnsForRelay(session);
		String requestJson = hubRelayRequestJson(overrideRes.answeredModel, patientUuid, priorTurns, question, true);
		HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
				.uri(URI.create(overrideRes.endpointUrl))
				.version(HttpClient.Version.HTTP_1_1)
				.timeout(Duration.ofSeconds(300))
				.header("Content-Type", "application/json")
				.header("Accept", "text/event-stream")
				.POST(HttpRequest.BodyPublishers.ofByteArray(
						requestJson.getBytes(StandardCharsets.UTF_8)));
		String apiKey = runtimeApiKey();
		if (apiKey != null && !apiKey.trim().isEmpty()) {
			requestBuilder.header("Authorization", "Bearer " + apiKey.trim());
		}
		HttpResponse<InputStream> hubResponse;
		try {
			hubResponse = HttpClient.newHttpClient().send(requestBuilder.build(),
					HttpResponse.BodyHandlers.ofInputStream());
		}
		catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException("Hub staged stream interrupted", e);
		}
		if (hubResponse.statusCode() < 200 || hubResponse.statusCode() >= 300) {
			String body = new String(hubResponse.body().readAllBytes(), StandardCharsets.UTF_8);
			log.warn("Hub staged stream returned HTTP {}: {}", hubResponse.statusCode(), body);
			writeSseEvent(out, "error", "Hub staged stream failed: HTTP " + hubResponse.statusCode());
			return;
		}

		final String[] assistantMessageUuid = new String[1];
		final boolean[] doneSeen = new boolean[1];
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(
				hubResponse.body(), StandardCharsets.UTF_8))) {
			String event = "";
			StringBuilder data = new StringBuilder();
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.isEmpty()) {
					handleHubStagedEvent(out, session, question, overrideRes.answeredModel,
							assistantMessageUuid, doneSeen, event, data.toString());
					event = "";
					data.setLength(0);
				} else if (line.startsWith("event:")) {
					event = line.substring("event:".length()).trim();
				} else if (line.startsWith("data:")) {
					if (data.length() > 0) {
						data.append('\n');
					}
					String raw = line.substring("data:".length());
					data.append(raw.startsWith(" ") ? raw.substring(1) : raw);
				} else if (line.startsWith(":")) {
					// Hub heartbeat during a stalled leg — forward it so a browser disconnect is
					// detected here (Client disconnected -> propagates -> closes the hub connection
					// -> frees its router slot) instead of only on the next real event.
					writeSseCommentOrThrow(out);
				}
			}
			if (data.length() > 0) {
				handleHubStagedEvent(out, session, question, overrideRes.answeredModel,
						assistantMessageUuid, doneSeen, event, data.toString());
			}
		}
		if (!doneSeen[0]) {
			writeSseEvent(out, "error", "Hub staged stream ended before final response.");
		}
	}

	/**
	 * Relay for a NON-staged remote model (e.g. bare parity — no answer/validation/in-depth
	 * decomposition): one blocking hub call, then a single {@code done} event. The hub still owns
	 * chart retrieval (patient ref) and reference resolution — this is NOT the old locally-
	 * orchestrated chatService.chatStreaming path; it is the low-level engine's single-completion
	 * contract, relayed rather than reimplemented.
	 */
	private void streamHubNonStagedChat(OutputStream out, ChatSession session, String patientUuid,
			String question, OverrideResolution overrideRes) throws IOException {
		Map<String, Object> wire;
		try {
			wire = hubRelayCompletionWire(session, patientUuid, question, overrideRes);
		}
		catch (IOException e) {
			log.warn("Hub relay (non-staged) failed for patient [id={}]: {}",
					session.getPatient().getPatientId(), e.getMessage());
			writeSseEvent(out, "error", e.getMessage());
			return;
		}
		ChatTurnResult result = chatService.persistHubStagedAnswer(session, question, wire);
		writeHubPayload(out, "done", wire, result.getSessionUuid(), result.getAssistantMessageUuid(),
				overrideRes.answeredModel);
	}

	/**
	 * One blocking (non-streaming) hub call for a non-staged turn — shared by the streaming
	 * relay ({@link #streamHubNonStagedChat}) and the sync {@code POST /chat} endpoint, so the
	 * harness's synchronous research client drains the SAME hub engine the product streams
	 * against, on whatever model/profile id it names (including the low-level `answer:`/
	 * `answer-review:`/`indepth-only:` legs). Returns the hub's answer wire; does NOT persist it
	 * (callers persist via whichever ChatService method matches their surface).
	 */
	@SuppressWarnings("unchecked")
	private Map<String, Object> hubRelayCompletionWire(ChatSession session, String patientUuid, String question,
			OverrideResolution overrideRes) throws IOException {
		List<ChatMessage> priorTurns = chatService.priorTurnsForRelay(session);
		String requestJson = hubRelayRequestJson(overrideRes.answeredModel, patientUuid, priorTurns, question, false);
		HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
				.uri(URI.create(overrideRes.endpointUrl))
				.version(HttpClient.Version.HTTP_1_1)
				.timeout(Duration.ofSeconds(300))
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofByteArray(
						requestJson.getBytes(StandardCharsets.UTF_8)));
		String apiKey = runtimeApiKey();
		if (apiKey != null && !apiKey.trim().isEmpty()) {
			requestBuilder.header("Authorization", "Bearer " + apiKey.trim());
		}
		HttpResponse<String> hubResponse;
		try {
			hubResponse = HttpClient.newHttpClient().send(requestBuilder.build(),
					HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
		}
		catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException("Hub relay interrupted", e);
		}
		if (hubResponse.statusCode() < 200 || hubResponse.statusCode() >= 300) {
			throw new IOException("Hub relay failed: HTTP " + hubResponse.statusCode()
					+ ": " + hubResponse.body());
		}
		Map<String, Object> completion = MAPPER.readValue(hubResponse.body(),
				new TypeReference<Map<String, Object>>() {});
		List<Object> choices = (List<Object>) completion.get("choices");
		if (choices == null || choices.isEmpty()) {
			throw new IOException("Hub relay returned no choices.");
		}
		Map<String, Object> message = (Map<String, Object>) ((Map<String, Object>) choices.get(0)).get("message");
		String content = message == null ? null : (String) message.get("content");
		if (content == null || content.isEmpty()) {
			throw new IOException("Hub relay returned an empty answer.");
		}
		return MAPPER.readValue(content, new TypeReference<Map<String, Object>>() {});
	}

	@SuppressWarnings("unchecked")
	private void handleHubStagedEvent(OutputStream out, ChatSession session, String question,
			String model, String[] assistantMessageUuid, boolean[] doneSeen, String event,
			String data) throws IOException {
		if (event == null || event.isEmpty() || data == null || data.isEmpty()) {
			return;
		}
		if ("token".equals(event) || "indepth_token".equals(event) || "error".equals(event)) {
			if ("error".equals(event)) {
				doneSeen[0] = true;
			}
			writeSseEvent(out, event, data);
			return;
		}
		Map<String, Object> payload = MAPPER.readValue(data,
				new TypeReference<Map<String, Object>>() {});
		if ("answer_done".equals(event)) {
			ChatTurnResult result = chatService.persistHubStagedAnswer(session, question, payload);
			assistantMessageUuid[0] = result.getAssistantMessageUuid();
			writeHubPayload(out, event, payload, result.getSessionUuid(), assistantMessageUuid[0], model);
			return;
		}
		if ("answer_validation".equals(event)) {
			ChatTurnResult result = chatService.updateHubStagedMessage(
					session, assistantMessageUuid[0], payload);
			writeHubPayload(out, event, payload, result.getSessionUuid(), assistantMessageUuid[0], model);
			return;
		}
		if ("indepth_pending".equals(event)) {
			payload.put("messageId", assistantMessageUuid[0]);
			writeSseEvent(out, event, MAPPER.writeValueAsString(payload));
			return;
		}
		if ("indepth_done".equals(event) || "indepth_error".equals(event)) {
			Map<String, Object> update = new LinkedHashMap<String, Object>();
			update.put("inDepth", payload);
			if (assistantMessageUuid[0] != null) {
				chatService.updateHubStagedMessage(session, assistantMessageUuid[0], update);
			}
			payload.put("messageId", assistantMessageUuid[0]);
			writeSseEvent(out, event, MAPPER.writeValueAsString(payload));
			return;
		}
		if ("done".equals(event)) {
			doneSeen[0] = true;
			ChatTurnResult result;
			if (assistantMessageUuid[0] == null) {
				result = chatService.persistHubStagedAnswer(session, question, payload);
				assistantMessageUuid[0] = result.getAssistantMessageUuid();
			} else {
				result = chatService.updateHubStagedMessage(session, assistantMessageUuid[0], payload);
			}
			writeHubPayload(out, event, payload, result.getSessionUuid(), assistantMessageUuid[0], model);
			return;
		}
		writeSseEvent(out, event, data);
	}

	private void writeHubPayload(OutputStream out, String event, Map<String, Object> payload,
			String sessionUuid, String assistantMessageUuid, String model) throws IOException {
		payload.put("session", sessionUuid);
		payload.put("messageId", assistantMessageUuid);
		payload.put("model", model);
		payload.put("disclaimer", DISCLAIMER);
		writeSseEvent(out, event, MAPPER.writeValueAsString(payload));
	}

	private String hubRelayRequestJson(String model, String patientUuid, List<ChatMessage> priorTurns,
			String question, boolean stream) throws IOException {
		Map<String, Object> root = new LinkedHashMap<String, Object>();
		root.put("model", model);
		root.put("stream", stream);
		root.put("patient", patientUuid);
		List<Map<String, Object>> messages = new ArrayList<Map<String, Object>>();
		// Prior turns: prose-only (priorTurnsForRelay's contract — never the raw stored JSON
		// envelope), chronological, excluding the current turn. The hub owns the chart and the
		// system prompt; it inserts both itself, so the relay sends conversation content only.
		if (priorTurns != null) {
			for (ChatMessage prior : priorTurns) {
				Map<String, Object> turn = new LinkedHashMap<String, Object>();
				turn.put("role", prior.getRole());
				turn.put("content", prior.getContent());
				messages.add(turn);
			}
		}
		Map<String, Object> user = new LinkedHashMap<String, Object>();
		user.put("role", "user");
		user.put("content", question);
		messages.add(user);
		root.put("messages", messages);
		root.put("response_format", chartAnswerResponseFormat());
		return MAPPER.writeValueAsString(root);
	}

	private Map<String, Object> chartAnswerResponseFormat() {
		Map<String, Object> root = new LinkedHashMap<String, Object>();
		root.put("type", "json_schema");
		Map<String, Object> jsonSchema = new LinkedHashMap<String, Object>();
		jsonSchema.put("name", "chart_answer");
		Map<String, Object> schema = new LinkedHashMap<String, Object>();
		schema.put("type", "object");
		Map<String, Object> properties = new LinkedHashMap<String, Object>();
		properties.put("answer", Collections.singletonMap("type", "string"));
		Map<String, Object> citations = new LinkedHashMap<String, Object>();
		citations.put("type", "array");
		citations.put("items", Collections.singletonMap("type", "integer"));
		properties.put("citations", citations);
		properties.put("blocks", Collections.singletonMap("type", "array"));
		schema.put("properties", properties);
		schema.put("required", java.util.Arrays.asList("answer", "citations", "blocks"));
		jsonSchema.put("schema", schema);
		root.put("json_schema", jsonSchema);
		return root;
	}

	private String runtimeApiKey() {
		Properties props = Context.getRuntimeProperties();
		return props == null ? null : props.getProperty(ChartSearchAiConstants.RP_LLM_REMOTE_API_KEY);
	}

	private boolean wantsStaged(Map<String, String> body) {
		return "true".equalsIgnoreCase(body.get("staged"));
	}

	/**
	 * Whether this turn should relay through the hub's phased-streaming engine — a capability
	 * lookup against the endpoint's /v1/models (ModelSwitchService#isStagedModel), never a
	 * name-prefix guess. The hub owns which profiles are staged (levels.yaml); the controller
	 * asks it, rather than re-deciding scaffolding from the model id string.
	 */
	private boolean canStage(OverrideResolution overrideRes) {
		return overrideRes != null
				&& overrideRes.endpointUrl != null
				&& overrideRes.answeredModel != null
				&& modelSwitchService.isStagedModel(overrideRes.endpointUrl, overrideRes.answeredModel);
	}

	private void validateStagedModels(OverrideResolution overrideRes) {
		modelSwitchService.validateEndpointAndModel(
				overrideRes.endpointUrl, overrideRes.answeredModel);
	}

	private void writeSseEvent(OutputStream out, String event, String data) throws IOException {
		StringBuilder sb = new StringBuilder();
		sb.append("event: ").append(event).append('\n');
		for (String line : data.split("\n", -1)) {
			sb.append("data: ").append(line).append('\n');
		}
		sb.append('\n');
		out.write(sb.toString().getBytes("UTF-8"));
		out.flush();
	}

	/**
	 * Forwards a hub SSE heartbeat/comment line to the browser and converts a write failure into
	 * the {@link RuntimeException} the streaming loop unwinds on. A stalled leg (long answer/
	 * review/in-depth call) otherwise gives the relay NO opportunity to notice a browser disconnect
	 * until the hub's next real event — this write on every heartbeat is what makes a mid-leg abort
	 * actually free the router slot promptly instead of blocking for the rest of the leg.
	 */
	private void writeSseCommentOrThrow(OutputStream out) {
		try {
			out.write(": hb\n\n".getBytes(StandardCharsets.UTF_8));
			out.flush();
		}
		catch (IOException e) {
			log.debug("Client disconnected during a hub staged heartbeat");
			throw new RuntimeException("Client disconnected", e);
		}
	}

	/**
	 * Writes an SSE event, converting a client-disconnect {@link IOException} into the
	 * {@link RuntimeException} the streaming loop unwinds on. Shared by the answer ({@code token})
	 * and reasoning ({@code thinking}) channels so both handle a mid-stream disconnect identically.
	 */
	private void writeSseEventOrThrow(OutputStream out, String event, String data) {
		try {
			writeSseEvent(out, event, data);
		}
		catch (IOException e) {
			log.debug("Client disconnected during streaming ({})", event);
			throw new RuntimeException("Client disconnected", e);
		}
	}

	private void writeJsonError(HttpServletResponse response, int status, String message)
			throws IOException {
		response.setStatus(status);
		response.setContentType("application/json");
		response.setCharacterEncoding("UTF-8");
		new ObjectMapper().writeValue(response.getOutputStream(), errorResponse(message));
	}

	private Map<String, String> errorResponse(String message) {
		Map<String, String> error = new HashMap<String, String>();
		error.put("error", message);
		return error;
	}

	/**
	 * The model that answered a chat request + whether a per-request override was
	 * applied. Produced by {@link #resolveOverride}.
	 */
	private static final class OverrideResolution {

		private final String endpointUrl;

		private final String answeredModel;

		private final boolean overridden;

		OverrideResolution(String endpointUrl, String answeredModel, boolean overridden) {
			this.endpointUrl = endpointUrl;
			this.answeredModel = answeredModel;
			this.overridden = overridden;
		}
	}

	/**
	 * Resolve the model that will answer this chat request and apply any
	 * per-request backend override, honoring the active LLM engine:
	 * <ul>
	 *   <li><b>remote</b>: the answering model is {@code GP_LLM_REMOTE_MODEL_NAME},
	 *       or the validated {@code {endpointUrl, modelName}} override when the
	 *       caller supplies one (sets {@link RequestLlmOverride}; the caller MUST
	 *       clear it in a finally when {@code overridden} is true).</li>
	 *   <li><b>local</b> (bundled llama-server): the answering model is the bundled
	 *       model file; a per-request remote override does not apply and is
	 *       rejected so the caller isn't misled into thinking it took effect.</li>
	 * </ul>
	 *
	 * @throws IllegalArgumentException on an invalid override, or an override sent
	 *         to a non-remote engine — callers map this to HTTP 400.
	 */
	/**
	 * Chat always relays through a remote engine (a configured hub endpoint) — there is no local
	 * bundled-engine chat fallback. A misconfigured engine GP fails cleanly here (400) rather than
	 * silently falling back to local orchestration.
	 */
	private OverrideResolution resolveOverride(Map<String, String> body) {
		String engine = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_ENGINE);
		boolean isRemote = ChartSearchAiConstants.LLM_ENGINE_REMOTE
				.equalsIgnoreCase(engine != null ? engine.trim() : "");
		if (!isRemote) {
			throw new IllegalArgumentException(
					"Chat requires the remote engine (a configured hub endpoint); active engine is '"
							+ (engine != null ? engine.trim() : "") + "'.");
		}

		String overrideUrl = body.get("endpointUrl");
		String overrideModel = body.get("modelName");
		boolean hasOverride = overrideUrl != null && !overrideUrl.trim().isEmpty()
				&& overrideModel != null && !overrideModel.trim().isEmpty();

		String endpointUrl = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_REMOTE_ENDPOINT_URL);
		String answeredModel = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_LLM_REMOTE_MODEL_NAME);
		if (hasOverride) {
			String[] valid = modelSwitchService.validateEndpointAndModel(overrideUrl, overrideModel);
			RequestLlmOverride.set(valid[0], valid[1]);
			return new OverrideResolution(valid[0], valid[1], true);
		}
		return new OverrideResolution(endpointUrl, answeredModel, false);
	}
}
