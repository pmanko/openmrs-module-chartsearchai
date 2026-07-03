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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.openmrs.Patient;
import org.openmrs.User;
import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.ChartSearchService.ChartAnswer;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.api.db.ChatDAO;
import org.openmrs.module.chartsearchai.model.ChartSearchAuditLog;
import org.openmrs.module.chartsearchai.model.ChatMessage;
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service("chartSearchAi.chatService")
@Transactional
public class ChatServiceImpl implements ChatService {

	private static final Logger log = LoggerFactory.getLogger(ChatServiceImpl.class);

	private static final String SEARCH_MODE_CHAT = "chat";

	private static final ObjectMapper MAPPER = new ObjectMapper();

	@Autowired
	private ChatDAO chatDAO;

	@Autowired
	private LlmInferenceService llmInferenceService;

	@Autowired
	private ChartSearchAiDAO auditDAO;

	@Override
	public ChatSession openOrLoadActiveSession(Patient patient) {
		User user = Context.getAuthenticatedUser();
		ChatSession existing = chatDAO.getLatestSession(patient, user);
		if (existing != null) {
			return existing;
		}
		return createSession(patient, user);
	}

	@Override
	public ChatSession loadByUuid(String uuid) {
		if (uuid == null || uuid.isEmpty()) {
			return null;
		}
		return chatDAO.getSessionByUuid(uuid);
	}

	@Override
	public ChatSession closeAndStartNew(Patient patient) {
		User user = Context.getAuthenticatedUser();
		ChatSession existing = chatDAO.getLatestSession(patient, user);
		if (existing != null) {
			existing.setStatus(ChatSession.STATUS_CLOSED);
			existing.setEndedAt(new Date());
			chatDAO.saveSession(existing);
		}
		return createSession(patient, user);
	}

	@Override
	public ChatSession refreshChartSnapshot(Patient patient) {
		User user = Context.getAuthenticatedUser();
		ChatSession existing = chatDAO.getLatestSession(patient, user);
		if (existing == null) {
			// Nothing to refresh — opening a session builds a fresh snapshot anyway.
			return createSession(patient, user);
		}
		// Rebuild only the three chart fields; the transcript (messages) is never
		// touched — this is the whole contrast with closeAndStartNew.
		populateChartSnapshot(existing, patient);
		return chatDAO.saveSession(existing);
	}

	@Override
	public List<ChatMessage> getMessages(ChatSession session) {
		return chatDAO.getMessages(session);
	}

	@Override
	public ChatTurnResult chat(ChatSession session, String question) {
		ensureChartSnapshot(session);
		String chartEnvelope = session.getChartSnapshot();
		List<RecordMapping> mappings = deserializeMappings(session.getChartMappingsJson());

		List<ChatMessage> priorTurns = priorsForLlm(chatDAO.getMessages(session));
		int nextOrdinal = chatDAO.getLastOrdinal(session) + 1;
		persistUserMessage(session, question, nextOrdinal);

		long startedMs = System.currentTimeMillis();
		ChartAnswer answer;
		try {
			answer = llmInferenceService.chat(chartEnvelope, mappings, priorTurns, question);
		}
		catch (RuntimeException e) {
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;

		ChatMessage assistant = persistAssistantTurn(session, answer, nextOrdinal + 1,
				ChatMessage.FINISH_STOP, question, elapsedMs);
		touchSession(session);

		return new ChatTurnResult(answer, session.getUuid(), assistant.getUuid());
	}

	@Override
	public ChatTurnResult chatStreaming(ChatSession session, String question,
			Consumer<String> tokenConsumer) {
		ensureChartSnapshot(session);
		String chartEnvelope = session.getChartSnapshot();
		List<RecordMapping> mappings = deserializeMappings(session.getChartMappingsJson());

		List<ChatMessage> priorTurns = priorsForLlm(chatDAO.getMessages(session));
		int nextOrdinal = chatDAO.getLastOrdinal(session) + 1;
		persistUserMessage(session, question, nextOrdinal);

		// Accumulate the streamed text so a mid-stream abort can persist the
		// partial answer the client already received (see catch below).
		StringBuilder streamed = new StringBuilder();
		Consumer<String> accumulating = token -> {
			streamed.append(token);
			tokenConsumer.accept(token);
		};

		long startedMs = System.currentTimeMillis();
		ChartAnswer answer;
		try {
			answer = llmInferenceService.chatStreaming(
					chartEnvelope, mappings, priorTurns, question, accumulating);
		}
		catch (RuntimeException e) {
			// Client disconnected (or the upstream stream died) mid-flight.
			// Persist the partial assistant turn with finish_reason='aborted'
			// BEFORE re-throwing so the transcript stays well-formed: the next
			// request resumes at the correct ordinal instead of appending a
			// second consecutive user message. (ChatService docstring contract.)
			long abortedMs = System.currentTimeMillis() - startedMs;
			ChartAnswer partial = new ChartAnswer(streamed.toString(), Collections.emptyList());
			persistAssistantTurn(session, partial, nextOrdinal + 1,
					ChatMessage.FINISH_ABORTED, question, abortedMs);
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;

		ChatMessage assistant = persistAssistantTurn(session, answer, nextOrdinal + 1,
				ChatMessage.FINISH_STOP, question, elapsedMs);
		touchSession(session);

		return new ChatTurnResult(answer, session.getUuid(), assistant.getUuid());
	}

	@Override
	public ChatTurnResult chatStagedAnswer(ChatSession session, String question,
			Consumer<String> tokenConsumer) {
		ensureChartSnapshot(session);
		String chartEnvelope = session.getChartSnapshot();
		List<RecordMapping> mappings = deserializeMappings(session.getChartMappingsJson());

		List<ChatMessage> priorTurns = priorsForLlm(chatDAO.getMessages(session));
		int nextOrdinal = chatDAO.getLastOrdinal(session) + 1;
		persistUserMessage(session, question, nextOrdinal);

		StringBuilder streamed = new StringBuilder();
		Consumer<String> accumulating = token -> {
			streamed.append(token);
			tokenConsumer.accept(token);
		};

		long startedMs = System.currentTimeMillis();
		ChartAnswer answer;
		try {
			answer = llmInferenceService.chatStreaming(
					chartEnvelope, mappings, priorTurns, question, accumulating);
		}
		catch (RuntimeException e) {
			long abortedMs = System.currentTimeMillis() - startedMs;
			ChartAnswer partial = new ChartAnswer(streamed.toString(), Collections.emptyList());
			persistAssistantTurn(session, partial, nextOrdinal + 1,
					ChatMessage.FINISH_ABORTED, question, abortedMs,
					answerValidationWire("unavailable",
							"Answer check unavailable because the answer stream aborted.",
							Collections.emptyList(), null),
					inDepthWire("failed", "", "Answer stream aborted before In-Depth could start."));
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;

		ChatMessage assistant = persistAssistantTurn(session, answer, nextOrdinal + 1,
				ChatMessage.FINISH_STOP, question, elapsedMs,
				answerValidationWire("validating", "Checking answer against chart and temporal/date rules.",
						Collections.emptyList(), null),
				inDepthWire("pending", "", ""));
		touchSession(session);

		return new ChatTurnResult(answer, session.getUuid(), assistant.getUuid());
	}

	@Override
	public ChatTurnResult completeStagedAnswerValidation(ChatSession session, String assistantMessageUuid,
			String originalQuestion, Consumer<String> tokenConsumer) {
		ensureChartSnapshot(session);
		ChatMessage assistant = chatDAO.getMessageByUuid(assistantMessageUuid);
		requireAssistantInSession(session, assistant, "Answer validation");

		String chartEnvelope = session.getChartSnapshot();
		List<RecordMapping> mappings = deserializeMappings(session.getChartMappingsJson());
		List<ChatMessage> priorTurns = priorsForLlm(chatDAO.getMessages(session));
		Map<String, Object> stored = assistantWire(assistant.getContent());
		String prompt = answerReviewPrompt(originalQuestion, stored);

		long startedMs = System.currentTimeMillis();
		ChartAnswer reviewed;
		try {
			reviewed = llmInferenceService.chatStreaming(
					chartEnvelope, mappings, priorTurns, prompt, tokenConsumer);
		}
		catch (RuntimeException e) {
			updateAssistantAnswerValidation(assistant,
					answerValidationWire("unavailable",
							"Answer check unavailable; the review model failed or timed out.",
							Collections.singletonList(e.getMessage()), null));
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;
		Map<String, Object> answerValidation = reviewed.getAnswerValidation();
		if (answerValidation == null) {
			answerValidation = answerValidationWire("checked",
					"Answer check completed, but the reviewer did not return lifecycle metadata.",
					Collections.emptyList(), null);
		}
		updateAssistantReviewedAnswer(assistant, reviewed, answerValidation);
		assistant.setOutputTokens((assistant.getOutputTokens() == null ? 0 : assistant.getOutputTokens())
				+ reviewed.getOutputTokens());
		touchSession(session);
		log.info("[timing] stagedAnswerValidation session={} message={} llmMs={} inputTokens={} outputTokens={}",
				session.getUuid(), assistant.getUuid(), elapsedMs,
				reviewed.getInputTokens(), reviewed.getOutputTokens());
		return new ChatTurnResult(reviewed, session.getUuid(), assistant.getUuid());
	}

	@Override
	public ChatTurnResult completeStagedInDepth(ChatSession session, String assistantMessageUuid,
			String prompt, Consumer<String> tokenConsumer) {
		ensureChartSnapshot(session);
		ChatMessage assistant = chatDAO.getMessageByUuid(assistantMessageUuid);
		requireAssistantInSession(session, assistant, "In-Depth");

		String chartEnvelope = session.getChartSnapshot();
		List<RecordMapping> mappings = deserializeMappings(session.getChartMappingsJson());
		List<ChatMessage> priorTurns = priorsForLlm(chatDAO.getMessages(session));

		StringBuilder streamed = new StringBuilder();
		Consumer<String> accumulating = token -> {
			streamed.append(token);
			tokenConsumer.accept(token);
		};

		long startedMs = System.currentTimeMillis();
		ChartAnswer inDepth;
		try {
			inDepth = llmInferenceService.chatStreaming(
					chartEnvelope, mappings, priorTurns, prompt, accumulating);
		}
		catch (RuntimeException e) {
			updateAssistantInDepth(assistant,
					inDepthWire("failed", stripInDepthHeader(streamed.toString()),
							"In-Depth generation failed before completion."));
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;
		updateAssistantInDepth(assistant,
				inDepthWire("complete", stripInDepthHeader(inDepth.getAnswer()), ""));
		assistant.setOutputTokens((assistant.getOutputTokens() == null ? 0 : assistant.getOutputTokens())
				+ inDepth.getOutputTokens());
		touchSession(session);
		log.info("[timing] stagedInDepth session={} message={} llmMs={} inputTokens={} outputTokens={}",
				session.getUuid(), assistant.getUuid(), elapsedMs,
				inDepth.getInputTokens(), inDepth.getOutputTokens());
		return new ChatTurnResult(inDepth, session.getUuid(), assistant.getUuid());
	}

	protected ChatSession createSession(Patient patient, User user) {
		ChatSession session = new ChatSession();
		session.setPatient(patient);
		session.setUser(user);
		Date now = new Date();
		session.setStartedAt(now);
		session.setLastActivityAt(now);
		session.setStatus(ChatSession.STATUS_ACTIVE);
		populateChartSnapshot(session, patient);
		return chatDAO.saveSession(session);
	}

	/**
	 * Backfill chart snapshot for sessions that lack one on first chat() call.
	 * Idempotent.
	 */
	protected void ensureChartSnapshot(ChatSession session) {
		if (session.getChartSnapshot() != null) {
			return;
		}
		populateChartSnapshot(session, session.getPatient());
		chatDAO.saveSession(session);
	}

	/**
	 * Build the full chart for the session's patient (bypassing pre-filter)
	 * and store envelope + mappings on the session row. The byte-stability
	 * of envelope across all turns is the load-bearing invariant of the
	 * chat design — the LLM's prompt cache hits on this prefix.
	 */
	protected void populateChartSnapshot(ChatSession session, Patient patient) {
		PatientChart chart = llmInferenceService.buildSessionChart(patient);
		session.setChartSnapshot(chart.getText());
		session.setChartMappingsJson(serializeMappings(chart.getMappings()));
		session.setChartBuiltAt(new Date());
	}

	/**
	 * Serialize {@link RecordMapping} list as JSON with epoch-ms dates so
	 * the round-trip is locale-free and deterministic. {@code RecordMapping}
	 * lacks a no-arg ctor so we ser/des via plain Map.
	 */
	private static String serializeMappings(List<RecordMapping> mappings) {
		if (mappings == null || mappings.isEmpty()) {
			return "[]";
		}
		List<Map<String, Object>> wire = new ArrayList<>(mappings.size());
		for (RecordMapping m : mappings) {
			Map<String, Object> e = new LinkedHashMap<>();
			e.put("index", m.getIndex());
			e.put("resourceType", m.getResourceType());
			e.put("resourceUuid", m.getResourceUuid());
			e.put("date", m.getDate() == null ? null : m.getDate().getTime());
			wire.add(e);
		}
		try {
			return MAPPER.writeValueAsString(wire);
		}
		catch (IOException ioe) {
			throw new APIException("Failed to serialize chart mappings: " + ioe.getMessage(), ioe);
		}
	}

	static List<RecordMapping> deserializeMappings(String json) {
		if (json == null || json.isEmpty() || "[]".equals(json)) {
			return Collections.emptyList();
		}
		try {
			List<Map<String, Object>> wire = MAPPER.readValue(
					json, new TypeReference<List<Map<String, Object>>>() {});
			List<RecordMapping> out = new ArrayList<>(wire.size());
			for (Map<String, Object> e : wire) {
				int index = ((Number) e.get("index")).intValue();
				String type = (String) e.get("resourceType");
				String uuid = (String) e.get("resourceUuid");
				Number dateMs = (Number) e.get("date");
				Date d = dateMs == null ? null : new Date(dateMs.longValue());
				out.add(new RecordMapping(index, type, uuid, d));
			}
			return out;
		}
		catch (IOException ioe) {
			throw new APIException("Failed to deserialize chart mappings: " + ioe.getMessage(), ioe);
		}
	}

	protected ChatMessage persistUserMessage(ChatSession session, String content, int ordinal) {
		ChatMessage msg = new ChatMessage();
		msg.setSession(session);
		msg.setOrdinal(ordinal);
		msg.setRole(ChatMessage.ROLE_USER);
		msg.setContent(content);
		msg.setCreatedAt(new Date());
		return chatDAO.saveMessage(msg);
	}

	protected ChatMessage persistAssistantTurn(ChatSession session, ChartAnswer answer, int ordinal,
			String finishReason, String questionForAudit, long responseTimeMs) {
		return persistAssistantTurn(session, answer, ordinal, finishReason, questionForAudit,
				responseTimeMs, null);
	}

	protected ChatMessage persistAssistantTurn(ChatSession session, ChartAnswer answer, int ordinal,
			String finishReason, String questionForAudit, long responseTimeMs,
			Map<String, Object> inDepth) {
		return persistAssistantTurn(session, answer, ordinal, finishReason, questionForAudit,
				responseTimeMs, answer.getAnswerValidation(), inDepth);
	}

	protected ChatMessage persistAssistantTurn(ChatSession session, ChartAnswer answer, int ordinal,
			String finishReason, String questionForAudit, long responseTimeMs,
			Map<String, Object> answerValidation, Map<String, Object> inDepth) {
		ChartSearchAuditLog audit = buildAuditRow(session, questionForAudit, answer, responseTimeMs);
		auditDAO.saveAuditLog(audit);

		ChatMessage msg = new ChatMessage();
		msg.setSession(session);
		msg.setOrdinal(ordinal);
		msg.setRole(ChatMessage.ROLE_ASSISTANT);
		msg.setContent(serializeAssistantContent(answer, answerValidation, inDepth));
		msg.setCreatedAt(new Date());
		msg.setAuditLog(audit);
		msg.setInputTokens(answer.getInputTokens());
		msg.setOutputTokens(answer.getOutputTokens());
		msg.setFinishReason(finishReason);
		return chatDAO.saveMessage(msg);
	}

	/**
	 * Serialize the assistant response (prose + citations + blocks) to JSON
	 * for storage on {@code chat_message.content}. The SPA hydration parses
	 * this JSON back into {@code {answer, blocks}}; LLM replay extracts just
	 * the prose answer via {@link #extractProseAnswer}.
	 *
	 * <p>When blocks is empty, we still store JSON (not plaintext) so the
	 * hydration parser only has to handle one canonical shape on newly-
	 * created rows. Legacy plaintext rows are detected and handled
	 * separately in {@link #extractProseAnswer}.
	 */
	private static String serializeAssistantContent(ChartAnswer answer) {
		return serializeAssistantContent(answer, answer.getAnswerValidation(), null);
	}

	private static String serializeAssistantContent(ChartAnswer answer, Map<String, Object> inDepth) {
		return serializeAssistantContent(answer, answer.getAnswerValidation(), inDepth);
	}

	private static String serializeAssistantContent(ChartAnswer answer,
			Map<String, Object> answerValidation, Map<String, Object> inDepth) {
		Map<String, Object> wire = new LinkedHashMap<>();
		wire.put("answer", answer.getAnswer());
		wire.put("references", referencesToWire(answer.getReferences()));
		// blocks rendered via the same helper used by the REST controller —
		// keep wire format identical across persistence and live response so
		// SPA hydration and SSE done events parse the same way.
		wire.put("blocks", blocksToWire(answer.getBlocks()));
		// Persist per-section confidence so a page reload rehydrates the same tag the live
		// stream showed; omitted when absent (LM Studio / parity lane) so no phantom tag.
		if (answer.getConfidence() != null) {
			wire.put("confidence", answer.getConfidence());
		}
		if (answerValidation != null) {
			wire.put("answerValidation", answerValidation);
		}
		if (inDepth != null) {
			wire.put("inDepth", inDepth);
		}
		try {
			return MAPPER.writeValueAsString(wire);
		}
		catch (IOException ioe) {
			// Fail soft: if serialization breaks, fall back to plain prose so
			// chat continues to function (just without blocks rendering on
			// hydration). The thrown error would otherwise abort the entire
			// assistant-message persistence and lose the answer.
			log.warn("Failed to serialize assistant content to JSON; storing prose only: {}",
					ioe.getMessage());
			return answer.getAnswer();
		}
	}

	private static Map<String, Object> inDepthWire(String status, String answer, String error) {
		Map<String, Object> wire = new LinkedHashMap<>();
		wire.put("status", status);
		wire.put("answer", answer == null ? "" : answer);
		if (error != null && !error.isEmpty()) {
			wire.put("error", error);
		}
		return wire;
	}

	private static Map<String, Object> answerValidationWire(String status, String summary,
			List<?> issues, String originalAnswer) {
		Map<String, Object> wire = new LinkedHashMap<>();
		wire.put("status", status);
		wire.put("label", answerValidationLabel(status));
		wire.put("summary", summary == null ? "" : summary);
		wire.put("issues", issues == null ? Collections.emptyList() : issues);
		if (!"validating".equals(status)) {
			java.text.SimpleDateFormat fmt = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
			fmt.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
			wire.put("completedAt", fmt.format(new Date()));
		}
		if (originalAnswer != null) {
			wire.put("originalAnswer", originalAnswer);
		}
		return wire;
	}

	private static String answerValidationLabel(String status) {
		if ("validating".equals(status)) {
			return "Checking answer";
		}
		if ("checked".equals(status)) {
			return "Checked";
		}
		if ("edited".equals(status)) {
			return "Updated after check";
		}
		if ("needs_review".equals(status)) {
			return "Needs review";
		}
		if ("unavailable".equals(status)) {
			return "Check unavailable";
		}
		return status;
	}

	private ChatMessage updateAssistantInDepth(ChatMessage assistant, Map<String, Object> inDepth) {
		Map<String, Object> wire = assistantWire(assistant.getContent());
		wire.put("inDepth", inDepth);
		return saveAssistantWire(assistant, wire, "staged In-Depth");
	}

	private ChatMessage updateAssistantAnswerValidation(ChatMessage assistant,
			Map<String, Object> answerValidation) {
		Map<String, Object> wire = assistantWire(assistant.getContent());
		wire.put("answerValidation", answerValidation);
		return saveAssistantWire(assistant, wire, "staged Answer validation");
	}

	private ChatMessage updateAssistantReviewedAnswer(ChatMessage assistant, ChartAnswer answer,
			Map<String, Object> answerValidation) {
		Map<String, Object> wire = assistantWire(assistant.getContent());
		wire.put("answer", answer.getAnswer());
		wire.put("references", referencesToWire(answer.getReferences()));
		wire.put("blocks", blocksToWire(answer.getBlocks()));
		if (answer.getConfidence() != null) {
			wire.put("confidence", answer.getConfidence());
		} else {
			wire.remove("confidence");
		}
		wire.put("answerValidation", answerValidation);
		return saveAssistantWire(assistant, wire, "staged Answer validation");
	}

	private ChatMessage saveAssistantWire(ChatMessage assistant, Map<String, Object> wire,
			String context) {
		try {
			assistant.setContent(MAPPER.writeValueAsString(wire));
		}
		catch (IOException ioe) {
			throw new APIException("Failed to serialize " + context + ": " + ioe.getMessage(), ioe);
		}
		return chatDAO.saveMessage(assistant);
	}

	private static Map<String, Object> assistantWire(String stored) {
		Map<String, Object> wire = new LinkedHashMap<>();
		if (stored != null && stored.trim().startsWith("{")) {
			try {
				wire.putAll(MAPPER.readValue(stored,
						new TypeReference<Map<String, Object>>() {}));
			}
			catch (IOException ignored) {
				wire.clear();
				wire.put("answer", stored);
				wire.put("blocks", Collections.emptyList());
			}
		} else {
			wire.put("answer", stored == null ? "" : stored);
			wire.put("blocks", Collections.emptyList());
		}
		if (!wire.containsKey("blocks")) {
			wire.put("blocks", Collections.emptyList());
		}
		if (!wire.containsKey("references")) {
			wire.put("references", Collections.emptyList());
		}
		return wire;
	}

	private static List<Map<String, Object>> referencesToWire(List<RecordReference> references) {
		List<Map<String, Object>> out = new ArrayList<>();
		if (references == null) {
			return out;
		}
		for (RecordReference ref : references) {
			Map<String, Object> refMap = new LinkedHashMap<>();
			refMap.put("index", ref.getIndex());
			refMap.put("resourceType", ref.getResourceType());
			refMap.put("resourceUuid", ref.getResourceUuid());
			refMap.put("date", ref.getDate() == null ? null : DateFormatUtil.formatDate(ref.getDate()));
			out.add(refMap);
		}
		return out;
	}

	@SuppressWarnings("unchecked")
	private static String answerReviewPrompt(String originalQuestion, Map<String, Object> stored) {
		Map<String, Object> payload = new LinkedHashMap<>();
		payload.put("schema_version", "answer_to_review.v1");
		payload.put("original_question", originalQuestion == null ? "" : originalQuestion);
		payload.put("answer", stored.get("answer") == null ? "" : stored.get("answer"));
		payload.put("citations", citationIndicesFromStoredWire(stored));
		payload.put("blocks", stored.get("blocks") instanceof List ? stored.get("blocks") : Collections.emptyList());
		if (stored.get("references") instanceof List) {
			payload.put("references", stored.get("references"));
		}
		if (stored.get("confidence") instanceof Map) {
			payload.put("confidence", stored.get("confidence"));
		}
		try {
			return "Review the already-visible clinical answer below. Return the chart_answer JSON "
					+ "envelope with answerValidation metadata.\n\n```json\n"
					+ MAPPER.writeValueAsString(payload) + "\n```";
		}
		catch (IOException ioe) {
			throw new APIException("Failed to serialize answer review payload: " + ioe.getMessage(), ioe);
		}
	}

	@SuppressWarnings("unchecked")
	private static List<Integer> citationIndicesFromStoredWire(Map<String, Object> stored) {
		java.util.Set<Integer> seen = new java.util.LinkedHashSet<Integer>();
		Object answer = stored.get("answer");
		if (answer instanceof String) {
			java.util.regex.Matcher marker = ChartSearchAiUtils.INLINE_CITATION.matcher((String) answer);
			while (marker.find()) {
				seen.add(Integer.valueOf(marker.group(1)));
			}
		}
		Object blocks = stored.get("blocks");
		collectBlockRefs(blocks, seen);
		return new ArrayList<Integer>(seen);
	}

	@SuppressWarnings("unchecked")
	private static void collectBlockRefs(Object value, java.util.Set<Integer> out) {
		if (value instanceof Map) {
			for (Object entryValue : ((Map<Object, Object>) value).values()) {
				collectBlockRefs(entryValue, out);
			}
		} else if (value instanceof List) {
			for (Object item : (List<Object>) value) {
				collectBlockRefs(item, out);
			}
		} else if (value instanceof Number) {
			out.add(((Number) value).intValue());
		}
	}

	private void requireAssistantInSession(ChatSession session, ChatMessage assistant, String operation) {
		requireOk(assistant != null, "Assistant message not found");
		requireOk(ChatMessage.ROLE_ASSISTANT.equals(assistant.getRole()),
				operation + " can only be attached to an assistant message");
		requireOk(assistant.getSession() != null
				&& session.getSessionId() != null
				&& session.getSessionId().equals(assistant.getSession().getSessionId()),
				"Assistant message does not belong to this chat session");
	}

	private static String stripInDepthHeader(String answer) {
		if (answer == null) {
			return "";
		}
		return answer.replaceFirst("(?is)^\\s*\\*\\*In\\s*Depth\\*\\*\\s*", "").trim();
	}

	private static List<Map<String, Object>> blocksToWire(List<ResponseBlock> blocks) {
		List<Map<String, Object>> out = new ArrayList<>();
		if (blocks == null) {
			return out;
		}
		for (ResponseBlock block : blocks) {
			Map<String, Object> blockMap = new LinkedHashMap<>();
			blockMap.put("kind", block.getKind());
			if (block.getTitle() != null) {
				blockMap.put("title", block.getTitle());
			}
			List<Map<String, Object>> columns = new ArrayList<>();
			for (ResponseBlock.Column c : block.getColumns()) {
				Map<String, Object> col = new LinkedHashMap<>();
				col.put("key", c.getKey());
				col.put("label", c.getLabel());
				columns.add(col);
			}
			blockMap.put("columns", columns);
			List<Map<String, Object>> rows = new ArrayList<>();
			for (ResponseBlock.Row row : block.getRows()) {
				Map<String, Object> cellsMap = new LinkedHashMap<>();
				for (Map.Entry<String, ResponseBlock.Cell> entry : row.getCells().entrySet()) {
					Map<String, Object> cellMap = new LinkedHashMap<>();
					cellMap.put("text", entry.getValue().getText());
					cellMap.put("refs", entry.getValue().getRefs());
					cellsMap.put(entry.getKey(), cellMap);
				}
				Map<String, Object> rowMap = new LinkedHashMap<>();
				rowMap.put("cells", cellsMap);
				rows.add(rowMap);
			}
			blockMap.put("rows", rows);
			out.add(blockMap);
		}
		return out;
	}

	/**
	 * Extract the prose-only answer from a stored assistant message content
	 * for LLM-replay. Handles both shapes:
	 * <ul>
	 *   <li>New: {@code {"answer": "<prose>", "blocks": [...]}}</li>
	 *   <li>Legacy: plain string</li>
	 * </ul>
	 * Sending the JSON envelope to the LLM in prior assistant turns confuses
	 * small models; the prose summary is enough context for follow-ups.
	 */
	/**
	 * Project the persisted prior turns into the shape the LLM should see.
	 * For assistant rows, replaces the stored JSON envelope with just the
	 * prose answer (see {@link #extractProseAnswer}). User rows pass
	 * through unchanged. Returns new transient {@link ChatMessage}
	 * instances rather than mutating the Hibernate-managed entities — the
	 * loaded rows are still in the session and a content change would
	 * persist on flush.
	 */
	private static List<ChatMessage> priorsForLlm(List<ChatMessage> raw) {
		List<ChatMessage> out = new ArrayList<>(raw.size());
		for (ChatMessage m : raw) {
			ChatMessage view = new ChatMessage();
			view.setRole(m.getRole());
			view.setContent(ChatMessage.ROLE_ASSISTANT.equals(m.getRole())
					? extractProseAnswer(m.getContent())
					: m.getContent());
			view.setOrdinal(m.getOrdinal());
			out.add(view);
		}
		return out;
	}

	static String extractProseAnswer(String storedContent) {
		if (storedContent == null || storedContent.isEmpty()) {
			return storedContent;
		}
		String trimmed = storedContent.trim();
		if (!trimmed.startsWith("{")) {
			return storedContent;
		}
		try {
			com.fasterxml.jackson.databind.JsonNode root = MAPPER.readTree(trimmed);
			com.fasterxml.jackson.databind.JsonNode answer = root.get("answer");
			if (answer != null && answer.isTextual()) {
				return answer.asText();
			}
		}
		catch (IOException ignored) {
			// Not JSON or malformed — treat as plaintext.
		}
		return storedContent;
	}

	protected ChatSession touchSession(ChatSession session) {
		session.setLastActivityAt(new Date());
		return chatDAO.saveSession(session);
	}

	private ChartSearchAuditLog buildAuditRow(ChatSession session, String question,
			ChartAnswer answer, long responseTimeMs) {
		ChartSearchAuditLog audit = new ChartSearchAuditLog();
		audit.setUser(session.getUser());
		audit.setPatient(session.getPatient());
		audit.setQuestion(question);
		audit.setAnswer(answer.getAnswer());
		audit.setReferenceCount(answer.getReferences() == null ? 0 : answer.getReferences().size());
		audit.setSearchMode(SEARCH_MODE_CHAT);
		audit.setResponseTimeMs(responseTimeMs);
		audit.setInputTokens(answer.getInputTokens());
		audit.setOutputTokens(answer.getOutputTokens());
		audit.setDateCreated(new Date());
		return audit;
	}

	/**
	 * Retention horizon for chat content rows. The {@link AuditLogPurgeTask}
	 * reads this to drive purgeBefore.
	 */
	public static int getChatRetentionDays() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_CHAT_RETENTION_DAYS);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed > 0) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid {} value '{}', using default",
						ChartSearchAiConstants.GP_CHAT_RETENTION_DAYS, value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_CHAT_RETENTION_DAYS;
	}

	void requireOk(boolean ok, String msg) {
		if (!ok) {
			throw new APIException(msg);
		}
	}
}
