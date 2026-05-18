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

import java.util.Date;
import java.util.List;
import java.util.function.Consumer;

import org.openmrs.Patient;
import org.openmrs.User;
import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.ChartSearchService.ChartAnswer;
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.api.db.ChatDAO;
import org.openmrs.module.chartsearchai.model.ChartSearchAuditLog;
import org.openmrs.module.chartsearchai.model.ChatMessage;
import org.openmrs.module.chartsearchai.model.ChatSession;
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
	public List<ChatMessage> getMessages(ChatSession session) {
		return chatDAO.getMessages(session);
	}

	@Override
	public ChatTurnResult chat(ChatSession session, String question) {
		List<ChatMessage> priorTurns = chatDAO.getMessages(session);
		int nextOrdinal = chatDAO.getLastOrdinal(session) + 1;
		persistUserMessage(session, question, nextOrdinal);

		long startedMs = System.currentTimeMillis();
		ChartAnswer answer;
		try {
			answer = llmInferenceService.chat(session.getPatient(), priorTurns, question);
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
		List<ChatMessage> priorTurns = chatDAO.getMessages(session);
		int nextOrdinal = chatDAO.getLastOrdinal(session) + 1;
		persistUserMessage(session, question, nextOrdinal);

		long startedMs = System.currentTimeMillis();
		ChartAnswer answer;
		String finishReason = ChatMessage.FINISH_STOP;
		try {
			answer = llmInferenceService.chatStreaming(
					session.getPatient(), priorTurns, question, tokenConsumer);
		}
		catch (RuntimeException e) {
			finishReason = ChatMessage.FINISH_ABORTED;
			touchSession(session);
			throw e;
		}
		long elapsedMs = System.currentTimeMillis() - startedMs;

		ChatMessage assistant = persistAssistantTurn(session, answer, nextOrdinal + 1,
				finishReason, question, elapsedMs);
		touchSession(session);

		return new ChatTurnResult(answer, session.getUuid(), assistant.getUuid());
	}

	protected ChatSession createSession(Patient patient, User user) {
		ChatSession session = new ChatSession();
		session.setPatient(patient);
		session.setUser(user);
		Date now = new Date();
		session.setStartedAt(now);
		session.setLastActivityAt(now);
		session.setStatus(ChatSession.STATUS_ACTIVE);
		return chatDAO.saveSession(session);
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
		ChartSearchAuditLog audit = buildAuditRow(session, questionForAudit, answer, responseTimeMs);
		auditDAO.saveAuditLog(audit);

		ChatMessage msg = new ChatMessage();
		msg.setSession(session);
		msg.setOrdinal(ordinal);
		msg.setRole(ChatMessage.ROLE_ASSISTANT);
		msg.setContent(answer.getAnswer());
		msg.setCreatedAt(new Date());
		msg.setAuditLog(audit);
		msg.setInputTokens(answer.getInputTokens());
		msg.setOutputTokens(answer.getOutputTokens());
		msg.setFinishReason(finishReason);
		return chatDAO.saveMessage(msg);
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
