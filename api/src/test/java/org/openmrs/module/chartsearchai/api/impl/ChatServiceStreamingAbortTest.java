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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.api.ChartSearchService.ChartAnswer;
import org.openmrs.module.chartsearchai.api.ChatService;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.api.db.ChatDAO;
import org.openmrs.module.chartsearchai.model.ChatMessage;
import org.openmrs.module.chartsearchai.model.ChatSession;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.openmrs.test.jupiter.BaseModuleContextSensitiveTest;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * Abort contract for streaming chat turns, exercised through the real
 * {@link ChatServiceImpl} + {@link ChatDAO} against a real patient (no mocks
 * of the persistence path — module rule).
 *
 * <p>The contract under test is the {@link ChatService#chatStreaming} docstring:
 * when the upstream LLM stream dies mid-flight (client disconnect →
 * RuntimeException out of {@code llmInferenceService.chatStreaming}), the
 * partial assistant text already streamed to the client MUST be persisted as
 * an assistant row with {@code finish_reason='aborted'} <em>before</em> the
 * exception propagates. Otherwise the transcript is left with a dangling user
 * message and no assistant reply, so the next turn appends a second consecutive
 * user message at the wrong ordinal — a malformed history the LLM then sees.
 *
 * <p>The LLM call is the only thing stubbed: a hand-built {@link ChatServiceImpl}
 * wired to the real DAOs plus a stub {@link LlmInferenceService} whose
 * {@code chatStreaming} emits two tokens through the supplied consumer and then
 * throws. We bypass the autowired {@code @Transactional} proxy on purpose — the
 * thrown exception must NOT mark the test transaction rollback-only, so the
 * already-persisted rows remain queryable.
 */
public class ChatServiceStreamingAbortTest extends BaseModuleContextSensitiveTest {

	@Autowired
	private ChatService chatService;

	@Autowired
	private ChatDAO chatDAO;

	@Autowired
	private ChartSearchAiDAO auditDAO;

	private Patient patient;

	@BeforeEach
	public void setUp() {
		patient = Context.getPatientService().getPatient(2);
		assertNotNull(patient, "standard test patient 2 must exist");
	}

	/**
	 * Build a ChatServiceImpl wired to the real DAOs and the given stub LLM,
	 * setting the @Autowired private fields directly so we drive the real
	 * persistence path without going through the transactional CGLIB proxy.
	 */
	private ChatServiceImpl newServiceWith(LlmInferenceService stubLlm) throws Exception {
		ChatServiceImpl impl = new ChatServiceImpl();
		setField(impl, "chatDAO", chatDAO);
		setField(impl, "auditDAO", auditDAO);
		setField(impl, "llmInferenceService", stubLlm);
		return impl;
	}

	private static void setField(Object target, String name, Object value) throws Exception {
		Field f = ChatServiceImpl.class.getDeclaredField(name);
		f.setAccessible(true);
		f.set(target, value);
	}

	@Test
	public void chatStreaming_whenStreamAborts_persistsAbortedAssistantTurnWithPartialText()
			throws Exception {
		ChatSession session = chatService.openOrLoadActiveSession(patient);
		assertNotNull(session.getChartSnapshot(), "a session opens with a chart snapshot");
		Context.flushSession();
		assertEquals(0, chatService.getMessages(session).size(), "precondition: empty transcript");

		// --- turn 1: stream dies after emitting two tokens ---
		ChatServiceImpl aborting = newServiceWith(new AbortingStreamStub());
		assertThrows(RuntimeException.class, () ->
				aborting.chatStreaming(session, "What infections does this patient have?",
						token -> { /* client sink; tokens streamed before the abort */ }));
		Context.flushSession();

		List<ChatMessage> afterAbort = chatService.getMessages(session);
		assertEquals(2, afterAbort.size(),
				"abort must persist BOTH the user message and a partial assistant row");

		ChatMessage user1 = afterAbort.get(0);
		ChatMessage assistant1 = afterAbort.get(1);
		assertEquals(ChatMessage.ROLE_USER, user1.getRole());
		assertEquals(0, user1.getOrdinal().intValue(), "first user turn is ordinal 0");

		assertEquals(ChatMessage.ROLE_ASSISTANT, assistant1.getRole(),
				"the aborted turn must be persisted as an assistant row");
		assertEquals(1, assistant1.getOrdinal().intValue(),
				"aborted assistant turn takes the next ordinal");
		assertEquals(ChatMessage.FINISH_ABORTED, assistant1.getFinishReason(),
				"aborted turn must carry finish_reason='aborted'");
		String partial = ChatServiceImpl.extractProseAnswer(assistant1.getContent());
		assertEquals(AbortingStreamStub.PARTIAL, partial,
				"aborted assistant row must carry the partial streamed text");

		// --- turn 2: a healthy stream must resume at the correct ordinal ---
		ChatServiceImpl healthy = newServiceWith(new SuccessfulStreamStub());
		ChatService.ChatTurnResult result = healthy.chatStreaming(session,
				"And what medications?", token -> { });
		assertNotNull(result.getAssistantMessageUuid());
		Context.flushSession();

		List<ChatMessage> afterRecovery = chatService.getMessages(session);
		assertEquals(4, afterRecovery.size(),
				"recovery turn appends a clean user+assistant pair");
		assertEquals(2, afterRecovery.get(2).getOrdinal().intValue(),
				"recovery user turn resumes at the correct next ordinal");
		assertEquals(ChatMessage.ROLE_USER, afterRecovery.get(2).getRole());
		assertEquals(3, afterRecovery.get(3).getOrdinal().intValue());
		assertEquals(ChatMessage.ROLE_ASSISTANT, afterRecovery.get(3).getRole());

		// well-formed history: never two consecutive user messages.
		for (int i = 1; i < afterRecovery.size(); i++) {
			boolean bothUser = ChatMessage.ROLE_USER.equals(afterRecovery.get(i - 1).getRole())
					&& ChatMessage.ROLE_USER.equals(afterRecovery.get(i).getRole());
			assertTrue(!bothUser, "history must not contain two consecutive user messages; "
					+ "the aborted assistant turn separates them");
		}
	}

	/** Emits two tokens through the consumer, then throws as if the client hung up. */
	private static final class AbortingStreamStub extends LlmInferenceService {

		static final String TOKEN_A = "Active ";

		static final String TOKEN_B = "tuberculosis [1].";

		static final String PARTIAL = TOKEN_A + TOKEN_B;

		@Override
		public ChartAnswer chatStreaming(String chartEnvelope, List<RecordMapping> mappings,
				List<ChatMessage> priorTurns, String question, Consumer<String> tokenConsumer) {
			tokenConsumer.accept(TOKEN_A);
			tokenConsumer.accept(TOKEN_B);
			throw new RuntimeException("client disconnected mid-stream");
		}
	}

	/** A clean stream that returns a full answer. */
	private static final class SuccessfulStreamStub extends LlmInferenceService {

		@Override
		public ChartAnswer chatStreaming(String chartEnvelope, List<RecordMapping> mappings,
				List<ChatMessage> priorTurns, String question, Consumer<String> tokenConsumer) {
			tokenConsumer.accept("Lisinopril 10mg daily.");
			return new ChartAnswer("Lisinopril 10mg daily.", Collections.emptyList());
		}
	}
}
