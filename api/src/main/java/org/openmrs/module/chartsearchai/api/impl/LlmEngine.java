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

import java.util.function.Consumer;

/**
 * Abstraction for LLM inference engines. Implementations handle the actual
 * model invocation (local or remote) while prompt construction and response
 * parsing remain in {@link LlmProvider}.
 */
public interface LlmEngine {

	/**
	 * Run inference and return the full generated text.
	 *
	 * @param systemPrompt the system prompt
	 * @param userMessage the user message (patient records + question)
	 * @param timeoutSeconds maximum wall-clock seconds for the request
	 * @return the inference result containing generated text and token counts
	 */
	InferenceResult infer(String systemPrompt, String userMessage, int timeoutSeconds);

	/**
	 * Run inference with streaming, calling the consumer for each token fragment.
	 *
	 * @param systemPrompt the system prompt
	 * @param userMessage the user message (patient records + question)
	 * @param timeoutSeconds maximum wall-clock seconds for the request
	 * @param tokenConsumer called with each token fragment as it is generated
	 * @return the inference result containing the full generated text and token counts
	 */
	InferenceResult inferStreaming(String systemPrompt, String userMessage, int timeoutSeconds,
			Consumer<String> tokenConsumer);

	/**
	 * Release resources (model, connections) but allow re-initialization on next call.
	 */
	void close();

	/**
	 * Permanently shut down background threads and release all resources.
	 * After this call, the engine should not be used again.
	 */
	void shutdown();

	/**
	 * Result of an LLM inference call.
	 */
	class InferenceResult {

		private final String text;

		private final int tokenCount;

		public InferenceResult(String text, int tokenCount) {
			this.text = text;
			this.tokenCount = tokenCount;
		}

		public String getText() {
			return text;
		}

		public int getTokenCount() {
			return tokenCount;
		}
	}
}
