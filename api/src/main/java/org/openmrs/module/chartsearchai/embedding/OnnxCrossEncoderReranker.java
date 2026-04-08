/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.embedding;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.embedding.WordPieceTokenizer.TokenizedInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Cross-encoder reranker using ONNX Runtime. Scores a (query, document) pair
 * by feeding both through a single BERT pass, producing a relevance logit.
 * Opt-in: only active when the reranker model path global property is configured.
 */
@Component("chartSearchAi.crossEncoderReranker")
public class OnnxCrossEncoderReranker {

	private static final Logger log = LoggerFactory.getLogger(OnnxCrossEncoderReranker.class);

	private OrtEnvironment env;

	private OrtSession session;

	private String loadedModelPath;

	private WordPieceTokenizer tokenizer;

	private String explicitModelPath;

	private String explicitVocabPath;

	private int explicitMaxSeqLen = -1;

	/**
	 * Default constructor for Spring context — resolves model paths from
	 * OpenMRS global properties.
	 */
	public OnnxCrossEncoderReranker() {
	}

	/**
	 * Test-friendly constructor that accepts explicit file paths, bypassing
	 * the OpenMRS Context dependency.
	 */
	public OnnxCrossEncoderReranker(String modelPath, String vocabPath) {
		this.explicitModelPath = modelPath;
		this.explicitVocabPath = vocabPath;
		this.explicitMaxSeqLen = ChartSearchAiConstants.DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH;
	}

	/**
	 * Scores a (query, document) pair. Returns the raw relevance logit from
	 * the cross-encoder model. Higher values indicate greater relevance.
	 */
	public synchronized double score(String query, String document) {
		Map<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();
		OrtSession.Result result = null;
		try {
			OrtSession ortSession = getSession();
			WordPieceTokenizer wpTokenizer = getTokenizer();

			TokenizedInput tokenized = wpTokenizer.tokenizePair(query, document);

			long[][] inputIdsArr = { tokenized.getInputIds() };
			long[][] attentionMaskArr = { tokenized.getAttentionMask() };
			long[][] tokenTypeIdsArr = { tokenized.getTokenTypeIds() };

			inputs.put("input_ids", OnnxTensor.createTensor(env, inputIdsArr));
			inputs.put("attention_mask", OnnxTensor.createTensor(env, attentionMaskArr));
			inputs.put("token_type_ids", OnnxTensor.createTensor(env, tokenTypeIdsArr));

			result = ortSession.run(inputs);

			float[][] output = (float[][]) result.get(0).getValue();
			if (output[0].length == 1) {
				// Single logit output (regression head)
				return output[0][0];
			} else {
				// Binary classification output — use index [1] (relevant class)
				return output[0][1];
			}
		}
		catch (OrtException e) {
			throw new RuntimeException("Failed to compute cross-encoder score", e);
		}
		finally {
			if (result != null) {
				try {
					result.close();
				}
				catch (Exception e) {
					log.warn("Error closing ONNX result", e);
				}
			}
			for (OnnxTensor tensor : inputs.values()) {
				try {
					tensor.close();
				}
				catch (Exception e) {
					log.warn("Error closing ONNX tensor", e);
				}
			}
		}
	}

	/**
	 * Returns true when the reranker model path is configured (opt-in).
	 */
	public boolean isAvailable() {
		if (explicitModelPath != null) {
			return true;
		}
		try {
			String configuredPath = Context.getAdministrationService()
					.getGlobalProperty(ChartSearchAiConstants.GP_RERANKER_MODEL_FILE_PATH);
			return configuredPath != null && !configuredPath.trim().isEmpty();
		}
		catch (Exception e) {
			return false;
		}
	}

	public synchronized void close() {
		if (session != null) {
			log.info("Closing ONNX cross-encoder session");
			try {
				session.close();
			}
			catch (OrtException e) {
				log.warn("Error closing ONNX session", e);
			}
			session = null;
			loadedModelPath = null;
			tokenizer = null;
			env = null;
		}
	}

	private synchronized OrtSession getSession() throws OrtException {
		String modelPath;
		if (explicitModelPath != null) {
			modelPath = explicitModelPath;
		} else {
			String configuredPath = Context.getAdministrationService()
					.getGlobalProperty(ChartSearchAiConstants.GP_RERANKER_MODEL_FILE_PATH);
			if (configuredPath == null || configuredPath.trim().isEmpty()) {
				throw new IllegalStateException(
						"Cross-encoder model path not configured. Set the global property: "
								+ ChartSearchAiConstants.GP_RERANKER_MODEL_FILE_PATH);
			}
			modelPath = ChartSearchAiUtils.resolveModelPath(
					configuredPath.trim(), ChartSearchAiConstants.GP_RERANKER_MODEL_FILE_PATH);
		}

		if (session != null && !modelPath.equals(loadedModelPath)) {
			log.info("Cross-encoder model path changed from {} to {}, reloading",
					loadedModelPath, modelPath);
			close();
		}

		if (session == null) {
			log.info("Loading ONNX cross-encoder model from {}", modelPath);
			env = OrtEnvironment.getEnvironment();
			session = env.createSession(modelPath);
			loadedModelPath = modelPath;
			log.info("ONNX cross-encoder model loaded successfully");
		}
		return session;
	}

	private synchronized WordPieceTokenizer getTokenizer() {
		if (tokenizer == null) {
			String vocabPath;
			if (explicitVocabPath != null) {
				vocabPath = explicitVocabPath;
			} else {
				String configuredPath = Context.getAdministrationService()
						.getGlobalProperty(ChartSearchAiConstants.GP_RERANKER_VOCAB_FILE_PATH);
				if (configuredPath == null || configuredPath.trim().isEmpty()) {
					throw new IllegalStateException(
							"Cross-encoder vocabulary path not configured. Set the global property: "
									+ ChartSearchAiConstants.GP_RERANKER_VOCAB_FILE_PATH);
				}
				vocabPath = ChartSearchAiUtils.resolveModelPath(
						configuredPath.trim(), ChartSearchAiConstants.GP_RERANKER_VOCAB_FILE_PATH);
			}
			int maxSeqLen = explicitMaxSeqLen > 0
					? explicitMaxSeqLen : getMaxSequenceLength();
			log.info("Loading cross-encoder vocabulary from {} (maxSequenceLength={})",
					vocabPath, maxSeqLen);
			try {
				tokenizer = new WordPieceTokenizer(vocabPath, maxSeqLen);
			}
			catch (IOException e) {
				throw new IllegalStateException(
						"Failed to load cross-encoder vocabulary from " + vocabPath, e);
			}
			log.info("Cross-encoder vocabulary loaded successfully");
		}
		return tokenizer;
	}

	private static int getMaxSequenceLength() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RERANKER_MAX_SEQUENCE_LENGTH);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed >= 32 && parsed <= 8192) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid reranker maxSequenceLength value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH;
	}
}
