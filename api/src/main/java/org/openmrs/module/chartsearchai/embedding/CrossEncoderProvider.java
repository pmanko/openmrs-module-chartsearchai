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
 * Cross-encoder relevance scorer using ONNX Runtime. A cross-encoder reads
 * the {@code (query, document)} pair jointly through a BERT-style model
 * and emits a single relevance logit — a much sharper signal than
 * dual-encoder cosine similarity, at the cost of running one inference
 * per candidate (no shared embedding cache).
 *
 * <p>Used by the retrieval pipeline as a precision gate: candidates with
 * very low cross-encoder scores are dropped (the model says "definitely
 * not relevant"), and candidates with very high scores override
 * upstream rejections (the model says "definitely relevant"). Middle
 * scores defer to the dual-encoder pipeline.</p>
 *
 * <p>Configuration is via two global properties:
 * {@link ChartSearchAiConstants#GP_CROSS_ENCODER_MODEL_FILE_PATH} and
 * {@link ChartSearchAiConstants#GP_CROSS_ENCODER_VOCAB_FILE_PATH}. When
 * either is unset the provider returns {@code false} from
 * {@link #isAvailable()} and {@link #scorePair} throws — callers should
 * gate the cross-encoder stage on availability.</p>
 */
@Component("chartSearchAi.crossEncoderProvider")
public class CrossEncoderProvider {

	private static final Logger log = LoggerFactory.getLogger(
			CrossEncoderProvider.class);

	private OrtEnvironment env;

	private OrtSession session;

	private OrtSession.SessionOptions sessionOptions;

	private String loadedModelPath;

	private WordPieceTokenizer tokenizer;

	private final String explicitModelPath;

	private final String explicitVocabPath;

	private final int explicitMaxSeqLen;

	/**
	 * Spring-managed constructor — resolves model and vocab paths from
	 * OpenMRS global properties on first use.
	 */
	public CrossEncoderProvider() {
		this.explicitModelPath = null;
		this.explicitVocabPath = null;
		this.explicitMaxSeqLen = -1;
	}

	/**
	 * Test-friendly constructor that accepts explicit file paths.
	 */
	public CrossEncoderProvider(String modelPath, String vocabPath) {
		this.explicitModelPath = modelPath;
		this.explicitVocabPath = vocabPath;
		this.explicitMaxSeqLen =
				ChartSearchAiConstants.DEFAULT_MAX_SEQUENCE_LENGTH;
	}

	/**
	 * Returns true when both the model and vocab paths are configured
	 * (via the test constructor or via the corresponding global
	 * properties). Callers should check before invoking
	 * {@link #scorePair}.
	 */
	public boolean isAvailable() {
		try {
			return resolveModelPath() != null && resolveVocabPath() != null;
		}
		catch (Exception e) {
			return false;
		}
	}

	/**
	 * Scores a {@code (query, document)} pair. Returns the model's raw
	 * relevance logit — higher means more relevant. Range and zero-point
	 * are model-specific; thresholds must be calibrated per model.
	 *
	 * @throws IllegalStateException if the cross-encoder is not configured
	 */
	public synchronized float scorePair(String query, String document) {
		try {
			OrtSession ortSession = getSession();
			TokenizedInput tok = getTokenizer().tokenizePair(
					query == null ? "" : query,
					document == null ? "" : document);

			long[][] inputIds = { tok.getInputIds() };
			long[][] attentionMask = { tok.getAttentionMask() };

			Map<String, OnnxTensor> inputs =
					new HashMap<String, OnnxTensor>();
			OrtSession.Result result = null;
			try {
				OrtEnvironment ortEnv = getOrCreateEnv();
				inputs.put("input_ids",
						OnnxTensor.createTensor(ortEnv, inputIds));
				inputs.put("attention_mask",
						OnnxTensor.createTensor(ortEnv, attentionMask));
				if (ortSession.getInputNames().contains("token_type_ids")) {
					long[][] tokenTypeIds = { tok.getTokenTypeIds() };
					inputs.put("token_type_ids",
							OnnxTensor.createTensor(ortEnv, tokenTypeIds));
				}
				result = ortSession.run(inputs);
				Object out = result.get(0).getValue();
				// BertForSequenceClassification: shape [batch, num_labels].
				// MedCPT cross-encoder has num_labels=1; MS-MARCO is also 1.
				if (out instanceof float[][]) {
					return ((float[][]) out)[0][0];
				}
				if (out instanceof float[][][]) {
					return ((float[][][]) out)[0][0][0];
				}
				throw new RuntimeException(
						"Unexpected cross-encoder output shape: "
						+ out.getClass());
			}
			finally {
				if (result != null) {
					try { result.close(); } catch (Exception ignored) {}
				}
				for (OnnxTensor t : inputs.values()) {
					try { t.close(); } catch (Exception ignored) {}
				}
			}
		}
		catch (OrtException e) {
			throw new RuntimeException(
					"Cross-encoder inference failed", e);
		}
	}

	public synchronized void close() {
		if (session != null) {
			try { session.close(); }
			catch (OrtException e) {
				log.warn("Error closing cross-encoder session", e);
			}
			session = null;
			if (sessionOptions != null) {
				sessionOptions.close();
				sessionOptions = null;
			}
			loadedModelPath = null;
			tokenizer = null;
		}
		env = null;
	}

	private OrtEnvironment getOrCreateEnv() {
		if (env == null) {
			env = OrtEnvironment.getEnvironment();
		}
		return env;
	}

	private synchronized OrtSession getSession() throws OrtException {
		String modelPath = resolveModelPath();
		if (modelPath == null) {
			throw new IllegalStateException(
					"Cross-encoder model path not configured");
		}
		if (session != null && !modelPath.equals(loadedModelPath)) {
			log.info("Cross-encoder model path changed, reloading");
			close();
		}
		if (session == null) {
			log.info("Loading cross-encoder ONNX model from {}",
					modelPath);
			OrtEnvironment ortEnv = getOrCreateEnv();
			sessionOptions = new OrtSession.SessionOptions();
			session = ortEnv.createSession(modelPath, sessionOptions);
			loadedModelPath = modelPath;
			log.info("Cross-encoder loaded ({} inputs, {} outputs)",
					session.getInputNames(), session.getOutputNames());
		}
		return session;
	}

	private synchronized WordPieceTokenizer getTokenizer() {
		if (tokenizer == null) {
			String vocabPath = resolveVocabPath();
			if (vocabPath == null) {
				throw new IllegalStateException(
						"Cross-encoder vocab path not configured");
			}
			int maxSeqLen = explicitMaxSeqLen > 0
					? explicitMaxSeqLen
					: ChartSearchAiConstants.DEFAULT_MAX_SEQUENCE_LENGTH;
			try {
				tokenizer = new WordPieceTokenizer(vocabPath, maxSeqLen);
			}
			catch (IOException e) {
				throw new IllegalStateException(
						"Failed to load cross-encoder vocab from "
						+ vocabPath, e);
			}
		}
		return tokenizer;
	}

	private String resolveModelPath() {
		if (explicitModelPath != null) return explicitModelPath;
		try {
			String configured = Context.getAdministrationService()
					.getGlobalProperty(
							ChartSearchAiConstants.GP_CROSS_ENCODER_MODEL_FILE_PATH);
			if (configured == null || configured.trim().isEmpty()) {
				return null;
			}
			return ChartSearchAiUtils.resolveModelPath(
					configured.trim(),
					ChartSearchAiConstants.GP_CROSS_ENCODER_MODEL_FILE_PATH);
		}
		catch (Exception e) {
			return null;
		}
	}

	private String resolveVocabPath() {
		if (explicitVocabPath != null) return explicitVocabPath;
		try {
			String configured = Context.getAdministrationService()
					.getGlobalProperty(
							ChartSearchAiConstants.GP_CROSS_ENCODER_VOCAB_FILE_PATH);
			if (configured == null || configured.trim().isEmpty()) {
				return null;
			}
			return ChartSearchAiUtils.resolveModelPath(
					configured.trim(),
					ChartSearchAiConstants.GP_CROSS_ENCODER_VOCAB_FILE_PATH);
		}
		catch (Exception e) {
			return null;
		}
	}
}
