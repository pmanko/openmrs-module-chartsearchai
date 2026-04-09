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
 * Embedding provider using ONNX Runtime. Supports any BERT-based sentence embedding
 * model (e.g. all-MiniLM-L6-v2 at 384 dimensions). The embedding dimensions are
 * detected automatically from the model output on first use. The ONNX model file path
 * is configured via the {@code chartsearchai.embedding.modelFilePath} global property.
 */
@Component("chartSearchAi.embeddingProvider")
public class OnnxEmbeddingProvider implements EmbeddingProvider {

	private static final Logger log = LoggerFactory.getLogger(OnnxEmbeddingProvider.class);

	private OrtEnvironment env;

	private OrtSession session;

	private String loadedModelPath;

	private WordPieceTokenizer tokenizer;

	private volatile int detectedDimensions = -1;

	private String explicitModelPath;

	private String explicitVocabPath;

	private int explicitMaxSeqLen = -1;

	/**
	 * Default constructor for Spring context — resolves model paths from
	 * OpenMRS global properties.
	 */
	public OnnxEmbeddingProvider() {
	}

	/**
	 * Test-friendly constructor that accepts explicit file paths, bypassing
	 * the OpenMRS Context dependency.
	 *
	 * @param modelPath absolute path to the ONNX model file
	 * @param vocabPath absolute path to the WordPiece vocabulary file
	 */
	public OnnxEmbeddingProvider(String modelPath, String vocabPath) {
		this.explicitModelPath = modelPath;
		this.explicitVocabPath = vocabPath;
		this.explicitMaxSeqLen = ChartSearchAiConstants.DEFAULT_MAX_SEQUENCE_LENGTH;
	}

	@Override
	public synchronized float[] embed(String text) {
		try {
			return runInference(getSession(), getTokenizer(), text);
		}
		catch (OrtException e) {
			throw new RuntimeException("Failed to compute embedding", e);
		}
	}

	@Override
	public int getDimensions() {
		return detectedDimensions;
	}

	public synchronized void close() {
		if (session != null) {
			log.info("Closing ONNX embedding session");
			try {
				session.close();
			}
			catch (OrtException e) {
				log.warn("Error closing ONNX session", e);
			}
			session = null;
			loadedModelPath = null;
			tokenizer = null;
			detectedDimensions = -1;
		}
		env = null;
	}

	private float[] runInference(OrtSession ortSession, WordPieceTokenizer wpTokenizer,
			String text) throws OrtException {
		Map<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();
		OrtSession.Result result = null;
		try {
			TokenizedInput tokenized = wpTokenizer.tokenize(text);
			int seqLen = tokenized.getLength();

			long[][] inputIdsArr = { tokenized.getInputIds() };
			long[][] attentionMaskArr = { tokenized.getAttentionMask() };
			long[][] tokenTypeIdsArr = { tokenized.getTokenTypeIds() };

			OrtEnvironment ortEnv = getOrCreateEnv();
			inputs.put("input_ids", OnnxTensor.createTensor(ortEnv, inputIdsArr));
			inputs.put("attention_mask", OnnxTensor.createTensor(ortEnv, attentionMaskArr));
			inputs.put("token_type_ids", OnnxTensor.createTensor(ortEnv, tokenTypeIdsArr));

			result = ortSession.run(inputs);

			float[][][] output = (float[][][]) result.get(0).getValue();
			int modelDimensions = output[0][0].length;
			if (detectedDimensions == -1) {
				detectedDimensions = modelDimensions;
				log.info("Detected embedding dimensions: {}", detectedDimensions);
			} else if (modelDimensions != detectedDimensions) {
				throw new OrtException("ONNX model output dimensions changed ("
						+ modelDimensions + " vs previously detected "
						+ detectedDimensions + "). Was the model swapped without reloading?");
			}

			// Mean pooling over attention-masked tokens
			float[] embedding = new float[modelDimensions];
			long[] attentionMask = tokenized.getAttentionMask();
			int tokenCount = 0;
			for (int i = 0; i < seqLen; i++) {
				if (attentionMask[i] == 1) {
					for (int j = 0; j < embedding.length; j++) {
						embedding[j] += output[0][i][j];
					}
					tokenCount++;
				}
			}
			if (tokenCount > 0) {
				for (int j = 0; j < embedding.length; j++) {
					embedding[j] /= tokenCount;
				}
			}

			// L2 normalize
			double norm = 0;
			for (float v : embedding) {
				norm += v * v;
			}
			norm = Math.sqrt(norm);
			if (norm > 0) {
				for (int i = 0; i < embedding.length; i++) {
					embedding[i] /= (float) norm;
				}
			}

			return embedding;
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

	private OrtEnvironment getOrCreateEnv() {
		if (env == null) {
			env = OrtEnvironment.getEnvironment();
		}
		return env;
	}

	private synchronized OrtSession getSession() throws OrtException {
		String modelPath;
		if (explicitModelPath != null) {
			modelPath = explicitModelPath;
		} else {
			String configuredPath = Context.getAdministrationService()
					.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_MODEL_FILE_PATH);
			if (configuredPath == null || configuredPath.trim().isEmpty()) {
				throw new IllegalStateException(
						"Embedding model path not configured. Set the global property: "
								+ ChartSearchAiConstants.GP_EMBEDDING_MODEL_FILE_PATH);
			}
			modelPath = ChartSearchAiUtils.resolveModelPath(
					configuredPath.trim(), ChartSearchAiConstants.GP_EMBEDDING_MODEL_FILE_PATH);
		}

		if (session != null && !modelPath.equals(loadedModelPath)) {
			log.info("Embedding model path changed from {} to {}, reloading",
					loadedModelPath, modelPath);
			close();
		}

		if (session == null) {
			log.info("Loading ONNX embedding model from {}", modelPath);
			OrtEnvironment ortEnv = getOrCreateEnv();
			session = ortEnv.createSession(modelPath);
			loadedModelPath = modelPath;
			log.info("ONNX embedding model loaded successfully");
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
						.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_VOCAB_FILE_PATH);
				if (configuredPath == null || configuredPath.trim().isEmpty()) {
					throw new IllegalStateException(
							"Embedding vocabulary path not configured. Set the global property: "
									+ ChartSearchAiConstants.GP_EMBEDDING_VOCAB_FILE_PATH);
				}
				vocabPath = ChartSearchAiUtils.resolveModelPath(
						configuredPath.trim(), ChartSearchAiConstants.GP_EMBEDDING_VOCAB_FILE_PATH);
			}
			int maxSeqLen = explicitMaxSeqLen > 0
					? explicitMaxSeqLen : getMaxSequenceLength();
			log.info("Loading WordPiece vocabulary from {} (maxSequenceLength={})",
					vocabPath, maxSeqLen);
			try {
				tokenizer = new WordPieceTokenizer(vocabPath, maxSeqLen);
			}
			catch (IOException e) {
				throw new IllegalStateException("Failed to load WordPiece vocabulary from "
						+ vocabPath, e);
			}
			log.info("WordPiece vocabulary loaded successfully");
		}
		return tokenizer;
	}

	private static int getMaxSequenceLength() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_MAX_SEQUENCE_LENGTH);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed >= 32 && parsed <= 8192) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid maxSequenceLength value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_MAX_SEQUENCE_LENGTH;
	}
}
