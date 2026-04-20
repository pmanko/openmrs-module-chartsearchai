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

import java.util.EnumSet;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
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

	private OrtSession.SessionOptions sessionOptions;

	private String loadedModelPath;

	private WordPieceTokenizer tokenizer;

	private volatile int detectedDimensions = -1;

	private OrtProvider activeProvider = OrtProvider.CPU;

	private String explicitModelPath;

	private String explicitVocabPath;

	private int explicitMaxSeqLen = -1;

	// Cached model identity to avoid redundant sentinel-string embedding
	private String cachedModelIdentity;

	private String identityCachedForModelPath;

	// Dual-encoder support: separate session for query encoding
	private OrtSession querySession;

	private OrtSession.SessionOptions querySessionOptions;

	private String loadedQueryModelPath;

	private String explicitQueryModelPath;

	/**
	 * Default constructor for Spring context — resolves model paths from
	 * OpenMRS global properties.
	 */
	public OnnxEmbeddingProvider() {
	}

	/**
	 * Test-friendly constructor that accepts explicit file paths, bypassing
	 * the OpenMRS Context dependency. Uses single-encoder mode.
	 *
	 * @param modelPath absolute path to the ONNX model file
	 * @param vocabPath absolute path to the WordPiece vocabulary file
	 */
	public OnnxEmbeddingProvider(String modelPath, String vocabPath) {
		this.explicitModelPath = modelPath;
		this.explicitVocabPath = vocabPath;
		this.explicitMaxSeqLen = ChartSearchAiConstants.DEFAULT_MAX_SEQUENCE_LENGTH;
	}

	/**
	 * Test-friendly constructor for dual-encoder models (e.g. MedCPT).
	 *
	 * @param articleModelPath path to the article/record encoder ONNX model
	 * @param queryModelPath path to the query encoder ONNX model
	 * @param vocabPath path to the shared WordPiece vocabulary file
	 */
	public OnnxEmbeddingProvider(String articleModelPath, String queryModelPath,
			String vocabPath) {
		this.explicitModelPath = articleModelPath;
		this.explicitQueryModelPath = queryModelPath;
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
	public synchronized float[] embedQuery(String text) {
		try {
			OrtSession qs = getQuerySession();
			if (qs != null) {
				return runInference(qs, getTokenizer(), text);
			}
			// Single-encoder fallback
			return runInference(getSession(), getTokenizer(), text);
		}
		catch (OrtException e) {
			throw new RuntimeException("Failed to compute query embedding", e);
		}
	}

	@Override
	public int getDimensions() {
		return detectedDimensions;
	}

	@Override
	public boolean isSubwordToken(String term) {
		try {
			return getTokenizer().isSplitWord(term);
		} catch (Exception e) {
			return false;
		}
	}

	@Override
	public String getModelName() {
		if (loadedModelPath != null) {
			return loadedModelPath;
		}
		return explicitModelPath;
	}

	/**
	 * Known model fingerprints: first 4 embedding values for the sentinel
	 * string {@link EmbeddingProvider#MODEL_IDENTITY_SENTINEL}, rounded to
	 * 2 decimal places and concatenated. Computed by profiling each model.
	 */
	private static final String FINGERPRINT_MEDEMBED_SMALL =
			"-0.03,0.04,0.02,-0.03";
	private static final String FINGERPRINT_L6_V2 =
			"-0.09,-0.01,-0.01,0.05";
	private static final String FINGERPRINT_PUBMEDBERT =
			"-0.01,0.04,-0.04,-0.02";
	private static final String FINGERPRINT_L12_V2 =
			"-0.09,0.06,-0.01,-0.01";
	private static final String FINGERPRINT_MEDCPT =
			"-0.01,0.00,-0.03,-0.01";

	@Override
	public synchronized String identifyModel() {
		try {
			// Ensure the current model is loaded (triggers reload if GP
			// changed). Without this, the cache would silently prevent
			// the model reload that getSession() normally performs.
			getSession();
			if (cachedModelIdentity != null && loadedModelPath != null
					&& loadedModelPath.equals(identityCachedForModelPath)) {
				return cachedModelIdentity;
			}
			float[] v = embed(MODEL_IDENTITY_SENTINEL);
			String fp = String.format("%.2f,%.2f,%.2f,%.2f",
					v[0], v[1], v[2], v[3]);
			String result;
			if (FINGERPRINT_MEDCPT.equals(fp)) {
				result = "medcpt";
			} else if (FINGERPRINT_MEDEMBED_SMALL.equals(fp)) {
				result = "medembed-small";
			} else if (FINGERPRINT_L6_V2.equals(fp)) {
				result = "all-MiniLM-L6-v2";
			} else if (FINGERPRINT_PUBMEDBERT.equals(fp)) {
				result = "pubmedbert";
			} else if (FINGERPRINT_L12_V2.equals(fp)) {
				result = "all-MiniLM-L12-v2";
			} else {
				log.info("Unknown embedding model fingerprint: {}", fp);
				result = null;
			}
			cachedModelIdentity = result;
			identityCachedForModelPath = loadedModelPath;
			return result;
		}
		catch (Exception e) {
			log.warn("Failed to fingerprint embedding model", e);
			return null;
		}
	}

	/**
	 * Returns the execution provider used by the ONNX session (e.g. CORE_ML, CUDA, or CPU).
	 * The provider is determined when the session is first created.
	 */
	public OrtProvider getActiveProvider() {
		return activeProvider;
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
			if (sessionOptions != null) {
				sessionOptions.close();
				sessionOptions = null;
			}
			loadedModelPath = null;
			cachedModelIdentity = null;
			identityCachedForModelPath = null;
			tokenizer = null;
			detectedDimensions = -1;
			activeProvider = OrtProvider.CPU;
		}
		if (querySession != null) {
			try {
				querySession.close();
			}
			catch (OrtException e) {
				log.warn("Error closing query ONNX session", e);
			}
			querySession = null;
			if (querySessionOptions != null) {
				querySessionOptions.close();
				querySessionOptions = null;
			}
			loadedQueryModelPath = null;
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

			OrtEnvironment ortEnv = getOrCreateEnv();
			inputs.put("input_ids", OnnxTensor.createTensor(ortEnv, inputIdsArr));
			inputs.put("attention_mask", OnnxTensor.createTensor(ortEnv, attentionMaskArr));

			// Only provide token_type_ids if the model expects it (e.g.
			// all-MiniLM-L6-v2 requires it, but e5-base-v2 does not).
			java.util.Set<String> expectedInputs = ortSession.getInputNames();
			if (expectedInputs.contains("token_type_ids")) {
				long[][] tokenTypeIdsArr = { tokenized.getTokenTypeIds() };
				inputs.put("token_type_ids", OnnxTensor.createTensor(ortEnv, tokenTypeIdsArr));
			}

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

	/**
	 * Creates an ONNX session, handling models with external data files
	 * (model.onnx.data). ONNX runtime resolves external data relative to the
	 * model file path, which fails when the path contains ".." segments.
	 * This method canonicalizes the path first; if that still fails (runtime
	 * bug), it falls back to loading from a byte array.
	 */
	private OrtSession createSessionWithExternalData(OrtEnvironment ortEnv,
			String modelPath, OrtSession.SessionOptions options) throws OrtException {
		// Try canonical path first (handles ".." segments)
		String canonical;
		try {
			canonical = new java.io.File(modelPath).getCanonicalPath();
		} catch (java.io.IOException e) {
			canonical = modelPath;
		}
		try {
			return ortEnv.createSession(canonical, options);
		} catch (OrtException e) {
			// External data resolution failed — try loading as byte array.
			// This works because the byte array approach doesn't need to
			// resolve a .data file path. However, it requires the model
			// to be self-contained (no external data), so this only works
			// for models without external initializers.
			log.warn("ONNX session creation failed with path '{}', "
					+ "trying byte array fallback: {}", canonical, e.getMessage());
			try {
				byte[] modelBytes = java.nio.file.Files.readAllBytes(
						java.nio.file.Paths.get(canonical));
				return ortEnv.createSession(modelBytes, options);
			} catch (java.io.IOException | OrtException fallbackE) {
				// Byte array fallback also failed — throw original error
				throw e;
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
			sessionOptions = new OrtSession.SessionOptions();
			EnumSet<OrtProvider> available = OrtEnvironment.getAvailableProviders();
			log.info("Available ONNX execution providers: {}", available);
			if (available.contains(OrtProvider.CORE_ML)) {
				sessionOptions.addCoreML();
				activeProvider = OrtProvider.CORE_ML;
				log.info("CoreML execution provider enabled (GPU acceleration)");
			} else if (available.contains(OrtProvider.CUDA)) {
				sessionOptions.addCUDA();
				activeProvider = OrtProvider.CUDA;
				log.info("CUDA execution provider enabled (GPU acceleration)");
			}
			session = createSessionWithExternalData(ortEnv, modelPath, sessionOptions);
			loadedModelPath = modelPath;
			log.info("ONNX embedding model loaded successfully");
		}
		return session;
	}

	/**
	 * Returns the query-encoder session for dual-encoder models, or null
	 * for single-encoder models. Lazily loads the query model on first call.
	 */
	private synchronized OrtSession getQuerySession() throws OrtException {
		String queryModelPath;
		if (explicitQueryModelPath != null) {
			queryModelPath = explicitQueryModelPath;
		} else {
			try {
				String configuredPath = Context.getAdministrationService()
						.getGlobalProperty(
								ChartSearchAiConstants.GP_EMBEDDING_QUERY_MODEL_FILE_PATH);
				if (configuredPath == null || configuredPath.trim().isEmpty()) {
					return null; // Single-encoder mode
				}
				queryModelPath = ChartSearchAiUtils.resolveModelPath(
						configuredPath.trim(),
						ChartSearchAiConstants.GP_EMBEDDING_QUERY_MODEL_FILE_PATH);
			}
			catch (Exception e) {
				return null; // Context not available — single-encoder fallback
			}
		}

		if (querySession != null && !queryModelPath.equals(loadedQueryModelPath)) {
			log.info("Query model path changed, reloading");
			if (querySession != null) {
				try { querySession.close(); } catch (OrtException e) { /* ignore */ }
			}
			if (querySessionOptions != null) {
				querySessionOptions.close();
			}
			querySession = null;
		}

		if (querySession == null) {
			log.info("Loading ONNX query encoder from {}", queryModelPath);
			OrtEnvironment ortEnv = getOrCreateEnv();
			querySessionOptions = new OrtSession.SessionOptions();
			// Use CPU for query encoder to avoid CoreML/CUDA conflicts
			// with the article encoder session. Query inference is fast
			// enough (~1ms) that GPU acceleration isn't needed.
			querySession = createSessionWithExternalData(ortEnv, queryModelPath,
					querySessionOptions);
			loadedQueryModelPath = queryModelPath;
			log.info("ONNX query encoder loaded successfully");
		}
		return querySession;
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
