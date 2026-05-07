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

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.embedding.WordPieceTokenizer.TokenizedInput;

/**
 * Phase 0 feasibility probe for cross-encoder reranking. Loads the MedCPT
 * cross-encoder directly and scores 12 diagnostic (query, record-text)
 * pairs to verify that scores cleanly separate true positives from
 * false positives.
 *
 * <p>Tagged "eval" so it doesn't run by default. Invoke with:</p>
 * <pre>mvn -pl api test -Dtest=CrossEncoderFeasibilityProbeTest \
 *     -Dsurefire.excludedGroups=</pre>
 */
@Tag("eval")
public class CrossEncoderFeasibilityProbeTest {

	private static final String MODELS_BASE = System.getProperty(
			"chartsearchai.models.base", "../../models");

	/** Pairs to probe: (label, query, candidate-record-text). The text is
	 *  modelled after what chartsearchai serializes — a short tag plus the
	 *  concept name. The actual production text varies but the concept
	 *  name dominates relevance scoring. */
	private static final String[][] PAIRS = {
		// 5 KNOWN WRONGS (filter currently returns these — should score low)
		{"WRONG", "TB treatment history",
		 "Condition: Personal history of blood transfusion (active)"},
		{"WRONG", "any cough?",
		 "Diagnosis: Chronic fatigue"},
		{"WRONG", "any cough?",
		 "Clinical observation: Respiratory rate 22 breaths/min"},
		{"WRONG", "azithromycin",
		 "Condition: Syphilitic Cirrhosis (active)"},
		{"WRONG", "dental problems",
		 "Condition: Female infertility (active)"},
		{"WRONG", "headache",
		 "Diagnosis: Chronic fatigue"},

		// 7 KNOWN PASSES with no kwTerm overlap (cosine ≈ wrongs)
		{"PASS", "fever",
		 "Clinical observation: Temperature (c) 37.5"},
		{"PASS", "wasting",
		 "Clinical observation: Weight (kg) 14.5"},
		{"PASS", "skin rash",
		 "Condition: Granuloma annulare (active)"},
		{"PASS", "skin rash",
		 "Condition: Scarring Alopecia (active)"},
		{"PASS", "liver problems",
		 "Condition: Syphilitic Cirrhosis (active)"},
		{"PASS", "is the patient anemic?",
		 "Condition: Personal history of blood transfusion (active)"},
		{"PASS", "any fracture?",
		 "Condition: Complete tear of ligament of ankle or foot (active)"},
		{"PASS", "mental health",
		 "Diagnosis: Mild depressive episode"},
		{"PASS", "any STD?",
		 "Condition: Syphilitic Cirrhosis (active)"},

		// 3 KNOWN HIGH-CONFIDENCE PASSES (sanity controls)
		{"PASS", "stroke",
		 "Condition: Nonparalytic stroke (active)"},
		{"PASS", "depression",
		 "Diagnosis: Mild depressive episode"},
		{"PASS", "hair loss",
		 "Condition: Scarring Alopecia (active)"},
	};

	@Test
	public void probeMedCptCrossEncoder() throws Exception {
		File modelDir = new File(MODELS_BASE, "MedCPT-Cross-Encoder")
				.getCanonicalFile();
		File modelFile = new File(modelDir, "model.onnx");
		if (!modelFile.exists()) {
			System.out.println("MedCPT cross-encoder not available at "
					+ modelDir + " — skipping");
			return;
		}
		System.out.println("\n=== MedCPT-Cross-Encoder probe ===");
		runProbe(modelFile);
	}

	@Test
	public void probeMsMarcoCrossEncoder() throws Exception {
		File modelDir = new File(MODELS_BASE, "cross-encoder-ms-marco-MiniLM-L-6-v2")
				.getCanonicalFile();
		File modelFile = new File(modelDir, "model.onnx");
		if (!modelFile.exists()) {
			System.out.println("MS-MARCO cross-encoder not available at "
					+ modelDir + " — skipping");
			return;
		}
		System.out.println("\n=== MS-MARCO-MiniLM-L-6-v2 cross-encoder probe ===");
		runProbe(modelFile);
	}

	private void runProbe(File modelFile) throws OrtException, IOException {
		String modelPath = modelFile.getAbsolutePath();
		String vocabPath = new File(modelFile.getParentFile(), "vocab.txt").getAbsolutePath();

		OrtEnvironment env = OrtEnvironment.getEnvironment();
		try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
				OrtSession session = env.createSession(modelPath, opts)) {
			System.out.println("Inputs:  " + session.getInputNames());
			System.out.println("Outputs: " + session.getOutputNames());

			// Cross-encoders trained on MS-MARCO use seq len 512;
			// the BERT base limit is 512. Use 256 for speed.
			WordPieceTokenizer tokenizer = new WordPieceTokenizer(vocabPath, 512);
			boolean usesTokenTypeIds = session.getInputNames().contains("token_type_ids");

			System.out.printf("%-7s %8s | %-30s | %s%n",
					"label", "score", "query", "record");
			System.out.println(repeat('-', 110));
			for (String[] row : PAIRS) {
				String label = row[0];
				String query = row[1];
				String record = row[2];
				double score = scorePair(env, session, tokenizer, usesTokenTypeIds, query, record);
				System.out.printf("%-7s %8.4f | %-30s | %s%n",
						label, score, truncate(query, 30), truncate(record, 60));
			}
		}
	}

	private double scorePair(OrtEnvironment env, OrtSession session,
			WordPieceTokenizer tokenizer, boolean usesTokenTypeIds,
			String query, String document)
			throws OrtException {
		TokenizedInput tok = tokenizer.tokenizePair(query, document);
		long[][] inputIds = { tok.getInputIds() };
		long[][] attentionMask = { tok.getAttentionMask() };
		long[][] tokenTypeIds = { tok.getTokenTypeIds() };

		Map<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();
		try {
			inputs.put("input_ids", OnnxTensor.createTensor(env, inputIds));
			inputs.put("attention_mask",
					OnnxTensor.createTensor(env, attentionMask));
			if (usesTokenTypeIds) {
				inputs.put("token_type_ids",
						OnnxTensor.createTensor(env, tokenTypeIds));
			}
			try (OrtSession.Result result = session.run(inputs)) {
				Object out = result.get(0).getValue();
				// BertForSequenceClassification: shape [batch, num_labels].
				// MedCPT cross-encoder has num_labels=1; MS-MARCO is also 1.
				if (out instanceof float[][]) {
					float[][] arr = (float[][]) out;
					return arr[0][0];
				}
				if (out instanceof float[][][]) {
					float[][][] arr = (float[][][]) out;
					return arr[0][0][0];
				}
				throw new RuntimeException("Unexpected output shape: " + out.getClass());
			}
		} finally {
			for (OnnxTensor t : inputs.values()) {
				try { t.close(); } catch (Exception ignored) {}
			}
		}
	}

	private static String truncate(String s, int max) {
		if (s == null || s.length() <= max) return s;
		return s.substring(0, max - 1) + "…";
	}

	private static String repeat(char c, int n) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++) sb.append(c);
		return sb.toString();
	}
}
