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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;

/**
 * Phase 0 feasibility probe for "whitening" the dual-encoder embeddings.
 * Tests whether subtracting the centroid of a generic-clinical reference
 * set from both query and record embeddings cleanly separates the 5 known
 * false positives from the rights — without crossing the boundary cases
 * that all prior approaches choked on.
 *
 * <p>Tagged "eval" so it doesn't run by default. Invoke with:</p>
 * <pre>mvn -pl api test -Dtest=WhiteningFeasibilityProbeTest \
 *     -Dsurefire.excludedGroups=</pre>
 */
@Tag("eval")
public class WhiteningFeasibilityProbeTest {

	private static final String MODELS_BASE = System.getProperty(
			"chartsearchai.models.base", "../../models");

	/** A diverse reference set of 50 generic clinical concept names spanning
	 *  multiple classes (Diagnosis, Drug, Finding, Procedure, Lab, Symptom,
	 *  Anatomy). Drawn so as not to overlap with the live patient's specific
	 *  conditions or with any of the 12 diagnostic queries — these are what
	 *  defines the "generic clinical query" centroid. */
	private static final String[] REFERENCE_SET = {
			"Hypertension", "Diabetes mellitus", "Asthma",
			"Pneumonia", "Bronchitis", "Hepatitis", "Malaria",
			"HIV infection", "Diarrhea", "Vomiting", "Nausea",
			"Constipation", "Insomnia", "Anxiety", "Schizophrenia",
			"Migraine", "Dermatitis", "Eczema", "Psoriasis",
			"Acne", "Conjunctivitis", "Otitis media", "Pharyngitis",
			"Tonsillitis", "Sinusitis", "Pneumothorax", "Appendicitis",
			"Cholecystitis", "Pancreatitis", "Gastritis",
			"Paracetamol", "Ibuprofen", "Aspirin", "Metformin",
			"Insulin", "Amoxicillin", "Penicillin", "Ciprofloxacin",
			"Complete blood count", "Urinalysis", "Chest X-ray",
			"Electrocardiogram", "Glucose level", "Cholesterol level",
			"Vaccination administration", "Suture removal",
			"Wound dressing change", "Catheter insertion",
			"Cervical biopsy", "Colonoscopy",
	};

	/** (label, query, candidate-record-text) triples — same as cross-encoder
	 *  probe, so we can compare. */
	private static final String[][] PAIRS = {
		// 5 KNOWN WRONGS
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

		// Marginal PASSES (lowest-cosine ones)
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

		// 3 HIGH-CONFIDENCE PASSES (sanity)
		{"PASS", "stroke",
		 "Condition: Nonparalytic stroke (active)"},
		{"PASS", "depression",
		 "Diagnosis: Mild depressive episode"},
		{"PASS", "hair loss",
		 "Condition: Scarring Alopecia (active)"},
	};

	@Test
	public void probeMedCpt() throws Exception {
		File base = new File(MODELS_BASE, "MedCPT").getCanonicalFile();
		File articleModel = new File(base, "Article-Encoder/model-merged.onnx");
		if (!articleModel.exists()) {
			System.out.println("MedCPT not available — skipping");
			return;
		}
		System.out.println("\n=== MedCPT whitening probe ===");
		String articlePath = articleModel.getAbsolutePath();
		String queryPath = new File(base, "Query-Encoder/model-merged.onnx").getAbsolutePath();
		String vocabPath = new File(base, "Query-Encoder/vocab.txt").getAbsolutePath();
		OnnxEmbeddingProvider provider = new OnnxEmbeddingProvider(
				articlePath, queryPath, vocabPath);
		try {
			runProbe(provider);
		} finally {
			provider.close();
		}
	}

	@Test
	public void probeL6V2() throws Exception {
		File base = new File(MODELS_BASE, "all-MiniLM-L6-v2").getCanonicalFile();
		File modelFile = new File(base, "model.onnx");
		if (!modelFile.exists()) {
			System.out.println("L6-v2 not available — skipping");
			return;
		}
		System.out.println("\n=== L6-v2 whitening probe ===");
		String modelPath = modelFile.getAbsolutePath();
		String vocabPath = new File(base, "vocab.txt").getAbsolutePath();
		OnnxEmbeddingProvider provider = new OnnxEmbeddingProvider(modelPath, vocabPath);
		try {
			runProbe(provider);
		} finally {
			provider.close();
		}
	}

	private void runProbe(OnnxEmbeddingProvider provider) {
		// For dual-encoder models (MedCPT), reference clinical names are
		// queries — embed them with the query encoder. For symmetric models
		// (L6-v2) embedQuery falls back to embed.
		int dim = -1;
		double[] sumQuery = null;
		double[] sumDoc = null;
		for (String name : REFERENCE_SET) {
			float[] q = provider.embedQuery(name);
			float[] d = provider.embed(name);
			if (sumQuery == null) {
				dim = q.length;
				sumQuery = new double[dim];
				sumDoc = new double[dim];
			}
			for (int i = 0; i < dim; i++) {
				sumQuery[i] += q[i];
				sumDoc[i] += d[i];
			}
		}
		float[] muQuery = new float[dim];
		float[] muDoc = new float[dim];
		for (int i = 0; i < dim; i++) {
			muQuery[i] = (float) (sumQuery[i] / REFERENCE_SET.length);
			muDoc[i] = (float) (sumDoc[i] / REFERENCE_SET.length);
		}
		System.out.printf("  Centroid computed from %d reference names (dim=%d)%n",
				REFERENCE_SET.length, dim);

		System.out.printf("%-7s %8s %10s %10s | %-30s | %s%n",
				"label", "raw_cos", "cent_cos", "delta", "query", "record");
		System.out.println(repeat('-', 130));
		for (String[] row : PAIRS) {
			String label = row[0];
			String query = row[1];
			String record = row[2];
			float[] q = provider.embedQuery(query);
			float[] r = provider.embed(record);
			double rawCos = ChartSearchAiUtils.cosineSimilarity(q, r);
			double centCos = ChartSearchAiUtils.cosineSimilarity(subtract(q, muQuery), subtract(r, muDoc));
			System.out.printf("%-7s %8.4f %10.4f %10.4f | %-30s | %s%n",
					label, rawCos, centCos, centCos - rawCos,
					truncate(query, 30), truncate(record, 50));
		}
	}

	private static float[] subtract(float[] a, float[] b) {
		float[] out = new float[a.length];
		for (int i = 0; i < a.length; i++) out[i] = a[i] - b[i];
		return out;
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
