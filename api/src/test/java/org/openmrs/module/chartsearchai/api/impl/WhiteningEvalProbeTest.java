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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider;

/**
 * Validates whether centered-cosine ("whitening") would break any of the
 * 497 MedCPT eval baseline cases. For each baseline (datasetIndex, query,
 * resultIndices), measures the centered cosine of every expected record
 * and reports the minimum — a threshold below this minimum is "safe".
 *
 * <p>Tagged "eval" so excluded by default. Invoke with:</p>
 * <pre>mvn -pl api test -Dtest=WhiteningEvalProbeTest \
 *     -Dsurefire.excludedGroups= -Dchartsearchai.eval.model=medcpt</pre>
 */
@Tag("eval")
public class WhiteningEvalProbeTest {

	private static final String MODELS_BASE = System.getProperty(
			"chartsearchai.models.base", "../../models");

	/** Same reference set as the live-patient probe — keeps the centroid
	 *  consistent across probes. */
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

	@Test
	public void probeAgainstFullMedCptEval() throws Exception {
		File base = new File(MODELS_BASE, "MedCPT").getCanonicalFile();
		File articleModel = new File(base, "Article-Encoder/model-merged.onnx");
		if (!articleModel.exists()) {
			System.out.println("MedCPT not available — skipping");
			return;
		}
		System.out.println("\n=== Whitening eval probe (MedCPT) ===");
		String articlePath = articleModel.getAbsolutePath();
		String queryPath = new File(base, "Query-Encoder/model-merged.onnx").getAbsolutePath();
		String vocabPath = new File(base, "Query-Encoder/vocab.txt").getAbsolutePath();
		OnnxEmbeddingProvider provider = new OnnxEmbeddingProvider(
				articlePath, queryPath, vocabPath);
		try {
			run(provider, "src/test/resources/eval/medcpt-retrieval-eval.json");
		} finally {
			provider.close();
		}
	}

	private void run(OnnxEmbeddingProvider provider, String baselineRelPath)
			throws IOException {
		System.out.println("Computing centroid from " + REFERENCE_SET.length + " refs...");
		long t0 = System.currentTimeMillis();
		float[] muQuery = null, muDoc = null;
		double[] sumQ = null, sumD = null;
		int dim = -1;
		for (String name : REFERENCE_SET) {
			float[] q = provider.embedQuery(name);
			float[] d = provider.embed(name);
			if (sumQ == null) { dim = q.length; sumQ = new double[dim]; sumD = new double[dim]; }
			for (int i = 0; i < dim; i++) { sumQ[i] += q[i]; sumD[i] += d[i]; }
		}
		muQuery = new float[dim]; muDoc = new float[dim];
		for (int i = 0; i < dim; i++) {
			muQuery[i] = (float) (sumQ[i] / REFERENCE_SET.length);
			muDoc[i]   = (float) (sumD[i] / REFERENCE_SET.length);
		}
		System.out.printf("  Centroid done in %d ms (dim=%d)%n",
				System.currentTimeMillis() - t0, dim);

		// Match the actual indexing path by routing each record through
		// stripDatasetPrefixAndDate before embedding.
		String[][] datasets = {
				TestDatasetHelper.FULL_PATIENT_DATASET,
				TestDatasetHelper.SECOND_PATIENT_DATASET,
				TestDatasetHelper.THIRD_PATIENT_DATASET,
				TestDatasetHelper.FOURTH_PATIENT_DATASET,
				TestDatasetHelper.FIFTH_PATIENT_DATASET,
		};
		float[][][] datasetEmb = new float[datasets.length][][];
		System.out.println("Embedding records...");
		t0 = System.currentTimeMillis();
		int totalRecords = 0;
		for (int d = 0; d < datasets.length; d++) {
			float[][] e = new float[datasets[d].length][];
			for (int i = 0; i < datasets[d].length; i++) {
				String text = TestDatasetHelper.stripDatasetPrefixAndDate(datasets[d][i]);
				e[i] = provider.embed(text);
				totalRecords++;
			}
			datasetEmb[d] = e;
		}
		System.out.printf("  Embedded %d records in %d ms%n",
				totalRecords, System.currentTimeMillis() - t0);

		ObjectMapper mapper = new ObjectMapper();
		JsonNode root = mapper.readTree(Files.newInputStream(Paths.get(baselineRelPath)));
		JsonNode cases = root.get("cases");
		System.out.printf("Loaded %d baseline cases%n", cases.size());

		List<Double> minPerCase = new ArrayList<Double>();
		Map<String, float[]> queryEmbCache = new HashMap<String, float[]>();
		double absoluteMin = 1.0;
		String absoluteMinCase = null;
		int tooLow = 0;     // expected records below 0.15
		int veryLow = 0;    // below 0.10
		int below005 = 0;
		List<String> belowThresholdSamples = new ArrayList<String>();

		for (int c = 0; c < cases.size(); c++) {
			JsonNode caseNode = cases.get(c);
			int dsIdx = caseNode.get("datasetIndex").asInt();
			String query = caseNode.get("query").asText();
			JsonNode indices = caseNode.get("resultIndices");
			if (indices == null || !indices.isArray() || indices.size() == 0) continue;

			float[] q = queryEmbCache.computeIfAbsent(query, provider::embedQuery);
			float[] qC = subtract(q, muQuery);

			double caseMin = 1.0;
			for (JsonNode idx : indices) {
				int recIdx = idx.asInt();
				if (recIdx < 0 || recIdx >= datasetEmb[dsIdx].length) continue;
				float[] r = datasetEmb[dsIdx][recIdx];
				float[] rC = subtract(r, muDoc);
				double cos = ChartSearchAiUtils.cosineSimilarity(qC, rC);
				if (cos < caseMin) caseMin = cos;
				if (cos < 0.15) tooLow++;
				if (cos < 0.10) veryLow++;
				if (cos < 0.05) below005++;
			}
			minPerCase.add(caseMin);
			if (caseMin < absoluteMin) {
				absoluteMin = caseMin;
				absoluteMinCase = "ds=" + dsIdx + " q='" + query + "'";
			}
			if (caseMin < 0.15) {
				belowThresholdSamples.add(String.format(
						"  caseMin=%.4f  ds=%d  q='%s'  (%d expected records)",
						caseMin, dsIdx, query, indices.size()));
			}
		}

		java.util.Collections.sort(minPerCase);
		System.out.println();
		System.out.println("=== centered-cosine MIN-per-case distribution ===");
		System.out.printf("  cases analyzed: %d%n", minPerCase.size());
		System.out.printf("  expected-record samples below 0.05: %d%n", below005);
		System.out.printf("  expected-record samples below 0.10: %d%n", veryLow);
		System.out.printf("  expected-record samples below 0.15: %d%n", tooLow);
		System.out.printf("  absolute MIN: %.4f at %s%n", absoluteMin, absoluteMinCase);
		System.out.printf("  10th-percentile case-min: %.4f%n",
				minPerCase.get(minPerCase.size() / 10));
		System.out.printf("  50th-percentile case-min: %.4f%n",
				minPerCase.get(minPerCase.size() / 2));
		System.out.printf("  90th-percentile case-min: %.4f%n",
				minPerCase.get(minPerCase.size() * 9 / 10));

		// First 15 cases that would break at threshold 0.15
		System.out.println();
		System.out.println("Cases at risk if threshold = 0.15:");
		int shown = 0;
		for (String s : belowThresholdSamples) {
			System.out.println(s);
			if (++shown >= 15) {
				System.out.printf("  ...and %d more%n", belowThresholdSamples.size() - 15);
				break;
			}
		}

		// Show the safe threshold range
		System.out.println();
		System.out.println("=== Safe threshold candidates ===");
		double[] candidates = {0.05, 0.08, 0.10, 0.12, 0.15};
		for (double t : candidates) {
			int wouldBreak = 0;
			for (double m : minPerCase) if (m < t) wouldBreak++;
			System.out.printf("  threshold %.2f → would break %d / %d baseline cases%n",
					t, wouldBreak, minPerCase.size());
		}
	}

	private static float[] subtract(float[] a, float[] b) {
		float[] o = new float[a.length];
		for (int i = 0; i < a.length; i++) o[i] = a[i] - b[i];
		return o;
	}
}
