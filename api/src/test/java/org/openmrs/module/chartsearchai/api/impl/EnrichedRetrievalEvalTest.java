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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.embedding.OnnxEmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Eval harness that runs every test query through the enriched production
 * pipeline (with concept-set category hints) and asserts minimum result
 * counts. Catches cross-query regressions where improving one query
 * (e.g. STD) silently breaks another (e.g. vital signs).
 *
 * <p>Uses a golden baseline file ({@code enriched-retrieval-eval.json})
 * with per-query expected minimums. Any pipeline change that reduces
 * results below the minimum triggers a test failure, forcing the
 * developer to review and either fix the regression or update the
 * baseline with justification.</p>
 *
 * <p>Skipped when ONNX model files are not present.</p>
 */
public class EnrichedRetrievalEvalTest {

	private static final Logger log = LoggerFactory.getLogger(EnrichedRetrievalEvalTest.class);

	private static final String[] DATASET = TestDatasetHelper.FULL_PATIENT_DATASET;

	private static OnnxEmbeddingProvider provider;

	private static List<ChartEmbedding> allEmbeddings;

	private static List<SerializedRecord> allRecords;

	private static void ensureInitialized() {
		if (provider == null) {
			provider = new OnnxEmbeddingProvider(
					TestDatasetHelper.MODEL_PATH, TestDatasetHelper.VOCAB_PATH);
			allRecords = TestDatasetHelper.toSerializedRecords(
					DATASET, TestDatasetHelper.FULL_DATASET_CATEGORY_HINTS);
			allEmbeddings = EmbeddingIndexer.buildEmbeddings(allRecords, provider);
		}
	}

	private static List<Integer> runQuery(String query) {
		ensureInitialized();
		List<SerializedRecord> results = LlmInferenceService.findRelevantRecords(
				allEmbeddings, allRecords, provider, query, 100,
				ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX,
				LlmInferenceService.PipelineConfig.defaults());
		List<Integer> indices = new ArrayList<>();
		if (results != null) {
			for (SerializedRecord r : results) {
				indices.add(r.getResourceId());
			}
		}
		Collections.sort(indices);
		return indices;
	}

	// --- Golden baseline loading ---

	private static JsonNode loadBaseline() {
		try {
			return new ObjectMapper().readTree(
					new InputStreamReader(
							EnrichedRetrievalEvalTest.class.getClassLoader()
									.getResourceAsStream("eval/enriched-retrieval-eval.json"),
							StandardCharsets.UTF_8));
		} catch (IOException e) {
			throw new RuntimeException("Failed to load enriched eval baseline", e);
		}
	}

	static Stream<Arguments> evalCases() {
		JsonNode root = loadBaseline();
		JsonNode cases = root.get("cases");
		List<Arguments> args = new ArrayList<>();
		for (JsonNode c : cases) {
			args.add(Arguments.of(
					c.get("query").asText(),
					c.get("expectedMinRecords").asInt(),
					c.has("mustContainText") ? c.get("mustContainText").asText() : null));
		}
		return args.stream();
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("evalCases")
	public void enrichedRetrieval_shouldMeetBaseline(String query,
			int expectedMinRecords, String mustContainText) {
		Assumptions.assumeTrue(TestDatasetHelper.modelFilesExist(),
				"Skipping: ONNX model not found");

		List<Integer> result = runQuery(query);

		assertTrue(result.size() >= expectedMinRecords,
				"'" + query + "' should return >= " + expectedMinRecords
				+ " records but got " + result.size() + ": " + result);

		if (mustContainText != null) {
			boolean found = false;
			for (int idx : result) {
				if (DATASET[idx].contains(mustContainText)) {
					found = true;
					break;
				}
			}
			assertTrue(found,
					"'" + query + "' should return a record containing '"
					+ mustContainText + "' but none of " + result + " do");
		}
	}

	/**
	 * Generator: run all 97 queries and output the baseline JSON.
	 * Enable this test ONCE to generate the golden file, then disable.
	 */
	@Test
	@org.junit.jupiter.api.Disabled("Run manually to regenerate the golden baseline")
	public void generateBaseline() throws IOException {
		Assumptions.assumeTrue(TestDatasetHelper.modelFilesExist(),
				"Skipping: ONNX model not found");

		String[] QUERIES = {
			"Any family history of cancer?","Does this patient use any family planning methods?",
			"HIV status and CD4 count","Hb results",
			"How have this patient's blood pressure, weight, and temperature trended across their last 7 visits?",
			"I need to know the patient's most recent blood pressure and weight readings",
			"I want to review the patient's complete medication history including dosages",
			"TB","TB treatment history",
			"Were all lab orders placed for this patient resulted?",
			"What are this patient's HB results over time, and are values moving toward or away from the normal range?",
			"What is the current CD4 Count?","What is the latest CD4 Count?",
			"What is the patient allergic to?",
			"When were each of this patient's active conditions first recorded, and have any resolved or escalated?",
			"allergies","allergies and current medications","anaemia","anemia",
			"any STD?","any allergies?","any blood problems?","any cancer?",
			"any conditions","any conditions?","any cough?","any episodes?",
			"any fracture?","any history of cancer?","any infections?",
			"any known conditions?","any prescriptions?",
			"any sexually transmitted disease?","any sexually transmitted diseases?",
			"azithromycin","blood pressure and pulse","blood sugar","bowel disease",
			"breathing problems",
			"can you tell me about the patient's respiratory symptoms and oxygen levels",
			"cancer","cardiovascular disease","cardiovascular risk factors",
			"cholesterol","dental problems","depression","diabetes",
			"disease problem","disease treatment","does he have cancer?",
			"does she have any blood problem?","does the patient have any allergies",
			"does the patient have any allergies?","does the patient have cancer?",
			"ever been immunized?","ever got pregnant?","fever","fracture",
			"hair loss",
			"has this patient ever been diagnosed with any sexually transmitted infections",
			"has this patient had a sexually transmitted disease?",
			"headache","how hot is the patient?","immunization",
			"is the patient anemic?","is the patient enrolled in any programs?",
			"is the patient immunocompromised?","is the patient on any medications?",
			"is the patient tired?","kidney function","kidney problems",
			"latest vital signs","list all allergies","liver problems",
			"mental health","musculoskeletal injuries","nutritional status",
			"opportunistic infections in HIV","oxygen saturation",
			"patient denies chest pain","pregnancy-related concerns","pulse rate",
			"recent temperature readings","signs of infection or inflammation",
			"skin rash","stroke","substance abuse","vital signs","wasting",
			"what are the most recent two heart rates?",
			"what chronic conditions has this patient been diagnosed with over the years",
			"what is the current cd4 count and when was it recorded?",
			"what is the latest heart rate?","what is the latest weight?",
			"what is the patient being treated for?",
			"what medications is the patient on?",
			"what tests have been ordered for this patient",
		};

		ObjectMapper mapper = new ObjectMapper();
		com.fasterxml.jackson.databind.node.ObjectNode root = mapper.createObjectNode();
		root.put("dataset", "enriched-retrieval-v1");
		root.put("description", "Baseline for enriched pipeline with concept-set category hints on FULL_PATIENT_DATASET");
		com.fasterxml.jackson.databind.node.ArrayNode cases = root.putArray("cases");

		for (String q : QUERIES) {
			List<Integer> result = runQuery(q);
			com.fasterxml.jackson.databind.node.ObjectNode c = cases.addObject();
			c.put("query", q);
			c.put("expectedMinRecords", result.size());
			com.fasterxml.jackson.databind.node.ArrayNode ids = c.putArray("resultIndices");
			for (int idx : result) {
				ids.add(idx);
			}
		}

		java.io.File outFile = new java.io.File(
				"src/test/resources/eval/enriched-retrieval-eval.json");
		mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, root);
		log.info("Generated baseline: {} queries, file: {}",
				QUERIES.length, outFile.getAbsolutePath());
	}
}
