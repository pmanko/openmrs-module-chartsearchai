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
import org.junit.jupiter.api.Tag;
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
@Tag("eval")
public class EnrichedRetrievalEvalTest {

	private static final Logger log = LoggerFactory.getLogger(EnrichedRetrievalEvalTest.class);

	private static final String[][] DATASETS = {
		TestDatasetHelper.FULL_PATIENT_DATASET,
		TestDatasetHelper.SECOND_PATIENT_DATASET,
		TestDatasetHelper.THIRD_PATIENT_DATASET,
		TestDatasetHelper.FOURTH_PATIENT_DATASET,
		TestDatasetHelper.FIFTH_PATIENT_DATASET,
	};

	private static final String[] DATASET_NAMES = {
		"FULL", "SECOND", "THIRD", "FOURTH", "FIFTH"
	};

	@SuppressWarnings("unchecked")
	private static final java.util.Map<Integer, List<String>>[] DATASET_HINTS = new java.util.Map[] {
		TestDatasetHelper.FULL_DATASET_CATEGORY_HINTS,
		TestDatasetHelper.SECOND_DATASET_CATEGORY_HINTS,
		TestDatasetHelper.THIRD_DATASET_CATEGORY_HINTS,
		TestDatasetHelper.FOURTH_DATASET_CATEGORY_HINTS,
		TestDatasetHelper.FIFTH_DATASET_CATEGORY_HINTS,
	};

	private static OnnxEmbeddingProvider provider;

	private static org.openmrs.module.chartsearchai.embedding.EmbeddingProvider cachingProvider;

	private static List<ChartEmbedding>[] datasetEmbeddings;

	private static List<SerializedRecord>[] datasetRecords;

	private static LlmInferenceService.PipelineConfig[] datasetConfigs;

	private static String activeModel;

	private static String activeBaseline;

	private static boolean medcptAvailable() {
		try {
			String base = new java.io.File(System.getProperty(
					"chartsearchai.models.base", "../../models")
					+ "/MedCPT").getCanonicalPath();
			return new java.io.File(base + "/Article-Encoder/model-merged.onnx").exists()
					&& new java.io.File(base + "/Query-Encoder/model-merged.onnx").exists();
		} catch (java.io.IOException e) {
			return false;
		}
	}

	@SuppressWarnings("unchecked")
	private static void ensureInitialized() {
		if (provider != null) return;

		// Detect model: use MedCPT if available and requested via system
		// property, otherwise fall back to L6-v2.
		String modelChoice = System.getProperty(
				"chartsearchai.eval.model", "auto");
		boolean useMedCPT = false;
		if ("medcpt".equalsIgnoreCase(modelChoice)) {
			useMedCPT = true;
		} else if ("auto".equalsIgnoreCase(modelChoice)) {
			// Auto: use whatever is configured via the model dir property
			useMedCPT = false;
		}

		if (useMedCPT && medcptAvailable()) {
			try {
				String base = new java.io.File(System.getProperty(
						"chartsearchai.models.base", "../../models")
						+ "/MedCPT").getCanonicalPath();
				provider = new OnnxEmbeddingProvider(
						base + "/Article-Encoder/model-merged.onnx",
						base + "/Query-Encoder/model-merged.onnx",
						base + "/Query-Encoder/vocab.txt");
				activeModel = "medcpt";
				activeBaseline = "eval/medcpt-retrieval-eval.json";
			} catch (java.io.IOException e) {
				throw new RuntimeException(e);
			}
		} else {
			provider = new OnnxEmbeddingProvider(
					TestDatasetHelper.MODEL_PATH, TestDatasetHelper.VOCAB_PATH);
			activeModel = "l6-v2";
			activeBaseline = "eval/enriched-retrieval-eval.json";
		}

		cachingProvider = TestDatasetHelper.cachingProvider(provider);
		LlmInferenceService.PipelineConfig modelConfig =
				LlmInferenceService.PipelineConfig.forModel(activeModel);

		datasetEmbeddings = new List[DATASETS.length];
		datasetRecords = new List[DATASETS.length];
		datasetConfigs = new LlmInferenceService.PipelineConfig[DATASETS.length];
		for (int d = 0; d < DATASETS.length; d++) {
			datasetRecords[d] = TestDatasetHelper.toSerializedRecords(
					DATASETS[d], DATASET_HINTS[d]);
			datasetEmbeddings[d] = TestDatasetHelper.buildOrLoadCachedEmbeddings(
					datasetRecords[d], provider);
			org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile noise =
					TestDatasetHelper.buildOrLoadCachedNoiseProfile(
							datasetEmbeddings[d], provider);
			datasetConfigs[d] = new LlmInferenceService.PipelineConfig(
					modelConfig.keywordWeight, modelConfig.scoreGapMultiplier,
					modelConfig.minScoreGap, modelConfig.gapValidationCosineThreshold,
					modelConfig.similarityRatio, noise, modelConfig.floorRescueMinZScore);
		}
		log.info("Eval initialized with model={}, baseline={}", activeModel, activeBaseline);
	}

	private static List<Integer> runQuery(String query, int datasetIndex) {
		ensureInitialized();
		List<SerializedRecord> results = LlmInferenceService.findRelevantRecords(
				datasetEmbeddings[datasetIndex], datasetRecords[datasetIndex],
				cachingProvider, query, 100,
				ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX,
				datasetConfigs[datasetIndex]);
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
		ensureInitialized();
		try {
			return new ObjectMapper().readTree(
					new InputStreamReader(
							EnrichedRetrievalEvalTest.class.getClassLoader()
									.getResourceAsStream(activeBaseline),
							StandardCharsets.UTF_8));
		} catch (IOException e) {
			throw new RuntimeException("Failed to load eval baseline: " + activeBaseline, e);
		}
	}

	static Stream<Arguments> evalCases() {
		JsonNode root = loadBaseline();
		JsonNode cases = root.get("cases");
		List<Arguments> args = new ArrayList<>();
		for (JsonNode c : cases) {
			int dsIdx = c.has("datasetIndex") ? c.get("datasetIndex").asInt() : 0;
			String dsName = dsIdx < DATASET_NAMES.length ? DATASET_NAMES[dsIdx] : "DS" + dsIdx;
			int expectedMax = c.has("expectedMaxRecords")
					? c.get("expectedMaxRecords").asInt() : -1;
			args.add(Arguments.of(
					dsName + ": " + c.get("query").asText(),
					c.get("query").asText(),
					dsIdx,
					c.get("expectedMinRecords").asInt(),
					expectedMax,
					c.has("mustContainText") ? c.get("mustContainText").asText() : null));
		}
		return args.stream();
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("evalCases")
	public void enrichedRetrieval_shouldMeetBaseline(String label,
			String query, int datasetIndex,
			int expectedMinRecords, int expectedMaxRecords,
			String mustContainText) {
		ensureInitialized();
		Assumptions.assumeTrue(provider != null,
				"Skipping: embedding model not found");

		List<Integer> result = runQuery(query, datasetIndex);

		assertTrue(result.size() >= expectedMinRecords,
				"'" + label + "' should return >= " + expectedMinRecords
				+ " records but got " + result.size() + ": " + result);

		if (expectedMaxRecords >= 0) {
			assertTrue(result.size() <= expectedMaxRecords,
					"'" + label + "' should return <= " + expectedMaxRecords
					+ " records but got " + result.size() + ": " + result);
		}

		if (mustContainText != null) {
			boolean found = false;
			for (int idx : result) {
				if (DATASETS[datasetIndex][idx].contains(mustContainText)) {
					found = true;
					break;
				}
			}
			assertTrue(found,
					"'" + label + "' should return a record containing '"
					+ mustContainText + "' but none of " + result + " do");
		}
	}

	@Test
	public void physicalInjury_shouldReturnOnlyInjuryRecords() {
		ensureInitialized();
		Assumptions.assumeTrue(provider != null,
				"Skipping: embedding model not found");

		for (int ds = 0; ds < DATASETS.length; ds++) {
			List<Integer> result = runQuery("any physical injury?", ds);
			StringBuilder details = new StringBuilder();
			for (int idx : result) {
				if (idx < DATASETS[ds].length) {
					details.append("\n  [").append(idx).append("] ")
							.append(DATASETS[ds][idx]);
				}
			}
			log.warn("[{}] 'any physical injury?' returned {} records: {}{}",
					DATASET_NAMES[ds], result.size(), result, details);

			// FOURTH and FIFTH contain "Crushing injury of thigh" and
			// "Nonunion of fracture" — expect those and nothing else.
			if (ds == 3 || ds == 4) {
				assertTrue(result.contains(93),
						DATASET_NAMES[ds] + ": should contain Crushing injury condition [93]");
				assertTrue(result.contains(97),
						DATASET_NAMES[ds] + ": should contain Crushing injury diagnosis [97]");
				assertTrue(result.contains(108),
						DATASET_NAMES[ds] + ": should contain Nonunion of fracture condition [108]");
				assertTrue(result.contains(110),
						DATASET_NAMES[ds] + ": should contain Nonunion of fracture diagnosis [110]");
			} else {
				// FULL, SECOND, THIRD have no injury records
				assertTrue(result.isEmpty(),
						DATASET_NAMES[ds] + ": should return 0 records for 'any physical injury?' but got "
						+ result.size() + ": " + details);
			}
		}
	}

	@Test
	public void latestBmi_shouldReturnHeightAndWeightRecords() {
		ensureInitialized();
		Assumptions.assumeTrue(provider != null,
				"Skipping: embedding model not found");

		for (int ds = 0; ds < DATASETS.length; ds++) {
			List<Integer> result = runQuery("What is the latest BMI?", ds);
			StringBuilder details = new StringBuilder();
			for (int idx : result) {
				if (idx < DATASETS[ds].length) {
					details.append("\n  [").append(idx).append("] ")
							.append(DATASETS[ds][idx]);
				}
			}
			log.warn("[{}] 'What is the latest BMI?' returned {} records: {}{}",
					DATASET_NAMES[ds], result.size(), result, details);

			// BMI = weight / height². Every dataset has Height and Weight
			// records, so the pipeline should return both.
			boolean hasHeight = false;
			boolean hasWeight = false;
			for (int idx : result) {
				if (idx < DATASETS[ds].length) {
					String text = DATASETS[ds][idx];
					if (text.contains("Height")) {
						hasHeight = true;
					}
					if (text.contains("Weight")) {
						hasWeight = true;
					}
				}
			}
			assertTrue(hasWeight,
					DATASET_NAMES[ds] + ": BMI query should return Weight records, got: " + result);
			assertTrue(hasHeight,
					DATASET_NAMES[ds] + ": BMI query should return Height records, got: " + result);
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
		root.put("dataset", "enriched-retrieval-v2");
		root.put("description", "Baseline for enriched pipeline across all 5 patient datasets");
		com.fasterxml.jackson.databind.node.ArrayNode cases = root.putArray("cases");

		for (int d = 0; d < DATASETS.length; d++) {
			for (String q : QUERIES) {
				List<Integer> result = runQuery(q, d);
				com.fasterxml.jackson.databind.node.ObjectNode c = cases.addObject();
				c.put("query", q);
				c.put("datasetIndex", d);
				c.put("expectedMinRecords", result.size());
				com.fasterxml.jackson.databind.node.ArrayNode ids = c.putArray("resultIndices");
				for (int idx : result) {
					ids.add(idx);
				}
			}
		}

		java.io.File outFile = new java.io.File(
				"src/test/resources/eval/enriched-retrieval-eval.json");
		mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, root);
		log.info("Generated baseline: {} queries, file: {}",
				QUERIES.length, outFile.getAbsolutePath());
	}
}
