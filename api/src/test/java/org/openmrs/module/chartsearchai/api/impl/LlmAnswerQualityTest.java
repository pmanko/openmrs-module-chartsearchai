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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;

import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Tests LLM answer quality for multi-concept trend queries by running the
 * actual GGUF model against the 28-record vitals dataset. Skipped automatically
 * when the model file is not found.
 *
 * <p>Uses a single LLM call with all 28 records and iterates through system
 * prompt variations to find the best citation coverage. Citations are collected
 * from both the answer text and the JSON citations array.</p>
 */
public class LlmAnswerQualityTest {

	// 28 records grouped by concept, matching production pipeline output.
	private static final String RECORDS =
			"Patient: 16-year-old Male\n\n"
			+ "[1] (2025-09-17) Test — Systolic Blood Pressure: 97.0 mmHg\n"
			+ "[2] (2025-06-29) Test — Systolic Blood Pressure: 122.0 mmHg\n"
			+ "[3] (2025-06-28) Test — Systolic Blood Pressure: 101.0 mmHg\n"
			+ "[4] (2023-04-30) Test — Systolic Blood Pressure: 123.0 mmHg\n"
			+ "[5] (2022-10-07) Test — Systolic Blood Pressure: 137.0 mmHg\n"
			+ "[6] (2022-01-16) Test — Systolic Blood Pressure: 147.0 mmHg\n"
			+ "[7] (2021-11-30) Test — Systolic Blood Pressure: 98.0 mmHg\n"
			+ "[8] (2025-09-17) Test — Diastolic Blood Pressure: 99.0 mmHg\n"
			+ "[9] (2025-09-17) Test — Diastolic Blood Pressure: 92.0 mmHg\n"
			+ "[10] (2025-06-28) Test — Diastolic Blood Pressure: 99.0 mmHg\n"
			+ "[11] (2023-04-29) Test — Diastolic Blood Pressure: 71.0 mmHg\n"
			+ "[12] (2022-10-07) Test — Diastolic Blood Pressure: 67.0 mmHg\n"
			+ "[13] (2022-04-27) Test — Diastolic Blood Pressure: 105.0 mmHg\n"
			+ "[14] (2022-01-16) Test — Diastolic Blood Pressure: 58.0 mmHg\n"
			+ "[15] (2025-10-30) Test — Weight (kg): 94.0 kg\n"
			+ "[16] (2025-09-17) Test — Weight (kg): 107.0 kg\n"
			+ "[17] (2025-09-17) Test — Weight (kg): 139.0 kg\n"
			+ "[18] (2023-04-29) Test — Weight (kg): 38.0 kg\n"
			+ "[19] (2022-10-07) Test — Weight (kg): 146.0 kg\n"
			+ "[20] (2022-01-16) Test — Weight (kg): 68.0 kg\n"
			+ "[21] (2021-11-30) Test — Weight (kg): 121.0 kg\n"
			+ "[22] (2025-10-30) Test — Temperature (C): 36.7 DEG C\n"
			+ "[23] (2025-09-17) Test — Temperature (C): 37.7 DEG C\n"
			+ "[24] (2025-06-29) Test — Temperature (C): 40.3 DEG C\n"
			+ "[25] (2022-10-07) Test — Temperature (C): 39.3 DEG C\n"
			+ "[26] (2022-01-16) Test — Temperature (C): 36.4 DEG C\n"
			+ "[27] (2021-11-30) Test — Temperature (C): 37.8 DEG C\n"
			+ "[28] (2021-09-16) Test — Temperature (C): 40.1 DEG C\n";

	private static final String QUESTION =
			"How have this patient's blood pressure, weight, and temperature "
			+ "trended across their last 7 visits?";

	private static final int TOTAL_RECORDS = 28;

	private static final int MIN_REQUIRED_CITATIONS = 20;

	private static final Set<String> REQUIRED_CONCEPTS = new HashSet<>(Arrays.asList(
			"Systolic Blood Pressure", "Diastolic Blood Pressure",
			"Weight", "Temperature"));

	private static final Pattern CITATION_PATTERN = Pattern.compile("\\[(\\d+)]");

	private static final String[] TEMPLATES_TO_TRY = { "llama3", "chatml" };

	private static final String LLM_MODEL_PATH = resolveLlmModelPath();

	private static String resolveLlmModelPath() {
		String explicit = System.getProperty("chartsearchai.llm.model.path");
		if (explicit != null) {
			return explicit;
		}
		String[] baseDirs = { "../models", "../../models" };
		for (String base : baseDirs) {
			java.io.File dir = new java.io.File(base);
			if (!dir.isDirectory()) {
				continue;
			}
			java.io.File[] subdirs = dir.listFiles(java.io.File::isDirectory);
			if (subdirs == null) {
				continue;
			}
			for (java.io.File subdir : subdirs) {
				java.io.File[] ggufFiles = subdir.listFiles(
						(d, name) -> name.endsWith(".gguf"));
				if (ggufFiles != null && ggufFiles.length > 0) {
					java.io.File smallest = ggufFiles[0];
					for (java.io.File f : ggufFiles) {
						if (f.length() < smallest.length()) {
							smallest = f;
						}
					}
					return smallest.getPath();
				}
			}
		}
		return "../models/model.gguf";
	}

	private static boolean llmModelFileExists() {
		return new java.io.File(LLM_MODEL_PATH).exists();
	}

	private static String runLlm(LlamaModel model, String systemPrompt, String templateName) {
		String userMessage = "Patient records (grouped by type, most recent first within each group):\n"
				+ RECORDS + "\nQuestion: " + QUESTION;

		String grammar = null;
		try {
			java.io.InputStream is = LlmProvider.class.getClassLoader()
					.getResourceAsStream("json-answer.gbnf");
			if (is != null) {
				grammar = new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
			}
		} catch (java.io.IOException e) {
			// proceed without grammar
		}

		String template = LlmProvider.PRESET_TEMPLATES.get(templateName);
		String prompt = template.replace("{system}", systemPrompt).replace("{user}", userMessage);
		InferenceParameters params = new InferenceParameters(prompt)
				.setStopStrings(LlmProvider.resolveStopStrings(templateName))
				.setTemperature(0.0f)
				.setSeed(42)
				.setCachePrompt(false)
				.setRepeatPenalty(ChartSearchAiConstants.DEFAULT_REPEAT_PENALTY);
		if (grammar != null) {
			params.setGrammar(grammar);
		}

		StringBuilder result = new StringBuilder();
		long deadline = System.currentTimeMillis() + 120_000;
		for (LlamaOutput output : model.generate(params)) {
			if (System.currentTimeMillis() > deadline) {
				break;
			}
			result.append(output);
		}
		return result.toString();
	}

	private static Set<Integer> extractTextCitations(String answer) {
		Set<Integer> citations = new HashSet<>();
		Matcher m = CITATION_PATTERN.matcher(answer);
		while (m.find()) {
			citations.add(Integer.parseInt(m.group(1)));
		}
		return citations;
	}

	private static Set<String> findMentionedConcepts(String answer) {
		Set<String> found = new HashSet<>();
		String lower = answer.toLowerCase();
		if (lower.contains("systolic")) found.add("Systolic Blood Pressure");
		if (lower.contains("diastolic")) found.add("Diastolic Blood Pressure");
		if (lower.contains("weight")) found.add("Weight");
		if (lower.contains("temperature") || lower.contains("temp")) found.add("Temperature");
		return found;
	}

	/**
	 * Validates that a single LLM call with all 28 records achieves good
	 * citation coverage. Iterates through prompt and template variations,
	 * collecting citations from both answer text and JSON citations array.
	 */
	@Test
	public void trendQuery_shouldCiteRecordsAndCoverAllConcepts() {
		org.junit.jupiter.api.Assumptions.assumeTrue(llmModelFileExists(),
				"Skipping: LLM model file not found at " + LLM_MODEL_PATH);

		ModelParameters modelParams = new ModelParameters()
				.setModel(LLM_MODEL_PATH)
				.setGpuLayers(-1)
				.setCtxSize(8192);
		LlamaModel model = new LlamaModel(modelParams);

		try {
			List<String> promptVariations = buildPromptVariations();

			String bestConfig = null;
			Set<Integer> bestCitations = new HashSet<>();
			Set<String> bestConcepts = new HashSet<>();
			String bestAnswer = null;

			for (String templateName : TEMPLATES_TO_TRY) {
				for (int p = 0; p < promptVariations.size(); p++) {
					String rawResponse = runLlm(model, promptVariations.get(p), templateName);
					LlmProvider.LlmResponse parsed = LlmProvider.extractResponse(rawResponse);
					String answer = parsed.getAnswer();

					Set<Integer> allCitations = new HashSet<>();
					allCitations.addAll(extractTextCitations(answer));
					for (int c : parsed.getCitations()) {
						allCitations.add(c);
					}
					// Only count valid citations (1-28)
					allCitations.removeIf(c -> c < 1 || c > TOTAL_RECORDS);

					Set<String> concepts = findMentionedConcepts(answer);

					String config = "[" + templateName + "] Prompt " + (p + 1);
					System.out.println("\n=== " + config + ": "
							+ allCitations.size() + "/" + TOTAL_RECORDS
							+ " citations, " + concepts.size() + " concepts ===");
					System.out.println("Text citations: " + extractTextCitations(answer));
					System.out.println("JSON citations: " + parsed.getCitations());
					System.out.println("All unique: " + allCitations);

					if (allCitations.size() > bestCitations.size()
							|| (allCitations.size() == bestCitations.size()
								&& concepts.size() > bestConcepts.size())) {
						bestCitations = allCitations;
						bestConcepts = concepts;
						bestConfig = config;
						bestAnswer = answer;
					}
				}
			}

			System.out.println("\n=== BEST: " + bestConfig + " ("
					+ bestCitations.size() + "/" + TOTAL_RECORDS + ") ===");
			System.out.println("Citations: " + bestCitations);
			System.out.println("Concepts: " + bestConcepts);
			System.out.println("Answer: " + bestAnswer);

			// All 4 concepts must be present
			for (String concept : REQUIRED_CONCEPTS) {
				assertTrue(bestConcepts.contains(concept),
						"Best answer should mention " + concept + ". Answer: " + bestAnswer);
			}

			// Minimum citation coverage
			Set<Integer> missing = new HashSet<>();
			for (int i = 1; i <= TOTAL_RECORDS; i++) {
				if (!bestCitations.contains(i)) {
					missing.add(i);
				}
			}
			assertTrue(bestCitations.size() >= MIN_REQUIRED_CITATIONS,
					"Best answer should cite at least " + MIN_REQUIRED_CITATIONS
					+ "/" + TOTAL_RECORDS + " records. Got: " + bestCitations.size()
					+ ". Missing: " + missing + ". Answer: " + bestAnswer);
		} finally {
			model.close();
		}
	}

	private static List<String> buildPromptVariations() {
		List<String> prompts = new ArrayList<>();

		// Variation 0: current production default
		prompts.add(LlmProvider.DEFAULT_SYSTEM_PROMPT);

		// Variation 1: anti-summarization
		prompts.add(LlmProvider.DEFAULT_SYSTEM_PROMPT.replace(
				"Keep your answer concise while including all relevant details.",
				"NEVER summarize multiple values into a range. List each measurement individually "
				+ "with its date, value, and citation. Every record MUST appear in the answer."));

		// Variation 2: strongest directive with arrow format
		prompts.add(LlmProvider.DEFAULT_SYSTEM_PROMPT.replace(
				"Include the relevant details from the records in your answer "
				+ "and cite EVERY record you use by its number in brackets (e.g. [1], [3]). "
				+ "Keep your answer concise while including all relevant details.",
				"CRITICAL REQUIREMENT: You must cite EVERY record provided. "
				+ "For each record [N], include: the exact value, date, and [N] citation. "
				+ "Do NOT summarize ranges. Do NOT skip records. "
				+ "Use the arrow format: val (date) [N] \\u2192 val (date) [N] \\u2192 ..."));

		// Variation 3: format directive with oldest-to-newest ordering
		prompts.add(LlmProvider.DEFAULT_SYSTEM_PROMPT
				.replace("Answer ONLY the specific question asked.",
						"Answer ONLY the specific question asked. When the question asks about trends, "
						+ "you MUST list every single value from oldest to newest.")
				.replace("Include the relevant details from the records in your answer "
						+ "and cite EVERY record you use by its number in brackets (e.g. [1], [3]). "
						+ "Keep your answer concise while including all relevant details.",
						"Cite EVERY record by its [N] number. Your answer must contain ALL record numbers. "
						+ "Format: val (date) [N] \\u2192 val (date) [N] \\u2192 ..."));

		return prompts;
	}
}
