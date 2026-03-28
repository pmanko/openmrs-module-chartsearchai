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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
import org.openmrs.module.chartsearchai.eval.EvalReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Eval suite for absent-data detection. Tests that when retrieval finds
 * no matching records, the system produces a clear "no records" answer
 * naming what was asked about.
 */
public class AbsentDataEvalTest {

	private static final Logger log = LoggerFactory.getLogger(AbsentDataEvalTest.class);

	private static EvalDataset dataset;

	private static EvalDataset getDataset() {
		if (dataset == null) {
			try {
				dataset = EvalDataset.load("eval/absent-data-eval-dataset.json");
			}
			catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return dataset;
	}

	static Stream<Arguments> absentCases() {
		List<Arguments> args = new ArrayList<>();
		for (EvalCase evalCase : getDataset().getCases()) {
			if (evalCase.isExpectedAbsent()) {
				args.add(Arguments.of(evalCase.getId(), evalCase));
			}
		}
		return args.stream();
	}

	static Stream<Arguments> presentCases() {
		List<Arguments> args = new ArrayList<>();
		for (EvalCase evalCase : getDataset().getCases()) {
			if (!evalCase.isExpectedAbsent()) {
				args.add(Arguments.of(evalCase.getId(), evalCase));
			}
		}
		return args.stream();
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("absentCases")
	public void absentDataDetection_perCase(String caseId, EvalCase evalCase) {
		String answer = LlmInferenceService.buildNoMatchAnswer(evalCase.getQuestion());

		log.info("[{}] question='{}' answer='{}'", caseId,
				evalCase.getQuestion(), answer);

		if (evalCase.getExpectedAnswerContains() != null) {
			for (String expected : evalCase.getExpectedAnswerContains()) {
				assertTrue(answer.toLowerCase().contains(expected.toLowerCase()),
						caseId + ": answer should contain '" + expected + "' but was: " + answer);
			}
		}

		if (evalCase.getExpectedAnswerNotContains() != null) {
			for (String banned : evalCase.getExpectedAnswerNotContains()) {
				assertFalse(answer.toLowerCase().contains(banned.toLowerCase()),
						caseId + ": answer should not contain '" + banned + "'");
			}
		}
	}

	@ParameterizedTest(name = "[{index}] {0}")
	@MethodSource("presentCases")
	public void presentData_perCase(String caseId, EvalCase evalCase) {
		String stripped = LlmInferenceService.stripQueryStopwords(evalCase.getQuestion());

		log.info("[{}] question='{}' stripped='{}'",
				caseId, evalCase.getQuestion(), stripped);

		assertFalse(stripped.isEmpty(),
				caseId + ": question should have searchable terms");
	}

	@Test
	public void absentDataDetection_shouldMeetAggregateThreshold() {
		int passed = 0;
		int total = 0;

		for (EvalCase evalCase : getDataset().getCases()) {
			if (!evalCase.isExpectedAbsent()) {
				continue;
			}
			total++;

			String answer = LlmInferenceService.buildNoMatchAnswer(evalCase.getQuestion());

			boolean allContained = true;
			if (evalCase.getExpectedAnswerContains() != null) {
				for (String expected : evalCase.getExpectedAnswerContains()) {
					if (!answer.toLowerCase().contains(expected.toLowerCase())) {
						log.warn("[{}] MISSING expected term '{}' in answer: {}",
								evalCase.getId(), expected, answer);
						allContained = false;
					}
				}
			}

			if (evalCase.getExpectedAnswerNotContains() != null) {
				for (String banned : evalCase.getExpectedAnswerNotContains()) {
					if (answer.toLowerCase().contains(banned.toLowerCase())) {
						allContained = false;
					}
				}
			}

			if (allContained) {
				passed++;
			}

			Map<String, Object> metrics = new LinkedHashMap<>();
			metrics.put("passed", allContained);
			EvalReporter.appendResult("absent-data", evalCase.getId(), metrics);
		}

		log.info("Absent-data eval: {}/{} cases passed", passed, total);

		Map<String, Object> summary = new LinkedHashMap<>();
		summary.put("passed", passed);
		summary.put("total", total);
		summary.put("rate", total > 0 ? String.format("%.3f", (double) passed / total) : "0");
		EvalReporter.appendSummary("absent-data", summary);

		assertTrue(passed >= total * 0.8,
				"At least 80% of absent-data cases should name the missing data type, "
				+ "but only " + passed + "/" + total + " passed");
	}
}
