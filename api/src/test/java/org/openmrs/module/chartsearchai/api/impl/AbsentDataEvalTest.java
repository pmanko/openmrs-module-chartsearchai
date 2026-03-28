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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
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

	@BeforeAll
	static void loadDataset() throws IOException {
		dataset = EvalDataset.load("eval/absent-data-eval-dataset.json");
	}

	@Test
	public void absentDataDetection_shouldNameMissingDataType() {
		int passed = 0;
		int total = 0;

		for (EvalCase evalCase : dataset.getCases()) {
			if (!evalCase.isExpectedAbsent()) {
				continue;
			}
			total++;

			String answer = LlmInferenceService.buildNoMatchAnswer(evalCase.getQuestion());

			log.info("[{}] question='{}' answer='{}'", evalCase.getId(),
					evalCase.getQuestion(), answer);

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
					assertFalse(answer.toLowerCase().contains(banned.toLowerCase()),
							evalCase.getId() + ": answer should not contain '" + banned + "'");
				}
			}

			if (allContained) {
				passed++;
			}
		}

		log.info("Absent-data eval: {}/{} cases passed", passed, total);
		assertTrue(passed >= total * 0.8,
				"At least 80% of absent-data cases should name the missing data type, "
				+ "but only " + passed + "/" + total + " passed");
	}

	@Test
	public void presentData_shouldNotProduceAbsentAnswer() {
		for (EvalCase evalCase : dataset.getCases()) {
			if (evalCase.isExpectedAbsent()) {
				continue;
			}

			String stripped = LlmInferenceService.stripQueryStopwords(evalCase.getQuestion());

			log.info("[{}] question='{}' stripped='{}'",
					evalCase.getId(), evalCase.getQuestion(), stripped);

			assertFalse(stripped.isEmpty(),
					evalCase.getId() + ": question should have searchable terms");
		}
	}
}
