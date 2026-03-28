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
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmrs.module.chartsearchai.api.ChartSearchService.RecordReference;
import org.openmrs.module.chartsearchai.eval.EvalCase;
import org.openmrs.module.chartsearchai.eval.EvalDataset;
import org.openmrs.module.chartsearchai.eval.EvalMetrics;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Eval suite for citation accuracy. Tests that the LLM response parsing
 * pipeline ({@code extractResponse} + {@code extractCitedReferences})
 * correctly extracts citations from simulated LLM JSON output.
 */
public class CitationEvalTest {

	private static final Logger log = LoggerFactory.getLogger(CitationEvalTest.class);

	private static EvalDataset dataset;

	@BeforeAll
	static void loadDataset() throws IOException {
		dataset = EvalDataset.load("eval/citation-eval-dataset.json");
	}

	@Test
	public void citationAccuracy_shouldMeetMinimumF1() {
		List<RecordMapping> mappings = new ArrayList<>();
		for (EvalDataset.EvalRecord record : dataset.getRecords()) {
			mappings.add(new RecordMapping(record.getIndex(), record.getResourceType(),
					record.getResourceId(), null));
		}

		int totalCases = 0;
		int exactMatches = 0;
		double totalF1 = 0;

		for (EvalCase evalCase : dataset.getCases()) {
			if (evalCase.getSimulatedLlmResponse() == null) {
				continue;
			}
			totalCases++;

			LlmProvider.LlmResponse llmResponse = LlmProvider.extractResponse(
					evalCase.getSimulatedLlmResponse());

			List<Integer> predictedCitations;
			if (evalCase.getSimulatedCitations() != null) {
				predictedCitations = evalCase.getSimulatedCitations();
			} else {
				predictedCitations = llmResponse.getCitations();
			}

			List<RecordReference> refs = LlmInferenceService.extractCitedReferences(
					predictedCitations, mappings);
			List<Integer> predictedIndices = new ArrayList<>();
			for (RecordReference ref : refs) {
				predictedIndices.add(ref.getIndex());
			}

			List<Integer> expected = evalCase.getExpectedRecordIndices();
			double f1 = EvalMetrics.f1(predictedIndices, expected);
			boolean exact = EvalMetrics.exactMatch(predictedIndices, expected);

			totalF1 += f1;
			if (exact) {
				exactMatches++;
			}

			log.info("[{}] F1={} exact={} predicted={} expected={}",
					evalCase.getId(), String.format("%.3f", f1), exact,
					predictedIndices, expected);
		}

		double avgF1 = totalCases > 0 ? totalF1 / totalCases : 0;
		log.info("Citation eval: avgF1={} exactMatch={}/{}", String.format("%.3f", avgF1),
				exactMatches, totalCases);

		assertTrue(avgF1 >= 0.7,
				"Citation F1 should be >= 0.7 but was " + String.format("%.3f", avgF1));
	}
}
