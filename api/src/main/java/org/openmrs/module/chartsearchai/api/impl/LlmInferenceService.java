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

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.api.ChartSearchService;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Answers natural language questions about a patient's chart using direct LLM inference.
 * Serializes the full patient chart and sends all records to the LLM with the question.
 */
@Service("chartSearchAi.llmInferenceService")
public class LlmInferenceService implements ChartSearchService {

	private static final Logger log = LoggerFactory.getLogger(LlmInferenceService.class);

	@Autowired
	private PatientChartSerializer chartSerializer;

	@Autowired
	private LlmProvider llmProvider;

	private static final Pattern CITATION_PATTERN = Pattern.compile(
			"\\[([A-Z][a-z]+ #\\d+(?:,\\s*\\d{4}-\\d{2}-\\d{2})?(?:,\\s*[A-Z][a-z]+ #\\d+(?:,\\s*\\d{4}-\\d{2}-\\d{2})?)*)\\]");

	private static final Pattern SINGLE_CITATION_PATTERN = Pattern.compile(
			"([A-Z][a-z]+) #(\\d+)(?:,\\s*\\d{4}-\\d{2}-\\d{2})?");

	@Override
	public ChartAnswer search(Patient patient, String question) {
		PatientChart chart = chartSerializer.serialize(patient);
		if (log.isTraceEnabled()) {
			log.trace("Serialized patient chart:\n{}", chart.getText());
		}

		String response = llmProvider.search(chart.getText(), question);

		return new ChartAnswer(response, extractCitedReferences(response));
	}

	@Override
	public ChartAnswer searchStreaming(Patient patient, String question,
			Consumer<String> tokenConsumer) {
		PatientChart chart = chartSerializer.serialize(patient);

		String response = llmProvider.searchStreaming(chart.getText(), question, tokenConsumer);

		return new ChartAnswer(response, extractCitedReferences(response));
	}

	static List<RecordReference> extractCitedReferences(String answer) {
		Set<String> seen = new LinkedHashSet<String>();
		Matcher matcher = CITATION_PATTERN.matcher(answer);
		while (matcher.find()) {
			for (String label : matcher.group(1).split(",")) {
				seen.add(label.trim());
			}
		}

		List<RecordReference> references = new ArrayList<RecordReference>();
		for (String label : seen) {
			Matcher m = SINGLE_CITATION_PATTERN.matcher(label);
			if (m.matches()) {
				String resourceType = m.group(1).toLowerCase();
				Integer resourceId = Integer.valueOf(m.group(2));
				references.add(new RecordReference(resourceType, resourceId));
			}
		}
		return references;
	}
}
