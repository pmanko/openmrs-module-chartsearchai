/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.eval;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * A single eval test case. Fields are optional depending on the eval type.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class EvalCase {

	private String id;

	private String question;

	private List<Integer> expectedRecordIndices;

	private boolean expectedAbsent;

	private List<String> expectedAnswerContains;

	private List<String> expectedAnswerNotContains;

	private String simulatedLlmResponse;

	private List<Integer> simulatedCitations;

	private String payload;

	private List<String> tags;

	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getQuestion() {
		return question;
	}

	public void setQuestion(String question) {
		this.question = question;
	}

	public List<Integer> getExpectedRecordIndices() {
		return expectedRecordIndices;
	}

	public void setExpectedRecordIndices(List<Integer> expectedRecordIndices) {
		this.expectedRecordIndices = expectedRecordIndices;
	}

	public boolean isExpectedAbsent() {
		return expectedAbsent;
	}

	public void setExpectedAbsent(boolean expectedAbsent) {
		this.expectedAbsent = expectedAbsent;
	}

	public List<String> getExpectedAnswerContains() {
		return expectedAnswerContains;
	}

	public void setExpectedAnswerContains(List<String> expectedAnswerContains) {
		this.expectedAnswerContains = expectedAnswerContains;
	}

	public List<String> getExpectedAnswerNotContains() {
		return expectedAnswerNotContains;
	}

	public void setExpectedAnswerNotContains(List<String> expectedAnswerNotContains) {
		this.expectedAnswerNotContains = expectedAnswerNotContains;
	}

	public String getSimulatedLlmResponse() {
		return simulatedLlmResponse;
	}

	public void setSimulatedLlmResponse(String simulatedLlmResponse) {
		this.simulatedLlmResponse = simulatedLlmResponse;
	}

	public List<Integer> getSimulatedCitations() {
		return simulatedCitations;
	}

	public void setSimulatedCitations(List<Integer> simulatedCitations) {
		this.simulatedCitations = simulatedCitations;
	}

	public String getPayload() {
		return payload;
	}

	public void setPayload(String payload) {
		this.payload = payload;
	}

	public List<String> getTags() {
		return tags;
	}

	public void setTags(List<String> tags) {
		this.tags = tags;
	}
}
