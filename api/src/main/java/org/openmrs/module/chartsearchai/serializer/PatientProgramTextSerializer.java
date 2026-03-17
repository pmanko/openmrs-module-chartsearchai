/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.serializer;

import org.openmrs.PatientProgram;
import org.openmrs.PatientState;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.springframework.stereotype.Component;

/**
 * Serializes a {@link PatientProgram} into embedding-friendly text.
 *
 * <p>Example output: {@code "Program: HIV Treatment. Enrolled: 2024-01-15. Status: Active.
 * Current state: On ART"}</p>
 */
@Component
public class PatientProgramTextSerializer implements ClinicalTextSerializer<PatientProgram> {

	@Override
	public String toText(PatientProgram patientProgram) {
		StringBuilder sb = new StringBuilder();

		if (patientProgram.getProgram() != null && patientProgram.getProgram().getName() != null) {
			sb.append("Program: ").append(patientProgram.getProgram().getName());
		}

		if (patientProgram.getDateEnrolled() != null) {
			sb.append(". Enrolled: ").append(DateFormatUtil.formatDate(patientProgram.getDateEnrolled()));
		}

		if (patientProgram.getDateCompleted() != null) {
			sb.append(". Completed: ").append(DateFormatUtil.formatDate(patientProgram.getDateCompleted()));
		} else {
			sb.append(". Status: Active");
		}

		if (patientProgram.getOutcome() != null) {
			String outcomeName = ConceptNameUtil.getName(patientProgram.getOutcome());
			if (!outcomeName.isEmpty()) {
				sb.append(". Outcome: ").append(outcomeName);
			}
		}

		PatientState currentState = patientProgram.getCurrentState(null);
		if (currentState != null && currentState.getState() != null
				&& currentState.getState().getConcept() != null) {
			String stateName = ConceptNameUtil.getName(currentState.getState().getConcept());
			if (!stateName.isEmpty()) {
				sb.append(". Current state: ").append(stateName);
			}
		}

		if (patientProgram.getLocation() != null && patientProgram.getLocation().getName() != null) {
			sb.append(". Location: ").append(patientProgram.getLocation().getName());
		}

		return sb.toString();
	}
}
