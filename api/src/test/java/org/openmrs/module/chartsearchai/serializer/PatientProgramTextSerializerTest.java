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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Calendar;
import java.util.Locale;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Concept;
import org.openmrs.ConceptName;
import org.openmrs.PatientProgram;
import org.openmrs.Program;
import org.openmrs.test.jupiter.BaseModuleContextSensitiveTest;

public class PatientProgramTextSerializerTest extends BaseModuleContextSensitiveTest {

	private PatientProgramTextSerializer serializer;

	@BeforeEach
	public void setUp() {
		serializer = new PatientProgramTextSerializer();
	}

	@Test
	public void toText_shouldSerializeActiveEnrollment() {
		PatientProgram pp = new PatientProgram();
		Program program = new Program();
		program.setName("HIV Treatment");
		pp.setProgram(program);

		Calendar cal = Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"));
		cal.set(2024, Calendar.JANUARY, 15, 12, 0, 0);
		pp.setDateEnrolled(cal.getTime());

		String result = serializer.toText(pp);
		assertTrue(result.contains("Program: HIV Treatment"));
		assertTrue(result.contains("Enrolled: 2024-01-15"));
		assertTrue(result.contains("Status: Active"));
	}

	@Test
	public void toText_shouldSerializeCompletedProgram() {
		PatientProgram pp = new PatientProgram();
		Program program = new Program();
		program.setName("TB Treatment");
		pp.setProgram(program);

		Calendar cal = Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"));
		cal.set(2023, Calendar.MARCH, 1, 12, 0, 0);
		pp.setDateEnrolled(cal.getTime());
		cal.set(2023, Calendar.SEPTEMBER, 15, 12, 0, 0);
		pp.setDateCompleted(cal.getTime());

		String result = serializer.toText(pp);
		assertTrue(result.contains("Program: TB Treatment"));
		assertTrue(result.contains("Completed: 2023-09-15"));
		assertTrue(!result.contains("Status: Active"));
	}

	@Test
	public void toText_shouldIncludeOutcome() {
		PatientProgram pp = new PatientProgram();
		Program program = new Program();
		program.setName("TB Treatment");
		pp.setProgram(program);

		Calendar cal = Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"));
		cal.set(2023, Calendar.MARCH, 1, 12, 0, 0);
		pp.setDateEnrolled(cal.getTime());
		cal.set(2023, Calendar.SEPTEMBER, 15, 12, 0, 0);
		pp.setDateCompleted(cal.getTime());

		Concept outcome = new Concept();
		outcome.addName(conceptName("Cured"));
		pp.setOutcome(outcome);

		String result = serializer.toText(pp);
		assertTrue(result.contains("Outcome: Cured"));
	}

	@Test
	public void toText_shouldHandleNullProgram() {
		PatientProgram pp = new PatientProgram();

		String result = serializer.toText(pp);
		assertTrue(result.contains("Status: Active"));
		assertTrue(!result.contains("Program:"));
	}

	private ConceptName conceptName(String name) {
		ConceptName cn = new ConceptName();
		cn.setName(name);
		cn.setLocale(Locale.ENGLISH);
		return cn;
	}
}
