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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Date;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Concept;
import org.openmrs.ConceptClass;
import org.openmrs.ConceptName;
import org.openmrs.ConceptNumeric;
import org.openmrs.Encounter;
import org.openmrs.EncounterType;
import org.openmrs.Obs;
import java.util.Locale;
import org.openmrs.test.jupiter.BaseModuleContextSensitiveTest;

public class ObsTextSerializerTest extends BaseModuleContextSensitiveTest {

	private ObsTextSerializer serializer;

	@BeforeEach
	public void setUp() {
		serializer = new ObsTextSerializer();
	}

	@Test
	public void toText_shouldSerializeObsWithEncounter() {
		Obs obs = new Obs();
		Encounter enc = new Encounter();
		EncounterType type = new EncounterType();
		type.setName("Outpatient Visit");
		enc.setEncounterType(type);
		enc.setEncounterDatetime(new Date());
		obs.setEncounter(enc);

		Concept concept = new Concept();
		concept.addName(conceptName("Weight"));
		obs.setConcept(concept);
		obs.setValueNumeric(70.0);

		String result = serializer.toText(obs);
		assertTrue(result.contains("Weight: 70.0"));
	}

	@Test
	public void toText_shouldIncludeNumericValueWithUnits() {
		Obs obs = new Obs();
		ConceptNumeric concept = new ConceptNumeric();
		concept.addName(conceptName("Temperature"));
		concept.setUnits("DEG C");
		obs.setConcept(concept);
		obs.setValueNumeric(37.5);

		String result = serializer.toText(obs);
		assertTrue(result.contains("37.5 DEG C"));
	}

	@Test
	public void toText_shouldIncludeCodedValue() {
		Obs obs = new Obs();
		Concept question = new Concept();
		question.addName(conceptName("Diagnosis"));
		obs.setConcept(question);

		Concept answer = new Concept();
		answer.addName(conceptName("Malaria"));
		obs.setValueCoded(answer);

		String result = serializer.toText(obs);
		assertTrue(result.contains("Diagnosis: Malaria"));
	}

	@Test
	public void toText_shouldIncludeInterpretation() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Heart Rate"));
		obs.setConcept(concept);
		obs.setValueNumeric(120.0);
		obs.setInterpretation(Obs.Interpretation.HIGH);

		String result = serializer.toText(obs);
		assertTrue(result.contains("(HIGH)"));
	}

	@Test
	public void toText_shouldIncludeComment() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("BP"));
		obs.setConcept(concept);
		obs.setValueNumeric(140.0);
		obs.setComment("Taken after exercise");

		String result = serializer.toText(obs);
		assertTrue(result.contains("Note: Taken after exercise"));
	}

	@Test
	public void toText_shouldFlattenGroupMembers() {
		Obs parent = new Obs();
		Concept parentConcept = new Concept();
		parentConcept.addName(conceptName("Vitals"));
		parent.setConcept(parentConcept);
		parent.setValueText("group");

		Obs child = new Obs();
		Concept childConcept = new Concept();
		childConcept.addName(conceptName("Pulse"));
		child.setConcept(childConcept);
		child.setValueNumeric(80.0);
		parent.addGroupMember(child);

		String result = serializer.toText(parent);
		assertTrue(result.contains("Vitals:"));
		assertTrue(result.contains("Pulse: 80.0"));
		assertTrue(!result.contains("group"), "Parent value should be skipped for group obs");
	}

	@Test
	public void toText_shouldHandleNullConcept() {
		Obs obs = new Obs();
		obs.setValueText("some text");

		String result = serializer.toText(obs);
		assertEquals("some text", result);
	}

	@Test
	public void toText_shouldIncludeDatetimeValue() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Date of Symptom Onset"));
		obs.setConcept(concept);

		java.util.Calendar cal = java.util.Calendar.getInstance();
		cal.set(2024, java.util.Calendar.FEBRUARY, 10);
		obs.setValueDatetime(cal.getTime());

		String result = serializer.toText(obs);
		assertTrue(result.contains("Date of Symptom Onset: 2024-02-10"));
	}

	@Test
	public void toText_shouldIncludeTextValue() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Clinical Notes"));
		obs.setConcept(concept);
		obs.setValueText("Patient reports improvement");

		String result = serializer.toText(obs);
		assertTrue(result.contains("Clinical Notes: Patient reports improvement"));
	}

	@Test
	public void toText_shouldHandleEncounterWithoutType() {
		Obs obs = new Obs();
		Encounter enc = new Encounter();
		enc.setEncounterDatetime(new Date());
		obs.setEncounter(enc);

		Concept concept = new Concept();
		concept.addName(conceptName("Weight"));
		obs.setConcept(concept);
		obs.setValueNumeric(70.0);

		String result = serializer.toText(obs);
		assertTrue(result.contains("Weight: 70.0"));
	}

	@Test
	public void toText_shouldHandleObsWithNoEncounter() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Temperature"));
		obs.setConcept(concept);
		obs.setValueNumeric(37.0);

		String result = serializer.toText(obs);
		assertTrue(result.startsWith("Temperature: 37.0"));
	}

	@Test
	public void toText_shouldIncludeValueModifier() {
		Obs obs = new Obs();
		ConceptNumeric concept = new ConceptNumeric();
		concept.addName(conceptName("Viral Load"));
		concept.setUnits("copies/mL");
		obs.setConcept(concept);
		obs.setValueNumeric(200.0);
		obs.setValueModifier(">");

		String result = serializer.toText(obs);
		assertTrue(result.contains(">200.0 copies/mL"));
	}

	@Test
	public void toText_shouldOmitModifierWhenNull() {
		Obs obs = new Obs();
		ConceptNumeric concept = new ConceptNumeric();
		concept.addName(conceptName("Hemoglobin"));
		concept.setUnits("g/dL");
		obs.setConcept(concept);
		obs.setValueNumeric(12.5);

		String result = serializer.toText(obs);
		assertTrue(result.contains("12.5 g/dL"));
		assertTrue(!result.contains(">") && !result.contains("<"));
	}

	@Test
	public void toText_shouldIncludeConceptClassPrefix() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Kaposi sarcoma oral"));
		ConceptClass cc = new ConceptClass();
		cc.setName("Finding");
		concept.setConceptClass(cc);
		obs.setConcept(concept);
		obs.setValueNumeric(3.91);

		String result = serializer.toText(obs);
		assertEquals("Finding — Kaposi sarcoma oral: 3.91", result);
	}

	@Test
	public void toText_shouldOmitQuestionConceptClass() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Do you smoke?"));
		ConceptClass cc = new ConceptClass();
		cc.setName("Question");
		concept.setConceptClass(cc);
		obs.setConcept(concept);
		Concept yes = new Concept();
		yes.addName(conceptName("Yes"));
		obs.setValueCoded(yes);

		String result = serializer.toText(obs);
		assertEquals("Assessment \u2014 Do you smoke?: Yes", result);
	}

	@Test
	public void toText_shouldOmitConceptClassWhenNull() {
		Obs obs = new Obs();
		Concept concept = new Concept();
		concept.addName(conceptName("Weight"));
		obs.setConcept(concept);
		obs.setValueNumeric(70.0);

		String result = serializer.toText(obs);
		assertTrue(result.startsWith("Weight: 70.0"));
	}

	@Test
	public void toText_shouldSkipGroupMemberWhenValueCodedIsSameConcept() {
		Obs parent = new Obs();
		Concept parentConcept = new Concept();
		parentConcept.addName(conceptName("Urine microscopy panel"));
		ConceptClass labSet = new ConceptClass();
		labSet.setName("LabSet");
		parentConcept.setConceptClass(labSet);
		parent.setConcept(parentConcept);

		// Member where valueCoded == concept (self-referencing, no real result)
		Concept bacteriuria = new Concept();
		bacteriuria.addName(conceptName("Bacteriuria test"));
		Obs selfRef = new Obs();
		selfRef.setConcept(bacteriuria);
		selfRef.setValueCoded(bacteriuria);
		parent.addGroupMember(selfRef);

		// Member with a real numeric value
		Obs realResult = new Obs();
		Concept leukocytes = new Concept();
		leukocytes.addName(conceptName("Leukocytes"));
		realResult.setConcept(leukocytes);
		realResult.setValueNumeric(5.0);
		parent.addGroupMember(realResult);

		String result = serializer.toText(parent);
		assertTrue(!result.contains("Bacteriuria test"),
				"Should skip member where valueCoded is the same concept");
		assertTrue(result.contains("Leukocytes: 5.0"),
				"Should include member with a real value");
	}

	@Test
	public void toText_shouldReturnEmptyWhenAllGroupMembersAreSelfReferencing() {
		Obs parent = new Obs();
		Concept parentConcept = new Concept();
		parentConcept.addName(conceptName("Urine microscopy panel"));
		ConceptClass labSet = new ConceptClass();
		labSet.setName("LabSet");
		parentConcept.setConceptClass(labSet);
		parent.setConcept(parentConcept);

		Concept test1 = new Concept();
		test1.addName(conceptName("Bacteriuria test"));
		Obs member1 = new Obs();
		member1.setConcept(test1);
		member1.setValueCoded(test1);
		parent.addGroupMember(member1);

		Concept test2 = new Concept();
		test2.addName(conceptName("Yeast presence"));
		Obs member2 = new Obs();
		member2.setConcept(test2);
		member2.setValueCoded(test2);
		parent.addGroupMember(member2);

		String result = serializer.toText(parent);
		assertEquals("", result, "Should return empty when all members are self-referencing");
	}

	@Test
	public void toText_shouldNotSkipGroupMemberWhenValueCodedIsDifferentConcept() {
		Obs parent = new Obs();
		Concept parentConcept = new Concept();
		parentConcept.addName(conceptName("Lab Panel"));
		parent.setConcept(parentConcept);

		Concept question = new Concept();
		question.addName(conceptName("Blood Type"));
		Concept answer = new Concept();
		answer.addName(conceptName("Type A"));
		Obs member = new Obs();
		member.setConcept(question);
		member.setValueCoded(answer);
		parent.addGroupMember(member);

		String result = serializer.toText(parent);
		assertTrue(result.contains("Blood Type: Type A"),
				"Should include member when valueCoded is a different concept");
	}

	private ConceptName conceptName(String name) {
		ConceptName cn = new ConceptName();
		cn.setName(name);
		cn.setLocale(Locale.ENGLISH);
		return cn;
	}
}
