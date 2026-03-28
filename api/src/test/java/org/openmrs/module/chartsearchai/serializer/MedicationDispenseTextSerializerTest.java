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
import org.openmrs.Drug;
import org.openmrs.MedicationDispense;
import org.openmrs.test.jupiter.BaseModuleContextSensitiveTest;

public class MedicationDispenseTextSerializerTest extends BaseModuleContextSensitiveTest {

	private MedicationDispenseTextSerializer serializer;

	@BeforeEach
	public void setUp() {
		serializer = new MedicationDispenseTextSerializer();
	}

	@Test
	public void toText_shouldSerializeBasicDispense() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		dispense.setDrug(drug);
		dispense.setQuantity(30.0);

		Concept quantityUnits = new Concept();
		quantityUnits.addName(conceptName("Tablet(s)"));
		dispense.setQuantityUnits(quantityUnits);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Dispensed: Metformin 500mg"));
		assertTrue(result.contains("Quantity: 30.0 Tablet(s)"));
	}

	@Test
	public void toText_shouldIncludeStatus() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		dispense.setDrug(drug);

		Concept status = new Concept();
		status.addName(conceptName("Completed"));
		dispense.setStatus(status);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Status: Completed"));
	}

	@Test
	public void toText_shouldIncludeDeclinedStatus() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Amoxicillin 250mg");
		dispense.setDrug(drug);

		Concept status = new Concept();
		status.addName(conceptName("Declined"));
		dispense.setStatus(status);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Status: Declined"));
	}

	@Test
	public void toText_shouldIncludeDoseAndRoute() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Amoxicillin 250mg");
		dispense.setDrug(drug);
		dispense.setDose(1.0);

		Concept doseUnits = new Concept();
		doseUnits.addName(conceptName("Capsule(s)"));
		dispense.setDoseUnits(doseUnits);

		Concept route = new Concept();
		route.addName(conceptName("Oral"));
		dispense.setRoute(route);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Dose: 1.0 Capsule(s) Oral"));
	}

	@Test
	public void toText_shouldIncludeDosingInstructions() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Ibuprofen 400mg");
		dispense.setDrug(drug);
		dispense.setDosingInstructions("Take with food");

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Instructions: Take with food"));
	}

	@Test
	public void toText_shouldIncludeDateHandedOver() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Lisinopril 10mg");
		dispense.setDrug(drug);

		Calendar cal = Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"));
		cal.set(2024, Calendar.JUNE, 15, 12, 0, 0);
		dispense.setDateHandedOver(cal.getTime());

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Handed over: 2024-06-15"));
	}

	@Test
	public void toText_shouldIncludeStatusReason() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		dispense.setDrug(drug);

		Concept statusReason = new Concept();
		statusReason.addName(conceptName("Out of stock"));
		dispense.setStatusReason(statusReason);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Status reason: Out of stock"));
	}

	@Test
	public void toText_shouldIncludeSubstitutionDetails() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		dispense.setDrug(drug);
		dispense.setWasSubstituted(true);

		Concept subType = new Concept();
		subType.addName(conceptName("Generic equivalent"));
		dispense.setSubstitutionType(subType);

		Concept subReason = new Concept();
		subReason.addName(conceptName("Cost"));
		dispense.setSubstitutionReason(subReason);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Substituted: Generic equivalent"));
		assertTrue(result.contains("Substitution reason: Cost"));
	}

	@Test
	public void toText_shouldNotIncludeSubstitutionWhenNotSubstituted() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		dispense.setDrug(drug);
		dispense.setWasSubstituted(false);

		String result = serializer.toText(dispense);
		assertTrue(!result.contains("Substituted"));
	}

	@Test
	public void toText_shouldFallBackToConceptName() {
		MedicationDispense dispense = new MedicationDispense();
		Concept concept = new Concept();
		concept.addName(conceptName("Paracetamol"));
		dispense.setConcept(concept);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Dispensed: Paracetamol"));
	}

	private ConceptName conceptName(String name) {
		ConceptName cn = new ConceptName();
		cn.setName(name);
		cn.setLocale(Locale.ENGLISH);
		return cn;
	}
}
