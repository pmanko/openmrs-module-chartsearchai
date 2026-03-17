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

		Calendar cal = Calendar.getInstance();
		cal.set(2024, Calendar.JUNE, 15);
		dispense.setDateHandedOver(cal.getTime());

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Handed over: 2024-06-15"));
	}

	@Test
	public void toText_shouldIncludeSubstitution() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Generic Atorvastatin");
		dispense.setDrug(drug);
		dispense.setWasSubstituted(true);

		Concept subType = new Concept();
		subType.addName(conceptName("Generic substitution"));
		dispense.setSubstitutionType(subType);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Substituted: Generic substitution"));
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

	@Test
	public void toText_shouldIncludeStatus() {
		MedicationDispense dispense = new MedicationDispense();
		Drug drug = new Drug();
		drug.setName("Aspirin 100mg");
		dispense.setDrug(drug);

		Concept status = new Concept();
		status.addName(conceptName("Completed"));
		dispense.setStatus(status);

		String result = serializer.toText(dispense);
		assertTrue(result.contains("Status: Completed"));
	}

	private ConceptName conceptName(String name) {
		ConceptName cn = new ConceptName();
		cn.setName(name);
		cn.setLocale(Locale.ENGLISH);
		return cn;
	}
}
