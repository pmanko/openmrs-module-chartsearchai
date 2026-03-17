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

import java.util.Date;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Concept;
import org.openmrs.ConceptName;
import org.openmrs.Drug;
import org.openmrs.DrugOrder;
import org.openmrs.Order;
import org.openmrs.ServiceOrder;
import org.openmrs.TestOrder;
import org.openmrs.ReferralOrder;
import java.util.Locale;
import org.openmrs.test.jupiter.BaseModuleContextSensitiveTest;
import org.springframework.test.util.ReflectionTestUtils;

public class OrderTextSerializerTest extends BaseModuleContextSensitiveTest {

	private OrderTextSerializer serializer;

	@BeforeEach
	public void setUp() {
		serializer = new OrderTextSerializer();
	}

	@Test
	public void toText_shouldSerializeBasicOrder() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Complete Blood Count"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.ROUTINE);
		order.setDateActivated(new Date());

		String result = serializer.toText(order);
		assertTrue(result.contains("Order: Complete Blood Count"));
		assertTrue(result.contains("Action: NEW"));
		assertTrue(result.contains("Urgency: ROUTINE"));
	}

	@Test
	public void toText_shouldIncludeInstructions() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("X-Ray Chest"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.STAT);
		order.setInstructions("PA and lateral views");
		order.setDateActivated(new Date());

		String result = serializer.toText(order);
		assertTrue(result.contains("Instructions: PA and lateral views"));
	}

	@Test
	public void toText_shouldIncludeCodedReason() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Hemoglobin"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.ROUTINE);
		order.setDateActivated(new Date());

		Concept reason = new Concept();
		reason.addName(conceptName("Suspected anemia"));
		order.setOrderReason(reason);

		String result = serializer.toText(order);
		assertTrue(result.contains("Reason: Suspected anemia"));
	}

	@Test
	public void toText_shouldIncludeNonCodedReason() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Urinalysis"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.ROUTINE);
		order.setDateActivated(new Date());
		order.setOrderReasonNonCoded("Follow-up after UTI treatment");

		String result = serializer.toText(order);
		assertTrue(result.contains("Reason: Follow-up after UTI treatment"));
	}

	@Test
	public void toText_shouldIncludeUrgency() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Blood Culture"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.STAT);
		order.setDateActivated(new Date());

		String result = serializer.toText(order);
		assertTrue(result.contains("Urgency: STAT"));
	}

	@Test
	public void toText_shouldIncludeDateStopped() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Metformin"));
		order.setConcept(concept);
		order.setAction(Order.Action.DISCONTINUE);
		order.setUrgency(Order.Urgency.ROUTINE);
		order.setDateActivated(new Date());

		java.util.Calendar cal = java.util.Calendar.getInstance();
		cal.set(2024, java.util.Calendar.JUNE, 20);
		ReflectionTestUtils.setField(order, "dateStopped", cal.getTime());

		String result = serializer.toText(order);
		assertTrue(result.contains("Stopped: 2024-06-20"));
	}

	@Test
	public void toText_shouldOmitStoppedWhenNotSet() {
		Order order = new Order();
		Concept concept = new Concept();
		concept.addName(conceptName("Lisinopril"));
		order.setConcept(concept);
		order.setAction(Order.Action.NEW);
		order.setUrgency(Order.Urgency.ROUTINE);
		order.setDateActivated(new Date());

		String result = serializer.toText(order);
		assertTrue(!result.contains("Stopped:"));
	}

	@Test
	public void toText_shouldSerializeDrugOrderWithDoseAndRoute() {
		DrugOrder drugOrder = new DrugOrder();
		Drug drug = new Drug();
		drug.setName("Metformin 500mg");
		drugOrder.setDrug(drug);
		drugOrder.setDose(1.0);
		drugOrder.setAction(Order.Action.NEW);
		drugOrder.setUrgency(Order.Urgency.ROUTINE);
		drugOrder.setDateActivated(new Date());

		Concept doseUnits = new Concept();
		doseUnits.addName(conceptName("Tablet(s)"));
		drugOrder.setDoseUnits(doseUnits);

		Concept route = new Concept();
		route.addName(conceptName("Oral"));
		drugOrder.setRoute(route);

		String result = serializer.toText(drugOrder);
		assertTrue(result.contains("Drug order: Metformin 500mg"));
		assertTrue(result.contains("Dose: 1.0 Tablet(s) Oral"));
	}

	@Test
	public void toText_shouldIncludeDrugOrderDuration() {
		DrugOrder drugOrder = new DrugOrder();
		Drug drug = new Drug();
		drug.setName("Amoxicillin 250mg");
		drugOrder.setDrug(drug);
		drugOrder.setAction(Order.Action.NEW);
		drugOrder.setUrgency(Order.Urgency.ROUTINE);
		drugOrder.setDateActivated(new Date());
		drugOrder.setDuration(7);

		Concept durationUnits = new Concept();
		durationUnits.addName(conceptName("Day(s)"));
		drugOrder.setDurationUnits(durationUnits);

		String result = serializer.toText(drugOrder);
		assertTrue(result.contains("Duration: 7 Day(s)"));
	}

	@Test
	public void toText_shouldIncludeAsNeeded() {
		DrugOrder drugOrder = new DrugOrder();
		Drug drug = new Drug();
		drug.setName("Ibuprofen 400mg");
		drugOrder.setDrug(drug);
		drugOrder.setAction(Order.Action.NEW);
		drugOrder.setUrgency(Order.Urgency.ROUTINE);
		drugOrder.setDateActivated(new Date());
		drugOrder.setAsNeeded(true);
		drugOrder.setAsNeededCondition("for pain");

		String result = serializer.toText(drugOrder);
		assertTrue(result.contains("As needed (for pain)"));
	}

	@Test
	public void toText_shouldFallBackToNonCodedDrugName() {
		DrugOrder drugOrder = new DrugOrder();
		drugOrder.setDrugNonCoded("Custom herbal remedy");
		drugOrder.setAction(Order.Action.NEW);
		drugOrder.setUrgency(Order.Urgency.ROUTINE);
		drugOrder.setDateActivated(new Date());

		String result = serializer.toText(drugOrder);
		assertTrue(result.contains("Drug order: Custom herbal remedy"));
	}

	@Test
	public void toText_shouldIncludeDosingInstructions() {
		DrugOrder drugOrder = new DrugOrder();
		Drug drug = new Drug();
		drug.setName("Warfarin 5mg");
		drugOrder.setDrug(drug);
		drugOrder.setAction(Order.Action.NEW);
		drugOrder.setUrgency(Order.Urgency.ROUTINE);
		drugOrder.setDateActivated(new Date());
		drugOrder.setDosingInstructions("Monitor INR weekly");

		String result = serializer.toText(drugOrder);
		assertTrue(result.contains("Dosing: Monitor INR weekly"));
	}

	@Test
	public void toText_shouldIncludeLateralityForServiceOrder() {
		ServiceOrder serviceOrder = new TestOrder();
		Concept concept = new Concept();
		concept.addName(conceptName("X-Ray Knee"));
		serviceOrder.setConcept(concept);
		serviceOrder.setAction(Order.Action.NEW);
		serviceOrder.setUrgency(Order.Urgency.STAT);
		serviceOrder.setDateActivated(new Date());
		serviceOrder.setLaterality(ServiceOrder.Laterality.LEFT);

		String result = serializer.toText(serviceOrder);
		assertTrue(result.contains("Order: X-Ray Knee"));
		assertTrue(result.contains("Laterality: LEFT"));
	}

	@Test
	public void toText_shouldIncludeSpecimenSourceForServiceOrder() {
		ServiceOrder serviceOrder = new TestOrder();
		Concept concept = new Concept();
		concept.addName(conceptName("Blood Culture"));
		serviceOrder.setConcept(concept);
		serviceOrder.setAction(Order.Action.NEW);
		serviceOrder.setUrgency(Order.Urgency.ROUTINE);
		serviceOrder.setDateActivated(new Date());

		Concept specimen = new Concept();
		specimen.addName(conceptName("Venous blood"));
		serviceOrder.setSpecimenSource(specimen);

		String result = serializer.toText(serviceOrder);
		assertTrue(result.contains("Specimen: Venous blood"));
	}

	@Test
	public void toText_shouldIncludeClinicalHistoryForServiceOrder() {
		ServiceOrder serviceOrder = new TestOrder();
		Concept concept = new Concept();
		concept.addName(conceptName("Chest X-Ray"));
		serviceOrder.setConcept(concept);
		serviceOrder.setAction(Order.Action.NEW);
		serviceOrder.setUrgency(Order.Urgency.STAT);
		serviceOrder.setDateActivated(new Date());
		serviceOrder.setClinicalHistory("Persistent cough for 3 weeks");

		String result = serializer.toText(serviceOrder);
		assertTrue(result.contains("Clinical history: Persistent cough for 3 weeks"));
	}

	@Test
	public void toText_shouldHandleReferralOrder() {
		ReferralOrder referralOrder = new ReferralOrder();
		Concept concept = new Concept();
		concept.addName(conceptName("Cardiology Consultation"));
		referralOrder.setConcept(concept);
		referralOrder.setAction(Order.Action.NEW);
		referralOrder.setUrgency(Order.Urgency.ROUTINE);
		referralOrder.setDateActivated(new Date());
		referralOrder.setClinicalHistory("Chest pain on exertion");

		String result = serializer.toText(referralOrder);
		assertTrue(result.contains("Order: Cardiology Consultation"));
		assertTrue(result.contains("Clinical history: Chest pain on exertion"));
	}

	private ConceptName conceptName(String name) {
		ConceptName cn = new ConceptName();
		cn.setName(name);
		cn.setLocale(Locale.ENGLISH);
		return cn;
	}
}
