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

import org.openmrs.DrugOrder;
import org.openmrs.Order;
import org.openmrs.ReferralOrder;
import org.openmrs.ServiceOrder;
import org.openmrs.TestOrder;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.springframework.stereotype.Component;

/**
 * Serializes an {@link Order} into embedding-friendly text. Handles three order hierarchies:
 * {@link DrugOrder} (prescription details), {@link ServiceOrder} and its subclasses
 * ({@code TestOrder}, {@code ReferralOrder}) with laterality/specimen/clinical history,
 * and base {@link Order} for everything else.
 *
 * <p>Example output for a drug order: {@code "Drug order: Metformin 500mg. Dose: 1.0 Tablet(s)
 * Oral twice daily. Duration: 30 Day(s). Quantity: 60.0 Tablet(s). Action: NEW.
 * Urgency: ROUTINE"}</p>
 *
 * <p>Example output for a service/test/referral order: {@code "Order: X-Ray Chest. Laterality:
 * LEFT. Clinical history: Persistent cough for 3 weeks. Action: NEW. Urgency: STAT"}</p>
 *
 * <p><b>Date handling:</b> The activation date is excluded (it serves as the record
 * timestamp in {@link PatientRecordLoader}). The stop date is included as
 * {@code "Stopped: <date>"} because it is clinically significant — it indicates
 * when the order was discontinued.
 */
@Component
public class OrderTextSerializer implements ClinicalTextSerializer<Order> {

	@Override
	public String toText(Order order) {
		StringBuilder sb = new StringBuilder();

		if (order instanceof DrugOrder) {
			appendDrugOrderFields(sb, (DrugOrder) order);
		} else {
			String name = ConceptNameUtil.getName(order.getConcept());
			if (!name.isEmpty()) {
				if (order instanceof TestOrder) {
					sb.append("Test order: ");
				} else if (order instanceof ReferralOrder) {
					sb.append("Referral order: ");
				} else {
					sb.append("Order: ");
				}
				sb.append(name);
			}
			if (order instanceof ServiceOrder) {
				appendServiceOrderFields(sb, (ServiceOrder) order);
			}
		}

		if (sb.length() > 0) {
			sb.append(". ");
		}
		sb.append("Action: ").append(order.getAction());
		sb.append(". Urgency: ").append(order.getUrgency());

		if (order.getInstructions() != null && !order.getInstructions().trim().isEmpty()) {
			sb.append(". Instructions: ").append(order.getInstructions().trim());
		}
		if (order.getOrderReason() != null) {
			sb.append(". Reason: ").append(ConceptNameUtil.getName(order.getOrderReason()));
		} else if (order.getOrderReasonNonCoded() != null
				&& !order.getOrderReasonNonCoded().trim().isEmpty()) {
			sb.append(". Reason: ").append(order.getOrderReasonNonCoded().trim());
		}
		if (order.getDateStopped() != null) {
			sb.append(". Stopped: ").append(DateFormatUtil.formatDate(order.getDateStopped()));
		}

		return sb.toString();
	}

	private void appendServiceOrderFields(StringBuilder sb, ServiceOrder serviceOrder) {
		if (serviceOrder.getLaterality() != null) {
			sb.append(". Laterality: ").append(serviceOrder.getLaterality());
		}
		if (serviceOrder.getSpecimenSource() != null) {
			String specimen = ConceptNameUtil.getName(serviceOrder.getSpecimenSource());
			if (!specimen.isEmpty()) {
				sb.append(". Specimen: ").append(specimen);
			}
		}
		if (serviceOrder.getClinicalHistory() != null
				&& !serviceOrder.getClinicalHistory().trim().isEmpty()) {
			sb.append(". Clinical history: ").append(serviceOrder.getClinicalHistory().trim());
		}
	}

	private void appendDrugOrderFields(StringBuilder sb, DrugOrder drugOrder) {
		String drugName = getDrugName(drugOrder);
		if (!drugName.isEmpty()) {
			sb.append("Drug order: ").append(drugName);
		}

		if (drugOrder.getDose() != null) {
			sb.append(". Dose: ").append(drugOrder.getDose());
			if (drugOrder.getDoseUnits() != null) {
				String units = ConceptNameUtil.getName(drugOrder.getDoseUnits());
				if (!units.isEmpty()) {
					sb.append(" ").append(units);
				}
			}
			if (drugOrder.getRoute() != null) {
				String route = ConceptNameUtil.getName(drugOrder.getRoute());
				if (!route.isEmpty()) {
					sb.append(" ").append(route);
				}
			}
			if (drugOrder.getFrequency() != null && drugOrder.getFrequency().getConcept() != null) {
				String freq = ConceptNameUtil.getName(drugOrder.getFrequency().getConcept());
				if (!freq.isEmpty()) {
					sb.append(" ").append(freq);
				}
			}
		}

		if (drugOrder.getDuration() != null) {
			sb.append(". Duration: ").append(drugOrder.getDuration());
			if (drugOrder.getDurationUnits() != null) {
				String units = ConceptNameUtil.getName(drugOrder.getDurationUnits());
				if (!units.isEmpty()) {
					sb.append(" ").append(units);
				}
			}
		}

		if (drugOrder.getQuantity() != null) {
			sb.append(". Quantity: ").append(drugOrder.getQuantity());
			if (drugOrder.getQuantityUnits() != null) {
				String units = ConceptNameUtil.getName(drugOrder.getQuantityUnits());
				if (!units.isEmpty()) {
					sb.append(" ").append(units);
				}
			}
		}

		if (drugOrder.getAsNeeded() != null && drugOrder.getAsNeeded()) {
			sb.append(". As needed");
			if (drugOrder.getAsNeededCondition() != null
					&& !drugOrder.getAsNeededCondition().trim().isEmpty()) {
				sb.append(" (").append(drugOrder.getAsNeededCondition().trim()).append(")");
			}
		}

		if (drugOrder.getDosingInstructions() != null
				&& !drugOrder.getDosingInstructions().trim().isEmpty()) {
			sb.append(". Dosing: ").append(drugOrder.getDosingInstructions().trim());
		}
	}

	private String getDrugName(DrugOrder drugOrder) {
		if (drugOrder.getDrug() != null && drugOrder.getDrug().getName() != null) {
			return drugOrder.getDrug().getName();
		}
		if (drugOrder.getDrugNonCoded() != null
				&& !drugOrder.getDrugNonCoded().trim().isEmpty()) {
			return drugOrder.getDrugNonCoded().trim();
		}
		return ConceptNameUtil.getName(drugOrder.getConcept());
	}
}
