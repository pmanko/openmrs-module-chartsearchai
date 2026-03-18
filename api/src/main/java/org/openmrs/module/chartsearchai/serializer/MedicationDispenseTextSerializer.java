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

import org.openmrs.MedicationDispense;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.springframework.stereotype.Component;

/**
 * Serializes a {@link MedicationDispense} into embedding-friendly text.
 *
 * <p>Example output: {@code "Dispensed: Metformin 500mg. Status: Completed.
 * Quantity: 30 Tablet(s). Dose: 1 Tablet(s) Oral twice daily.
 * Handed over: 2024-06-15"}</p>
 *
 * <p>Example with substitution: {@code "Dispensed: Metformin 500mg. Status: Completed.
 * Status reason: Out of stock. Substituted: Generic equivalent.
 * Substitution reason: Cost. Handed over: 2024-06-15"}</p>
 */
@Component
public class MedicationDispenseTextSerializer implements ClinicalTextSerializer<MedicationDispense> {

	@Override
	public String toText(MedicationDispense dispense) {
		StringBuilder sb = new StringBuilder();

		String drugName = getDrugName(dispense);
		if (!drugName.isEmpty()) {
			sb.append("Dispensed: ").append(drugName);
		}

		if (dispense.getStatus() != null) {
			String status = ConceptNameUtil.getName(dispense.getStatus());
			if (!status.isEmpty()) {
				if (sb.length() > 0) {
					sb.append(". ");
				}
				sb.append("Status: ").append(status);
			}
		}

		if (dispense.getQuantity() != null) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Quantity: ").append(dispense.getQuantity());
			if (dispense.getQuantityUnits() != null) {
				String units = ConceptNameUtil.getName(dispense.getQuantityUnits());
				if (!units.isEmpty()) {
					sb.append(" ").append(units);
				}
			}
		}

		if (dispense.getDose() != null) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Dose: ").append(dispense.getDose());
			if (dispense.getDoseUnits() != null) {
				String units = ConceptNameUtil.getName(dispense.getDoseUnits());
				if (!units.isEmpty()) {
					sb.append(" ").append(units);
				}
			}
			if (dispense.getRoute() != null) {
				String route = ConceptNameUtil.getName(dispense.getRoute());
				if (!route.isEmpty()) {
					sb.append(" ").append(route);
				}
			}
			if (dispense.getFrequency() != null && dispense.getFrequency().getConcept() != null) {
				String freq = ConceptNameUtil.getName(dispense.getFrequency().getConcept());
				if (!freq.isEmpty()) {
					sb.append(" ").append(freq);
				}
			}
		}

		if (dispense.getDosingInstructions() != null
				&& !dispense.getDosingInstructions().trim().isEmpty()) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Instructions: ").append(dispense.getDosingInstructions().trim());
		}

		if (dispense.getStatusReason() != null) {
			String reason = ConceptNameUtil.getName(dispense.getStatusReason());
			if (!reason.isEmpty()) {
				if (sb.length() > 0) {
					sb.append(". ");
				}
				sb.append("Status reason: ").append(reason);
			}
		}

		if (dispense.getWasSubstituted() != null && dispense.getWasSubstituted()) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Substituted");
			if (dispense.getSubstitutionType() != null) {
				String type = ConceptNameUtil.getName(dispense.getSubstitutionType());
				if (!type.isEmpty()) {
					sb.append(": ").append(type);
				}
			}
			if (dispense.getSubstitutionReason() != null) {
				String reason = ConceptNameUtil.getName(dispense.getSubstitutionReason());
				if (!reason.isEmpty()) {
					sb.append(". Substitution reason: ").append(reason);
				}
			}
		}

		if (dispense.getDateHandedOver() != null) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Handed over: ").append(DateFormatUtil.formatDate(dispense.getDateHandedOver()));
		}

		return sb.toString();
	}

	private String getDrugName(MedicationDispense dispense) {
		if (dispense.getDrug() != null && dispense.getDrug().getName() != null) {
			return dispense.getDrug().getName();
		}
		if (dispense.getConcept() != null) {
			return ConceptNameUtil.getName(dispense.getConcept());
		}
		return "";
	}
}
