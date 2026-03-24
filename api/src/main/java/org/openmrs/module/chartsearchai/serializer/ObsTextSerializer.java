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

import org.openmrs.ConceptNumeric;
import org.openmrs.Obs;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.springframework.stereotype.Component;

/**
 * Serializes an {@link Obs} into embedding-friendly text.
 *
 * <p>Example output: {@code "Systolic Blood Pressure: 120 mmHg (ABNORMAL).
 * Note: Taken after exercise"}</p>
 */
@Component
public class ObsTextSerializer implements ClinicalTextSerializer<Obs> {

	@Override
	public String toText(Obs obs) {
		StringBuilder sb = new StringBuilder();

		String conceptClass = getConceptClassName(obs);
		if (!conceptClass.isEmpty()) {
			sb.append(conceptClass).append(" — ");
		}

		String conceptName = ConceptNameUtil.getName(obs.getConcept());

		// Flatten group members into parent text, skipping the parent's own value
		if (obs.hasGroupMembers()) {
			StringBuilder membersSb = new StringBuilder();
			boolean first = true;
			for (Obs member : obs.getGroupMembers()) {
				String memberValue = formatValue(member);
				String memberName = ConceptNameUtil.getName(member.getConcept());
				// Skip members with no useful value: empty, or valueCoded
				// is the same concept as the member itself (data quality
				// issue — a concept cannot be its own answer)
				boolean selfReferencing = member.getValueCoded() != null
						&& member.getConcept() != null
						&& member.getValueCoded().equals(member.getConcept());
				if (memberValue.isEmpty() || selfReferencing) {
					continue;
				}
				if (!first) {
					membersSb.append("; ");
				}
				first = false;
				if (!memberName.isEmpty()) {
					membersSb.append(memberName).append(": ");
				}
				membersSb.append(memberValue);
				if (member.getInterpretation() != null) {
					membersSb.append(" (").append(member.getInterpretation()).append(")");
				}
			}
			if (membersSb.length() == 0) {
				return "";
			}
			if (!conceptName.isEmpty()) {
				sb.append(conceptName).append(": ");
			}
			sb.append(membersSb);
		} else {
			String value = formatValue(obs);
			if (value.isEmpty()) {
				return "";
			}
			if (!conceptName.isEmpty()) {
				sb.append(conceptName).append(": ");
			}
			sb.append(value);

			if (obs.getInterpretation() != null) {
				sb.append(" (").append(obs.getInterpretation()).append(")");
			}
		}

		if (obs.getComment() != null && !obs.getComment().trim().isEmpty()) {
			sb.append(". Note: ").append(obs.getComment().trim());
		}

		return sb.toString();
	}

	private String formatValue(Obs obs) {
		if (obs.getValueCoded() != null) {
			return ConceptNameUtil.getName(obs.getValueCoded());
		}
		if (obs.getValueNumeric() != null) {
			String modifier = obs.getValueModifier();
			String units = getUnits(obs.getConcept());
			return (modifier != null ? modifier : "") + obs.getValueNumeric()
					+ (units != null ? " " + units : "");
		}
		if (obs.getValueText() != null) {
			return obs.getValueText();
		}
		if (obs.getValueDatetime() != null) {
			return DateFormatUtil.formatDate(obs.getValueDatetime());
		}
		if (obs.getValueDrug() != null && obs.getValueDrug().getName() != null) {
			return obs.getValueDrug().getName();
		}
		return "";
	}

	private String getConceptClassName(Obs obs) {
		if (obs.getConcept() != null && obs.getConcept().getConceptClass() != null) {
			String name = obs.getConcept().getConceptClass().getName();
			if (name != null && !name.trim().isEmpty()) {
				String trimmed = name.trim();
				// "Question" is a metadata class (form fields) that collides with the
				// "Question:" query separator in the LLM prompt. Map to "Assessment"
				// which is clinically appropriate and non-colliding.
				if ("Question".equalsIgnoreCase(trimmed)) {
					return "Assessment";
				}
				return trimmed;
			}
		}
		return "";
	}

	private String getUnits(org.openmrs.Concept concept) {
		if (concept instanceof ConceptNumeric) {
			return ((ConceptNumeric) concept).getUnits();
		}
		return null;
	}

}
