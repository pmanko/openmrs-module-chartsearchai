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

import java.util.List;

import org.openmrs.Allergy;
import org.openmrs.AllergyReaction;
import org.openmrs.AllergenType;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.springframework.stereotype.Component;

/**
 * Serializes an {@link Allergy} into embedding-friendly text.
 *
 * <p>Example output: {@code "Allergy: Penicillin (DRUG). Severity: Severe.
 * Reactions: Anaphylaxis, Rash"}</p>
 */
@Component
public class AllergyTextSerializer implements ClinicalTextSerializer<Allergy> {

	@Override
	public String toText(Allergy allergy) {
		StringBuilder sb = new StringBuilder();
		String name = getAllergenName(allergy);
		if (!name.isEmpty()) {
			sb.append("Allergy: ").append(name);
			if (allergy.getAllergen() != null && allergy.getAllergen().getAllergenType() != null) {
				sb.append(" (").append(formatAllergenType(allergy.getAllergen().getAllergenType())).append(")");
			}
		}
		if (allergy.getSeverity() != null) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Severity: ").append(ConceptNameUtil.getName(allergy.getSeverity()));
		}

		List<AllergyReaction> reactions = allergy.getReactions();
		if (reactions != null && !reactions.isEmpty()) {
			StringBuilder reactionSb = new StringBuilder();
			for (AllergyReaction reaction : reactions) {
				String reactionName = "";
				if (reaction.getReaction() != null) {
					reactionName = ConceptNameUtil.getName(reaction.getReaction());
				} else if (reaction.getReactionNonCoded() != null
						&& !reaction.getReactionNonCoded().trim().isEmpty()) {
					reactionName = reaction.getReactionNonCoded().trim();
				}
				if (!reactionName.isEmpty()) {
					if (reactionSb.length() > 0) {
						reactionSb.append(", ");
					}
					reactionSb.append(reactionName);
				}
			}
			if (reactionSb.length() > 0) {
				if (sb.length() > 0) {
					sb.append(". ");
				}
				sb.append("Reactions: ").append(reactionSb);
			}
		}
		if (allergy.getComments() != null && !allergy.getComments().trim().isEmpty()) {
			if (sb.length() > 0) {
				sb.append(". ");
			}
			sb.append("Comments: ").append(allergy.getComments().trim());
		}

		return sb.toString();
	}

	private static String formatAllergenType(AllergenType type) {
		switch (type) {
			case DRUG:
				return "drug allergen";
			case FOOD:
				return "food allergen";
			case ENVIRONMENT:
				return "environmental allergen";
			default:
				return type.toString();
		}
	}

	private String getAllergenName(Allergy allergy) {
		if (allergy.getAllergen() == null) {
			return "";
		}
		if (allergy.getAllergen().getCodedAllergen() != null) {
			return ConceptNameUtil.getName(allergy.getAllergen().getCodedAllergen());
		}
		if (allergy.getAllergen().getNonCodedAllergen() != null
				&& !allergy.getAllergen().getNonCodedAllergen().trim().isEmpty()) {
			return allergy.getAllergen().getNonCodedAllergen().trim();
		}
		return "";
	}
}
