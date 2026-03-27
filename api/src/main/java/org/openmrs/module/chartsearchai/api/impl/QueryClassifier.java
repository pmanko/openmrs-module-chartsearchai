/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.api.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Lightweight keyword-based classifier that maps a user query to the clinical
 * resource types most likely to contain the answer.
 *
 * <p>The classifier detects <b>types only</b>. It does <b>not</b> classify
 * queries as "category" vs "focused" — the retrieval layer handles that
 * distinction automatically by applying gap detection within type-matched
 * records. This data-driven approach eliminates the fragile word-pattern
 * heuristics that previously required manual patching for each new query
 * phrasing ("results", "each", "every", etc.).
 *
 * <h3>Vocabulary-driven design</h3>
 * All clinical terms are defined <b>once</b> in vocabulary arrays at the top
 * of this class. Type detection patterns are generated automatically from
 * these arrays.
 *
 * <h3>Adding new terms</h3>
 * <ul>
 *   <li><b>Generic terms</b>: add a {@code {"singular", "plural"}} pair to
 *       the relevant {@code *_TERMS} array. Both forms are included in the
 *       type detection regex automatically.</li>
 *   <li><b>Specific terms</b>: add to the relevant {@code *_SPECIFIC} array.
 *       Use for clinical concepts like "cd4" or "diabetes" that name specific
 *       items rather than record categories.</li>
 *   <li><b>Multi-word patterns</b> (regex): add to the relevant
 *       {@code *_MULTI_WORD} array. These may contain regex operators
 *       (e.g. {@code blood\\s*pressure}).</li>
 * </ul>
 *
 * <p>The classifier is intentionally simple — it uses regex keyword matching
 * rather than ML classification because the resource type vocabulary is small
 * and well-defined, and false positives (boosting an irrelevant type) are much
 * less harmful than false negatives (missing a relevant type).
 */
public class QueryClassifier {

	/**
	 * Classification result containing the matched resource types.
	 */
	public static class QueryIntent {

		private final Set<String> targetTypes;

		private final boolean categoryQuery;

		QueryIntent(Set<String> targetTypes, boolean categoryQuery) {
			this.targetTypes = Collections.unmodifiableSet(targetTypes);
			this.categoryQuery = categoryQuery;
		}

		/** Resource types the query is likely asking about (may be empty). */
		public Set<String> getTargetTypes() {
			return targetTypes;
		}

		/**
		 * Always returns {@code false}. Category vs focused distinction is
		 * now handled by the retrieval layer's gap detection algorithm.
		 *
		 * @deprecated Retained for API compatibility. Do not branch on this.
		 */
		public boolean isCategoryQuery() {
			return categoryQuery;
		}
	}

	// =====================================================================
	// VOCABULARY — single source of truth for all clinical term patterns.
	//
	// Each type group defines terms in up to three arrays:
	//
	//   *_TERMS (String[][]):
	//     Generic {singular, plural} pairs. These appear in type detection
	//     AND category detection. Category queries use PLURAL forms; the
	//     "every/each" pattern uses SINGULAR forms. Adding a row here
	//     updates all three automatically.
	//
	//   *_SPECIFIC (String[]):
	//     Concepts for type detection/boosting ONLY. These never trigger
	//     category expansion. Use for clinical terms that name specific
	//     items (cd4, diabetes, hemoglobin) rather than record categories.
	//
	//   *_MULTI_WORD (String[]):
	//     Regex patterns for compound terms (e.g. "blood\\s*pressure").
	//     Type detection only. Cannot appear in category patterns because
	//     the proximity regex requires single-word type anchors.
	// =====================================================================

	// --- Medications ---------------------------------------------------
	private static final String[][] MEDICATION_TERMS = {
			{"medication", "medications"},
			{"drug", "drugs"},
			{"prescription", "prescriptions"},
			{"med", "meds"},
			{"pill", "pills"},
			{"tablet", "tablets"},
	};
	private static final String[] MEDICATION_SPECIFIC = {
			"capsule", "capsules", "rx", "dose", "dosing", "dosage",
			"regimen", "arv", "arvs", "antiretroviral"
	};

	// --- Dispense (type detection only — no generic pairs, no category)
	private static final String[] DISPENSE_SPECIFIC = {
			"dispense", "dispensed", "dispensing", "filled",
			"refill", "refills", "pharmacy"
	};

	// --- Allergies -----------------------------------------------------
	private static final String[][] ALLERGY_TERMS = {
			{"allergy", "allergies"},
			{"reaction", "reactions"},
	};
	private static final String[] ALLERGY_SPECIFIC = {
			"allergic", "allergen", "allergens",
			"anaphylaxis", "hypersensitivity", "intolerance"
	};

	// --- Labs / Tests --------------------------------------------------
	private static final String[][] LAB_TERMS = {
			{"lab", "labs"},
			{"test", "tests"},
			{"panel", "panels"},
			{"bloodwork", "bloodwork"},
	};
	private static final String[] LAB_SPECIFIC = {
			"result", "results",
			"level", "levels", "count", "counts", "specimen",
			"hba1c", "hemoglobin", "haemoglobin", "hb",
			"glucose", "creatinine",
			"wbc", "rbc", "platelet", "platelets", "cd4"
	};
	private static final String[] LAB_MULTI_WORD = {"blood\\s*work"};

	// --- Conditions ----------------------------------------------------
	private static final String[][] CONDITION_TERMS = {
			{"condition", "conditions"},
			{"disease", "diseases"},
			{"illness", "illnesses"},
			{"problem", "problems"},
			{"comorbidity", "comorbidities"},
	};
	private static final String[] CONDITION_SPECIFIC = {
			"chronic", "hypertension", "diabetes", "hiv", "tb", "malaria"
	};

	// --- Diagnoses -----------------------------------------------------
	private static final String[][] DIAGNOSIS_TERMS = {
			{"diagnosis", "diagnoses"},
	};
	private static final String[] DIAGNOSIS_SPECIFIC = {"diagnosed"};

	// --- Programs ------------------------------------------------------
	private static final String[][] PROGRAM_TERMS = {
			{"program", "programs"},
	};
	private static final String[] PROGRAM_SPECIFIC = {
			"enrolled", "enrollment", "enrolment"
	};
	private static final String[] PROGRAM_MULTI_WORD = {
			"treatment\\s*program", "care\\s*program"
	};

	// --- Vitals --------------------------------------------------------
	private static final String[][] VITALS_TERMS = {
			{"vital", "vitals"},
	};
	private static final String[] VITALS_SPECIFIC = {
			"bp", "systolic", "diastolic", "temperature", "temp",
			"pulse", "weight", "height", "bmi", "spo2"
	};
	private static final String[] VITALS_MULTI_WORD = {
			"blood\\s*pressure", "heart\\s*rate",
			"respiratory\\s*rate", "oxygen\\s*saturation"
	};

	// =====================================================================
	// TYPE DETECTION PATTERNS — generated from vocabulary above
	// =====================================================================

	private static final Pattern MEDICATION_PATTERN = buildTypePattern(
			MEDICATION_TERMS, MEDICATION_SPECIFIC, null);

	private static final Pattern DISPENSE_PATTERN = buildTypePattern(
			null, DISPENSE_SPECIFIC, null);

	private static final Pattern ALLERGY_PATTERN = buildTypePattern(
			ALLERGY_TERMS, ALLERGY_SPECIFIC, null);

	private static final Pattern LAB_PATTERN = buildTypePattern(
			LAB_TERMS, LAB_SPECIFIC, LAB_MULTI_WORD);

	private static final Pattern CONDITION_PATTERN = buildTypePattern(
			CONDITION_TERMS, CONDITION_SPECIFIC, null);

	private static final Pattern DIAGNOSIS_PATTERN = buildTypePattern(
			DIAGNOSIS_TERMS, DIAGNOSIS_SPECIFIC, null);

	private static final Pattern PROGRAM_PATTERN = buildTypePattern(
			PROGRAM_TERMS, PROGRAM_SPECIFIC, PROGRAM_MULTI_WORD);

	private static final Pattern VITALS_PATTERN = buildTypePattern(
			VITALS_TERMS, VITALS_SPECIFIC, VITALS_MULTI_WORD);

	// =====================================================================
	// PATTERN BUILDING HELPERS
	// =====================================================================

	/**
	 * Builds a type detection regex from generic pairs, specific terms, and
	 * optional multi-word patterns. All alternatives are combined into a
	 * single {@code \\b(?:...)\\b} pattern.
	 */
	private static Pattern buildTypePattern(String[][] pairs, String[] specific,
			String[] multiWord) {
		List<String> terms = new ArrayList<String>();
		if (pairs != null) {
			for (String[] pair : pairs) {
				terms.add(pair[0]);
				if (!pair[0].equals(pair[1])) {
					terms.add(pair[1]);
				}
			}
		}
		if (specific != null) {
			for (String s : specific) {
				terms.add(s);
			}
		}
		if (multiWord != null) {
			for (String mw : multiWord) {
				terms.add(mw);
			}
		}
		return Pattern.compile("\\b(?:" + joinPipe(terms) + ")\\b");
	}

	private static String joinPipe(List<String> items) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < items.size(); i++) {
			if (i > 0) {
				sb.append("|");
			}
			sb.append(items.get(i));
		}
		return sb.toString();
	}

	private QueryClassifier() {
	}

	/**
	 * Classifies the raw user query and returns the inferred type intent.
	 *
	 * @param rawQuery the original user question
	 * @return the classification result (never null; targetTypes may be empty)
	 */
	public static QueryIntent classify(String rawQuery) {
		if (rawQuery == null || rawQuery.trim().isEmpty()) {
			return new QueryIntent(Collections.emptySet(), false);
		}
		String lower = rawQuery.toLowerCase();

		Set<String> types = new HashSet<String>();
		boolean hasMedication = MEDICATION_PATTERN.matcher(lower).find();
		boolean hasDispense = DISPENSE_PATTERN.matcher(lower).find();
		boolean hasAllergy = ALLERGY_PATTERN.matcher(lower).find();
		boolean hasLab = LAB_PATTERN.matcher(lower).find();
		boolean hasCondition = CONDITION_PATTERN.matcher(lower).find();
		boolean hasDiagnosis = DIAGNOSIS_PATTERN.matcher(lower).find();
		boolean hasProgram = PROGRAM_PATTERN.matcher(lower).find();
		boolean hasVitals = VITALS_PATTERN.matcher(lower).find();

		if (hasMedication) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_ORDER);
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE);
		}
		if (hasDispense) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE);
			if (!hasMedication) {
				types.add(ChartSearchAiConstants.RESOURCE_TYPE_ORDER);
			}
		}
		if (hasAllergy) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY);
		}
		if (hasLab) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_OBS);
		}
		if (hasCondition) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION);
		}
		if (hasDiagnosis) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS);
		}
		if (hasProgram) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM);
		}
		if (hasVitals) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_OBS);
		}

		// Category detection is NOT done here. The retrieval layer applies
		// gap detection within type-matched records to auto-determine how
		// many to include — no query wording heuristics needed.
		return new QueryIntent(types, false);
	}
}
