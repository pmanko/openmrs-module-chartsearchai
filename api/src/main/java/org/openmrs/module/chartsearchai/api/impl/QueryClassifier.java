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

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;

/**
 * Lightweight keyword-based classifier that maps a user query to the clinical
 * resource types most likely to contain the answer. Also detects broad "category"
 * queries (e.g., "any medications?") that expect exhaustive results rather than
 * just the top few matches.
 *
 * <p>The classifier is intentionally simple — it uses regex keyword matching
 * rather than ML classification because the resource type vocabulary is small
 * and well-defined, and false positives (boosting an irrelevant type) are much
 * less harmful than false negatives (missing a relevant type).
 */
public class QueryClassifier {

	/**
	 * Classification result containing the matched resource types and whether
	 * the query appears to be a broad category query.
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
		 * True if the query is a broad category query expecting exhaustive
		 * results (e.g., "any medications?", "list all conditions"). These
		 * queries should bypass topK limits for matched resource types.
		 */
		public boolean isCategoryQuery() {
			return categoryQuery;
		}
	}

	// Each entry maps a keyword pattern to resource types it implies.
	// Patterns are tested against the lowercased, stopword-stripped query.

	private static final Pattern MEDICATION_PATTERN = Pattern.compile(
			"\\b(?:medication|medications|drug|drugs|prescription|prescriptions"
					+ "|med|meds|rx|dose|dosing|dosage|pill|pills|tablet|tablets"
					+ "|capsule|capsules|regimen|arv|arvs|antiretroviral)\\b");

	private static final Pattern DISPENSE_PATTERN = Pattern.compile(
			"\\b(?:dispens|dispensed|dispensing|filled|refill|refills|pharmacy)\\b");

	private static final Pattern ALLERGY_PATTERN = Pattern.compile(
			"\\b(?:allergy|allergies|allergic|allergen|allergens"
					+ "|reaction|reactions|anaphylaxis|hypersensitivity|intolerance)\\b");

	private static final Pattern LAB_PATTERN = Pattern.compile(
			"\\b(?:lab|labs|test|tests|result|results|level|levels|count|counts"
					+ "|panel|panels|bloodwork|blood\\s*work|specimen|hba1c|hemoglobin"
					+ "|glucose|creatinine|wbc|rbc|platelet|platelets|cd4)\\b");

	private static final Pattern CONDITION_PATTERN = Pattern.compile(
			"\\b(?:condition|conditions|diagnosis|diagnoses|diagnosed"
					+ "|disease|diseases|illness|problem|problems|comorbidity"
					+ "|comorbidities|chronic|hypertension|diabetes|hiv|tb|malaria)\\b");

	private static final Pattern PROGRAM_PATTERN = Pattern.compile(
			"\\b(?:program|programs|enrolled|enrollment|enrolment"
					+ "|treatment\\s*program|care\\s*program)\\b");

	private static final Pattern VITALS_PATTERN = Pattern.compile(
			"\\b(?:vital|vitals|blood\\s*pressure|bp|systolic|diastolic"
					+ "|temperature|temp|pulse|heart\\s*rate|respiratory\\s*rate"
					+ "|weight|height|bmi|spo2|oxygen\\s*saturation)\\b");

	// Broad category indicators — words that suggest the user wants ALL
	// records of a type, not just the most relevant few.
	private static final Pattern CATEGORY_INDICATOR = Pattern.compile(
			"\\b(?:any|all|list|what|which|every|show|tell)\\b");

	private QueryClassifier() {
	}

	/**
	 * Classifies the normalized (stopword-stripped) query and returns the
	 * inferred intent.
	 *
	 * @param normalizedQuery the query after stopword removal and lowercasing
	 * @return the classification result (never null; targetTypes may be empty)
	 */
	public static QueryIntent classify(String normalizedQuery) {
		String lower = normalizedQuery.toLowerCase();

		Set<String> types = new HashSet<String>();
		boolean hasMedication = MEDICATION_PATTERN.matcher(lower).find();
		boolean hasDispense = DISPENSE_PATTERN.matcher(lower).find();
		boolean hasAllergy = ALLERGY_PATTERN.matcher(lower).find();
		boolean hasLab = LAB_PATTERN.matcher(lower).find();
		boolean hasCondition = CONDITION_PATTERN.matcher(lower).find();
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
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS);
		}
		if (hasProgram) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM);
		}
		if (hasVitals) {
			types.add(ChartSearchAiConstants.RESOURCE_TYPE_OBS);
		}

		// A category query has a broad indicator AND targets specific types.
		// "any medications?" → category. "tell me about the patient" → not
		// category (no specific type matched).
		boolean isCategory = !types.isEmpty()
				&& CATEGORY_INDICATOR.matcher(lower).find();

		return new QueryIntent(types, isCategory);
	}
}
