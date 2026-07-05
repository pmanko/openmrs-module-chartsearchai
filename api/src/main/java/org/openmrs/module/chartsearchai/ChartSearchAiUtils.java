/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai;

import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_CONDITION;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_OBS;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_ORDER;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.openmrs.Concept;
import org.openmrs.ConceptSet;
import org.openmrs.api.context.Context;

public class ChartSearchAiUtils {

	/**
	 * Builds a composite key from a resource type and resource UUID.
	 * This is the single canonical format for resource keys used across
	 * retrieval pipelines, filter methods, and result sets.
	 *
	 * @param resourceType the resource type constant
	 * @param resourceUuid the resource UUID
	 * @return a key in the format "resourceType:resourceUuid"
	 */
	public static String resourceKey(String resourceType, String resourceUuid) {
		return resourceType + ":" + resourceUuid;
	}

	/**
	 * Returns a semantic prefix for the given resource type and text, used when computing
	 * embeddings to help the embedding model distinguish between record types.
	 * This prefix is only prepended to the text for embedding computation, not
	 * for display in the LLM prompt.
	 *
	 * @param resourceType the resource type constant
	 * @param text the serialized record text, used to refine the prefix for types
	 *        that have sub-types (e.g. drug orders vs test orders)
	 * @return a descriptive prefix ending with ": "
	 */
	private static String getEmbeddingPrefix(String resourceType, String text) {
		switch (resourceType) {
			case RESOURCE_TYPE_OBS:
				return "Clinical observation: ";
			case RESOURCE_TYPE_CONDITION:
				return "Medical condition: ";
			case RESOURCE_TYPE_ALLERGY:
				return "Patient allergy: ";
			case RESOURCE_TYPE_DIAGNOSIS:
				return "Clinical diagnosis: ";
			case RESOURCE_TYPE_ORDER:
				if (text != null && text.startsWith("Drug order:")) {
					return "Medication prescription: ";
				}
				if (text != null && text.startsWith("Test order:")) {
					return "Lab or diagnostic test: ";
				}
				if (text != null && text.startsWith("Referral order:")) {
					return "Clinical referral: ";
				}
				return "Clinical order: ";
			case RESOURCE_TYPE_PROGRAM:
				return "Program enrollment: ";
			case RESOURCE_TYPE_MEDICATION_DISPENSE:
				return "Medication dispensed: ";
			default:
				return "";
		}
	}

	/**
	 * Builds the full prefixed text used for embedding and keyword matching.
	 * This is the single source of truth for the
	 * {@code getEmbeddingPrefix(resourceType, text) + text} pattern.
	 *
	 * @param resourceType the resource type constant
	 * @param text the serialized record text
	 * @return the prefixed text ready for embedding or keyword scoring
	 */
	public static String buildPrefixedText(String resourceType, String text) {
		return buildPrefixedText(resourceType, text, Collections.<String>emptyList());
	}

	/**
	 * Builds the prefixed embedding text with optional category hints injected
	 * between the structural prefix and the serialized text. Hints come from
	 * OpenMRS concept metadata (currently {@code getSetsContainingConcept}) and
	 * help the embedding model bridge category-name queries (e.g. "vital signs"
	 * → Temperature/BP/Pulse) when the literal category word does not appear
	 * in the serialized record text. Empty hints produce identical output to
	 * the 2-arg overload.
	 *
	 * <p>Example output with hints {@code ["Vital signs"]}:
	 * {@code "Clinical observation: Vital signs / Finding — Temperature: 36.7"}.
	 *
	 * @param resourceType the resource type constant
	 * @param text the serialized record text
	 * @param categoryHints concept-set names (or other category metadata)
	 *        derived from the source domain object; may be empty
	 * @return the prefixed text ready for embedding or keyword scoring
	 */
	public static String buildPrefixedText(String resourceType, String text,
			List<String> categoryHints) {
		return getEmbeddingPrefix(resourceType, text)
				+ injectCategoryHints(text, categoryHints);
	}

	/**
	 * Prepends category hints to the body text without adding a structural
	 * prefix. Used to enrich a record's serialized text so any consumer that
	 * re-prefixes the hint-augmented body gets a consistent string. The 2-arg
	 * {@link #buildPrefixedText(String, String)} called on hint-injected body
	 * produces the same prefixed text as the 3-arg overload called on the
	 * raw body with hints.
	 *
	 * <p>Empty or null hints return the body unchanged.</p>
	 *
	 * @param body the serialized record body (no structural prefix)
	 * @param categoryHints hints to inject
	 * @return body with hints prepended (e.g. "Vital signs / Finding — Temp: 37"),
	 *         or unchanged body if hints are empty
	 */
	public static String injectCategoryHints(String body, List<String> categoryHints) {
		if (categoryHints == null || categoryHints.isEmpty()) {
			return body;
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < categoryHints.size(); i++) {
			if (i > 0) {
				sb.append(" / ");
			}
			sb.append(categoryHints.get(i));
		}
		sb.append(" / ").append(body);
		return sb.toString();
	}

	/**
	 * Strips category hint prefixes from text that was enriched by
	 * {@link #injectCategoryHints}. The hint format is
	 * {@code "hint1 / hint2 / ... / originalBody"}. This method finds
	 * the original body by scanning for the first occurrence of a known
	 * record-body pattern (em-dash for Obs, "Condition:", "Diagnosis:",
	 * etc.) and taking the " / " boundary just before it.
	 *
	 * @param text the potentially hint-enriched text
	 * @return the original body without hint prefixes, or the input
	 *         unchanged if no hints are detected
	 */
	public static String stripCategoryHints(String text) {
		if (text == null || !text.contains(" / ")) {
			return text;
		}
		// Find the earliest known record-body pattern
		int earliest = text.length();
		// Obs: "TYPE — CONCEPT:"
		int emDash = text.indexOf(" \u2014 ");
		if (emDash >= 0 && emDash < earliest) {
			earliest = emDash;
		}
		// Condition/Diagnosis/Order/Allergy/Program patterns
		String[] patterns = { "Condition: ", "Diagnosis: ", "Drug order: ",
				"Test order: ", "Referral order: ", "Dispensed: ",
				"Allergy: ", "Program: " };
		for (String p : patterns) {
			int idx = text.indexOf(p);
			if (idx >= 0 && idx < earliest) {
				earliest = idx;
			}
		}
		if (earliest == text.length()) {
			return text; // no known pattern found
		}
		// Find the " / " boundary just before the pattern
		String prefix = text.substring(0, earliest);
		int lastSlash = prefix.lastIndexOf(" / ");
		if (lastSlash >= 0) {
			return text.substring(lastSlash + 3);
		}
		return text;
	}

	/**
	 * Extracts category hints for a concept by looking up the concept sets
	 * (CIEL convention: e.g. concept 1114 "Vital signs" contains Temperature,
	 * BP, Pulse, RR, SpO2). The returned list contains the names of the
	 * containing set concepts and is intended to be passed to the 3-arg
	 * {@link #buildPrefixedText(String, String, List)} so the literal
	 * category word ends up in the embedding input.
	 *
	 * <p>Returns an empty list when the concept is null, has no containing
	 * sets, or when the OpenMRS context is unavailable (e.g. during tests
	 * that bypass Spring). This is intentional — callers should not need to
	 * special-case the no-hints scenario.
	 *
	 * <p>Only concept-set names are used as hints. Concept descriptions are
	 * deliberately excluded — they can restate the concept name with
	 * different vocabulary, creating asymmetric semantic bias between
	 * related concepts (e.g. "Patient's weight in kilograms" has more
	 * overlap with "BMI" than "Patient's height in centimeters", causing
	 * Height to be dropped from BMI queries).
	 *
	 * @param concept the source concept
	 * @return list of containing-set names, or empty list
	 */
	public static List<String> extractCategoryHints(Concept concept) {
		if (concept == null) {
			return Collections.emptyList();
		}
		List<String> hints = new ArrayList<String>();

		// Concept-set membership (e.g. Temperature → "Vital signs")
		try {
			List<ConceptSet> sets = Context.getConceptService()
					.getSetsContainingConcept(concept);
			if (sets != null) {
				for (ConceptSet cs : sets) {
					Concept setConcept = cs.getConceptSet();
					if (setConcept == null || setConcept.getName() == null) {
						continue;
					}
					String name = setConcept.getName().getName();
					if (name != null && !name.trim().isEmpty()) {
						hints.add(name.trim());
					}
				}
			}
		}
		catch (Exception e) {
			// Context unavailable (test bypass) or transient API failure
		}

		return hints;
	}

	/**
	 * Returns the complete set of structural embedding prefixes used by
	 * {@link #getEmbeddingPrefix} across all supported resource types and
	 * sub-types. Used by keyword scoring to identify "type indicator"
	 * query terms — words that appear in any structural prefix and so
	 * should only match the prefix portion of records, not narrative
	 * body text. The set is the static prefix vocabulary, independent
	 * of which resource types appear in any particular dataset.
	 *
	 * @return set of all possible prefix strings (each ends with ": ")
	 */
	public static Set<String> getAllEmbeddingPrefixes() {
		Set<String> prefixes = new java.util.HashSet<String>();
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_OBS, ""));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_CONDITION, ""));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_ALLERGY, ""));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_DIAGNOSIS, ""));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_PROGRAM, ""));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_MEDICATION_DISPENSE, ""));
		// ORDER has sub-type prefixes triggered by body text — enumerate them.
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_ORDER, "Drug order:"));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_ORDER, "Test order:"));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_ORDER, "Referral order:"));
		prefixes.add(getEmbeddingPrefix(RESOURCE_TYPE_ORDER, ""));
		prefixes.remove("");
		return prefixes;
	}

	private ChartSearchAiUtils() {
	}
}
