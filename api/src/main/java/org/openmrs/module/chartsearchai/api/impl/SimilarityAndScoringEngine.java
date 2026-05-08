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

import java.util.Set;

/**
 * Stateless scoring and statistical helpers used by the retrieval pipeline:
 * keyword overlap, semantic z-score, the Gumbel-threshold family used by
 * outlier gates, and the data-derived gap/ratio helpers used by refinement.
 * Pure functions over their parameters — no instance state, no OpenMRS
 * Context access.
 */
final class SimilarityAndScoringEngine {

	private SimilarityAndScoringEngine() {
	}

	/**
	 * Computes a keyword overlap score for a record's text content.
	 * Returns a value between 0.0 (no terms match) and 1.0 (all terms match).
	 * Uses case-insensitive substring matching so that a query term "metformin"
	 * matches "Drug order: Metformin 500mg". Also tries a simple plural stem
	 * (stripping trailing 's') so that "conditions" matches "Condition: ...".
	 *
	 * @param queryTerms the extracted query terms (lowercased)
	 * @param textContent the full serialized record text
	 * @return the fraction of query terms found in the text
	 */
	static double computeKeywordScore(String[] queryTerms, String textContent) {
		if (queryTerms.length == 0 || textContent == null || textContent.isEmpty()) {
			return 0.0;
		}
		String lowerText = textContent.toLowerCase();
		String[] textWords = lowerText.split("\\s+");
		int matched = 0;
		for (String term : queryTerms) {
			if (termMatchesText(term, lowerText, textWords)) {
				matched++;
			}
		}
		return (double) matched / queryTerms.length;
	}

	/**
	 * Variant of {@link #computeKeywordScore} where {@code typeIndicatorTerms}
	 * may only match in the structural prefix portion of the text
	 * (i.e. {@code prefixedText} minus {@code body}). Other terms match
	 * in the full text as usual. This prevents type-indicator words like
	 * "medication" from matching narrative body text such as "Medication
	 * adjusted" in encounter notes that aren't medication records.
	 */
	static double computeKeywordScoreRestricted(String[] queryTerms,
			String prefixedText, String body,
			Set<String> typeIndicatorTerms) {
		if (queryTerms.length == 0 || prefixedText == null
				|| prefixedText.isEmpty()) {
			return 0.0;
		}
		String lowerFull = prefixedText.toLowerCase();
		String[] fullWords = lowerFull.split("\\s+");
		String prefix = body == null
				? prefixedText
				: prefixedText.substring(0,
						prefixedText.length() - body.length());
		String lowerPrefix = prefix.toLowerCase();
		String[] prefixWords = lowerPrefix.split("\\s+");
		// When type indicators exist, first check if this record
		// matches ANY of them. If it doesn't, this record is the
		// wrong type — non-type-indicator matches (e.g. "active"
		// matching "Status: ACTIVE" on an obs record when the
		// query asks for "conditions") are coincidental and
		// should not contribute to keyword scoring.
		boolean matchesType = false;
		for (String term : queryTerms) {
			if (typeIndicatorTerms.contains(term)
					&& termMatchesText(term, lowerPrefix,
							prefixWords)) {
				matchesType = true;
				break;
			}
		}
		if (!matchesType) {
			return 0.0;
		}
		int matched = 0;
		for (String term : queryTerms) {
			boolean isTypeIndicator = typeIndicatorTerms.contains(term);
			if (isTypeIndicator) {
				if (termMatchesText(term, lowerPrefix, prefixWords)) {
					matched++;
				}
			} else {
				if (termMatchesText(term, lowerFull, fullWords)) {
					matched++;
				}
			}
		}
		return (double) matched / queryTerms.length;
	}

	/**
	 * Checks whether a single query term matches within the given text
	 * using three matching strategies: exact substring, plural stem,
	 * and morphological stem. This is the shared matching logic used
	 * by both {@link #computeKeywordScore} and the keyword-tier
	 * subset check in the filter pipeline.
	 *
	 * @param term the lowercase query term to match
	 * @param lowerText the lowercase full text to search
	 * @param textWords the lowercase text split into words
	 * @return true if the term matches the text by any strategy
	 */
	static boolean termMatchesText(String term, String lowerText, String[] textWords) {
		// 1. Exact substring match
		if (lowerText.contains(term)) {
			return true;
		}

		// 2. Plural stem: strip trailing 's' and match as a whole word
		// (not substring) so "order" matches "order:" but not "ordered".
		if (term.length() > 3
				&& term.endsWith("s") && !term.endsWith("ss")) {
			String stem = term.substring(0, term.length() - 1);
			for (String word : textWords) {
				// Strip trailing punctuation so "order:" becomes "order"
				String clean = word.replaceAll("[^a-z0-9]+$", "");
				if (clean.equals(stem)) {
					return true;
				}
			}
		}

		// 3. Morphological stem: trim 2-3 trailing characters to handle
		// derivational variants (allergic/allergy, prescribed/prescription).
		// Uses word-prefix matching instead of substring to avoid false
		// positives from compound words (e.g. "allerg" inside "Photoallergy").
		if (term.length() >= 7) {
			for (int trim = 2; trim <= 3; trim++) {
				String stem = term.substring(0, term.length() - trim);
				if (stem.length() >= 5) {
					for (String word : textWords) {
						if (word.startsWith(stem)) {
							return true;
						}
					}
				}
			}
		}

		// 4. Shared-prefix match for long compound medical terms:
		// "immunocompromised" and "immunodeficiency" share the root
		// "immuno" but diverge after that — no simple stemmer catches
		// this. For terms AND text words both >= 12 characters, match
		// if they share a prefix of >= 6 characters. The length
		// constraints prevent false positives on short common words
		// (e.g. "immunization" at 12 chars wouldn't false-match
		// because its prefix "immuni" differs from "immuno").
		if (term.length() >= 12) {
			String termPrefix = term.substring(0, 6);
			for (String word : textWords) {
				String clean = word.replaceAll("[^a-z0-9]+$", "");
				if (clean.length() >= 12 && clean.startsWith(termPrefix)) {
					return true;
				}
			}
		}

		return false;
	}

}
