/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.openmrs.Concept;
import org.openmrs.ConceptName;
import org.openmrs.api.context.Context;

/**
 * Resolves a concept's preferred name enriched with same-locale synonyms so that embeddings
 * capture alternative terms clinicians may search for (e.g. "Hypertension (HTN, High Blood
 * Pressure)"). Limits synonyms to three to keep serialized text concise.
 */
public final class ConceptNameUtil {

	private static final int MAX_SYNONYMS = 3;

	private ConceptNameUtil() {
	}

	// Matches synonym parentheses added by getName(), e.g. " (syn. HTN, High Blood Pressure)"
	// The "syn. " prefix distinguishes these from structural parens like "(food allergen)".
	private static final Pattern SYNONYM_PARENS = Pattern.compile(" \\(syn\\. [^)]*\\)");

	/**
	 * Condenses synonym parentheses into a compact slash format so the LLM receives
	 * concise but synonym-aware records. Keeps only the first synonym to minimize noise.
	 *
	 * <p>For example, {@code "Diarrhea (syn. Diarrhoea, Loose bowels)"} becomes
	 * {@code "Diarrhea/Diarrhoea"}. Structural parentheses like "(food allergen)" are
	 * preserved.</p>
	 *
	 * @param text the serialized record text
	 * @return text with synonym parentheses condensed to slash format
	 */
	public static String condenseSynonyms(String text) {
		if (text == null) {
			return null;
		}
		Matcher matcher = SYNONYM_PARENS.matcher(text);
		StringBuffer sb = new StringBuffer();
		while (matcher.find()) {
			// Extract content after "syn. ", take only the first synonym
			String content = matcher.group();
			String inner = content.substring(" (syn. ".length(), content.length() - 1);
			String firstSynonym = inner.split(",")[0].trim();
			matcher.appendReplacement(sb, Matcher.quoteReplacement("/" + firstSynonym));
		}
		matcher.appendTail(sb);
		return sb.toString();
	}

	/**
	 * Strips synonym parentheses from serialized text entirely.
	 *
	 * @param text the serialized record text
	 * @return text with synonym parentheses removed
	 */
	public static String stripSynonyms(String text) {
		if (text == null) {
			return null;
		}
		return SYNONYM_PARENS.matcher(text).replaceAll("");
	}

	/**
	 * Returns just the preferred concept name without synonyms. Useful when concise text
	 * is needed, such as in allergy reaction and severity fields where synonyms would add
	 * noise that hurts small-LLM citation accuracy.
	 *
	 * @param concept the concept to resolve
	 * @return the preferred name, or empty string if unavailable
	 */
	public static String getPreferredName(Concept concept) {
		if (concept == null) {
			return "";
		}
		ConceptName preferred = concept.getName();
		return preferred != null ? preferred.getName() : "";
	}

	/**
	 * Returns the preferred concept name followed by up to three same-locale synonyms in
	 * parentheses.
	 *
	 * @param concept the concept to resolve
	 * @return a name string such as {@code "Hypertension (HTN, High Blood Pressure)"}
	 */
	public static String getName(Concept concept) {
		if (concept == null) {
			return "";
		}
		ConceptName preferred = concept.getName();
		if (preferred == null) {
			return "";
		}

		String preferredText = preferred.getName();
		Locale locale = Context.getLocale();
		if (locale == null) {
			locale = Locale.ENGLISH;
		}
		String localeLanguage = locale.getLanguage();
		// Sort candidate names alphabetically for deterministic synonym ordering
		// across JVM restarts (concept.getNames() returns a HashSet).
		List<String> candidateNames = new ArrayList<>();
		for (ConceptName cn : concept.getNames()) {
			if (cn.getLocale() != null && !cn.getLocale().getLanguage().equals(localeLanguage)) {
				continue;
			}
			String name = cn.getName();
			if (name != null && !name.equals(preferredText)) {
				candidateNames.add(name);
			}
		}
		Collections.sort(candidateNames);

		Set<String> synonyms = new LinkedHashSet<>();
		for (String name : candidateNames) {
			if (synonyms.size() >= MAX_SYNONYMS) {
				break;
			}
			synonyms.add(name);
		}

		if (synonyms.isEmpty()) {
			return preferredText;
		}
		return preferredText + " (syn. " + String.join(", ", synonyms) + ")";
	}
}
