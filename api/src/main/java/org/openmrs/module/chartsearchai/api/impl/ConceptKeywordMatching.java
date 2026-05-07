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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Concept-name aware keyword helpers used by the retrieval pipeline:
 * embedding-similarity-based query expansion (when no record contains the
 * verbatim query terms but a concept name is semantically close), and
 * type-indicator matching against a record's resource-type prefix
 * (with sister-resource-type fallback for condition ↔ diagnosis).
 */
final class ConceptKeywordMatching {

	private ConceptKeywordMatching() {
	}

	private static final Logger log = LoggerFactory.getLogger(ConceptKeywordMatching.class);

	private static final Set<String> CONCEPT_EXPANSION_FUNCTION_WORDS =
			Collections.unmodifiableSet(new HashSet<String>(
					Arrays.asList(
							"of", "to", "and", "or", "the", "a", "an",
							"in", "on", "for", "with", "by", "as")));

	/**
	 * Resource types that are structurally equivalent in the data model.
	 * Conditions and diagnoses both represent clinical findings about a
	 * patient's health. When a type indicator matches one, the other
	 * should also be kept — they are two representations of the same
	 * concept type.
	 */
	private static final Map<String, String> SISTER_RESOURCE_TYPES;
	static {
		Map<String, String> m = new HashMap<String, String>();
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION,
				ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS);
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS,
				ChartSearchAiConstants.RESOURCE_TYPE_CONDITION);
		SISTER_RESOURCE_TYPES = Collections.unmodifiableMap(m);
	}

	/**
	 * Embedding-based query expansion. When the user's query phrasing
	 * doesn't appear verbatim in any record but a concept name in the
	 * patient's chart is semantically very close to the query (per the
	 * embedding model's query encoder), replaces {@code kwTerms} with
	 * that concept's tokens so keyword scoring can anchor on the
	 * actual record vocabulary.
	 *
	 * <p>Activation rule (all required, designed to avoid breaking
	 * queries that already work): {@code kwTerms.length} between 1 and
	 * {@link ChartSearchAiConstants#CONCEPT_EXPANSION_MAX_KW_TERMS}; no
	 * record contains every {@code kwTerms} substring; the closest
	 * concept's similarity AND its margin over the second-best concept
	 * satisfy one of two model-tuned paths (very-high absolute similarity
	 * with modest margin, or moderate similarity with a larger margin);
	 * and the concept's tokens differ from {@code kwTerms}.
	 *
	 * <p>Cost: one query-encoder call per unique concept name in the
	 * patient's chart (typically 20-40), plus one cosine pass. Returns
	 * the same reference as {@code kwTerms} when no expansion applies,
	 * so callers can detect via identity comparison.
	 */
	static String[] expandKwTermsViaConceptSimilarity(String[] kwTerms,
			float[] queryVector, ChartEmbedding[] embeddings,
			EmbeddingProvider provider,
			PipelineConfig config) {
		if (kwTerms.length == 0
				|| kwTerms.length > ChartSearchAiConstants.CONCEPT_EXPANSION_MAX_KW_TERMS
				|| embeddings == null || embeddings.length == 0
				|| provider == null || queryVector == null) {
			return kwTerms;
		}
		// Skip when at least one record already contains every kwTerm —
		// keyword scoring is already finding full matches, replacing
		// would only narrow what we already capture correctly.
		for (ChartEmbedding ce : embeddings) {
			String text = ce.getTextContent();
			if (text == null) continue;
			String lower = text.toLowerCase();
			boolean allFound = true;
			for (String t : kwTerms) {
				if (!lower.contains(t)) {
					allFound = false;
					break;
				}
			}
			if (allFound) {
				return kwTerms;
			}
		}
		Set<String> conceptNames = new LinkedHashSet<String>();
		for (ChartEmbedding ce : embeddings) {
			String name = ConceptNameUtil.extractConceptName(ce.getTextContent());
			if (name != null) {
				String trimmed = name.trim();
				if (!trimmed.isEmpty()) {
					conceptNames.add(trimmed);
				}
			}
		}
		if (conceptNames.isEmpty()) {
			return kwTerms;
		}
		// Find the closest concept (by query-encoder cosine) AND track
		// the second-best score to assess "clear winner" margin.
		String bestConcept = null;
		double bestSim = -1;
		double secondSim = -1;
		for (String name : conceptNames) {
			float[] vec;
			try {
				vec = provider.embedQuery(name);
			} catch (Exception e) {
				continue;
			}
			if (vec == null || vec.length != queryVector.length) {
				continue;
			}
			double sim = ChartSearchAiUtils.cosineSimilarity(queryVector, vec);
			if (sim > bestSim) {
				secondSim = bestSim;
				bestSim = sim;
				bestConcept = name;
			} else if (sim > secondSim) {
				secondSim = sim;
			}
		}
		double margin = bestSim - secondSim;
		boolean veryHighWithModestMargin =
				bestSim >= config.conceptExpansionVeryHighMinSim
				&& margin >= config.conceptExpansionVeryHighMinMargin;
		boolean moderateWithLargeMargin =
				bestSim >= config.conceptExpansionModerateMinSim
				&& margin >= config.conceptExpansionModerateMinMargin;
		if (bestConcept == null
				|| (!veryHighWithModestMargin && !moderateWithLargeMargin)) {
			return kwTerms;
		}
		// Vocabulary-overlap guard: below the bypass threshold, require
		// at least one original kwTerm to appear in the concept name.
		if (bestSim < config.conceptExpansionVocabBypassThreshold) {
			String conceptLower = bestConcept.toLowerCase();
			boolean hasOverlap = false;
			for (String t : kwTerms) {
				if (t.length() >= ChartSearchAiConstants.CONCEPT_EXPANSION_MIN_OVERLAP_LENGTH
						&& conceptLower.contains(t)) {
					hasOverlap = true;
					break;
				}
			}
			if (!hasOverlap) {
				return kwTerms;
			}
		}
		// Tokenise the concept name. Strip non-alphanumerics, drop
		// single-character tokens that would match almost any text.
		String[] rawTokens = bestConcept.toLowerCase()
				.replaceAll("[^a-z0-9 ]+", " ").trim().split("\\s+");
		// Filter the concept tokens to keep only specific, content-bearing
		// terms: drop English function words, then IDF-filter by document
		// frequency.
		int idfThreshold = Math.max(1, (int) Math.ceil(embeddings.length
				* ChartSearchAiConstants.CONCEPT_EXPANSION_IDF_FRACTION));
		List<String> conceptTokens = new ArrayList<String>();
		for (String t : rawTokens) {
			if (t.length() < 2) continue;
			if (CONCEPT_EXPANSION_FUNCTION_WORDS.contains(t)) continue;
			int df = 0;
			for (ChartEmbedding ce : embeddings) {
				String text = ce.getTextContent();
				if (text != null && text.toLowerCase().contains(t)) {
					df++;
				}
			}
			if (df > 0 && df <= idfThreshold) {
				conceptTokens.add(t);
			}
		}
		if (conceptTokens.isEmpty()) {
			return kwTerms;
		}
		// Skip the no-op case: expansion produces the same set as the
		// original kwTerms.
		Set<String> existing = new HashSet<String>();
		for (String t : kwTerms) existing.add(t.toLowerCase());
		Set<String> conceptSet = new HashSet<String>(conceptTokens);
		if (existing.equals(conceptSet)) {
			return kwTerms;
		}
		log.warn("Concept-similarity expansion: kwTerms {} -> {} "
				+ "(concept '{}' sim={} margin={})",
				Arrays.toString(kwTerms),
				conceptTokens, bestConcept,
				String.format("%.4f", bestSim),
				String.format("%.4f", margin));
		return conceptTokens.toArray(new String[0]);
	}

	/**
	 * Checks whether a chart embedding's structural prefix contains any of
	 * the given type indicator terms. Uses the actual text content to derive
	 * the correct prefix (important for ORDER records whose prefix depends
	 * on body text, e.g. "Drug order:" → "Medication prescription:").
	 * Also checks sister resource types: condition ↔ diagnosis.
	 */
	static boolean matchesTypeIndicator(ChartEmbedding ce,
			Set<String> typeIndicatorTerms) {
		String textContent = ce.getTextContent() != null
				? ce.getTextContent() : "";
		String fullPrefixed = ChartSearchAiUtils.buildPrefixedText(
				ce.getResourceType(), textContent);
		String prefix = fullPrefixed.substring(0,
				fullPrefixed.length() - textContent.length())
				.toLowerCase();
		String[] prefixWords = prefix.split("\\s+");
		for (String term : typeIndicatorTerms) {
			if (SimilarityAndScoringEngine.termMatchesText(term, prefix, prefixWords)) {
				return true;
			}
		}
		// Check sister resource type — condition ↔ diagnosis are
		// structurally equivalent (both represent clinical findings).
		String sisterType = SISTER_RESOURCE_TYPES.get(
				ce.getResourceType());
		if (sisterType != null) {
			String sisterPrefix = ChartSearchAiUtils.buildPrefixedText(
					sisterType, "").toLowerCase();
			String[] sisterWords = sisterPrefix.split("\\s+");
			for (String term : typeIndicatorTerms) {
				if (SimilarityAndScoringEngine.termMatchesText(term, sisterPrefix, sisterWords)) {
					return true;
				}
			}
		}
		return false;
	}
}
