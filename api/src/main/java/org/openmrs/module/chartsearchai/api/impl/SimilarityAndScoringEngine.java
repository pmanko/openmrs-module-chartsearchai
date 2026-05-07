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

import org.openmrs.module.chartsearchai.util.ConceptNameUtil;

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

	/**
	 * Computes the z-score of the top semantic score relative to the full
	 * score distribution. A high z-score means the best match is a
	 * statistical outlier — genuine signal rather than noise floor.
	 */
	static double computeSemanticZScore(List<ScoredEmbedding> scored,
			double maxSemanticScore) {
		double sumSem = 0;
		for (ScoredEmbedding se : scored) {
			sumSem += se.semanticScore;
		}
		double meanSem = sumSem / scored.size();
		double sumSqDiff = 0;
		for (ScoredEmbedding se : scored) {
			double diff = se.semanticScore - meanSem;
			sumSqDiff += diff * diff;
		}
		double stddev = Math.sqrt(sumSqDiff / scored.size());
		return stddev == 0 ? 0 : (maxSemanticScore - meanSem) / stddev;
	}

	/**
	 * Returns true if the score distribution has enough variance for
	 * z-score statistics to be meaningful. Replaces the hardcoded
	 * MIN_RECORDS_FOR_Z_SCORE (30) with a data-driven check: the
	 * standard deviation must be non-trivial (&gt; 1e-6) and there must
	 * be at least 4 records (minimum for IQR-based statistics).
	 */
	static boolean hasStatisticalVariance(List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return false;
		}
		double sum = 0;
		for (ScoredEmbedding se : scored) {
			sum += se.semanticScore;
		}
		double mean = sum / scored.size();
		double sqSum = 0;
		for (ScoredEmbedding se : scored) {
			double d = se.semanticScore - mean;
			sqSum += d * d;
		}
		double std = Math.sqrt(sqSum / scored.size());
		return std > 1e-6;
	}

	/**
	 * Gumbel extreme value threshold adjusted for correlated observations.
	 * Semantic scores have two levels of correlation:
	 * <ol>
	 *   <li>Records within the same concept produce nearly identical
	 *       scores (reducing N to uniqueConcepts)</li>
	 *   <li>Concepts within the same domain (e.g. all medical concepts)
	 *       share embedding space structure, partially correlating their
	 *       query cosines (further reducing effectiveN)</li>
	 * </ol>
	 * The effective degrees of freedom is estimated as
	 * ln(uniqueConcepts) — the log captures the second-level
	 * correlation from domain structure.
	 */
	static double effectiveGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				Math.log(Math.max(2, uniqueConcepts.size())));
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the cluster z-score gate
	 * (Phase 2). Uses the raw unique concept count as effective N —
	 * after gap detection and coherence filtering isolate a candidate
	 * cluster, the surviving concepts act more independently than
	 * in the full score distribution (where intra-domain correlation
	 * is high). This produces thresholds of ~2.5-2.7 for typical
	 * datasets (30-45 unique concepts), matching the discrimination
	 * needed to separate genuine clusters from noise.
	 */
	static double clusterGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0, uniqueConcepts.size());
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the cluster z-score gate
	 * when the representative is the MEDIAN (K&ge;4 candidates). The
	 * median of K top-scoring records corresponds to the (N/2)-th
	 * order statistic, so the effective N for the Gumbel formula is
	 * halved. This produces thresholds of ~2.15-2.50 for typical
	 * datasets, appropriately more lenient than the max-based
	 * threshold since the median is naturally lower than the max.
	 */
	static double medianGumbelThreshold(List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				uniqueConcepts.size() / 2.0);
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Data-derived Gumbel threshold for the floor rescue z-score
	 * gate. Uses uniqueConcepts^(2/3) as effective N — an
	 * interpolation between the initial gate's heavy-correlation
	 * model (ln(N)) and the cluster gate's independence model (N).
	 * The floor rescue operates in an intermediate regime: it checks
	 * a single score (like the initial gate) but must be stricter
	 * because it's allowing below-floor results through. A secondary
	 * cluster density check provides an additional safety net.
	 * Produces thresholds of ~2.0-2.2 for typical datasets
	 * (20-45 unique concepts).
	 */
	static double floorRescueGumbelThreshold(
			List<ScoredEmbedding> scored) {
		Set<String> uniqueConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String name = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (name != null) {
				uniqueConcepts.add(name);
			}
		}
		double effectiveN = Math.max(2.0,
				Math.pow(Math.max(2, uniqueConcepts.size()),
						2.0 / 3.0));
		return Math.sqrt(2 * Math.log(effectiveN));
	}

	/**
	 * Computes a data-derived minimum gap for the second-pass gap
	 * detector from the median of consecutive sorted-score gaps.
	 * The median gap represents the "typical" spacing between
	 * adjacent scores in this distribution. Using half the median
	 * gap as the floor ensures we only detect gaps that are
	 * meaningfully larger than typical spacing, while staying
	 * sensitive enough for compressed distributions.
	 */
	static double secondPassMinGap(List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return 0;
		}
		List<Double> sortedScores = new ArrayList<Double>(
				scored.size());
		for (ScoredEmbedding se : scored) {
			sortedScores.add(se.semanticScore);
		}
		Collections.sort(sortedScores, Collections.reverseOrder());
		List<Double> gaps = new ArrayList<Double>();
		for (int i = 1; i < sortedScores.size(); i++) {
			double gap = sortedScores.get(i - 1)
					- sortedScores.get(i);
			if (gap > 0) {
				gaps.add(gap);
			}
		}
		if (gaps.isEmpty()) {
			return 0;
		}
		Collections.sort(gaps);
		double medianGap = gaps.get(gaps.size() / 2);
		return medianGap / 2.0;
	}

	/**
	 * Computes a data-derived adaptive gap ratio for the refinement
	 * path's second-pass gap detection. Uses the standard deviation
	 * of the top quartile of scores divided by the max score. The
	 * top quartile captures the region where refinement gaps matter
	 * — its std measures the natural spread among the highest-scoring
	 * records. Dividing by max normalizes across score scales.
	 */
	static double refinementAdaptiveGapRatio(
			List<ScoredEmbedding> scored) {
		if (scored.size() < 4) {
			return 0.10;
		}
		List<Double> sortedScores = new ArrayList<Double>(
				scored.size());
		for (ScoredEmbedding se : scored) {
			sortedScores.add(se.semanticScore);
		}
		Collections.sort(sortedScores, Collections.reverseOrder());
		double max = sortedScores.get(0);
		if (max <= 0) {
			return 0.10;
		}
		// Top quartile standard deviation
		int topN = Math.max(4, sortedScores.size() / 4);
		double sum = 0;
		for (int i = 0; i < topN; i++) {
			sum += sortedScores.get(i);
		}
		double mean = sum / topN;
		double sqSum = 0;
		for (int i = 0; i < topN; i++) {
			double d = sortedScores.get(i) - mean;
			sqSum += d * d;
		}
		double std = Math.sqrt(sqSum / topN);
		double ratio = std / max;
		return Math.max(0.05, Math.min(0.15, ratio));
	}

	/**
	 * Computes a data-derived semantic ratio for refinement filtering.
	 * Returns 1 - CV (coefficient of variation) of the candidate
	 * scores, clamped to [0.50, 0.90]. When scores are tightly
	 * clustered (low CV → ratio near 1.0), most candidates are
	 * relevant. When scores are spread (high CV → ratio near 0.50),
	 * only the top candidates matter. This replaces the hardcoded
	 * REFINEMENT_SEMANTIC_RATIO (0.70).
	 */
	static double refinementSemanticRatio(
			List<ScoredEmbedding> candidates) {
		if (candidates.size() < 2) {
			return 0.70;
		}
		List<Double> scores = new ArrayList<Double>(
				candidates.size());
		for (ScoredEmbedding se : candidates) {
			scores.add(se.semanticScore);
		}
		Collections.sort(scores);
		double max = scores.get(scores.size() - 1);
		if (max <= 0) {
			return 0.70;
		}
		// Use Q1/max as the semantic ratio. Q1 captures the lower
		// bound of the "bulk" of the distribution, excluding the
		// bottom quartile (likely noise). This naturally adapts:
		// tight sets (all scores similar) → high ratio (strict
		// floor); wide multi-concept sets → low ratio (permissive).
		// Clamped to [0.60, 0.90] — records below 60% of the best
		// score are noise; above 90% is over-strict for any set.
		double q1 = scores.get(scores.size() / 4);
		double ratio = q1 / max;
		return Math.max(0.60, Math.min(0.90, ratio));
	}
}
