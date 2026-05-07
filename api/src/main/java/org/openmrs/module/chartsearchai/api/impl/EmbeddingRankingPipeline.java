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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The retrieval/ranking pipeline. Holds the static helpers reachable from
 * {@code findSimilarWithProfile} that aren't already covered by
 * {@link SimilarityAndScoringEngine} (pure scoring) or
 * {@link QueryPreprocessor} (query normalization). This class will absorb
 * the gates, refinement, and entry points across subsequent extractions;
 * for now it owns the leaf utilities (clustering, coherence, concept-name
 * type-indicator matching, and concept-similarity keyword expansion).
 */
final class EmbeddingRankingPipeline {

	private EmbeddingRankingPipeline() {
	}

	private static final Logger log = LoggerFactory.getLogger(EmbeddingRankingPipeline.class);

	private static final Set<String> CONCEPT_EXPANSION_FUNCTION_WORDS =
			Collections.unmodifiableSet(new HashSet<String>(
					Arrays.asList(
							"of", "to", "and", "or", "the", "a", "an",
							"in", "on", "for", "with", "by", "as")));

	/**
	 * Grows a cluster starting from the initial {@code seedSize} records by
	 * adding candidates whose embedding has cosine similarity &ge; threshold
	 * with at least one existing cluster member. Iterates until no more
	 * records can be added (fixed-point). Records are tested in score order
	 * (highest first) so higher-scoring candidates are prioritized.
	 *
	 * @param candidates all candidates (sorted by combined score descending)
	 * @param seedSize number of initial cluster records (the top seedSize)
	 * @param cosineThreshold minimum cosine similarity to join the cluster
	 * @return the grown cluster, preserving the original score order
	 */
	static List<ScoredEmbedding> growCluster(List<ScoredEmbedding> candidates,
			int seedSize, double cosineThreshold) {
		if (seedSize >= candidates.size()) {
			return candidates;
		}
		boolean[] inCluster = new boolean[candidates.size()];
		float[][] vectors = new float[candidates.size()][];
		for (int i = 0; i < candidates.size(); i++) {
			vectors[i] = candidates.get(i).embedding.getEmbeddingVector();
			inCluster[i] = i < seedSize;
		}

		// Iteratively add candidates that are coherent with the cluster
		boolean added = true;
		while (added) {
			added = false;
			for (int i = seedSize; i < candidates.size(); i++) {
				if (inCluster[i] || vectors[i] == null) {
					continue;
				}
				for (int j = 0; j < candidates.size(); j++) {
					if (!inCluster[j] || vectors[j] == null
							|| vectors[j].length != vectors[i].length) {
						continue;
					}
					if (ChartSearchAiUtils.cosineSimilarity(vectors[i], vectors[j]) >= cosineThreshold) {
						inCluster[i] = true;
						added = true;
						break;
					}
				}
			}
		}

		List<ScoredEmbedding> result = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < candidates.size(); i++) {
			if (inCluster[i]) {
				result.add(candidates.get(i));
			}
		}
		return result;
	}

	/**
	 * Validates whether a gap in the score distribution at the given cutoff
	 * position is intra-topic (both sides belong to the same broad topic)
	 * by computing the average cosine similarity between records just above
	 * and just below the gap boundary. Uses a small window (up to 3
	 * records on each side) to get a robust estimate that doesn't depend
	 * on a single pair.
	 *
	 * <p>When the average cross-boundary cosine meets or exceeds the
	 * threshold, the gap is intra-topic and should NOT be used as a cutoff.
	 *
	 * @param scored the sorted candidate list
	 * @param cutoff the gap position (records 0..cutoff-1 above, cutoff+ below)
	 * @param cosineThreshold minimum average cosine for the gap to be intra-topic
	 * @return true if the gap is intra-topic (should NOT cut here)
	 */
	static boolean isGapCoherent(List<ScoredEmbedding> scored, int cutoff,
			double cosineThreshold) {
		if (cutoff <= 0 || cutoff >= scored.size()) {
			return false;
		}

		// Use a small window on each side of the gap for robustness
		int windowSize = Math.min(3, Math.min(cutoff, scored.size() - cutoff));
		double sumCosine = 0;
		int pairs = 0;

		for (int a = cutoff - windowSize; a < cutoff; a++) {
			float[] vecA = scored.get(a).embedding.getEmbeddingVector();
			if (vecA == null) {
				continue;
			}
			for (int b = cutoff; b < cutoff + windowSize; b++) {
				float[] vecB = scored.get(b).embedding.getEmbeddingVector();
				if (vecB == null || vecB.length != vecA.length) {
					continue;
				}
				sumCosine += ChartSearchAiUtils.cosineSimilarity(vecA, vecB);
				pairs++;
			}
		}

		if (pairs == 0) {
			return false;
		}

		double avgCosine = sumCosine / pairs;
		return avgCosine >= cosineThreshold;
	}

	/**
	 * Filters the final candidate set by inter-candidate coherence to
	 * remove "topic outliers" — records that scored similarly to the query
	 * by coincidence but are conceptually unrelated to the other results.
	 * Delegates the per-vector coherence math to
	 * {@link ChartSearchAiUtils#filterByCoherence(float[][])}.
	 *
	 * @param candidates the post-filtered candidate set
	 * @return the coherence-filtered subset (preserving original order)
	 */
	static List<ScoredEmbedding> filterByCoherence(List<ScoredEmbedding> candidates) {
		int n = candidates.size();

		float[][] vectors = new float[n][];
		for (int i = 0; i < n; i++) {
			byte[] raw = candidates.get(i).embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				return candidates;
			}
			vectors[i] = candidates.get(i).embedding.getEmbeddingVector();
		}

		boolean[] keep = ChartSearchAiUtils.filterByCoherence(vectors);

		int keepCount = 0;
		for (boolean k : keep) {
			if (k) keepCount++;
		}
		if (keepCount == n) {
			return candidates;
		}

		log.debug("Coherence filter removed {} outlier(s)", n - keepCount);

		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < n; i++) {
			if (keep[i]) {
				filtered.add(candidates.get(i));
			}
		}
		return filtered;
	}

	/**
	 * Checks whether a small set of candidates (n &le; 3) are mutually
	 * coherent by computing mean pairwise cosine similarity. If the mean
	 * is below the gap validation threshold, all candidates are rejected
	 * as false positives — they are unrelated topics that individually
	 * scored well against the query but are not coherent as a group.
	 *
	 * <p>This complements {@link #filterByCoherence}, which uses gap
	 * detection and cannot remove candidates when coherence values are
	 * uniformly low (stddev ≈ 0).
	 */
	static List<ScoredEmbedding> filterByMeanCoherence(
			List<ScoredEmbedding> candidates, PipelineConfig config) {
		boolean allHaveEmbeddings = true;
		for (ScoredEmbedding se : candidates) {
			byte[] raw = se.embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				allHaveEmbeddings = false;
				break;
			}
		}
		if (!allHaveEmbeddings) {
			return candidates;
		}

		double sumCosine = 0;
		int pairCount = 0;
		for (int i = 0; i < candidates.size(); i++) {
			for (int j = i + 1; j < candidates.size(); j++) {
				sumCosine += ChartSearchAiUtils.cosineSimilarity(
						candidates.get(i).embedding.getEmbeddingVector(),
						candidates.get(j).embedding.getEmbeddingVector());
				pairCount++;
			}
		}
		double meanCoherence = sumCosine / pairCount;
		if (meanCoherence < config.gapValidationCosineThreshold) {
			log.debug("{} candidates are not mutually coherent "
					+ "(meanCosine={}, threshold={}), returning empty",
					candidates.size(),
					String.format("%.4f", meanCoherence),
					config.gapValidationCosineThreshold);
			return Collections.emptyList();
		}
		return candidates;
	}

	/**
	 * Rescues records that scored below the similarity floor but are
	 * coherent with the surviving candidate cluster. Computes each
	 * below-floor record's average cosine similarity to the cluster
	 * members and includes it if that coherence is at least as high as
	 * the lowest coherence within the cluster itself.
	 *
	 * @param candidates the post-coherence-filtered cluster
	 * @param scored all scored records (including below-floor ones)
	 * @param adaptiveCutoff the first-pass cutoff (records above the floor)
	 * @return the original candidates plus any rescued below-floor records
	 */
	static List<ScoredEmbedding> rescueBelowFloor(List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored, int adaptiveCutoff) {
		if (adaptiveCutoff >= scored.size()) {
			return candidates;
		}

		float[][] clusterVecs = new float[candidates.size()][];
		boolean allValid = true;
		for (int i = 0; i < candidates.size(); i++) {
			byte[] raw = candidates.get(i).embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				allValid = false;
				break;
			}
			clusterVecs[i] = candidates.get(i).embedding.getEmbeddingVector();
		}
		if (!allValid) {
			return candidates;
		}

		// Compute the minimum within-cluster coherence as the rescue threshold.
		// A below-floor record must be at least as coherent with the cluster
		// as the least coherent existing member.
		double minClusterCoherence = Double.MAX_VALUE;
		for (int i = 0; i < clusterVecs.length; i++) {
			double sum = 0;
			for (int j = 0; j < clusterVecs.length; j++) {
				if (j != i) {
					sum += ChartSearchAiUtils.cosineSimilarity(clusterVecs[i], clusterVecs[j]);
				}
			}
			double coherence = sum / (clusterVecs.length - 1);
			if (coherence < minClusterCoherence) {
				minClusterCoherence = coherence;
			}
		}

		Set<Integer> inCluster = new HashSet<Integer>();
		for (ScoredEmbedding se : candidates) {
			inCluster.add(se.embedding.getEmbeddingId());
		}

		List<ScoredEmbedding> rescued = new ArrayList<ScoredEmbedding>();
		for (int i = adaptiveCutoff; i < scored.size(); i++) {
			ScoredEmbedding se = scored.get(i);
			if (inCluster.contains(se.embedding.getEmbeddingId())) {
				continue;
			}
			byte[] raw = se.embedding.getEmbedding();
			if (raw == null || raw.length == 0) {
				continue;
			}
			float[] vec = se.embedding.getEmbeddingVector();
			double sum = 0;
			for (float[] cv : clusterVecs) {
				sum += ChartSearchAiUtils.cosineSimilarity(vec, cv);
			}
			double coherence = sum / clusterVecs.length;
			if (coherence >= minClusterCoherence) {
				rescued.add(se);
			}
		}

		if (rescued.isEmpty()) {
			return candidates;
		}

		log.debug("Below-floor rescue: recovered {} records "
				+ "(minClusterCoherence={})",
				rescued.size(), String.format("%.4f", minClusterCoherence));
		List<ScoredEmbedding> result = new ArrayList<ScoredEmbedding>(candidates);
		result.addAll(rescued);
		return result;
	}

	/**
	 * After Phase 1's coherence filter runs, restore any candidates whose
	 * keyword evidence covers a query term NOT matched by the survivors.
	 * Coherence filtering uses only embedding geometry and can drop
	 * legitimate cross-concept matches in multi-concept queries (e.g.
	 * Weight records in "blood pressure and weight" — Weight is less
	 * coherent with the BP cluster but matches "weight" which BP doesn't
	 * cover). Records whose keyword matches duplicate the survivors'
	 * coverage remain filtered.
	 */
	static List<ScoredEmbedding> preserveUniqueCoverage(
			List<ScoredEmbedding> original,
			List<ScoredEmbedding> filtered, String[] queryTerms,
			double bonusThreshold) {
		if (queryTerms == null || queryTerms.length == 0) {
			return filtered;
		}
		Set<Integer> filteredIds = new HashSet<Integer>();
		Set<String> coveredTerms = new HashSet<String>();
		for (ScoredEmbedding se : filtered) {
			filteredIds.add(se.embedding.getResourceId());
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			for (String term : queryTerms) {
				if (SimilarityAndScoringEngine.termMatchesText(term, text, words)) {
					coveredTerms.add(term);
				}
			}
		}
		List<ScoredEmbedding> result =
				new ArrayList<ScoredEmbedding>(filtered);
		Set<Integer> restoredIds = new HashSet<Integer>(filteredIds);
		// Pass 1: restore records that cover query terms NOT already
		// covered by coherence survivors.
		for (ScoredEmbedding se : original) {
			if (se.keywordScore <= 0
					|| restoredIds.contains(
							se.embedding.getResourceId())) {
				continue;
			}
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			for (String term : queryTerms) {
				if (!coveredTerms.contains(term)
						&& SimilarityAndScoringEngine.termMatchesText(term, text, words)) {
					result.add(se);
					restoredIds.add(se.embedding.getResourceId());
					break;
				}
			}
		}
		// Pass 2: restore records with strong keyword relevance even
		// when their matched terms overlap with survivors. The bonus
		// threshold (min(2,N)/N query terms) ensures only records with
		// strong keyword evidence are restored.
		for (ScoredEmbedding se : original) {
			if (restoredIds.contains(se.embedding.getResourceId())) {
				continue;
			}
			if (se.keywordScore >= bonusThreshold) {
				result.add(se);
				restoredIds.add(se.embedding.getResourceId());
			}
		}
		return result;
	}

	/**
	 * Removes concept-pair outliers from a semantic core by comparing
	 * each concept's average inter-concept cosine to the patient's
	 * cross-concept noise baseline. A concept whose mean cosine to
	 * non-pair core members is at or below {@code noiseProfile.noiseMean}
	 * is, by construction, statistically indistinguishable from a
	 * random unrelated pair.
	 *
	 * <p>Only fires when there are &ge; 3 distinct concepts in the core
	 * and &ge; 2 of them have &ge; 2 members.
	 */
	static List<ScoredEmbedding> pruneCoherenceOutlierConcepts(
			List<ScoredEmbedding> core,
			ModelNoiseProfile noiseProfile) {
		if (core.size() < 4) {
			return core;
		}
		Map<String, List<ScoredEmbedding>> byConcept =
				new LinkedHashMap<String, List<ScoredEmbedding>>();
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null) {
				cn = "__unnamed_" + se.embedding.getResourceId();
			}
			List<ScoredEmbedding> list = byConcept.get(cn);
			if (list == null) {
				list = new ArrayList<ScoredEmbedding>();
				byConcept.put(cn, list);
			}
			list.add(se);
		}
		if (byConcept.size() < 3) {
			return core;
		}
		int multiMember = 0;
		for (List<ScoredEmbedding> g : byConcept.values()) {
			if (g.size() >= 2) {
				multiMember++;
			}
		}
		if (multiMember < 2) {
			return core;
		}
		Map<String, Double> conceptInterAvg =
				new LinkedHashMap<String, Double>();
		for (Map.Entry<String, List<ScoredEmbedding>> e : byConcept.entrySet()) {
			String concept = e.getKey();
			List<ScoredEmbedding> group = e.getValue();
			double sum = 0;
			int count = 0;
			for (ScoredEmbedding member : group) {
				float[] mv = member.embedding.getEmbeddingVector();
				if (mv == null) {
					continue;
				}
				for (ScoredEmbedding other : core) {
					String ocn = ConceptNameUtil.extractConceptName(
							other.embedding.getTextContent());
					if (ocn != null && ocn.equals(concept)) {
						continue;
					}
					float[] ov = other.embedding.getEmbeddingVector();
					if (ov == null) {
						continue;
					}
					sum += ChartSearchAiUtils.cosineSimilarity(mv, ov);
					count++;
				}
			}
			if (count > 0) {
				conceptInterAvg.put(concept, sum / count);
			}
		}
		if (conceptInterAvg.size() < 3) {
			return core;
		}
		double cutoff = noiseProfile.noiseMean;
		Set<String> drop = new HashSet<String>();
		for (Map.Entry<String, Double> e : conceptInterAvg.entrySet()) {
			if (e.getValue() <= cutoff) {
				drop.add(e.getKey());
			}
		}
		if (drop.isEmpty()) {
			return core;
		}
		int kept = 0;
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null || !drop.contains(cn)) {
				kept++;
			}
		}
		if (kept < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			return core;
		}
		List<ScoredEmbedding> pruned = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : core) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn == null || !drop.contains(cn)) {
				pruned.add(se);
			}
		}
		log.debug("Coherence-outlier concept pruning: dropped {} concepts ({}), kept {} of {} core members (cutoff=noiseMean={})",
				drop.size(), drop, pruned.size(), core.size(),
				String.format("%.4f", cutoff));
		return pruned;
	}

	/**
	 * Filters out lower-keyword-tier records whose matched query terms
	 * are already fully covered by the higher-keyword tier. Records that
	 * match unique terms not covered by the higher tier are kept.
	 *
	 * @param candidates the refined candidate set with non-uniform keywords
	 * @param queryTerms the query terms after stopword removal
	 * @param kwMax the maximum keyword score in the candidate set
	 * @return filtered candidates, or the original list if filtering
	 *         would produce fewer than {@code ADAPTIVE_MIN_RECORDS}
	 */
	static List<ScoredEmbedding> filterRedundantKeywordTier(
			List<ScoredEmbedding> candidates, String[] queryTerms,
			double kwMax, double bonusThreshold) {
		Set<String> coveredTerms = new HashSet<String>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						ConceptNameUtil.stripSynonyms(
								se.embedding.getTextContent()));
				String lower = text.toLowerCase();
				String[] words = lower.split("\\s+");
				for (String term : queryTerms) {
					if (SimilarityAndScoringEngine.termMatchesText(term, lower, words)) {
						coveredTerms.add(term);
					}
				}
			}
		}

		// Semantic floor for zero-keyword rescued records.
		double maxSemantic = 0;
		for (ScoredEmbedding se : candidates) {
			if (se.semanticScore > maxSemantic) {
				maxSemantic = se.semanticScore;
			}
		}
		double semanticFloor = maxSemantic
				* SimilarityAndScoringEngine.refinementSemanticRatio(candidates);

		List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= kwMax - 0.01) {
				filtered.add(se);
			} else if (se.keywordScore >= bonusThreshold) {
				filtered.add(se);
			} else if (se.keywordScore == 0
					&& se.semanticScore >= semanticFloor) {
				// Zero-keyword record with strong semantic signal —
				// the embedding model says this is relevant but no
				// query term appears in the text (e.g. "Pulse" for
				// "heart rate").
				filtered.add(se);
			} else {
				String text = ChartSearchAiUtils.buildPrefixedText(
						se.embedding.getResourceType(),
						ConceptNameUtil.stripSynonyms(
								se.embedding.getTextContent()));
				String lower = text.toLowerCase();
				String[] words = lower.split("\\s+");
				boolean addsNewCoverage = false;
				for (String term : queryTerms) {
					if (!coveredTerms.contains(term)
							&& SimilarityAndScoringEngine.termMatchesText(term, lower, words)) {
						addsNewCoverage = true;
						break;
					}
				}
				if (addsNewCoverage) {
					filtered.add(se);
				}
			}
		}
		if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			log.debug("Keyword tier subset filter: {} -> {} "
					+ "(coveredTerms={})",
					candidates.size(), filtered.size(), coveredTerms);
			return filtered;
		}
		return candidates;
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
	 * Resource types that are structurally equivalent in the data model.
	 * Conditions and diagnoses both represent clinical findings about a
	 * patient's health. When a type indicator matches one, the other
	 * should also be kept — they are two representations of the same
	 * concept type.
	 */
	private static final Map<String, String> SISTER_RESOURCE_TYPES;
	static {
		Map<String, String> m = new java.util.HashMap<String, String>();
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION,
				ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS);
		m.put(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS,
				ChartSearchAiConstants.RESOURCE_TYPE_CONDITION);
		SISTER_RESOURCE_TYPES = Collections.unmodifiableMap(m);
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

	/**
	 * Floor gate — stage 1 of the filter pipeline.
	 *
	 * <p>If the highest score in the candidate set is below the embedding
	 * model's absolute similarity floor, the candidate set is below the
	 * noise boundary. Attempts a z-score + cluster-density rescue that
	 * recovers vocabulary-mismatch queries (e.g. "how hot is the patient?"
	 * → Temperature records) where the model correctly ranks relevant
	 * records first but cosine similarity is inherently low.
	 *
	 * @return {@code null} if the gate rejects (the caller should return
	 *         an empty result); otherwise a boxed boolean indicating
	 *         whether the rescue fired.
	 */
	static Boolean applyFloorGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, double maxBaseScore,
			int queryTermCount, int keywordMatchCount,
			PipelineConfig config) {
		double floorScore = Math.max(maxSemanticScore, maxBaseScore);
		if (floorScore >= config.noiseProfile.absoluteSimilarityFloor()) {
			return Boolean.FALSE;
		}
		boolean belowFloorRescued = false;
		if (queryTermCount > 0
				&& SimilarityAndScoringEngine.hasStatisticalVariance(scored)
				&& keywordMatchCount < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			double zScore = SimilarityAndScoringEngine.computeSemanticZScore(scored, maxSemanticScore);
			double floorRescueZThreshold =
					SimilarityAndScoringEngine.floorRescueGumbelThreshold(scored);
			if (zScore >= floorRescueZThreshold) {
				// Verify the signal comes from a genuine cluster, not
				// an isolated outlier. Count records within a tight band
				// of the max score — vocabulary-mismatch queries produce
				// many records at similar scores, while false positives
				// only have 1-2 outlier conditions in that band.
				Set<String> floorConcepts = new HashSet<String>();
				for (ScoredEmbedding se : scored) {
					String cn = ConceptNameUtil
							.extractConceptName(
									se.embedding
											.getTextContent());
					if (cn != null) {
						floorConcepts.add(cn);
					}
				}
				int nConcepts = Math.max(2, floorConcepts.size());
				double densityBand = 1.0 / nConcepts;
				int minCluster = (int) Math.ceil(
						(double) scored.size() / nConcepts);
				double densityFloor = maxSemanticScore
						* (1 - densityBand);
				int clusterDensity = 0;
				for (ScoredEmbedding se : scored) {
					if (se.semanticScore >= densityFloor) {
						clusterDensity++;
					}
				}
				belowFloorRescued = clusterDensity >= minCluster;
				log.warn("Floor gate z-score rescue: zScore={}, "
						+ "density={} (band={}, floor={}), "
						+ "minCluster={}, rescued={}",
						String.format("%.2f", zScore),
						clusterDensity,
						String.format("%.4f", densityBand),
						String.format("%.4f", densityFloor),
						minCluster,
						belowFloorRescued);
			}
		}
		if (!belowFloorRescued) {
			log.debug("Top score {} (semantic={}, combined={}) is below "
					+ "absolute floor {}, returning empty",
					String.format("%.4f", floorScore),
					String.format("%.4f", maxSemanticScore),
					String.format("%.4f", maxBaseScore),
					config.noiseProfile.absoluteSimilarityFloor());
			return null;
		}
		return Boolean.TRUE;
	}

	/**
	 * Slim-margin gate — stage 2 of the filter pipeline.
	 *
	 * <p>When the top semantic score is above the absolute floor but
	 * within one {@code minScoreGap} of it, with zero keyword matches,
	 * require at least 2 records above the floor. A single record just
	 * above the floor is likely a coincidental near-miss, not genuine
	 * signal.
	 *
	 * @return {@code true} to continue, {@code false} to short-circuit
	 *         with an empty result
	 */
	static boolean applySlimMarginGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount, boolean belowFloorRescued,
			PipelineConfig config) {
		if (belowFloorRescued || keywordMatchCount != 0 || queryTermCount <= 0) {
			return true;
		}
		double smFloor = config.noiseProfile.absoluteSimilarityFloor();
		if (maxSemanticScore >= smFloor + config.minScoreGap) {
			return true;
		}
		int aboveFloorCount = 0;
		for (ScoredEmbedding se : scored) {
			if (se.semanticScore >= smFloor) {
				aboveFloorCount++;
			}
		}
		if (aboveFloorCount < 2) {
			// Allow a single record if its score is in the upper half
			// of the slim-margin zone — records barely above the floor
			// are likely noise, records further above are more likely
			// genuine.
			if (aboveFloorCount == 1
					&& maxSemanticScore >= smFloor + config.minScoreGap / 2) {
				if (log.isDebugEnabled()) {
					log.debug("Slim-margin gate: single record at {} is in "
							+ "upper half of margin zone [{}, {}), allowing",
							String.format("%.4f", maxSemanticScore),
							String.format("%.4f", smFloor + config.minScoreGap / 2),
							String.format("%.4f", smFloor + config.minScoreGap));
				}
				return true;
			}
			log.warn("Slim-margin gate: maxSem={} is within "
					+ "{} of floor {}, zero keywords, and only "
					+ "{} record(s) above floor — returning empty",
					String.format("%.4f", maxSemanticScore),
					config.minScoreGap,
					String.format("%.4f", smFloor),
					aboveFloorCount);
			return false;
		}
		return true;
	}

	/**
	 * Initial z-score gate — stage 3 of the filter pipeline. When too few
	 * records match any query keyword, require the top semantic score to
	 * be a statistical outlier. Captures the z-score and threshold values
	 * for reuse by the downstream cluster z-score gate.
	 *
	 * @return {@code null} if the gate rejects; otherwise a 2-element
	 *         array {@code [zScore, threshold]} ({@code -1} when not
	 *         computed).
	 */
	static double[] applyInitialZScoreGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount) {
		return applyInitialZScoreGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, -1);
	}

	static double[] applyInitialZScoreGate(List<ScoredEmbedding> scored,
			double maxSemanticScore, int queryTermCount,
			int keywordMatchCount, double maxZScoreThreshold) {
		double[] result = { -1, -1 };
		if (queryTermCount <= 0 || !SimilarityAndScoringEngine.hasStatisticalVariance(scored)) {
			return result;
		}
		if (keywordMatchCount >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			return result;
		}
		double zScore = SimilarityAndScoringEngine.computeSemanticZScore(scored, maxSemanticScore);
		double threshold = SimilarityAndScoringEngine.effectiveGumbelThreshold(scored);
		// Allow model-specific configs to cap the threshold for
		// compressed-score models where the Gumbel threshold is too
		// aggressive.
		if (maxZScoreThreshold > 0 && threshold > maxZScoreThreshold) {
			threshold = maxZScoreThreshold;
		}
		result[0] = zScore;
		result[1] = threshold;
		if (zScore < threshold) {
			log.debug("Only {} keyword match(es) and top semantic "
					+ "z-score {} is below threshold {}, "
					+ "returning empty (maxSem={})",
					keywordMatchCount,
					String.format("%.2f", zScore),
					String.format("%.2f", threshold),
					String.format("%.4f", maxSemanticScore));
			return null;
		}
		return result;
	}

	/**
	 * Phase 1 (outlier removal) — stage 6 of the filter pipeline.
	 *
	 * <p>Removes individual candidates that are topically unrelated to
	 * the majority via {@link #filterByCoherence}. Skipped when the
	 * partial-keyword semantic-core path already curated the set, every
	 * candidate matches ALL query terms, or the set is a compound-keyword
	 * match (see {@link #isCompoundKeywordMatch}).
	 *
	 * <p>At n=3 coherence filtering only fires when scores are tightly
	 * clustered (min/max ratio ≥ 0.90).
	 */
	static List<ScoredEmbedding> applyOutlierRemovalPhase1(
			List<ScoredEmbedding> candidates, String[] queryTerms,
			boolean partialKwValidated) {
		boolean allFullKeywordMatch = true;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore < 1.0) {
				allFullKeywordMatch = false;
				break;
			}
		}
		boolean isCompoundKeywordMatch = isCompoundKeywordMatch(
				candidates, queryTerms);
		double bonusThreshold = queryTerms.length == 0 ? 1.0
				: (double) Math.min(2, queryTerms.length) / queryTerms.length;
		if (!partialKwValidated && !allFullKeywordMatch
				&& !isCompoundKeywordMatch
				&& candidates.size() >= 4) {
			return preserveUniqueCoverage(candidates,
					filterByCoherence(candidates), queryTerms,
					bonusThreshold);
		}
		if (!partialKwValidated && !allFullKeywordMatch
				&& !isCompoundKeywordMatch
				&& candidates.size() == 3) {
			double topSemantic = candidates.get(0).semanticScore;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore > topSemantic) {
					topSemantic = se.semanticScore;
				}
			}
			double lowestSemantic = candidates.get(0).semanticScore;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore < lowestSemantic) {
					lowestSemantic = se.semanticScore;
				}
			}
			if (topSemantic > 0 && lowestSemantic / topSemantic >= 0.90) {
				return preserveUniqueCoverage(candidates,
						filterByCoherence(candidates), queryTerms,
						bonusThreshold);
			}
		}
		return candidates;
	}

	/**
	 * Phase 2 (zero-keyword validation) — stage 7 of the filter pipeline.
	 * When no surviving candidate has keyword support, the result set is
	 * purely semantic and must pass two orthogonal confidence checks
	 * (mean coherence and cluster z-score) plus a small-cluster coherence
	 * gate and a single-candidate structural gate. Candidates with any
	 * keyword match bypass this phase entirely.
	 *
	 * <p>The z-score gate is skipped when tight-cluster evidence already
	 * validated signal — captured via {@code belowFloorRescued},
	 * {@code firstPassGapDetected} vs concept count, or the ratio-floor
	 * tight-cluster signal.
	 */
	static List<ScoredEmbedding> applyZeroKeywordValidationPhase2(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored, PipelineConfig config,
			boolean partialKwValidated, boolean belowFloorRescued,
			boolean firstPassGapDetected, int adaptiveCutoff,
			int ratioFloorCandidateCount, double initialZScore,
			double initialZThreshold, double maxSemanticScore) {
		boolean candidatesHaveKeywords = false;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore > 0) {
				candidatesHaveKeywords = true;
				break;
			}
		}
		if (log.isDebugEnabled()) {
			StringBuilder sb = new StringBuilder();
			for (ScoredEmbedding se : candidates) {
				if (sb.length() > 0) sb.append(", ");
				sb.append(se.embedding.getResourceType())
					.append(':').append(se.embedding.getResourceId())
					.append(" sem=").append(String.format("%.4f", se.semanticScore))
					.append(" kw=").append(String.format("%.4f", se.keywordScore));
			}
			log.debug("Phase 2: candidatesHaveKeywords={}, candidates={}: [{}]",
					candidatesHaveKeywords, candidates.size(), sb);
		}
		log.debug("Phase2 entry: candidates={}, hasKw={}, partialKwValidated={}, belowFloorRescued={}, gapDetected={}, ratioFloorCount={}",
				candidates.size(), candidatesHaveKeywords, partialKwValidated,
				belowFloorRescued, firstPassGapDetected, ratioFloorCandidateCount);
		if (partialKwValidated || candidatesHaveKeywords || candidates.isEmpty()) {
			return candidates;
		}
		if (candidates.size() >= 2) {
			int preMeanCoherence = candidates.size();
			candidates = filterByMeanCoherence(candidates, config);
			log.debug("Phase2 meanCoherence: {} -> {}", preMeanCoherence, candidates.size());
		}
		// Small-cluster coherence gate: for ≤3 zero-keyword candidates,
		// apply a stricter coherence threshold derived from the patient's
		// own embedding statistics.
		if (candidates.size() == 3) {
			double intraFloor = config.noiseProfile.intraConceptMean
					- config.noiseProfile.intraConceptStd;
			double sumCos = 0;
			int nPairs = 0;
			for (int ci = 0; ci < candidates.size(); ci++) {
				float[] vi = candidates.get(ci).embedding.getEmbeddingVector();
				if (vi == null) {
					continue;
				}
				for (int cj = ci + 1; cj < candidates.size(); cj++) {
					float[] vj = candidates.get(cj).embedding.getEmbeddingVector();
					if (vj == null) {
						continue;
					}
					sumCos += ChartSearchAiUtils.cosineSimilarity(vi, vj);
					nPairs++;
				}
			}
			if (nPairs > 0) {
				double mc = sumCos / nPairs;
				if (mc < intraFloor) {
					// Extract the coherent same-concept pair if one exists.
					double[] avgCoh = new double[3];
					for (int ci = 0; ci < 3; ci++) {
						double s = 0;
						int cnt = 0;
						float[] vi = candidates.get(ci).embedding.getEmbeddingVector();
						if (vi == null) {
							continue;
						}
						for (int cj = 0; cj < 3; cj++) {
							if (ci == cj) {
								continue;
							}
							float[] vj = candidates.get(cj).embedding.getEmbeddingVector();
							if (vj == null) {
								continue;
							}
							s += ChartSearchAiUtils.cosineSimilarity(vi, vj);
							cnt++;
						}
						avgCoh[ci] = cnt > 0 ? s / cnt : 0;
					}
					int worstIdx = 0;
					for (int ci = 1; ci < 3; ci++) {
						if (avgCoh[ci] < avgCoh[worstIdx]) {
							worstIdx = ci;
						}
					}
					int p1 = -1, p2 = -1;
					for (int ci = 0; ci < 3; ci++) {
						if (ci != worstIdx) {
							if (p1 < 0) {
								p1 = ci;
							} else {
								p2 = ci;
							}
						}
					}
					float[] v1 = candidates.get(p1).embedding.getEmbeddingVector();
					float[] v2 = candidates.get(p2).embedding.getEmbeddingVector();
					double pairCos = (v1 != null && v2 != null)
							? ChartSearchAiUtils.cosineSimilarity(v1, v2)
							: 0;
					String cn1 = ConceptNameUtil.extractConceptName(
							candidates.get(p1).embedding.getTextContent());
					String cn2 = ConceptNameUtil.extractConceptName(
							candidates.get(p2).embedding.getTextContent());
					boolean sameConcept = cn1 != null && cn1.equals(cn2);
					if (pairCos >= intraFloor && sameConcept) {
						List<ScoredEmbedding> pair = new ArrayList<ScoredEmbedding>();
						pair.add(candidates.get(p1));
						pair.add(candidates.get(p2));
						log.debug("Small-cluster gate: removed outlier [{}], kept pair cos={}, concept={}",
								candidates.get(worstIdx).embedding.getResourceId(),
								String.format("%.4f", pairCos), cn1);
						candidates = pair;
					} else {
						log.debug("Small-cluster coherence gate: meanCoherence={} < intraFloor={}, returning empty",
								String.format("%.4f", mc),
								String.format("%.4f", intraFloor));
						candidates = Collections.emptyList();
					}
				}
			}
		}
		// Tight-cluster detection: the z-score gate is skipped when
		// structural signals already confirm the cluster.
		Set<String> tightCheckConcepts = new HashSet<String>();
		for (ScoredEmbedding se : scored) {
			String cn = ConceptNameUtil.extractConceptName(se.embedding.getTextContent());
			if (cn != null) {
				tightCheckConcepts.add(cn);
			}
		}
		int tightThreshold = tightCheckConcepts.size();
		Set<String> p2CandidateConcepts = new HashSet<String>();
		for (ScoredEmbedding se : candidates) {
			String cn = ConceptNameUtil.extractConceptName(
					se.embedding.getTextContent());
			if (cn != null) {
				p2CandidateConcepts.add(cn);
			}
		}
		boolean ratioFloorSelective = ratioFloorCandidateCount >= 0
				&& ratioFloorCandidateCount <= scored.size() / 4;
		boolean tightClusterDetected = belowFloorRescued
				|| (firstPassGapDetected && adaptiveCutoff < tightThreshold)
				|| (ratioFloorSelective
						&& ratioFloorCandidateCount < tightThreshold
						&& initialZScore >= config.floorRescueMinZScore
						&& maxSemanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ config.minScoreGap)
				// Multi-concept umbrella bypass: when the ratio floor
				// produced a tight cluster spanning 3+ distinct concepts
				// AND the initial z-score passed its own gate, the concept
				// diversity provides the structural confidence that
				// floorRescueMinZScore provides for single-concept clusters.
				|| (ratioFloorSelective
						&& ratioFloorCandidateCount < tightThreshold
						&& p2CandidateConcepts.size() >= 3
						&& initialZThreshold > 0
						&& initialZScore >= initialZThreshold
						&& maxSemanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ config.minScoreGap);
		// Single-candidate structural gate.
		if (!tightClusterDetected
				&& candidates.size() == 1
				&& ratioFloorCandidateCount == 1) {
			log.debug("Single zero-keyword candidate with no tight-cluster support and ratioFloor=1, returning empty");
			candidates = Collections.emptyList();
		}
		if (!tightClusterDetected && !candidates.isEmpty()
				&& SimilarityAndScoringEngine.hasStatisticalVariance(scored)) {
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
			if (std > 0) {
				double[] zScores = new double[candidates.size()];
				for (int i = 0; i < candidates.size(); i++) {
					zScores[i] = (candidates.get(i).semanticScore - mean) / std;
				}
				Arrays.sort(zScores);
				double zRepresentative = candidates.size() <= 3
						? zScores[zScores.length - 1]
						: zScores[zScores.length / 2];
				double clusterZThreshold;
				if (candidates.size() <= 3) {
					clusterZThreshold = SimilarityAndScoringEngine.clusterGumbelThreshold(scored);
				} else {
					clusterZThreshold = SimilarityAndScoringEngine.medianGumbelThreshold(scored);
				}
				log.debug("Phase2 z-score: mean={}, std={}, z={} ({}), threshold={}, tightCluster={}, candidates={}",
						String.format("%.4f", mean),
						String.format("%.4f", std),
						String.format("%.2f", zRepresentative),
						candidates.size() <= 3 ? "max" : "median",
						String.format("%.2f", clusterZThreshold),
						tightClusterDetected, candidates.size());
				if (zRepresentative < clusterZThreshold) {
					if (candidates.size() <= 3
							&& initialZThreshold > 0
							&& initialZScore >= initialZThreshold) {
						log.debug("Phase2: z-score below threshold but initial gate validated, keeping");
					} else {
						log.debug("Phase2: REJECTED by z-score gate (z={} < threshold={})",
								String.format("%.2f", zRepresentative),
								String.format("%.2f", clusterZThreshold));
						candidates = Collections.emptyList();
					}
				}
			}
		}
		return candidates;
	}

	/**
	 * Returns true when the candidate set as a whole covers all query
	 * terms but no single record matches them all — the structural
	 * signature of a multi-concept query like "HIV and CD4 count" where
	 * HIV records match {hiv} and CD4 records match {cd4, count}.
	 *
	 * <p>Used to bypass {@link #filterByCoherence}, which would otherwise
	 * drop the minority concept cluster as outliers despite keyword
	 * evidence that those records belong to the queried set.
	 */
	static boolean isCompoundKeywordMatch(
			List<ScoredEmbedding> candidates, String[] queryTerms) {
		if (queryTerms == null || queryTerms.length < 2) {
			return false;
		}
		Set<String> unionCoverage = new HashSet<String>();
		boolean anyRecordMatchesAll = false;
		for (ScoredEmbedding se : candidates) {
			String text = ChartSearchAiUtils.buildPrefixedText(
					se.embedding.getResourceType(),
					ConceptNameUtil.stripSynonyms(
							se.embedding.getTextContent()))
					.toLowerCase();
			String[] words = text.split("\\s+");
			int matchCount = 0;
			for (String term : queryTerms) {
				if (SimilarityAndScoringEngine.termMatchesText(term, text, words)) {
					unionCoverage.add(term);
					matchCount++;
				}
			}
			if (matchCount == queryTerms.length) {
				anyRecordMatchesAll = true;
			}
		}
		if (anyRecordMatchesAll) {
			return false;
		}
		for (String term : queryTerms) {
			if (!unionCoverage.contains(term)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Output of {@link #buildCandidates}: the candidate list plus the
	 * scalar outputs ({@code minScore}, {@code adaptiveCutoff}) that
	 * downstream stages still need.
	 */
	static final class CandidateBuildResult {
		final List<ScoredEmbedding> candidates;
		final int adaptiveCutoff;
		final double minScore;

		CandidateBuildResult(List<ScoredEmbedding> candidates,
				int adaptiveCutoff, double minScore) {
			this.candidates = candidates;
			this.adaptiveCutoff = adaptiveCutoff;
			this.minScore = minScore;
		}
	}

	/**
	 * Overload that accepts a term count instead of actual query terms.
	 * The keyword-tier subset check is skipped since the actual terms
	 * are not available.
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			int queryTermCount, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				null, queryTermCount, config);
	}

	/**
	 * Pure filtering pipeline: scores, gates, gap detection, keyword
	 * refinement, coherence filtering, and below-floor rescue. Takes
	 * pre-computed scores and config parameters — no dependency on
	 * OpenMRS Context, making it directly testable.
	 *
	 * @param semanticScores cosine similarity between query and each record
	 * @param keywordScores keyword overlap fraction for each record
	 * @param embeddings the chart embeddings (parallel to score arrays)
	 * @param queryTerms query terms after stopword removal (null-safe:
	 *        when null, the keyword-tier subset check is skipped)
	 * @param config pipeline tuning parameters
	 * @return filtered list of relevant embeddings, or empty list
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, queryTerms != null ? queryTerms.length : 0,
				config);
	}

	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int queryTermCount,
			PipelineConfig config) {
		return filterPipeline(semanticScores, keywordScores, embeddings,
				queryTerms, queryTermCount, config, null);
	}

	/**
	 * @param outGapCutoff if non-null, receives the gap-analysis candidate
	 *        count in [0]. Used by the noise-saturation check to detect
	 *        queries where the dataset has no genuine signal.
	 */
	static List<ChartEmbedding> filterPipeline(double[] semanticScores,
			double[] keywordScores, ChartEmbedding[] embeddings,
			String[] queryTerms, int queryTermCount,
			PipelineConfig config, int[] outGapCutoff) {

		double maxBaseScore = 0;
		double maxSemanticScore = 0;

		// Require ≥2 term matches for the keyword bonus so that single
		// coincidental matches (e.g. "history" in "immunization history"
		// for "history of cancer", or "lab" in "Labs ordered." for
		// "lab orders placed resulted") don't inflate scores.
		double bonusThreshold = queryTermCount == 0 ? 1.0
				: (double) Math.min(2, queryTermCount) / queryTermCount;
		List<ScoredEmbedding> scored = new ArrayList<ScoredEmbedding>();
		for (int i = 0; i < embeddings.length; i++) {
			double semanticScore = semanticScores[i];
			double keywordScore = keywordScores[i];

			// Additive keyword bonus increases the score when enough terms
			// match. For N≤2 queries, a partial keyword match (below the
			// bonus threshold) applies a penalty instead.
			double keywordBonus = keywordScore >= bonusThreshold ? keywordScore : 0.0;
			double keywordPenalty = 0.0;
			if (queryTermCount <= 2 && keywordScore > 0 && keywordScore < bonusThreshold) {
				keywordPenalty = keywordScore;
			}
			double baseScore = semanticScore + config.keywordWeight * keywordBonus
					- config.keywordWeight * keywordPenalty;

			if (semanticScore > maxSemanticScore) {
				maxSemanticScore = semanticScore;
			}
			if (baseScore > maxBaseScore) {
				maxBaseScore = baseScore;
			}

			scored.add(new ScoredEmbedding(embeddings[i], baseScore, keywordScore, semanticScore));
		}

		Collections.sort(scored, new Comparator<ScoredEmbedding>() {
			@Override
			public int compare(ScoredEmbedding a, ScoredEmbedding b) {
				return Double.compare(b.score, a.score);
			}
		});

		if (scored.isEmpty()) {
			return Collections.emptyList();
		}

		int keywordMatchCount = 0;
		for (ScoredEmbedding se : scored) {
			if (se.keywordScore > 0) {
				keywordMatchCount++;
			}
		}

		// Keyword flooding guard: when > 50% of records match keywords,
		// the matching terms are too generic to distinguish relevant
		// from irrelevant records. Zero out keyword scores and re-sort
		// by semantic score only.
		if (keywordMatchCount > scored.size() / 2
				&& queryTermCount > 2) {
			log.warn("Keyword flooding: {}/{} records match keywords,"
					+ " zeroing keyword scores for {}",
					keywordMatchCount, scored.size(),
					Arrays.toString(queryTerms));
			maxBaseScore = 0;
			for (int i = 0; i < scored.size(); i++) {
				ScoredEmbedding se = scored.get(i);
				scored.set(i, new ScoredEmbedding(se.embedding,
						se.semanticScore, 0, se.semanticScore));
				if (se.semanticScore > maxBaseScore) {
					maxBaseScore = se.semanticScore;
				}
			}
			keywordMatchCount = 0;
			Collections.sort(scored, new Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.score, a.score);
				}
			});
		}

		// Floor gate.
		Boolean floorResult = applyFloorGate(scored, maxSemanticScore,
				maxBaseScore, queryTermCount, keywordMatchCount, config);
		if (floorResult == null) {
			return Collections.emptyList();
		}
		boolean belowFloorRescued = floorResult;

		// Slim-margin gate.
		if (!applySlimMarginGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, belowFloorRescued,
				config)) {
			return Collections.emptyList();
		}

		// Initial z-score gate.
		double[] zScoreState = applyInitialZScoreGate(scored,
				maxSemanticScore, queryTermCount, keywordMatchCount,
				config.floorRescueMinZScore);
		if (zScoreState == null) {
			return Collections.emptyList();
		}
		double initialZScore = zScoreState[0];
		double initialZThreshold = zScoreState[1];

		// Gap detection + candidate building.
		CandidateBuildResult built = buildCandidates(scored,
				maxSemanticScore, bonusThreshold, config);
		double minScore = built.minScore;
		int adaptiveCutoff = built.adaptiveCutoff;
		if (outGapCutoff != null && outGapCutoff.length > 0) {
			outGapCutoff[0] = adaptiveCutoff;
		}
		List<ScoredEmbedding> candidates = built.candidates;

		// Keyword refinement.
		List<ScoredEmbedding> preRefinementCandidates = candidates;
		boolean refinementActivated = false;
		boolean selectiveKwRescued = false;
		if (config.keywordWeight > 0) {
			List<ScoredEmbedding> refined = refineByKeywords(candidates, queryTermCount);
			refinementActivated = refined.size() < candidates.size();
			if (refinementActivated) {
				double rMaxSem = 0;
				double rMaxKw = 0;
				for (ScoredEmbedding se : refined) {
					if (se.semanticScore > rMaxSem) {
						rMaxSem = se.semanticScore;
					}
					if (se.keywordScore > rMaxKw) {
						rMaxKw = se.keywordScore;
					}
				}
				// Partial-match semantic floor: when keyword refinement
				// kept only PARTIAL matches AND the best semantic score
				// is below max(absoluteFloor, noiseMean - 1 std), the
				// keyword overlap is coincidental — the embedding model
				// sees no topical relevance.
				double partialKwFloor = Math.max(
						config.noiseProfile.absoluteSimilarityFloor(),
						config.noiseProfile.noiseMean
								- config.noiseProfile.noiseStd);
				int selectiveKwMaxRecords =
						ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
				if (scored.size()
						> ChartSearchAiConstants.LARGE_CORPUS_SELECTIVE_RESCUE_MIN) {
					selectiveKwMaxRecords = Math.max(
							selectiveKwMaxRecords,
							(int) Math.ceil(scored.size()
									* ChartSearchAiConstants.LARGE_CORPUS_SELECTIVE_KW_FRACTION));
				}
				boolean selectiveKwMatch = refined.size()
						<= selectiveKwMaxRecords;
				if (selectiveKwMatch && rMaxKw < bonusThreshold
						&& rMaxSem < partialKwFloor) {
					// Selective-keyword rescue: route through the
					// non-refinement path (ratio floor + z-score guards)
					// instead of the refinement expansion path that
					// would flood on compressed score models.
					refinementActivated = false;
					selectiveKwRescued = true;
					log.debug("Selective-keyword rescue: keeping {} "
							+ "records via non-refinement path "
							+ "(maxSem={}, floor={})",
							refined.size(),
							String.format("%.4f", rMaxSem),
							String.format("%.4f", partialKwFloor));
				} else if (rMaxKw < bonusThreshold
						&& rMaxSem < partialKwFloor) {
					Set<Integer> badIds =
							new HashSet<Integer>();
					for (ScoredEmbedding se : refined) {
						badIds.add(
								se.embedding.getResourceId());
					}
					List<ScoredEmbedding> cleaned =
							new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (!badIds.contains(
								se.embedding
										.getResourceId())) {
							cleaned.add(se);
						}
					}
					log.warn("Partial-match semantic floor: "
							+ "removing {} below-floor partial "
							+ "kw records (maxSem={}, floor={})",
							badIds.size(),
							String.format("%.4f", rMaxSem),
							String.format("%.4f", partialKwFloor));
					refined = cleaned;
					refinementActivated = false;
				}
			}
			candidates = refined;
		}

		log.warn("Pipeline stages {}: maxSem={}, maxBase={}, floor={}, kwCount={}, "
				+ "termCount={}, gapCutoff={}, candidates={}, refined={}",
				Arrays.toString(queryTerms), String.format("%.4f", maxSemanticScore),
				String.format("%.4f", maxBaseScore),
				String.format("%.4f", config.noiseProfile.absoluteSimilarityFloor()),
				keywordMatchCount, queryTermCount,
				adaptiveCutoff, candidates.size(), refinementActivated);

		boolean partialKwValidated = false;
		boolean firstPassGapDetected = adaptiveCutoff < scored.size();
		int ratioFloorCandidateCount = -1;

		// Post-processing with two paths.
		if (refinementActivated) {
			boolean[] pkvOut = { false };
			candidates = applyRefinementPath(candidates,
					preRefinementCandidates, scored, queryTerms,
					queryTermCount, bonusThreshold, minScore,
					config, pkvOut);
			partialKwValidated = pkvOut[0];
		} else {
			int[] rfccOut = { -1 };
			candidates = applyNonRefinementPath(candidates, scored,
					maxSemanticScore, maxBaseScore, minScore,
					config, rfccOut);
			ratioFloorCandidateCount = rfccOut[0];
		}

		// Keyword-dominance rescue: when all post-processed candidates
		// have keyword matches and the set is small, scan the full
		// scored list for non-keyword records that are above the ratio
		// floor and individually coherent with the keyword set above a
		// data-derived threshold.
		if (candidates.size()
				<= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS
				&& !candidates.isEmpty()) {
			boolean allKw = true;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore == 0) {
					allKw = false;
					break;
				}
			}
			if (allKw) {
				double kwMaxSem = 0;
				double kwMaxScore = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > kwMaxSem) {
						kwMaxSem = se.semanticScore;
					}
					if (se.keywordScore > kwMaxScore) {
						kwMaxScore = se.keywordScore;
					}
				}
				double kwConf = queryTermCount > 0
						? kwMaxScore * queryTermCount
								/ Math.max(bonusThreshold, 0.01)
						: 1.0;
				double perRecordThreshold = Math.sqrt(
						config.noiseProfile.noiseMean
						* config.noiseProfile.noiseP95) * kwConf;
				double rescueRatio = selectiveKwRescued
						? config.similarityRatio
								* config.similarityRatio
						: config.similarityRatio;
				double mergeFloor = kwMaxSem * rescueRatio;
				Set<Integer> inCandidates = new HashSet<Integer>();
				for (ScoredEmbedding se : candidates) {
					inCandidates.add(
							se.embedding.getResourceId());
				}

				// For the selective-keyword rescue path, compute a
				// distribution-based coherence threshold instead of
				// the fixed noise-profile one.
				double effectiveThreshold = perRecordThreshold;
				if (selectiveKwRescued) {
					List<Double> cohValues =
							new ArrayList<Double>();
					for (ScoredEmbedding se : scored) {
						if (inCandidates.contains(
								se.embedding.getResourceId())
								|| se.keywordScore > 0) {
							continue;
						}
						float[] vec = se.embedding
								.getEmbeddingVector();
						if (vec == null) {
							continue;
						}
						double cs = 0;
						int cn = 0;
						for (ScoredEmbedding kw : candidates) {
							float[] kwVec = kw.embedding
									.getEmbeddingVector();
							if (kwVec != null
									&& kwVec.length
									== vec.length) {
								cs += ChartSearchAiUtils
										.cosineSimilarity(
												vec, kwVec);
								cn++;
							}
						}
						if (cn > 0) {
							cohValues.add(cs / cn);
						}
					}
					if (cohValues.size() >= 4) {
						double cohSum = 0;
						for (double v : cohValues) {
							cohSum += v;
						}
						double cohMean = cohSum
								/ cohValues.size();
						double cohSqSum = 0;
						for (double v : cohValues) {
							double d = v - cohMean;
							cohSqSum += d * d;
						}
						double cohStd = Math.sqrt(cohSqSum
								/ cohValues.size());
						effectiveThreshold = cohMean + cohStd / 2;
						log.debug("Selective rescue coherence "
								+ "threshold: mean={}, std={}, "
								+ "effective={} (fixed was {})",
								String.format("%.4f", cohMean),
								String.format("%.4f", cohStd),
								String.format("%.4f",
										effectiveThreshold),
								String.format("%.4f",
										perRecordThreshold));
					}
				}

				List<ScoredEmbedding> rescued =
						new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : scored) {
					if (inCandidates.contains(
							se.embedding.getResourceId())
							|| se.keywordScore > 0
							|| se.semanticScore < mergeFloor) {
						continue;
					}
					float[] vec = se.embedding
							.getEmbeddingVector();
					if (vec == null) {
						continue;
					}
					double cosSum = 0;
					int cnt = 0;
					for (ScoredEmbedding kw : candidates) {
						float[] kwVec = kw.embedding
								.getEmbeddingVector();
						if (kwVec != null
								&& kwVec.length == vec.length) {
							cosSum += ChartSearchAiUtils
									.cosineSimilarity(
											vec, kwVec);
							cnt++;
						}
					}
					if (cnt > 0
							&& cosSum / cnt
							>= effectiveThreshold) {
						rescued.add(se);
					}
				}
				if (!rescued.isEmpty()) {
					// Concept-pairing for rescued records.
					Set<String> rescuedConcepts =
							new HashSet<String>();
					Set<Integer> rescuedIds =
							new HashSet<Integer>();
					for (ScoredEmbedding se : rescued) {
						rescuedIds.add(
								se.embedding.getResourceId());
						String cn = ConceptNameUtil
								.extractConceptName(
										se.embedding
												.getTextContent());
						if (cn != null) {
							rescuedConcepts.add(cn);
						}
					}
					if (selectiveKwRescued
						&& !rescuedConcepts.isEmpty()) {
						for (ScoredEmbedding se : scored) {
							int rid = se.embedding
									.getResourceId();
							if (inCandidates.contains(rid)
									|| rescuedIds.contains(rid)) {
								continue;
							}
							String cn = ConceptNameUtil
									.extractConceptName(
											se.embedding
													.getTextContent());
							if (cn != null
									&& rescuedConcepts
											.contains(cn)) {
								rescued.add(se);
								rescuedIds.add(rid);
							}
						}
					}
					log.warn("Keyword-dominance rescue: merging "
							+ "{} non-kw records (threshold={}, "
							+ "kwConf={}) with {} kw records",
							rescued.size(),
							String.format("%.4f",
									effectiveThreshold),
							String.format("%.2f", kwConf),
							candidates.size());
					List<ScoredEmbedding> merged =
							new ArrayList<ScoredEmbedding>(
									candidates);
					merged.addAll(rescued);
					candidates = merged;
					partialKwValidated = true;
				}
			}
		}

		// Phase 1: Outlier removal.
		candidates = applyOutlierRemovalPhase1(candidates, queryTerms,
				partialKwValidated);
		// Phase 2: Zero-keyword validation.
		candidates = applyZeroKeywordValidationPhase2(candidates,
				scored, config, partialKwValidated, belowFloorRescued,
				firstPassGapDetected, adaptiveCutoff,
				ratioFloorCandidateCount, initialZScore,
				initialZThreshold, maxSemanticScore);

		List<ChartEmbedding> results = new ArrayList<ChartEmbedding>();
		for (ScoredEmbedding se : candidates) {
			results.add(se.embedding);
		}

		// Safety net: if all downstream gates rejected every record,
		// but the initial z-score gate validated the top score as a
		// genuine statistical outlier (≥ 2x threshold) AND the top
		// semantic score clears floor + 2*minScoreGap, return the top
		// record as a last-resort fallback.
		if (results.isEmpty() && !scored.isEmpty()
				&& initialZScore >= 0
				&& initialZScore >= 2 * initialZThreshold
				&& scored.get(0).semanticScore
						>= config.noiseProfile.absoluteSimilarityFloor()
								+ 2 * config.minScoreGap) {
			log.warn("Safety net: all gates rejected {} records but top "
					+ "semantic score {} >= floor+2*gap {} and z-score "
					+ "{} >= 2*threshold {}, returning top record as "
					+ "fallback",
					scored.size(),
					String.format("%.4f", scored.get(0).semanticScore),
					String.format("%.4f",
							config.noiseProfile.absoluteSimilarityFloor()
									+ 2 * config.minScoreGap),
					String.format("%.2f", initialZScore),
					String.format("%.2f", 2 * initialZThreshold));
			results = Collections.singletonList(scored.get(0).embedding);
		}

		int logLimit = Math.min(20, scored.size());
		StringBuilder scores = new StringBuilder();
		for (int i = 0; i < logLimit; i++) {
			if (i > 0) {
				scores.append(", ");
			}
			ScoredEmbedding se = scored.get(i);
			scores.append(se.embedding.getResourceType())
					.append(":").append(se.embedding.getResourceId())
					.append("=").append(String.format("%.4f", se.score));
		}
		log.warn("Similarity scores: [{}], minScore: {}, adaptiveCutoff: {}",
				scores, String.format("%.4f", minScore), adaptiveCutoff);
		log.warn("Returning {} {} of {} candidates (topScore={})",
				Arrays.toString(queryTerms), results.size(), scored.size(),
				String.format("%.4f", maxBaseScore));
		return results;
	}

	/**
	 * Finds the adaptive cutoff point in a sorted list of scored embeddings
	 * by detecting a significant gap in raw semantic similarity scores. Uses
	 * semantic scores (not keyword-inflated combined scores) so that keyword
	 * bonuses don't create artificial cluster boundaries — this ensures
	 * multi-concept queries (e.g. "blood pressure, weight, and temperature")
	 * don't get split by uneven keyword matching across concepts. Always
	 * includes at least {@link ChartSearchAiConstants#ADAPTIVE_MIN_RECORDS}
	 * records (if available above the similarity floor) so the LLM has enough
	 * context.
	 *
	 * @param scored sorted list of scored embeddings (highest combined score first)
	 * @param limit the maximum number of candidates to consider
	 * @param minScore the absolute similarity floor (applied to semantic scores)
	 * @param gapMultiplier how many times larger than the average gap a score
	 *        drop must be to trigger a cutoff
	 * @param minGap the minimum absolute gap required to trigger a cutoff,
	 *        regardless of the multiplier condition
	 * @return the number of records to include in the primary cluster
	 */
	static int findAdaptiveCutoff(List<ScoredEmbedding> scored, int limit, double minScore,
			double gapMultiplier, double minGap) {
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;

		List<Double> semanticSorted = new ArrayList<Double>();
		for (int i = 0; i < limit; i++) {
			semanticSorted.add(scored.get(i).semanticScore);
		}
		Collections.sort(semanticSorted, Collections.reverseOrder());

		int aboveFloor = 0;
		for (int i = 0; i < semanticSorted.size(); i++) {
			if (semanticSorted.get(i) >= minScore) {
				aboveFloor++;
			} else {
				break;
			}
		}

		if (aboveFloor == 0) {
			return 0;
		}

		double gapSum = 0;
		int semanticCutoff = aboveFloor;
		for (int i = 1; i < aboveFloor; i++) {
			double gap = semanticSorted.get(i - 1) - semanticSorted.get(i);

			if (i >= minRecords && i >= 2) {
				double avgGap = gapSum / (i - 1);
				if (gap > avgGap * gapMultiplier && gap > minGap) {
					semanticCutoff = i;
					log.debug("Score gap detected at position {}: gap={}, avgGap={}, multiplier={}",
							i, String.format("%.4f", gap), String.format("%.4f", avgGap),
							gapMultiplier);
					break;
				}
			}
			gapSum += gap;
		}

		return Math.max(semanticCutoff, Math.min(minRecords, aboveFloor));
	}

	/**
	 * Gap detection + candidate building — stage 4 of the filter pipeline.
	 * Computes the permissive {@code minScore} floor, an IQR-based
	 * adaptive {@code minGap} for compressed-distribution models, then
	 * calls {@link #findAdaptiveCutoff} to locate the semantic cluster
	 * boundary. Builds the candidate set from records at or above the
	 * boundary OR with a full keyword match (kwScore ≥ bonusThreshold).
	 */
	static CandidateBuildResult buildCandidates(
			List<ScoredEmbedding> scored, double maxSemanticScore,
			double bonusThreshold, PipelineConfig config) {
		// Permissive floor: gap detection handles the real cutoff.
		double minScore = Math.min(
				config.noiseProfile.absoluteSimilarityFloor() / 2,
				maxSemanticScore / 2);

		// Adaptive min gap: scale to score-range so models with
		// compressed distributions (e.g. MedCPT) can detect gaps that
		// are large relative to their range but small in absolute terms.
		double firstPassMinGap = config.minScoreGap;
		if (scored.size() >= 4) {
			List<Double> semScores = new ArrayList<Double>(scored.size());
			for (ScoredEmbedding se : scored) {
				if (se.semanticScore >= minScore) {
					semScores.add(se.semanticScore);
				}
			}
			Collections.sort(semScores);
			if (semScores.size() >= 4) {
				double q1 = semScores.get(semScores.size() / 4);
				double q3 = semScores.get(3 * semScores.size() / 4);
				double iqr = q3 - q1;
				if (iqr < config.minScoreGap / 2) {
					Set<String> gapConcepts = new HashSet<String>();
					for (ScoredEmbedding se : scored) {
						String cn = ConceptNameUtil
								.extractConceptName(
										se.embedding
												.getTextContent());
						if (cn != null) {
							gapConcepts.add(cn);
						}
					}
					double fraction = (double) gapConcepts.size()
							/ scored.size();
					firstPassMinGap = iqr * fraction;
				}
			}
		}

		int adaptiveCutoff = findAdaptiveCutoff(scored, scored.size(),
				minScore, config.scoreGapMultiplier, firstPassMinGap);

		// Build the candidate set from the semantic cluster.
		List<ScoredEmbedding> candidates;
		if (adaptiveCutoff >= scored.size()) {
			candidates = new ArrayList<ScoredEmbedding>(scored);
		} else {
			List<Double> semanticDesc = new ArrayList<Double>(scored.size());
			for (ScoredEmbedding se : scored) {
				semanticDesc.add(se.semanticScore);
			}
			Collections.sort(semanticDesc, Collections.reverseOrder());
			double clusterThreshold = semanticDesc.get(adaptiveCutoff - 1);
			candidates = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : scored) {
				// Two independent inclusion signals: semantic cluster
				// or full keyword match. The embedding model may assign
				// low similarity to obscure medical terminology but the
				// keyword match is conclusive evidence the record
				// belongs to the queried category.
				if (se.semanticScore >= clusterThreshold
						|| (config.keywordWeight > 0
								&& se.keywordScore >= bonusThreshold)) {
					candidates.add(se);
				}
			}
			Collections.sort(candidates, new Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.score, a.score);
				}
			});
		}

		return new CandidateBuildResult(candidates, adaptiveCutoff, minScore);
	}

	/**
	 * Refinement post-processing path — stage 5a of the filter pipeline.
	 * Runs when keyword refinement identified a relevant subset. Computes
	 * uniform-keyword and semantic-dominance signals, then branches into
	 * the partial-keyword semantic-core path or the non-uniform second-pass
	 * gap detection path.
	 *
	 * @param partialKwValidatedOut set to {@code true} when the semantic
	 *        core expansion path fired — used downstream to skip
	 *        coherence filtering that would double-filter the core.
	 */
	static List<ScoredEmbedding> applyRefinementPath(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> preRefinementCandidates,
			List<ScoredEmbedding> scored,
			String[] queryTerms, int queryTermCount,
			double bonusThreshold, double minScore,
			PipelineConfig config, boolean[] partialKwValidatedOut) {
		double kwMin = Double.MAX_VALUE;
		double kwMax = 0;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore > 0) {
				if (se.keywordScore < kwMin) {
					kwMin = se.keywordScore;
				}
				if (se.keywordScore > kwMax) {
					kwMax = se.keywordScore;
				}
			}
		}
		if (kwMin == Double.MAX_VALUE) {
			kwMin = kwMax;
		}
		boolean uniformKeywords = (kwMax - kwMin) < 0.01;
		boolean semanticDominance = false;
		double maxKwSem = 0;
		double maxNonKwSem = 0;
		if (uniformKeywords && kwMax < bonusThreshold
				&& queryTermCount > 0) {
			double refMinKw = 1.0 / queryTermCount;
			for (ScoredEmbedding se : preRefinementCandidates) {
				if (se.keywordScore >= refMinKw) {
					if (se.semanticScore > maxKwSem) {
						maxKwSem = se.semanticScore;
					}
				} else {
					if (se.semanticScore > maxNonKwSem) {
						maxNonKwSem = se.semanticScore;
					}
				}
			}
			semanticDominance = maxNonKwSem > maxKwSem;
		}
		boolean partialKeywordMatch = uniformKeywords
				&& kwMax < bonusThreshold;
		log.warn("Pipeline refinement: kwMin={}, kwMax={}, uniform={}, partialKw={}, semDom={}",
				String.format("%.4f", kwMin),
				String.format("%.4f", kwMax), uniformKeywords,
				partialKeywordMatch, semanticDominance);
		if (partialKeywordMatch) {
			List<ScoredEmbedding> nonKeyword = new ArrayList<ScoredEmbedding>();
			for (ScoredEmbedding se : preRefinementCandidates) {
				if (se.keywordScore == 0) {
					nonKeyword.add(se);
				}
			}
			Collections.sort(nonKeyword,
					new Comparator<ScoredEmbedding>() {
				@Override
				public int compare(ScoredEmbedding a, ScoredEmbedding b) {
					return Double.compare(b.semanticScore, a.semanticScore);
				}
			});
			List<ScoredEmbedding> semanticCore = new ArrayList<ScoredEmbedding>();
			boolean coreFallbackUsed = false;
			int coreCutoff = nonKeyword.size();
			if (nonKeyword.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				double maxSemNk = nonKeyword.get(0).semanticScore;
				double nkMinGap = Math.max(
						maxSemNk * SimilarityAndScoringEngine.refinementAdaptiveGapRatio(scored),
						SimilarityAndScoringEngine.secondPassMinGap(scored));
				coreCutoff = findAdaptiveCutoff(nonKeyword, nonKeyword.size(),
						config.noiseProfile.absoluteSimilarityFloor(),
						config.scoreGapMultiplier, nkMinGap);
				int fisherCut = nonKeyword.size();
				double totalSumF = 0;
				double totalSumSqF = 0;
				for (ScoredEmbedding se : nonKeyword) {
					totalSumF += se.semanticScore;
					totalSumSqF += se.semanticScore * se.semanticScore;
				}
				int nF = nonKeyword.size();
				double leftSumF = 0;
				double leftSumSqF = 0;
				double rightSumF = totalSumF;
				double rightSumSqF = totalSumSqF;
				double maxGF = 0;
				for (int i = 1; i < nF; i++) {
					double s = nonKeyword.get(i - 1).semanticScore;
					leftSumF += s;
					leftSumSqF += s * s;
					rightSumF -= s;
					rightSumSqF -= s * s;
					if (i < ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
						continue;
					}
					int nR = nF - i;
					if (nR < 2) {
						break;
					}
					double lm = leftSumF / i;
					double rm = rightSumF / nR;
					double lv = leftSumSqF / i - lm * lm;
					double rv = rightSumSqF / nR - rm * rm;
					double den = lv / (i - 1) + rv / (nR - 1) + 1e-12;
					double diff = lm - rm;
					double t2 = diff * diff / den;
					double bGap = nonKeyword.get(i - 1).semanticScore
							- nonKeyword.get(i).semanticScore;
					double gf = bGap * t2;
					if (gf > maxGF) {
						maxGF = gf;
						fisherCut = i;
					}
				}
				log.debug("Fisher cluster cut at {} (score={})",
						fisherCut, String.format("%.2f", maxGF));
				if (fisherCut < coreCutoff) {
					double bGapF = nonKeyword.get(fisherCut - 1).semanticScore
							- nonKeyword.get(fisherCut).semanticScore;
					double prevGapF = nonKeyword.get(fisherCut - 2).semanticScore
							- nonKeyword.get(fisherCut - 1).semanticScore;
					if (bGapF > prevGapF) {
						coreFallbackUsed = true;
						coreCutoff = fisherCut;
					} else {
						log.debug("Fisher cut at {} suppressed: boundary gap {} <= prev gap {}",
								fisherCut,
								String.format("%.4f", bGapF),
								String.format("%.4f", prevGapF));
					}
				}
				for (int i = 0; i < coreCutoff; i++) {
					semanticCore.add(nonKeyword.get(i));
				}
				log.warn("Semantic core: {} non-keyword records, primary gap at {}, core size {}",
						nonKeyword.size(), coreCutoff, semanticCore.size());
				// Drop concepts whose mean inter-concept cosine to non-pair
				// core members is at or below the patient's noiseMean.
				semanticCore = pruneCoherenceOutlierConcepts(
						semanticCore, config.noiseProfile);
			}
			boolean coreHasStructure = !semanticCore.isEmpty()
					&& coreCutoff < nonKeyword.size();
			boolean coreRelevant = false;
			if (coreHasStructure && !semanticCore.isEmpty()
					&& !candidates.isEmpty()) {
				float[] coreVec = semanticCore.get(0).embedding.getEmbeddingVector();
				double maxCoreKwCosine = 0;
				for (ScoredEmbedding se : candidates) {
					double cos = ChartSearchAiUtils.cosineSimilarity(coreVec,
							se.embedding.getEmbeddingVector());
					if (cos > maxCoreKwCosine) {
						maxCoreKwCosine = cos;
					}
				}
				coreRelevant = maxCoreKwCosine
						>= ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE;
				log.warn("Core topical check: maxCos={} vs threshold={}, relevant={}",
						String.format("%.4f", maxCoreKwCosine),
						ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE,
						coreRelevant);
			}
			log.warn("Core validation: structure={}, relevant={}, cutoff={}/{}, semDom={}",
					coreHasStructure, coreRelevant,
					coreCutoff, nonKeyword.size(), semanticDominance);
			if (semanticDominance || coreRelevant) {
				if (!semanticCore.isEmpty()) {
					double coreMinSem = semanticCore.get(
							semanticCore.size() - 1).semanticScore;
					double coreMaxSem = semanticCore.get(0).semanticScore;
					double expansionRatio = coreFallbackUsed
							? (coreMaxSem > 0
									? coreMinSem / coreMaxSem
									: ChartSearchAiConstants.SEMANTIC_CORE_SCORE_RATIO)
							: ChartSearchAiConstants.SEMANTIC_CORE_SCORE_RATIO;
					double scoreFloor = coreMinSem * expansionRatio;
					List<ScoredEmbedding> expanded =
							new ArrayList<ScoredEmbedding>(semanticCore);
					Set<Integer> expandedIds = new HashSet<Integer>();
					for (ScoredEmbedding se : semanticCore) {
						expandedIds.add(se.embedding.getResourceId());
					}
					if (!semanticDominance) {
						for (ScoredEmbedding se : preRefinementCandidates) {
							if (expandedIds.contains(se.embedding.getResourceId())) {
								continue;
							}
							if (se.semanticScore < scoreFloor) {
								continue;
							}
							float[] vec = se.embedding.getEmbeddingVector();
							double maxCos = 0;
							for (ScoredEmbedding core : semanticCore) {
								double cos = ChartSearchAiUtils.cosineSimilarity(vec,
										core.embedding.getEmbeddingVector());
								if (cos > maxCos) {
									maxCos = cos;
								}
							}
							if (maxCos >= ChartSearchAiConstants.SEMANTIC_CORE_MIN_COSINE) {
								expanded.add(se);
								expandedIds.add(se.embedding.getResourceId());
							}
						}
					}
					// Concept-pair rescue for keyword-matched anchors:
					// when a kw-matched record from the refinement input
					// shares its concept name with a record already in
					// expanded, restore the missing partner.
					Set<String> expandedConcepts = new HashSet<String>();
					for (ScoredEmbedding se : expanded) {
						String cn = ConceptNameUtil.extractConceptName(
								se.embedding.getTextContent());
						if (cn != null) {
							expandedConcepts.add(cn);
						}
					}
					for (ScoredEmbedding se : candidates) {
						if (expandedIds.contains(se.embedding.getResourceId())) {
							continue;
						}
						if (se.keywordScore <= 0) {
							continue;
						}
						String cn = ConceptNameUtil.extractConceptName(
								se.embedding.getTextContent());
						if (cn != null && expandedConcepts.contains(cn)) {
							expanded.add(se);
							expandedIds.add(se.embedding.getResourceId());
							log.debug("Concept-pair rescue: re-added kw-matched [{}] (concept={})",
									se.embedding.getResourceId(), cn);
						}
					}
					candidates = expanded;
					log.debug("Semantic core expansion: core={}, floor={}, result={}",
							semanticCore.size(),
							String.format("%.4f", scoreFloor),
							candidates.size());
					partialKwValidatedOut[0] = true;
				} else {
					double maxSemanticKw = 0;
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore > maxSemanticKw) {
							maxSemanticKw = se.semanticScore;
						}
					}
					double semFloor = maxSemanticKw
							* SimilarityAndScoringEngine.refinementSemanticRatio(candidates);
					List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore >= semFloor) {
							filtered.add(se);
						}
					}
					if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
						log.debug("Partial-kw semantic floor: {} -> {} (floor={})",
								candidates.size(), filtered.size(),
								String.format("%.4f", semFloor));
						candidates = filtered;
					}
				}
			} else {
				double maxSemanticKw = 0;
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore > maxSemanticKw) {
						maxSemanticKw = se.semanticScore;
					}
				}
				double semFloor = maxSemanticKw * config.similarityRatio;
				List<ScoredEmbedding> filtered = new ArrayList<ScoredEmbedding>();
				for (ScoredEmbedding se : candidates) {
					if (se.semanticScore >= semFloor) {
						filtered.add(se);
					}
				}
				if (filtered.size() >= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
					log.debug("Low-coverage kw semantic floor: {} -> {} (floor={})",
							candidates.size(), filtered.size(),
							String.format("%.4f", semFloor));
					candidates = filtered;
				}
			}
		} else if (!uniformKeywords) {
			double maxSemanticRefined = 0;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore > maxSemanticRefined) {
					maxSemanticRefined = se.semanticScore;
				}
			}
			double adaptiveMinGap = Math.max(
					maxSemanticRefined * SimilarityAndScoringEngine.refinementAdaptiveGapRatio(scored),
					SimilarityAndScoringEngine.secondPassMinGap(scored));
			int refinedCutoff = findAdaptiveCutoff(candidates,
					candidates.size(), minScore,
					config.scoreGapMultiplier, adaptiveMinGap);
			if (refinedCutoff < candidates.size()) {
				if (isGapCoherent(candidates, refinedCutoff,
						config.gapValidationCosineThreshold)) {
					double semanticFloor = maxSemanticRefined
							* SimilarityAndScoringEngine.refinementSemanticRatio(candidates);
					List<ScoredEmbedding> floored = new ArrayList<ScoredEmbedding>();
					for (ScoredEmbedding se : candidates) {
						if (se.semanticScore >= semanticFloor) {
							floored.add(se);
						}
					}
					log.debug("Refinement gap is intra-topic, using semantic floor {} instead: {} -> {}",
							String.format("%.4f", semanticFloor),
							candidates.size(), floored.size());
					candidates = floored;
				} else if (queryTerms != null) {
					List<ScoredEmbedding> rescued =
							filterRedundantKeywordTier(candidates,
									queryTerms, kwMax, bonusThreshold);
					if (rescued.size() >= refinedCutoff) {
						log.debug("Refinement gap at {} is a concept boundary: rescued {} records with new coverage",
								refinedCutoff, rescued.size() - refinedCutoff);
						candidates = rescued;
					} else {
						candidates = new ArrayList<ScoredEmbedding>(
								candidates.subList(0, refinedCutoff));
					}
				} else {
					candidates = new ArrayList<ScoredEmbedding>(
							candidates.subList(0, refinedCutoff));
				}
			} else if (queryTerms != null) {
				candidates = filterRedundantKeywordTier(
						candidates, queryTerms, kwMax, bonusThreshold);
			}
		}
		return candidates;
	}

	/**
	 * Non-refinement post-processing path — stage 5b of the filter pipeline.
	 * Sensitive second-pass gap detection, ratio floor, and concept-pairing
	 * rescue.
	 *
	 * @param ratioFloorCandidateCountOut receives the strict-floor count
	 *        (before concept-pairing rescue) — used by the cluster z-score
	 *        gate downstream.
	 */
	static List<ScoredEmbedding> applyNonRefinementPath(
			List<ScoredEmbedding> candidates,
			List<ScoredEmbedding> scored,
			double maxSemanticScore, double maxBaseScore,
			double minScore, PipelineConfig config,
			int[] ratioFloorCandidateCountOut) {
		// Sensitive second-pass: same multiplier as the first pass, but
		// with a much lower absolute floor derived from the score
		// distribution's std.
		int secondCutoff = findAdaptiveCutoff(candidates, candidates.size(),
				minScore, config.scoreGapMultiplier,
				SimilarityAndScoringEngine.secondPassMinGap(scored));
		if (secondCutoff < candidates.size()) {
			if (isGapCoherent(candidates, secondCutoff,
					config.gapValidationCosineThreshold)) {
				log.debug("Second-pass gap at {} is intra-topic, skipping cut",
						secondCutoff);
			} else {
				candidates = new ArrayList<ScoredEmbedding>(
						candidates.subList(0, secondCutoff));
			}
		}
		// Use the lower of maxBaseScore and maxSemanticScore for the
		// floor. Bonuses inflate combined scores; penalties deflate.
		// The per-record check uses min(combined, semantic) so bonuses
		// are stripped (checking semantic) and penalties are preserved.
		//
		// Uniform-keyword exception: when every candidate has the same
		// keyword score, the penalty/bonus is applied equally, so it
		// cancels out in relative ranking but distorts the absolute
		// floor. Fall back to pure-semantic comparison.
		boolean uniformCandidateKw = true;
		double firstKw = candidates.isEmpty() ? 0
				: candidates.get(0).keywordScore;
		for (ScoredEmbedding se : candidates) {
			if (Math.abs(se.keywordScore - firstKw) > 1e-9) {
				uniformCandidateKw = false;
				break;
			}
		}
		// Uniform-keyword tight-cluster bypass: when every candidate
		// shares the same (non-zero) keyword score AND their semantic
		// scores form a tight cluster (within 2× cross-concept noise
		// std), skip the ratio floor — it would arbitrarily drop the
		// lower-scoring members of a cohesive same-topic group.
		boolean tightUniformCluster = false;
		if (uniformCandidateKw && firstKw > 0
				&& candidates.size()
				>= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
			double maxSem = 0;
			double minSem = Double.MAX_VALUE;
			for (ScoredEmbedding se : candidates) {
				if (se.semanticScore > maxSem) maxSem = se.semanticScore;
				if (se.semanticScore < minSem) minSem = se.semanticScore;
			}
			double spread = maxSem - minSem;
			double noiseSpreadBand = 2.0 * config.noiseProfile.noiseStd;
			if (maxSem > 0 && spread <= noiseSpreadBand) {
				tightUniformCluster = true;
				log.debug("Uniform-keyword tight cluster detected: spread={}, noiseBand={}, kw={}, n={} — bypassing ratio floor",
						String.format("%.4f", spread),
						String.format("%.4f", noiseSpreadBand),
						String.format("%.3f", firstKw),
						candidates.size());
			}
		}
		double ratioFloor = uniformCandidateKw
				? maxSemanticScore * config.similarityRatio
				: Math.min(maxBaseScore, maxSemanticScore)
						* config.similarityRatio;
		List<ScoredEmbedding> strict = new ArrayList<ScoredEmbedding>();
		List<ScoredEmbedding> nearMiss = new ArrayList<ScoredEmbedding>();
		for (ScoredEmbedding se : candidates) {
			if (tightUniformCluster) {
				strict.add(se);
				continue;
			}
			double check = uniformCandidateKw ? se.semanticScore
					: Math.min(se.score, se.semanticScore);
			if (check >= ratioFloor) {
				strict.add(se);
			} else {
				nearMiss.add(se);
			}
		}
		ratioFloorCandidateCountOut[0] = strict.size();
		log.debug("Non-refinement: ratioFloor={}, strict={}, nearMiss={}, total={}",
				String.format("%.4f", ratioFloor), strict.size(),
				nearMiss.size(), candidates.size());
		// Concept-pairing rescue: when a record barely misses the
		// ratio floor but its same-concept partner survived, the floor
		// is splitting a natural concept pair. Rescue the near-miss if
		// its gap from the floor is within the noise resolution.
		if (!strict.isEmpty() && !nearMiss.isEmpty()) {
			Map<String, Integer> conceptCounts = new HashMap<String, Integer>();
			Set<String> survivorConcepts = new HashSet<String>();
			for (ScoredEmbedding se : strict) {
				String cn = ConceptNameUtil
						.extractConceptName(se.embedding.getTextContent());
				if (cn != null) {
					survivorConcepts.add(cn);
					Integer count = conceptCounts.get(cn);
					conceptCounts.put(cn, count == null ? 1 : count + 1);
				}
			}
			boolean withinConceptCalibration = false;
			for (int count : conceptCounts.values()) {
				if (count >= 2) {
					withinConceptCalibration = true;
					break;
				}
			}
			if (withinConceptCalibration) {
				Set<String> allConcepts = new HashSet<String>();
				for (ScoredEmbedding se : scored) {
					String cn = ConceptNameUtil
							.extractConceptName(se.embedding.getTextContent());
					if (cn != null) {
						allConcepts.add(cn);
					}
				}
				double rescueTolerance = config.noiseProfile.noiseStd
						/ Math.max(1, allConcepts.size());
				for (ScoredEmbedding se : nearMiss) {
					double eff = Math.min(se.score, se.semanticScore);
					double gap = ratioFloor - eff;
					if (gap > rescueTolerance) {
						continue;
					}
					String cn = ConceptNameUtil
							.extractConceptName(se.embedding.getTextContent());
					if (cn != null && survivorConcepts.contains(cn)) {
						strict.add(se);
						log.debug("Concept-pairing rescue: [{}] gap={}, concept={}",
								se.embedding.getResourceId(),
								String.format("%.4f", gap), cn);
					}
				}
			}
		}
		candidates = strict;
		return candidates;
	}

	/**
	 * Refines the gap-detected result set by keeping only records that
	 * have strong keyword matches, when those matches identify a clear
	 * subset. Catches cases where the score distribution is too smooth
	 * for gap detection to find a cutoff but keyword matching provides
	 * a discriminative type signal.
	 */
	static List<ScoredEmbedding> refineByKeywords(List<ScoredEmbedding> candidates,
			int queryTermCount) {
		if (queryTermCount == 0) {
			return candidates;
		}
		// Require at least 1 matching query term — any keyword
		// relevance is sufficient. Gap detection is the primary noise
		// filter; refinement separates "has keyword evidence" from
		// "has no keyword evidence".
		double minKwScore = 1.0 / queryTermCount;

		List<ScoredEmbedding> keywordMatched = new ArrayList<ScoredEmbedding>();
		double kwMinScore = Double.MAX_VALUE;
		for (ScoredEmbedding se : candidates) {
			if (se.keywordScore >= minKwScore) {
				keywordMatched.add(se);
				if (se.keywordScore < kwMinScore) {
					kwMinScore = se.keywordScore;
				}
			}
		}
		int minRecords = ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
		if (keywordMatched.size() >= minRecords && keywordMatched.size() < candidates.size()) {
			// Rescue non-keyword records that the semantic model ranks
			// above the WEAKEST keyword tier.
			double maxMinTierSemantic = 0;
			for (ScoredEmbedding se : keywordMatched) {
				if (se.keywordScore <= kwMinScore + 0.01
						&& se.semanticScore > maxMinTierSemantic) {
					maxMinTierSemantic = se.semanticScore;
				}
			}
			List<ScoredEmbedding> refined =
					new ArrayList<ScoredEmbedding>(keywordMatched);
			int rescued = 0;
			for (ScoredEmbedding se : candidates) {
				if (se.keywordScore < minKwScore
						&& se.semanticScore > maxMinTierSemantic) {
					refined.add(se);
					rescued++;
				}
			}
			if (rescued > 0) {
				log.debug("Keyword refinement: keeping {} of {} "
						+ "(minKwScore={}, rescued {} semantic-"
						+ "dominant non-keyword records, "
						+ "minTierSem={})",
						refined.size(), candidates.size(),
						String.format("%.3f", minKwScore), rescued,
						String.format("%.4f", maxMinTierSemantic));
			} else {
				log.debug("Keyword refinement: keeping {} of {} "
						+ "(minKwScore={})",
						keywordMatched.size(), candidates.size(),
						String.format("%.3f", minKwScore));
			}
			return refined;
		}
		return candidates;
	}
}
