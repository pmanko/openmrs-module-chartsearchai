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
}
