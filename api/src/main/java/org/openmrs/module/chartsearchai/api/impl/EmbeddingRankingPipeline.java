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
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.openmrs.module.chartsearchai.api.impl.LlmInferenceService.FindSimilarResult;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.util.ConceptNameUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The retrieval/ranking pipeline. Owns everything reachable from the
 * {@link #findSimilar} / {@link #findSimilarWithProfile} entry points that
 * isn't covered by {@link SimilarityAndScoringEngine} (pure scoring),
 * {@link QueryPreprocessor} (query normalization),
 * {@link ConceptRescueAndFilter} (concept-name floor / category-hint rescue),
 * or {@link ChartBuildingStrategy} (chart assembly). Internally groups into
 * coherence/clustering primitives, keyword/concept matchers, candidate
 * gates, the filter-pipeline orchestration with refinement paths, and the
 * static query entry points consumed by {@link LlmInferenceService}.
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
		Boolean floorResult = RankingPipelineGates.applyFloorGate(scored, maxSemanticScore,
				maxBaseScore, queryTermCount, keywordMatchCount, config);
		if (floorResult == null) {
			return Collections.emptyList();
		}
		boolean belowFloorRescued = floorResult;

		// Slim-margin gate.
		if (!RankingPipelineGates.applySlimMarginGate(scored, maxSemanticScore,
				queryTermCount, keywordMatchCount, belowFloorRescued,
				config)) {
			return Collections.emptyList();
		}

		// Initial z-score gate.
		double[] zScoreState = RankingPipelineGates.applyInitialZScoreGate(scored,
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
		candidates = RankingPipelineGates.applyOutlierRemovalPhase1(candidates, queryTerms,
				partialKwValidated);
		// Phase 2: Zero-keyword validation.
		candidates = RankingPipelineGates.applyZeroKeywordValidationPhase2(candidates,
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

	/**
	 * Static query pipeline that runs the exact same logic as
	 * {@link LlmInferenceService#findSimilar(org.openmrs.Patient, String)}
	 * but without DAO or Spring dependencies. Accepts pre-loaded embeddings
	 * and an embedding provider, making it directly callable from
	 * integration tests with zero simulation.
	 */
	static FindSimilarResult findSimilar(List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		return findSimilarWithProfile(allEmbeddings, provider, question,
				queryPrefix, config);
	}

	static FindSimilarResult findSimilarWithProfile(
			List<ChartEmbedding> allEmbeddings,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		String normalizedQuery = QueryPreprocessor.stripQueryStopwords(question);
		String[] queryTerms = QueryPreprocessor.extractQueryTerms(normalizedQuery);
		String embeddingQuery = QueryPreprocessor.buildEmbeddingQuery(normalizedQuery);
		float[] queryVector = provider.embedQuery(queryPrefix + embeddingQuery);

		// Identify "type indicator" query terms — terms that appear in any
		// structural embedding prefix. For these terms, only matches in the
		// prefix portion of a record's text count — body-text matches are
		// demoted as coincidental narrative mentions. Content terms that
		// don't appear in any prefix still match freely in the body.
		Set<String> typeIndicatorTerms = new HashSet<String>();
		if (queryTerms.length > 0) {
			for (String prefix
					: ChartSearchAiUtils.getAllEmbeddingPrefixes()) {
				String lowerPrefix = prefix.toLowerCase();
				String[] prefixWords = lowerPrefix.split("\\s+");
				for (String term : queryTerms) {
					if (typeIndicatorTerms.contains(term)) {
						continue;
					}
					if (SimilarityAndScoringEngine.termMatchesText(term, lowerPrefix, prefixWords)) {
						typeIndicatorTerms.add(term);
					}
				}
			}
		}

		// IDF-based term filtering: terms matching > 20% of records are
		// too generic and would flood keyword scoring with noise.
		String[] kwTerms = queryTerms;
		if (queryTerms.length > 2) {
			int totalDocs = 0;
			int[] docFreq = new int[queryTerms.length];
			for (int i = 0; i < allEmbeddings.size(); i++) {
				ChartEmbedding ce = allEmbeddings.get(i);
				if (ce.getEmbeddingVector() == null) continue;
				totalDocs++;
				String text = (ce.getTextContent() != null
						? ce.getTextContent() : "").toLowerCase();
				for (int t = 0; t < queryTerms.length; t++) {
					if (text.contains(queryTerms[t])) {
						docFreq[t]++;
					}
				}
			}
			if (totalDocs > 0) {
				double threshold = totalDocs * 0.2;
				List<String> filtered = new ArrayList<String>();
				int genericCount = 0;
				for (int t = 0; t < queryTerms.length; t++) {
					if (docFreq[t] <= threshold) {
						filtered.add(queryTerms[t]);
					} else {
						genericCount++;
					}
				}
				// Only activate when the query is dominated by generic
				// terms (> 50% filtered).
				if (!filtered.isEmpty()
						&& genericCount > queryTerms.length / 2) {
					kwTerms = filtered.toArray(
							new String[filtered.size()]);
					log.warn("IDF filter: {} -> {} terms (dropped "
							+ "generic terms from {})",
							queryTerms.length, kwTerms.length,
							Arrays.toString(queryTerms));
				}
			}
		}

		double[] semanticScores = new double[allEmbeddings.size()];
		double[] keywordScores = new double[allEmbeddings.size()];
		ChartEmbedding[] embeddings = new ChartEmbedding[allEmbeddings.size()];
		int validCount = 0;
		// First pass: collect valid records and compute semantic scores.
		// Keyword scoring is deferred until after concept-similarity
		// expansion so the (possibly replaced) kwTerms are scored.
		for (int i = 0; i < allEmbeddings.size(); i++) {
			ChartEmbedding ce = allEmbeddings.get(i);
			float[] vector = ce.getEmbeddingVector();
			if (vector == null || vector.length != queryVector.length) {
				continue;
			}
			embeddings[validCount] = ce;
			semanticScores[validCount] = ChartSearchAiUtils.cosineSimilarity(queryVector, vector);
			validCount++;
		}
		if (validCount < embeddings.length) {
			embeddings = Arrays.copyOf(embeddings, validCount);
		}
		// Concept-similarity expansion: when the query phrasing isn't in
		// any record but a concept name is semantically very close to the
		// query, replace kwTerms with the concept's tokens.
		String[] expandedKwTerms = expandKwTermsViaConceptSimilarity(
				kwTerms, queryVector, embeddings, provider, config);
		if (expandedKwTerms != kwTerms) {
			kwTerms = expandedKwTerms;
			// Type-indicator restriction was computed from the original
			// query terms. After replacement clear it so keyword matching
			// is unconstrained on the replacement tokens.
			typeIndicatorTerms = new HashSet<String>();
		}
		// Second pass: compute keyword scores with the (possibly
		// expanded) kwTerms.
		for (int i = 0; i < embeddings.length; i++) {
			ChartEmbedding ce = embeddings[i];
			// Strip synonym parentheticals before keyword scoring so that
			// "(syn. ...)" doesn't cause false matches.
			String body = ConceptNameUtil.stripSynonyms(ce.getTextContent());
			String keywordText = ChartSearchAiUtils.buildPrefixedText(
					ce.getResourceType(), body);
			if (typeIndicatorTerms.isEmpty()) {
				keywordScores[i] = SimilarityAndScoringEngine.computeKeywordScore(
						kwTerms, keywordText);
			} else {
				keywordScores[i] = SimilarityAndScoringEngine.computeKeywordScoreRestricted(
						kwTerms, keywordText, body,
						typeIndicatorTerms);
			}
		}
		if (validCount < semanticScores.length) {
			semanticScores = Arrays.copyOf(semanticScores, validCount);
			keywordScores = Arrays.copyOf(keywordScores, validCount);
		}

		// Compute noise profile using hint-stripped re-embeddings for
		// cross-concept similarity. Reuse pre-computed profile from config
		// when available.
		ModelNoiseProfile noiseProfile = config.noiseProfile != null
				&& !config.noiseProfile.isConservativeDefault()
				? config.noiseProfile
				: ModelNoiseProfile.compute(embeddings, provider);
		log.warn("NoiseProfile: Q1={} median={} mean={} P95={} "
				+ "intraMean={} floor={}",
				String.format("%.4f", noiseProfile.noiseQ1),
				String.format("%.4f", noiseProfile.noiseMedian),
				String.format("%.4f", noiseProfile.noiseMean),
				String.format("%.4f", noiseProfile.noiseP95),
				String.format("%.4f", noiseProfile.intraConceptMean),
				String.format("%.4f", noiseProfile.absoluteSimilarityFloor()));
		PipelineConfig profiledConfig =
				config.withNoiseProfile(noiseProfile);

		int[] gapCutoffOut = new int[1];
		List<ChartEmbedding> pipelineResult = filterPipeline(
				semanticScores, keywordScores, embeddings,
				queryTerms, queryTerms.length, profiledConfig,
				gapCutoffOut);
		int gapCutoff = gapCutoffOut[0];

		int keywordMatchCount = 0;
		for (int i = 0; i < validCount; i++) {
			if (keywordScores[i] > 0) {
				keywordMatchCount++;
			}
		}

		// Query truncation retry: when the pipeline returns empty for a
		// long query (>3 terms), retry with the first 2 core terms and a
		// fresh embedding.
		if ((pipelineResult == null || pipelineResult.isEmpty())
				&& queryTerms.length > 3) {
			String[] coreTerms = Arrays.copyOf(
					queryTerms, Math.min(2, queryTerms.length));
			String coreQuery = String.join(" ", coreTerms);
			float[] coreVector = provider.embedQuery(
					queryPrefix + coreQuery);

			double[] coreSemantic = new double[validCount];
			double[] coreKw = new double[validCount];
			for (int i = 0; i < validCount; i++) {
				coreSemantic[i] = ChartSearchAiUtils.cosineSimilarity(
						coreVector, embeddings[i].getEmbeddingVector());
				String body = ConceptNameUtil.stripSynonyms(
						embeddings[i].getTextContent());
				String kwText = ChartSearchAiUtils.buildPrefixedText(
						embeddings[i].getResourceType(), body);
				coreKw[i] = SimilarityAndScoringEngine.computeKeywordScore(coreTerms, kwText);
			}

			int[] retryGapCutoff = new int[1];
			pipelineResult = filterPipeline(coreSemantic, coreKw,
					embeddings, coreTerms, coreTerms.length,
					profiledConfig, retryGapCutoff);
			if (pipelineResult != null && !pipelineResult.isEmpty()) {
				gapCutoff = retryGapCutoff[0];
				log.warn("Query truncation retry: '{}' -> '{}', "
						+ "found {} results",
						String.join(" ", queryTerms), coreQuery,
						pipelineResult.size());
				keywordMatchCount = 0;
				for (int i = 0; i < validCount; i++) {
					if (coreKw[i] > 0) keywordMatchCount++;
					semanticScores[i] = coreSemantic[i];
				}
			}
		}

		// Dilution rescue: long queries (>3 terms) may dilute the
		// embedding. Embed the first 2 terms as a focused query and check
		// if the improvement in max similarity exceeds 1 std of the
		// focused score distribution — a statistically significant
		// signal gain that indicates the full query lost a concept to
		// dilution.
		if (pipelineResult != null && !pipelineResult.isEmpty()
				&& queryTerms.length > 3) {
			double maxFullSem = 0;
			for (int i = 0; i < validCount; i++) {
				if (semanticScores[i] > maxFullSem) {
					maxFullSem = semanticScores[i];
				}
			}
			String[] coreTerms2 = Arrays.copyOf(
					queryTerms, Math.min(2, queryTerms.length));
			String coreQuery2 = String.join(" ", coreTerms2);
			float[] coreVector2 = provider.embedQuery(
					queryPrefix + coreQuery2);
			double[] coreSem2 = new double[validCount];
			double maxCoreSem = 0;
			double sumCoreSem = 0;
			double sumCoreSemSq = 0;
			for (int i = 0; i < validCount; i++) {
				double sim = ChartSearchAiUtils.cosineSimilarity(coreVector2,
						embeddings[i].getEmbeddingVector());
				coreSem2[i] = sim;
				if (sim > maxCoreSem) maxCoreSem = sim;
				sumCoreSem += sim;
				sumCoreSemSq += sim * sim;
			}
			double meanCore = validCount > 0
					? sumCoreSem / validCount : 0;
			double stdCore = validCount > 1
					? Math.sqrt(Math.max(0,
							sumCoreSemSq / validCount
									- meanCore * meanCore))
					: 0;
			if (maxCoreSem - maxFullSem > stdCore) {
				double[] coreKw2 = new double[validCount];
				for (int i = 0; i < validCount; i++) {
					String body = ConceptNameUtil.stripSynonyms(
							embeddings[i].getTextContent());
					String kwText =
							ChartSearchAiUtils.buildPrefixedText(
									embeddings[i].getResourceType(),
									body);
					coreKw2[i] = SimilarityAndScoringEngine.computeKeywordScore(
							coreTerms2, kwText);
				}
				int[] mergeGapCutoff = new int[1];
				List<ChartEmbedding> mergeResult = filterPipeline(
						coreSem2, coreKw2, embeddings, coreTerms2,
						coreTerms2.length, profiledConfig,
						mergeGapCutoff);
				if (mergeResult != null
						&& !mergeResult.isEmpty()
						&& mergeResult.size()
								<= ChartSearchAiConstants
										.ADAPTIVE_MIN_RECORDS) {
					Set<ChartEmbedding> existing =
							Collections.newSetFromMap(
									new IdentityHashMap<
											ChartEmbedding,
											Boolean>());
					existing.addAll(pipelineResult);
					boolean anyOverlap = false;
					for (ChartEmbedding ce : mergeResult) {
						if (existing.contains(ce)) {
							anyOverlap = true;
							break;
						}
					}
					if (!anyOverlap) {
						List<ChartEmbedding> merged =
								new ArrayList<>(pipelineResult);
						merged.addAll(mergeResult);
						log.warn("Dilution rescue: merged {}"
								+ " focused results into {} "
								+ "pipeline results for '{}'",
								mergeResult.size(),
								pipelineResult.size(),
								coreQuery2);
						pipelineResult = merged;
					}
				}
			}
		}

		// When type indicators are present, remove straggler records
		// whose structural prefix doesn't match any type indicator term.
		// Only fires when >75% of pipeline results already match — this
		// targets the case where a few wrong-type records leak through
		// on semantic similarity alone.
		int typeMatchCount = -1;
		if (!typeIndicatorTerms.isEmpty()
				&& pipelineResult != null && !pipelineResult.isEmpty()) {
			typeMatchCount = 0;
			for (ChartEmbedding ce : pipelineResult) {
				if (matchesTypeIndicator(ce, typeIndicatorTerms)) {
					typeMatchCount++;
				}
			}
			if (typeMatchCount > 0
					&& typeMatchCount > pipelineResult.size() * 2 / 3) {
				int beforeFilter = pipelineResult.size();
				List<ChartEmbedding> filtered = new ArrayList<>();
				for (ChartEmbedding ce : pipelineResult) {
					if (matchesTypeIndicator(ce, typeIndicatorTerms)) {
						filtered.add(ce);
					}
				}
				pipelineResult = filtered;
				if (pipelineResult.size() != beforeFilter) {
					log.warn("Type-indicator filter {}: {} -> {}",
							typeIndicatorTerms, beforeFilter,
							pipelineResult.size());
				}
			}
		}

		// Concept-name re-ranking for zero-keyword queries.
		boolean noTypeMatch = !typeIndicatorTerms.isEmpty()
				&& typeMatchCount == 0;
		if (pipelineResult != null && !pipelineResult.isEmpty()
				&& keywordMatchCount == 0 && queryTerms.length > 0) {
			List<ChartEmbedding> preGateCandidates = pipelineResult;
			int beforeRerank = pipelineResult.size();
			pipelineResult = ConceptRescueAndFilter.rerankByConceptName(pipelineResult,
					embeddings, validCount, queryVector, provider,
					noiseProfile, profiledConfig, noTypeMatch);
			if (pipelineResult.size() != beforeRerank) {
				StringBuilder kept = new StringBuilder();
				for (ChartEmbedding ce : pipelineResult) {
					if (kept.length() > 0) kept.append(", ");
					kept.append(ce.getResourceType()).append(":")
							.append(ce.getResourceId());
				}
				log.warn("Concept-name rerank {}: {} -> {} [{}]",
						Arrays.toString(queryTerms),
						beforeRerank, pipelineResult.size(), kept);
			}

			// Type-indicator rescue: when the gate rejects all
			// candidates but type indicators exist, rescue
			// candidates whose record type matches the indicated
			// type from the pre-gate set.
			if (pipelineResult.isEmpty()
					&& !typeIndicatorTerms.isEmpty()) {
				List<ChartEmbedding> rescued = new ArrayList<>();
				for (ChartEmbedding ce : preGateCandidates) {
					if (matchesTypeIndicator(ce,
							typeIndicatorTerms)) {
						rescued.add(ce);
					}
				}
				if (!rescued.isEmpty()) {
					pipelineResult = rescued;
					log.warn("Type-indicator rescue after "
							+ "concept-name gate: 0 -> {} for {}",
							rescued.size(), typeIndicatorTerms);
				}
			}

		}

		// No-type-match check: when type indicators are present but ALL
		// pipeline results are wrong type AND no records of the indicated
		// type exist in the dataset, the patient has no records of the
		// queried type.
		if (noTypeMatch
				&& pipelineResult != null
				&& !pipelineResult.isEmpty()) {
			boolean bodyMentionsType = false;
			for (ChartEmbedding ce : pipelineResult) {
				String body = (ce.getTextContent() != null
						? ce.getTextContent() : "").toLowerCase();
				String[] bodyWords = body.split("\\s+");
				for (String term : typeIndicatorTerms) {
					if (SimilarityAndScoringEngine.termMatchesText(term, body, bodyWords)) {
						bodyMentionsType = true;
						break;
					}
				}
				if (bodyMentionsType) break;
			}
			if (!bodyMentionsType) {
				List<ChartEmbedding> typeMatching =
						new ArrayList<>();
				for (int i = 0; i < validCount; i++) {
					if (matchesTypeIndicator(embeddings[i],
							typeIndicatorTerms)) {
						typeMatching.add(embeddings[i]);
					}
				}
				if (typeMatching.isEmpty()) {
					log.warn("No-type-match empty: {} wrong-type "
							+ "results, 0 type-matching in "
							+ "dataset — returning empty for {}",
							pipelineResult.size(),
							typeIndicatorTerms);
					pipelineResult = Collections.emptyList();
				} else if (typeMatching.size()
						<= ChartSearchAiConstants
								.ADAPTIVE_MIN_RECORDS) {
					log.warn("No-type-match replacement: "
							+ "replacing {} wrong-type with "
							+ "{} type-matching records for {}",
							pipelineResult.size(),
							typeMatching.size(),
							typeIndicatorTerms);
					pipelineResult = typeMatching;
				}
			}
		}

		// Type-indicator full-scan rescue: when the pipeline returns
		// empty but type indicators exist, scan ALL embeddings for
		// records matching the indicated type.
		if ((pipelineResult == null || pipelineResult.isEmpty())
				&& !typeIndicatorTerms.isEmpty()) {
			List<ChartEmbedding> rescued = new ArrayList<>();
			for (int i = 0; i < validCount; i++) {
				if (matchesTypeIndicator(embeddings[i],
						typeIndicatorTerms)) {
					rescued.add(embeddings[i]);
				}
			}
			if (!rescued.isEmpty()
					&& rescued.size()
					<= ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS) {
				pipelineResult = rescued;
				log.warn("Type-indicator full-scan rescue: "
						+ "found {} records for {}",
						rescued.size(), typeIndicatorTerms);
			}
		}

		// Concept-name floor check: when pipeline results have no
		// keyword anchoring, the results rely entirely on embedding
		// similarity. If all results share a single concept whose name
		// is below the noise floor, return empty.
		if (!pipelineResult.isEmpty()
				&& queryTerms.length > 0) {
			boolean pipelineHasKwAnchoring = false;
			if (keywordMatchCount > 0) {
				Set<ChartEmbedding> kwFloorSet =
						Collections.newSetFromMap(
								new IdentityHashMap<
										ChartEmbedding, Boolean>());
				kwFloorSet.addAll(pipelineResult);
				for (int i = 0; i < validCount; i++) {
					if (kwFloorSet.contains(embeddings[i])
							&& keywordScores[i] > 0) {
						pipelineHasKwAnchoring = true;
						break;
					}
				}
			}
			if (!pipelineHasKwAnchoring) {
				Set<String> seenConcepts = new HashSet<>();
				for (ChartEmbedding ce : pipelineResult) {
					String name = ConceptNameUtil.extractConceptName(
							ce.getTextContent());
					if (name != null) {
						seenConcepts.add(name);
					}
				}
				if (seenConcepts.size() == 1) {
					String conceptName =
							seenConcepts.iterator().next();
					float[] nameVec =
							provider.embedQuery(conceptName);
					double conceptSim = ChartSearchAiUtils.cosineSimilarity(
							queryVector, nameVec);
					double floorWithMargin = noiseProfile
							.absoluteSimilarityFloor()
							* profiledConfig.conceptFloorMargin;
					if (conceptSim < floorWithMargin) {
						log.warn("Concept-name floor check: "
								+ "sim({})={} < floor*{}={}, "
								+ "returning empty for {}",
								conceptName,
								String.format("%.4f", conceptSim),
								String.format("%.2f",
										profiledConfig.conceptFloorMargin),
								String.format("%.4f",
										floorWithMargin),
								Arrays.toString(
										queryTerms));
						pipelineResult = Collections.emptyList();
					}
				}
			}
		}

		// Gap-saturation check: when kwCount=0 and the gap analysis
		// found no meaningful gap, the embedding scores are
		// undifferentiated — every record looks equally relevant.
		// Catches queries about topics not in the patient's chart.
		if (keywordMatchCount == 0
				&& !pipelineResult.isEmpty()
				&& validCount > 0
				&& profiledConfig.gapSaturationThreshold > 0) {
			double gapRatio = (double) gapCutoff / validCount;
			if (gapRatio > profiledConfig.gapSaturationThreshold) {
				log.warn("Gap-saturation check: "
						+ "gapCutoff={}/{} ({}) — "
						+ "returning empty for {}",
						gapCutoff, validCount,
						String.format("%.0f%%",
								gapRatio * 100),
						Arrays.toString(
								queryTerms));
				pipelineResult = Collections.emptyList();
			}
		}

		return new FindSimilarResult(pipelineResult, noiseProfile,
				keywordMatchCount);
	}

	/**
	 * Composed retrieval pipeline: runs {@link #findSimilar}, then
	 * {@link LlmInferenceService#postRetrievalPipeline} — the same
	 * post-retrieval sequence that production's {@code filterAndSerialize}
	 * runs before handing records to {@code chartSerializer.serialize}.
	 */
	static List<SerializedRecord> findRelevantRecords(
			List<ChartEmbedding> allEmbeddings,
			List<SerializedRecord> allRecords,
			EmbeddingProvider provider, String question,
			String queryPrefix, PipelineConfig config) {
		FindSimilarResult fsResult = findSimilar(allEmbeddings, provider,
				question, queryPrefix, config);
		List<ChartEmbedding> similar = fsResult.records;
		if (similar == null) {
			return null;
		}
		if (similar.isEmpty()) {
			return Collections.emptyList();
		}
		Set<String> relevantKeys = new HashSet<String>();
		for (ChartEmbedding ce : similar) {
			relevantKeys.add(ChartSearchAiUtils.resourceKey(
					ce.getResourceType(), ce.getResourceId()));
		}

		log.warn("findRelevantRecords: {} embeddings -> {} keys",
				similar.size(), relevantKeys.size());

		return ConceptRescueAndFilter.postRetrievalPipeline(allRecords, relevantKeys, question,
				similar.size(), fsResult.keywordMatchCount,
				provider,
				ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX);
	}
}
