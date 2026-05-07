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

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.embedding.ModelNoiseProfile;

/**
 * Tunable parameters for the retrieval/ranking pipeline. Built once per
 * search by composing model-specific defaults with admin-customized
 * global properties; thereafter passed through the pipeline as an
 * immutable bundle. Use {@link #forModel(String)} for the model-tuned
 * defaults and {@link #buildEffective} to overlay global-property
 * overrides on top.
 */
class PipelineConfig {

	final double keywordWeight;
	final double scoreGapMultiplier;
	final double minScoreGap;
	final double gapValidationCosineThreshold;
	final double similarityRatio;
	final ModelNoiseProfile noiseProfile;
	/** Minimum z-score for tight-cluster bypass of the
	 * zero-keyword z-score gate. */
	final double floorRescueMinZScore;
	/** Minimum candidate count for the concept-name outlier
	 * gate to fire. Dual-encoder models (MedCPT) produce
	 * reliable concept-name z-scores even for small sets;
	 * single-encoder models (L6-v2) need larger sets. */
	final int conceptNameGateMinCandidates;
	/** Margin multiplier for the concept-name floor check.
	 * The floor check rejects single-concept results whose
	 * concept-name similarity is below
	 * absoluteSimilarityFloor * conceptFloorMargin. */
	final double conceptFloorMargin;
	/** Gap-saturation threshold: when gapCutoff/validCount
	 * exceeds this ratio and kwCount=0, embedding scores are
	 * undifferentiated and results are discarded. Set to 0.0
	 * to disable (e.g. for MedCPT where medical embeddings
	 * produce meaningful similarity even for adjacent topics). */
	final double gapSaturationThreshold;
	/** Concept-similarity expansion — Path A "very-high
	 * similarity" gate: bestSim of the closest chart concept
	 * to the query must reach this absolute threshold for the
	 * expansion to fire on a modest margin. */
	final double conceptExpansionVeryHighMinSim;
	/** Concept-similarity expansion — Path A margin: how far
	 * the best concept must beat the second-best to trigger
	 * the very-high path. */
	final double conceptExpansionVeryHighMinMargin;
	/** Concept-similarity expansion — Path B "moderate
	 * similarity" gate: bestSim required when the cluster of
	 * candidate concepts is well-separated. */
	final double conceptExpansionModerateMinSim;
	/** Concept-similarity expansion — Path B margin: required
	 * separation from the second-best concept on the moderate
	 * path. Larger than Path A's because moderate similarity
	 * alone is not enough — the winner must clearly stand out. */
	final double conceptExpansionModerateMinMargin;
	/** Concept-similarity expansion vocabulary-overlap bypass:
	 * when bestSim is at or above this threshold, the
	 * substring-overlap guard is skipped (the embedding match
	 * is strong enough to trust without lexical anchoring). */
	final double conceptExpansionVocabBypassThreshold;

	PipelineConfig(double keywordWeight, double scoreGapMultiplier,
			double minScoreGap, double gapValidationCosineThreshold,
			double similarityRatio) {
		this(keywordWeight, scoreGapMultiplier, minScoreGap,
				gapValidationCosineThreshold, similarityRatio,
				ModelNoiseProfile.conservativeDefault(),
				ChartSearchAiConstants.FLOOR_RESCUE_MIN_Z_SCORE,
				10, 0.85, 0.95,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	PipelineConfig(double keywordWeight, double scoreGapMultiplier,
			double minScoreGap, double gapValidationCosineThreshold,
			double similarityRatio,
			ModelNoiseProfile noiseProfile,
			double floorRescueMinZScore,
			int conceptNameGateMinCandidates) {
		this(keywordWeight, scoreGapMultiplier, minScoreGap,
				gapValidationCosineThreshold, similarityRatio,
				noiseProfile, floorRescueMinZScore,
				conceptNameGateMinCandidates, 0.85, 0.95,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	PipelineConfig(double keywordWeight, double scoreGapMultiplier,
			double minScoreGap, double gapValidationCosineThreshold,
			double similarityRatio,
			ModelNoiseProfile noiseProfile,
			double floorRescueMinZScore,
			int conceptNameGateMinCandidates,
			double conceptFloorMargin,
			double gapSaturationThreshold) {
		this(keywordWeight, scoreGapMultiplier, minScoreGap,
				gapValidationCosineThreshold, similarityRatio,
				noiseProfile, floorRescueMinZScore,
				conceptNameGateMinCandidates,
				conceptFloorMargin, gapSaturationThreshold,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	PipelineConfig(double keywordWeight, double scoreGapMultiplier,
			double minScoreGap, double gapValidationCosineThreshold,
			double similarityRatio,
			ModelNoiseProfile noiseProfile,
			double floorRescueMinZScore,
			int conceptNameGateMinCandidates,
			double conceptFloorMargin,
			double gapSaturationThreshold,
			double conceptExpansionVeryHighMinSim,
			double conceptExpansionVeryHighMinMargin,
			double conceptExpansionModerateMinSim,
			double conceptExpansionModerateMinMargin,
			double conceptExpansionVocabBypassThreshold) {
		this.keywordWeight = keywordWeight;
		this.scoreGapMultiplier = scoreGapMultiplier;
		this.minScoreGap = minScoreGap;
		this.gapValidationCosineThreshold = gapValidationCosineThreshold;
		this.similarityRatio = similarityRatio;
		this.noiseProfile = noiseProfile;
		this.floorRescueMinZScore = floorRescueMinZScore;
		this.conceptNameGateMinCandidates =
				conceptNameGateMinCandidates;
		this.conceptFloorMargin = conceptFloorMargin;
		this.gapSaturationThreshold = gapSaturationThreshold;
		this.conceptExpansionVeryHighMinSim =
				conceptExpansionVeryHighMinSim;
		this.conceptExpansionVeryHighMinMargin =
				conceptExpansionVeryHighMinMargin;
		this.conceptExpansionModerateMinSim =
				conceptExpansionModerateMinSim;
		this.conceptExpansionModerateMinMargin =
				conceptExpansionModerateMinMargin;
		this.conceptExpansionVocabBypassThreshold =
				conceptExpansionVocabBypassThreshold;
	}

	/**
	 * Returns a copy of this config with only the noise
	 * profile replaced. All other parameters (including
	 * model-specific values like conceptNameGateMinCandidates)
	 * are preserved. Use this instead of constructing a new
	 * PipelineConfig when updating the noise profile — it
	 * prevents config drift where new fields are forgotten
	 * in manual constructor calls.
	 */
	PipelineConfig withNoiseProfile(
			ModelNoiseProfile newNoiseProfile) {
		return new PipelineConfig(keywordWeight,
				scoreGapMultiplier, minScoreGap,
				gapValidationCosineThreshold,
				similarityRatio, newNoiseProfile,
				floorRescueMinZScore,
				conceptNameGateMinCandidates,
				conceptFloorMargin,
				gapSaturationThreshold,
				conceptExpansionVeryHighMinSim,
				conceptExpansionVeryHighMinMargin,
				conceptExpansionModerateMinSim,
				conceptExpansionModerateMinMargin,
				conceptExpansionVocabBypassThreshold);
	}

	/** Returns a config using all default constant values
	 * (tuned for all-MiniLM-L6-v2). */
	static PipelineConfig defaults() {
		return new PipelineConfig(
				ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
				ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
				ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP,
				ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
				ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO,
				ModelNoiseProfile.conservativeDefault(),
				ChartSearchAiConstants.FLOOR_RESCUE_MIN_Z_SCORE,
				10, 0.85, 0.95,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	/**
	 * Returns a model-specific config based on the model path.
	 * Falls back to {@link #defaults()} for unrecognized models.
	 *
	 * @param modelPath the embedding model file path or directory
	 *        name (e.g. "pubmedbert-onnx", "/path/to/pubmedbert-onnx/model.onnx")
	 */
	static PipelineConfig forModel(String modelPath) {
		if (modelPath == null) {
			return defaults();
		}
		String lower = modelPath.toLowerCase();
		if (lower.contains("medcpt")) {
			return medcptDefaults();
		}
		if (lower.contains("medembed")) {
			return medembedDefaults();
		}
		if (lower.contains("pubmedbert")) {
			return pubmedbertDefaults();
		}
		return defaults();
	}

	/**
	 * Pipeline defaults for MedEmbed (medical IR fine-tune of
	 * all-MiniLM-L12-v2). MedEmbed produces higher absolute cosine
	 * similarities with a compressed score range — noise mean ~0.59
	 * vs L6-v2's ~0.26. Parameters are adjusted for this tighter
	 * distribution:
	 * <ul>
	 * <li>minScoreGap lowered: meaningful cluster gaps are 0.01–0.05
	 *     vs L6-v2's 0.05–0.15</li>
	 * <li>similarityRatio raised: noise mean is close to relevant
	 *     scores, so the ratio floor must be tighter</li>
	 * <li>gapValidationCosineThreshold raised: baseline cosines are
	 *     higher, so gap validation needs a higher bar</li>
	 * </ul>
	 */
	static PipelineConfig medembedDefaults() {
		// MedEmbed produces compressed score distributions (noise
		// mean ~0.59 vs L6-v2's ~0.26). Two parameters need to
		// differ:
		// - minScoreGap: meaningful cluster gaps are 0.01-0.05
		//   vs L6-v2's 0.05-0.15
		// - similarityRatio: noise scores (0.55-0.60) are close
		//   to relevant scores (0.64-0.65), so the ratio floor
		//   must be tighter to separate them. At 0.80 (L6-v2
		//   default), floor = 0.52 which includes all noise.
		//   At 0.95, floor = 0.61 which separates signal from
		//   noise while keeping related records.
		return new PipelineConfig(
				ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
				ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
				0.03,   // minScoreGap — lower for compressed range
				ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
				0.95,   // similarityRatio — tighter for compressed range
				ModelNoiseProfile.conservativeDefault(),
				1.5,    // floorRescueMinZScore — lower for compressed range
				10,     // conceptNameGateMinCandidates — same as L6-v2
				0.85, 0.95,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	/**
	 * Pipeline defaults for pubmedbert-onnx / pubmedbert-matryoshka.
	 * Currently identical to L6-v2 defaults — the data-derived
	 * {@link ModelNoiseProfile} adapts thresholds automatically.
	 */
	static PipelineConfig pubmedbertDefaults() {
		return defaults();
	}

	/**
	 * Pipeline defaults for MedCPT (dual-encoder, PubMedBERT-based).
	 * MedCPT uses a medical tokenizer that correctly handles
	 * abbreviations (BMI, STD, COPD) as single tokens. Its score
	 * distribution characteristics need to be profiled and tuned.
	 * Starting with L6-v2 defaults; the dynamic
	 * {@link ModelNoiseProfile} adapts automatically.
	 */
	static PipelineConfig medcptDefaults() {
		// MedCPT (dual-encoder, PubMedBERT-based) produces highly
		// compressed score distributions (noise mean ~0.67). Signal
		// is only 3-5% above noise, requiring very tight parameters:
		// - minScoreGap 0.01: gaps between clusters are 0.005-0.02
		// - similarityRatio 0.98: floor must be very close to top
		//   score to exclude noise at 0.669 when signal is at 0.700
		// - floorRescueMinZScore 1.0: z-scores are compressed too
		// - conceptNameGateMinCandidates 2: MedCPT's dual-encoder
		//   query encoder produces reliable concept-name z-scores
		//   even for small result sets, so the outlier gate can
		//   fire at 2+ candidates (vs L6-v2's 10+).
		// - gapSaturationThreshold 0.0: disabled — MedCPT's medical
		//   embeddings produce meaningful similarity for clinically
		//   adjacent topics (e.g. COPD → Respiratory rate), so a
		//   saturated gap doesn't imply no signal.
		return new PipelineConfig(
				ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
				ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
				0.01,
				ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
				0.98,
				ModelNoiseProfile.conservativeDefault(),
				1.0,
				ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS,
				0.85, 0.0,
				0.90, 0.07, 0.80, 0.14, 0.92);
	}

	/**
	 * Builds the per-search effective config by overlaying any
	 * explicitly-customized global property values on top of the
	 * model-specific defaults in {@code baseConfig}. Every model-specific
	 * field of {@code baseConfig} that is not exposed as a global property
	 * (e.g. {@code conceptFloorMargin}, {@code gapSaturationThreshold})
	 * is passed through unchanged, so that compressed-distribution
	 * models (e.g. MedCPT) don't silently fall back to L6-v2 floor values.
	 */
	static PipelineConfig buildEffective(PipelineConfig baseConfig,
			double gpKeywordWeight, double gpScoreGapMultiplier,
			double gpMinScoreGap, double gpGapValidationCosineThreshold,
			double gpSimilarityRatio, ModelNoiseProfile cachedProfile) {
		return new PipelineConfig(
				overrideIfCustomized(gpKeywordWeight,
						ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT,
						baseConfig.keywordWeight),
				overrideIfCustomized(gpScoreGapMultiplier,
						ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER,
						baseConfig.scoreGapMultiplier),
				overrideIfCustomized(gpMinScoreGap,
						ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP,
						baseConfig.minScoreGap),
				overrideIfCustomized(gpGapValidationCosineThreshold,
						ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD,
						baseConfig.gapValidationCosineThreshold),
				overrideIfCustomized(gpSimilarityRatio,
						ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO,
						baseConfig.similarityRatio),
				cachedProfile != null ? cachedProfile
						: ModelNoiseProfile.conservativeDefault(),
				baseConfig.floorRescueMinZScore,
				baseConfig.conceptNameGateMinCandidates,
				baseConfig.conceptFloorMargin,
				baseConfig.gapSaturationThreshold,
				baseConfig.conceptExpansionVeryHighMinSim,
				baseConfig.conceptExpansionVeryHighMinMargin,
				baseConfig.conceptExpansionModerateMinSim,
				baseConfig.conceptExpansionModerateMinMargin,
				baseConfig.conceptExpansionVocabBypassThreshold);
	}

	/**
	 * Returns the model-specific default unless the admin has explicitly
	 * customized the global property (i.e., the GP value differs from the
	 * L6-v2 default that ships as the initial value).
	 */
	private static double overrideIfCustomized(double gpValue,
			double l6v2Default, double modelDefault) {
		if (Math.abs(gpValue - l6v2Default) > 1e-9) {
			return gpValue;
		}
		return modelDefault;
	}
}
