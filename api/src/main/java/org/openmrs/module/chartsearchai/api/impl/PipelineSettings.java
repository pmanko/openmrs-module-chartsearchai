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

import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Resolves {@link PipelineConfig} field values from OpenMRS global
 * properties, with type-specific validation and per-field defaults.
 * Centralizes the read-validate-fallback pattern that
 * {@link ChartBuildingStrategy} consults at every {@code PipelineConfig}
 * construction site.
 */
final class PipelineSettings {

	private PipelineSettings() {
	}

	private static final Logger log = LoggerFactory.getLogger(PipelineSettings.class);

	static boolean usePreFilter() {
		String mode = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER, "false");
		return !"false".equalsIgnoreCase(mode.trim());
	}

	static int getTopK() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_TOP_K);
		if (value != null && !value.trim().isEmpty()) {
			try {
				int parsed = Integer.parseInt(value.trim());
				if (parsed > 0) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid topK value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_RETRIEVAL_TOP_K;
	}

	static double getSimilarityRatio() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_SIMILARITY_RATIO);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed > 0 && parsed < 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid similarityRatio value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_SIMILARITY_RATIO;
	}

	static double getScoreGapMultiplier() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_SCORE_GAP_MULTIPLIER);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed > 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid scoreGapMultiplier value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_SCORE_GAP_MULTIPLIER;
	}

	static double getMinScoreGap() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_MIN_SCORE_GAP);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid minScoreGap value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_MIN_SCORE_GAP;
	}

	static double getGapValidationCosineThreshold() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(
						ChartSearchAiConstants.GP_EMBEDDING_GAP_VALIDATION_COSINE_THRESHOLD);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid gapValidationCosineThreshold value '{}', "
						+ "using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_GAP_VALIDATION_COSINE_THRESHOLD;
	}

	static double getKeywordWeight() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_KEYWORD_WEIGHT);
		if (value != null && !value.trim().isEmpty()) {
			try {
				double parsed = Double.parseDouble(value.trim());
				if (parsed >= 0 && parsed <= 1) {
					return parsed;
				}
			}
			catch (NumberFormatException e) {
				log.warn("Invalid keywordWeight value '{}', using default", value);
			}
		}
		return ChartSearchAiConstants.DEFAULT_KEYWORD_WEIGHT;
	}

	static String getQueryPrefix() {
		String value = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_QUERY_PREFIX);
		if (value != null && !value.trim().isEmpty()) {
			return value;
		}
		return ChartSearchAiConstants.DEFAULT_QUERY_EMBEDDING_PREFIX;
	}
}
