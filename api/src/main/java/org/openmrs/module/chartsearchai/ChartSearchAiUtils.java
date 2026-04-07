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

import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.ADAPTIVE_MIN_RECORDS;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.COHERENCE_ADAPTIVE_GAP_RATIO;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.COHERENCE_GAP_MULTIPLIER;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.COHERENCE_REFERENCE_N;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.COHERENCE_SAME_TOPIC_FLOOR;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.PIPELINE_HYBRID;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.PIPELINE_LUCENE;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_CONDITION;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_OBS;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_ORDER;
import static org.openmrs.module.chartsearchai.ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM;

import java.io.File;
import java.io.IOException;

import org.openmrs.util.OpenmrsUtil;

public class ChartSearchAiUtils {

	/**
	 * Builds a composite key from a resource type and resource ID.
	 * This is the single canonical format for resource keys used across
	 * retrieval pipelines, filter methods, and result sets.
	 *
	 * @param resourceType the resource type constant
	 * @param resourceId the resource ID
	 * @return a key in the format "resourceType:resourceId"
	 */
	public static String resourceKey(String resourceType, int resourceId) {
		return resourceType + ":" + resourceId;
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
		return getEmbeddingPrefix(resourceType, text) + text;
	}

	/**
	 * Resolves a model path relative to the OpenMRS application data directory.
	 * Rejects paths containing ".." to prevent path traversal and verifies the
	 * resolved path stays within the application data directory.
	 *
	 * @param relativePath the relative path from the global property (e.g. "chartsearchai/model.gguf")
	 * @param globalPropertyName the global property name, used in error messages
	 * @return the absolute path to the model file
	 * @throws IllegalStateException if the path is invalid, traverses outside the data directory,
	 *         or the file does not exist
	 */
	public static String resolveModelPath(String relativePath, String globalPropertyName) {
		if (relativePath == null || relativePath.trim().isEmpty()) {
			throw new IllegalStateException(
					"Model path is not configured: " + globalPropertyName);
		}
		if (relativePath.contains("..")) {
			throw new IllegalStateException(
					"Model path must not contain '..': " + globalPropertyName);
		}

		File appDataDir = new File(OpenmrsUtil.getApplicationDataDirectory());
		File modelFile = new File(appDataDir, relativePath);

		try {
			String canonicalPath = modelFile.getCanonicalPath();
			String canonicalDataDir = appDataDir.getCanonicalPath();
			if (!canonicalPath.startsWith(canonicalDataDir + File.separator)) {
				throw new IllegalStateException(
						"Model path must resolve to within the OpenMRS application data directory: "
								+ globalPropertyName);
			}
		}
		catch (IOException e) {
			throw new IllegalStateException(
					"Failed to resolve model path for " + globalPropertyName, e);
		}

		if (!modelFile.exists()) {
			throw new IllegalStateException(
					"Model file not found: " + modelFile.getAbsolutePath()
							+ ". Set the correct relative path in " + globalPropertyName);
		}

		return modelFile.getAbsolutePath();
	}

	/**
	 * Computes cosine similarity between two embedding vectors.
	 *
	 * @param a first embedding vector
	 * @param b second embedding vector
	 * @return cosine similarity in [-1, 1], or 0 if either vector is
	 *         null, empty, or the vectors differ in length
	 */
	public static double cosineSimilarity(float[] a, float[] b) {
		if (a == null || b == null || a.length == 0 || b.length == 0
				|| a.length != b.length) {
			return 0;
		}
		double dot = 0, normA = 0, normB = 0;
		for (int i = 0; i < a.length; i++) {
			dot += a[i] * b[i];
			normA += a[i] * a[i];
			normB += b[i] * b[i];
		}
		double denom = Math.sqrt(normA) * Math.sqrt(normB);
		return denom == 0 ? 0 : dot / denom;
	}

	/**
	 * Computes a dynamic similarity floor using z-score gating. Returns
	 * the higher of {@code absoluteFloor} and {@code mean + zThreshold * stddev}.
	 *
	 * @param scores all scores in the distribution
	 * @param absoluteFloor minimum floor regardless of distribution
	 * @param zThreshold number of standard deviations above mean
	 * @return the effective minimum score threshold
	 */
	public static double zScoreFloor(double[] scores, double absoluteFloor,
			double zThreshold) {
		if (scores.length == 0) {
			return absoluteFloor;
		}
		double sum = 0;
		for (double s : scores) {
			sum += s;
		}
		double mean = sum / scores.length;
		double variance = 0;
		for (double s : scores) {
			variance += (s - mean) * (s - mean);
		}
		double stddev = Math.sqrt(variance / scores.length);
		double zFloor = mean + zThreshold * stddev;
		return Math.max(absoluteFloor, zFloor);
	}

	/**
	 * Filters a set of embedding vectors by pairwise coherence, removing
	 * topic outliers. Returns a boolean array where {@code true} means
	 * the vector at that index should be kept.
	 *
	 * <p>Computes average pairwise cosine similarity for each vector against
	 * all others. Detects a gap in the coherence distribution using the
	 * same gap-multiplier approach as the embedding pipeline's coherence
	 * filter: a gap must exceed both {@code COHERENCE_GAP_MULTIPLIER} times
	 * the running average gap AND an adaptive minimum gap proportional to
	 * the coherence range. Vectors below the gap are marked for removal.
	 *
	 * @param vectors the embedding vectors to filter
	 * @return boolean array indicating which vectors to keep
	 */
	public static boolean[] filterByCoherence(float[][] vectors) {
		int n = vectors.length;
		boolean[] keep = new boolean[n];
		if (n < 3) {
			java.util.Arrays.fill(keep, true);
			return keep;
		}

		// Compute average coherence per vector without allocating O(n^2)
		// matrix -- accumulate pairwise sums directly into per-vector totals.
		double[] coherence = new double[n];
		double[] pairSum = new double[n];
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				double sim = cosineSimilarity(vectors[i], vectors[j]);
				pairSum[i] += sim;
				pairSum[j] += sim;
			}
		}
		for (int i = 0; i < n; i++) {
			coherence[i] = pairSum[i] / (n - 1);
		}

		// Sort by coherence descending
		Integer[] sortedIdx = new Integer[n];
		for (int i = 0; i < n; i++) {
			sortedIdx[i] = i;
		}
		java.util.Arrays.sort(sortedIdx, new java.util.Comparator<Integer>() {
			@Override
			public int compare(Integer a, Integer b) {
				return Double.compare(coherence[b], coherence[a]);
			}
		});

		// Gap detection
		double maxCoherence = coherence[sortedIdx[0]];
		double minCoherence = coherence[sortedIdx[n - 1]];
		double coherenceRange = maxCoherence - minCoherence;
		int refPairs = COHERENCE_REFERENCE_N - 1;
		double scaleFactor = Math.max(1.0,
				Math.sqrt((double) refPairs / (n - 1)));
		double base = n <= COHERENCE_REFERENCE_N
				? coherenceRange : maxCoherence;
		double coherenceMinGap = base
				* COHERENCE_ADAPTIVE_GAP_RATIO * scaleFactor;

		int keepCount = n;
		double gapSum = 0;
		for (int i = 1; i < n; i++) {
			double gap = coherence[sortedIdx[i - 1]] - coherence[sortedIdx[i]];
			if (i >= ADAPTIVE_MIN_RECORDS && i >= 2) {
				double avgGap = gapSum / (i - 1);
				if (gap > avgGap * COHERENCE_GAP_MULTIPLIER
						&& gap > coherenceMinGap) {
					keepCount = i;
					break;
				}
			}
			gapSum += gap;
		}

		// Same-topic floor check
		if (keepCount < n) {
			double highestRemovedCoherence = coherence[sortedIdx[keepCount]];
			if (highestRemovedCoherence >= COHERENCE_SAME_TOPIC_FLOOR) {
				keepCount = n;
			}
		}

		java.util.Arrays.fill(keep, false);
		for (int i = 0; i < keepCount; i++) {
			keep[sortedIdx[i]] = true;
		}
		return keep;
	}

	/**
	 * Returns true if the given pipeline value uses a Lucene index
	 * (either the pure Lucene pipeline or the hybrid pipeline that
	 * combines Lucene BM25 with embedding kNN).
	 */
	public static boolean usesLuceneIndex(String pipeline) {
		if (pipeline == null) {
			return false;
		}
		String trimmed = pipeline.trim();
		return PIPELINE_LUCENE.equalsIgnoreCase(trimmed)
				|| PIPELINE_HYBRID.equalsIgnoreCase(trimmed);
	}

	private ChartSearchAiUtils() {
	}
}
