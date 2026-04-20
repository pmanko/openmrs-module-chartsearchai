/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.embedding;

/**
 * Computes vector embeddings from text using ONNX Runtime.
 *
 * <p>Supports both single-encoder models (e.g. all-MiniLM-L6-v2, MedEmbed)
 * where the same model encodes both queries and records, and dual-encoder
 * models (e.g. MedCPT) where separate models encode queries and records.
 * Single-encoder models need only implement {@link #embed(String)};
 * dual-encoder models should also override {@link #embedQuery(String)}.
 */
public interface EmbeddingProvider {

	/**
	 * Compute a vector embedding for record/document text. For dual-encoder
	 * models, this uses the article/document encoder.
	 *
	 * @param text the text to embed
	 * @return a float array representing the text's embedding vector
	 */
	float[] embed(String text);

	/**
	 * Compute a vector embedding for query text. For dual-encoder models
	 * (e.g. MedCPT), this uses the query encoder which is trained to map
	 * questions into the same vector space as record embeddings. For
	 * single-encoder models, delegates to {@link #embed(String)}.
	 *
	 * @param text the query text to embed
	 * @return a float array representing the query's embedding vector
	 */
	default float[] embedQuery(String text) {
		return embed(text);
	}

	/**
	 * @return the dimensionality of vectors produced by this provider
	 */
	int getDimensions();

	/**
	 * Returns the model name or path, used to select model-specific
	 * pipeline configuration. Returns {@code null} by default.
	 */
	default String getModelName() {
		return null;
	}

	/**
	 * Sentinel string embedded at startup to fingerprint the model.
	 * Different models produce distinct, deterministic vectors for
	 * the same input — the first 4 values uniquely identify the model.
	 */
	String MODEL_IDENTITY_SENTINEL = "model identity probe";

	/**
	 * Identifies the model by embedding a sentinel string and matching
	 * the output fingerprint against known models. More reliable than
	 * path-based detection since the model file can be renamed.
	 * Returns a canonical model name (e.g. "medembed-small") or
	 * {@code null} if the model is unrecognized.
	 */
	default String identifyModel() {
		return null;
	}
}
