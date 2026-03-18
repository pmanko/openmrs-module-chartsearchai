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
 * Stub embedding provider for tests. Uses hash-based vectors so tests
 * do not require ONNX model files. Dimensions default to 384 but can
 * be changed via the constructor to simulate different embedding models.
 */
public class StubEmbeddingProvider implements EmbeddingProvider {

	private final int dimensions;

	public StubEmbeddingProvider() {
		this(384);
	}

	public StubEmbeddingProvider(int dimensions) {
		this.dimensions = dimensions;
	}

	@Override
	public float[] embed(String text) {
		float[] embedding = new float[dimensions];

		String[] tokens = text.toLowerCase().split("\\W+");
		for (String token : tokens) {
			if (token.isEmpty()) {
				continue;
			}
			int index = (token.hashCode() & 0x7FFFFFFF) % dimensions;
			embedding[index] += 1.0f;
		}

		double norm = 0;
		for (float v : embedding) {
			norm += v * v;
		}
		norm = Math.sqrt(norm);
		if (norm > 0) {
			for (int i = 0; i < embedding.length; i++) {
				embedding[i] /= (float) norm;
			}
		}

		return embedding;
	}

	@Override
	public int getDimensions() {
		return dimensions;
	}
}
