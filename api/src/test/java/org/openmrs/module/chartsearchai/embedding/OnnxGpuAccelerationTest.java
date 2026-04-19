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

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumSet;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtProvider;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public class OnnxGpuAccelerationTest {

	private static final String MODEL_DIR = System.getProperty(
			"chartsearchai.embedding.model.dir", "../models/all-MiniLM-L6-v2");
	private static final String MODEL_PATH = MODEL_DIR + "/model.onnx";
	private static final String VOCAB_PATH = MODEL_DIR + "/vocab.txt";

	private static boolean modelFilesExist() {
		return new java.io.File(MODEL_PATH).exists()
				&& new java.io.File(VOCAB_PATH).exists();
	}

	@Test
	public void shouldUseGpuProviderWhenAvailable() {
		Assumptions.assumeTrue(modelFilesExist(),
				"Skipping: ONNX model not found at " + MODEL_PATH);

		EnumSet<OrtProvider> available = OrtEnvironment.getAvailableProviders();
		boolean gpuAvailable = available.contains(OrtProvider.CORE_ML)
				|| available.contains(OrtProvider.CUDA);
		Assumptions.assumeTrue(gpuAvailable,
				"Skipping: no GPU provider available (found: " + available + ")");

		OnnxEmbeddingProvider provider =
				new OnnxEmbeddingProvider(MODEL_PATH, VOCAB_PATH);
		try {
			// Trigger session creation by computing an embedding
			float[] embedding = provider.embed("test");
			assertNotNull(embedding);
			assertTrue(embedding.length > 0);

			// Verify a GPU provider was activated
			OrtProvider active = provider.getActiveProvider();
			assertNotEquals(OrtProvider.CPU, active,
					"Expected GPU provider but got CPU. Available providers: " + available);
			assertTrue(
					active == OrtProvider.CORE_ML || active == OrtProvider.CUDA,
					"Expected CORE_ML or CUDA but got: " + active);
		} finally {
			provider.close();
		}
	}
}
