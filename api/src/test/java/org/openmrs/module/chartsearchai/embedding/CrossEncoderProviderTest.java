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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;

import org.junit.jupiter.api.Test;

/**
 * Validates {@link CrossEncoderProvider} against the real MedCPT cross-
 * encoder ONNX model. The test asserts only the qualitative property
 * the production pipeline relies on: a clearly-relevant pair scores
 * higher than a clearly-irrelevant one. Absolute logit values are
 * model-specific and are calibrated separately in the eval suite.
 */
public class CrossEncoderProviderTest {

	private static final String MODELS_BASE = System.getProperty(
			"chartsearchai.models.base", "../../models");

	@Test
	public void scorePair_shouldRankRelevantAboveIrrelevant() throws Exception {
		File modelDir = new File(MODELS_BASE, "MedCPT-Cross-Encoder")
				.getCanonicalFile();
		File modelFile = new File(modelDir, "model.onnx");
		File vocabFile = new File(modelDir, "vocab.txt");
		if (!modelFile.exists() || !vocabFile.exists()) {
			System.out.println("MedCPT cross-encoder not available — skipping");
			return;
		}

		CrossEncoderProvider provider = new CrossEncoderProvider(
				modelFile.getAbsolutePath(),
				vocabFile.getAbsolutePath());
		try {
			assertTrue(provider.isAvailable(),
					"Provider should report available with explicit paths");

			float relevant = provider.scorePair(
					"stroke",
					"Condition: Nonparalytic stroke (active)");
			float irrelevant = provider.scorePair(
					"dental problems",
					"Condition: Female infertility (active)");

			System.out.printf("relevant=%.4f, irrelevant=%.4f%n",
					relevant, irrelevant);

			assertTrue(relevant > irrelevant,
					"Cross-encoder must rank a topical match above a "
					+ "random clinical pair (relevant=" + relevant
					+ ", irrelevant=" + irrelevant + ")");
		} finally {
			provider.close();
		}
	}

	@Test
	public void isAvailable_shouldReturnFalseWhenNotConfigured() {
		CrossEncoderProvider provider = new CrossEncoderProvider();
		// No explicit paths AND no OpenMRS Context — both resolvers return null
		assertFalse(provider.isAvailable(),
				"Unconfigured provider must report unavailable");
	}
}
