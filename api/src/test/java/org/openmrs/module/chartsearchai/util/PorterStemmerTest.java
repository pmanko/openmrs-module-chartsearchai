/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class PorterStemmerTest {

	@Test
	public void stem_shouldReduceAllergyNounForms() {
		assertEquals(PorterStemmer.stem("allergy"), PorterStemmer.stem("allergies"));
	}

	@Test
	public void stem_shouldReduceMedicationVariantsToSameRoot() {
		String root = PorterStemmer.stem("medication");
		assertEquals(root, PorterStemmer.stem("medications"));
	}

	@Test
	public void stem_shouldStemDiagnosisVariants() {
		// Porter stems these to closely related but not identical forms
		// ("diagnosi" vs "diagnos") — close enough for embedding similarity
		assertTrue(PorterStemmer.stem("diagnosis").startsWith("diagnos"));
		assertTrue(PorterStemmer.stem("diagnoses").startsWith("diagnos"));
	}

	@Test
	public void stem_shouldReducePrescriptionNounForms() {
		assertEquals(PorterStemmer.stem("prescription"), PorterStemmer.stem("prescriptions"));
	}

	@Test
	public void stem_shouldStripAdjectiveSuffix() {
		// "allergic" → "allerg" (adjective -ic stripped by step4)
		assertEquals("allerg", PorterStemmer.stem("allergic"));
	}

	@Test
	public void stem_shouldHandleShortWords() {
		assertEquals("cd", PorterStemmer.stem("cd"));
		assertEquals("a", PorterStemmer.stem("a"));
	}

	@Test
	public void stem_shouldHandleNull() {
		assertNull(PorterStemmer.stem(null));
	}

	@Test
	public void stem_shouldBeCaseInsensitive() {
		assertEquals(PorterStemmer.stem("allergies"), PorterStemmer.stem("ALLERGIES"));
	}
}
