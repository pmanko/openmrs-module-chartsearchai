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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Unit tests for {@link LlmInferenceService#shouldRunWarmup}, the pure-logic
 * decision for whether the current pipeline mode produces a chart prefix that
 * warmup can usefully prime.
 *
 * <p>The helper only decides chart-byte-stability semantics. The operational
 * kill switches (`chartsearchai.warmupEnabled` and `LlmProvider.supportsWarmup`)
 * are checked at the {@code warmup()} call site, not here — so they can short-
 * circuit before the GP-reading gates this helper decides on. The call-site
 * wiring is locked by {@link LlmInferenceServiceWarmupIntegrationTest}.
 */
public class LlmInferenceServiceWarmupTest {

	@Test
	public void shouldRunWarmup_shouldReturnFalse_whenPreFilterEnabled() {
		// Pre-filter chooses different records per question — chart is question-dependent.
		assertFalse(LlmInferenceService.shouldRunWarmup(true, false),
				"pre-filter pipelines produce question-dependent charts; warmup can't help");
	}

	@Test
	public void shouldRunWarmup_shouldReturnTrue_whenFullChartPathWithoutQueryStore() {
		// querystore off + prefilter off: chartSerializer.serialize(patient) produces the
		// full chart deterministically (byte-identical across questions for the same
		// patient). Warmup primes the real prefix. WORKS.
		assertTrue(LlmInferenceService.shouldRunWarmup(false, false),
				"the full-chart path is the original happy case for warmup; this must keep working");
	}

	@Test
	public void shouldRunWarmup_shouldReturnTrue_whenFullChartPathWithQueryStore() {
		// querystore on + prefilter off: chartsearchai dispatches to getPatientChart
		// (querystore Decision 15), which returns the patient's full indexed projection —
		// byte-identical across questions, same shape as the legacy serializer path.
		// Warmup primes the real prefix here too. Pre-Decision-15 this case returned false
		// because the dispatch went through searchByPatient (question-conditioned, varying
		// per query); the dispatch change makes it warmup-eligible.
		assertTrue(LlmInferenceService.shouldRunWarmup(false, true),
				"with the Decision 15 dispatch, querystore's full-chart mode produces "
				+ "question-independent bytes — warmup must run to prime them");
	}

	@Test
	public void shouldRunWarmup_shouldReturnFalse_whenPreFilterEnabledRegardlessOfQuerystore() {
		// Pins that preFilter is the sole gate. A refactor that re-introduced a queryStore
		// check (e.g., "skip if querystore on") could silently drop warmup in the
		// (preFilter=false, queryStore=true) case where this slice explicitly enables it.
		// This test plus the truth-table parameterised test below catch that regression.
		assertFalse(LlmInferenceService.shouldRunWarmup(true, true),
				"preFilter=true must skip warmup regardless of queryStore — chart varies per question");
	}

	/**
	 * Exhaustive truth table — all 2² = 4 combinations. Expected = {@code !preFilterEnabled};
	 * the {@code queryStoreEnabled} parameter is non-load-bearing after the Decision 15
	 * dispatch change. Locks against a refactor that re-introduces a queryStore-dependent
	 * gate (a regression that would silently disable warmup in the full-chart-with-
	 * querystore configuration).
	 */
	@ParameterizedTest(name = "[{index}] preFilter={0} queryStore={1} → {2}")
	@CsvSource({
			"false, false, true",   // full-chart, no querystore — original happy path
			"false, true,  true",   // full-chart via Decision 15 getPatientChart — also warmup-eligible
			"true,  false, false",  // preFilter on — chart varies per query
			"true,  true,  false",  // preFilter on regardless of querystore — chart varies per query
	})
	public void shouldRunWarmup_exhaustiveTruthTable(boolean preFilterEnabled,
			boolean queryStoreEnabled, boolean expected) {
		assertEquals(expected,
				LlmInferenceService.shouldRunWarmup(preFilterEnabled, queryStoreEnabled),
				"truth-table row failed; expected = (!preFilterEnabled), queryStoreEnabled "
				+ "must not affect the outcome");
	}
}
