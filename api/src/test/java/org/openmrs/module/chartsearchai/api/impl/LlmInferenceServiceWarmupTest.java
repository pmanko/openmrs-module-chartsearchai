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
	public void shouldRunWarmup_shouldReturnTrue_whenFullChartPath() {
		// querystore off + prefilter off: ChartCache returns the full chart, byte-identical
		// across questions for the same patient. Warmup primes the real prefix. WORKS.
		assertTrue(LlmInferenceService.shouldRunWarmup(false, false),
				"the full-chart path is the original happy case for warmup; this must keep working");
	}

	@Test
	public void shouldRunWarmup_shouldReturnFalse_whenQuerystoreEnabled() {
		// The bug fix: today this returned true and warmup ran with buildChart(patient,"")
		// → empty chart → primed the wrong bytes. Real querystore queries with topK records
		// hit a completely different prefix. Wasted compute, zero cache hit. Must skip.
		assertFalse(LlmInferenceService.shouldRunWarmup(false, true),
				"querystore on means chart varies per question; warmup primes bytes that "
				+ "won't match real queries — must skip to avoid wasting compute");
	}

	@Test
	public void shouldRunWarmup_shouldReturnFalse_whenBothPreFilterAndQuerystoreEnabled() {
		// Locks the disjunction semantics: EITHER gate alone is sufficient. A refactor that
		// accidentally ANDs them ("if (preFilter && querystore) return false") would still
		// pass shouldRunWarmup_shouldReturnFalse_whenPreFilterEnabled AND
		// shouldRunWarmup_shouldReturnFalse_whenQuerystoreEnabled — and silently regress
		// to running warmup on the bug-on-bug combination.
		assertFalse(LlmInferenceService.shouldRunWarmup(true, true),
				"both question-dependent gates on must still skip; their relationship is OR, "
				+ "not AND — either alone makes the chart bytes question-dependent");
	}

	/**
	 * Exhaustive truth table — all 2² = 4 combinations. Expected = (!preFilterEnabled
	 * AND !queryStoreEnabled). Only the (false,false) row should return true; every
	 * other row returns false. Locks against any refactor that re-orders the gates,
	 * drops one, or replaces the short-circuit chain with a misformed boolean expression.
	 */
	@ParameterizedTest(name = "[{index}] preFilter={0} queryStore={1} → {2}")
	@CsvSource({
			"false, false, true",   // ONLY this row is true — full-chart happy path
			"false, true,  false",
			"true,  false, false",
			"true,  true,  false",
	})
	public void shouldRunWarmup_exhaustiveTruthTable(boolean preFilterEnabled,
			boolean queryStoreEnabled, boolean expected) {
		assertEquals(expected,
				LlmInferenceService.shouldRunWarmup(preFilterEnabled, queryStoreEnabled),
				"truth-table row failed; the only row that should return true is "
				+ "(preFilter=false, queryStore=false)");
	}
}
