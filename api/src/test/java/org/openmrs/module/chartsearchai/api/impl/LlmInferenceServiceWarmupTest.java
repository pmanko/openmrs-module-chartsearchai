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
 *
 * <p>Focus-hint dispatch update: with the focus-hint contract,
 * {@code preFilter=true, queryStore=true} now produces a stable chart prefix
 * (full chart) plus a small trailing focus hint, so warmup is viable there too.
 * The single remaining unstable mode is preFilter+legacy (no querystore).
 */
public class LlmInferenceServiceWarmupTest {

	@Test
	public void shouldRunWarmup_shouldReturnFalse_whenPreFilterEnabledWithoutQueryStore() {
		// Legacy preFilter dispatch (embedding/lucene/elasticsearch, no querystore) still picks
		// question-dependent records inline. Chart varies per query → warmup can't help.
		assertFalse(LlmInferenceService.shouldRunWarmup(true, false),
				"preFilter without querystore goes through the legacy filter pipeline; chart bytes vary");
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
		assertTrue(LlmInferenceService.shouldRunWarmup(false, true),
				"querystore's full-chart mode produces question-independent bytes — warmup must run");
	}

	@Test
	public void shouldRunWarmup_shouldReturnTrue_whenPreFilterEnabledWithQueryStore() {
		// Focus-hint contract: preFilter+querystore now calls getPatientChart for the chart
		// (stable bytes) and renders the searchByPatient hits as a trailing focus hint. The
		// chart prefix is byte-identical across queries for the same patient, so warmup
		// primes the real prefix — same as the no-prefilter cases above. Pre-focus-hint this
		// returned false because the chart bytes varied per query.
		assertTrue(LlmInferenceService.shouldRunWarmup(true, true),
				"preFilter+querystore is the focus-hint configuration; chart bytes are stable per patient");
	}

	/**
	 * Exhaustive truth table — all 2² = 4 combinations. Expected =
	 * {@code !preFilterEnabled || queryStoreEnabled}; the only mode that still produces
	 * question-dependent chart bytes is {@code preFilter=true, queryStore=false} (the
	 * legacy embedding/lucene/elasticsearch dispatch).
	 */
	@ParameterizedTest(name = "[{index}] preFilter={0} queryStore={1} → {2}")
	@CsvSource({
			"false, false, true",   // full-chart, legacy serializer — original happy path
			"false, true,  true",   // full-chart via Decision 15 getPatientChart
			"true,  false, false",  // legacy preFilter dispatch — chart varies per query
			"true,  true,  true",   // focus-hint preFilter — chart is full + stable, hint at end
	})
	public void shouldRunWarmup_exhaustiveTruthTable(boolean preFilterEnabled,
			boolean queryStoreEnabled, boolean expected) {
		assertEquals(expected,
				LlmInferenceService.shouldRunWarmup(preFilterEnabled, queryStoreEnabled),
				"truth-table row failed; expected = !preFilter || queryStore");
	}
}
