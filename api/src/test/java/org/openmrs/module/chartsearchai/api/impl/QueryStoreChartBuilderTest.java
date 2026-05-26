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

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.querystore.api.QueryStoreService;
import org.openmrs.module.querystore.backend.WriteResult;
import org.openmrs.module.querystore.model.QueryDocument;

/**
 * Pure unit tests for {@link QueryStoreChartBuilder}.
 *
 * <p>The class branches on {@code chartsearchai.embedding.preFilter} between
 * {@code QueryStoreService.searchByPatient} (preFilter=true, ranked top-K) and
 * {@code QueryStoreService.getPatientChart} (preFilter=false, unfiltered full chart
 * per querystore Decision 15). These tests pin the dispatch correctness on both
 * branches, the input-guard contract (null patient / null uuid in both modes; blank
 * question only short-circuits in preFilter mode), and the record-conversion loop.
 */
public class QueryStoreChartBuilderTest {

	private static Patient patient(int id) {
		Patient p = new Patient();
		p.setPatientId(id);
		p.setUuid("uuid-" + id);
		return p;
	}

	private CountingQueryStore queryStore;
	private TestableQueryStoreChartBuilder builder;

	@BeforeEach
	public void setUp() {
		queryStore = new CountingQueryStore();
		builder = new TestableQueryStoreChartBuilder(queryStore);
		builder.setChartSerializer(new PatientChartSerializer());
	}

	@Test
	public void build_shouldReturnEmptyChartAndSkipQueryStore_whenQuestionIsBlank() {
		PatientChart chart = builder.build(patient(1), "   ");

		assertEquals(0, chart.getMappings().size(),
				"blank question must produce an empty chart — the empty-question early-return "
				+ "is what makes LlmInferenceService.warmup() safe to call with an empty string "
				+ "before chart open, and what protects querystore from spurious calls");
		assertEquals(0, queryStore.getCallCount(),
				"blank question must not consume a querystore call — that costs RPC + a Gemma "
				+ "encoder pass for no useful output");
	}

	@Test
	public void build_shouldReturnEmptyChart_whenQuestionIsNull() {
		PatientChart chart = builder.build(patient(1), null);

		assertEquals(0, chart.getMappings().size(),
				"null question must not NPE — caller contract is empty chart");
		assertEquals(0, queryStore.getCallCount());
	}

	@Test
	public void build_shouldReturnEmptyChart_whenPatientIsNull() {
		PatientChart chart = builder.build(null, "any allergies?");

		assertEquals(0, chart.getMappings().size(),
				"null patient must not NPE — caller contract is empty chart");
		assertEquals(0, queryStore.getCallCount());
	}

	@Test
	public void build_shouldReturnEmptyChart_whenBothPatientAndQuestionAreNull() {
		// Belt-and-braces: locks the guard against a future refactor that swaps the order
		// of clauses in the OR-chain and accidentally dereferences a null patient before
		// short-circuiting (e.g. `question == null || patient.getUuid() == null` would NPE).
		PatientChart chart = builder.build(null, null);

		assertEquals(0, chart.getMappings().size(),
				"both-null must short-circuit cleanly without NPE — the guard's OR-chain "
				+ "must check patient==null BEFORE dereferencing patient.getUuid()");
		assertEquals(0, queryStore.getCallCount());
	}

	@Test
	public void build_shouldReturnEmptyChartAndSkipQueryStore_whenPatientUuidIsNull() {
		// OpenMRS BaseOpenmrsObject auto-generates a UUID in the no-arg constructor, so
		// new Patient() already has a non-null uuid. Explicit setUuid(null) is what reaches
		// the guard — defensive coverage for deserialization paths or buggy callers.
		Patient unidentifiedPatient = new Patient();
		unidentifiedPatient.setPatientId(99);
		unidentifiedPatient.setUuid(null);

		PatientChart chart = builder.build(unidentifiedPatient, "any allergies?");

		assertEquals(0, chart.getMappings().size(),
				"patient without uuid must short-circuit — passing a null uuid to querystore "
				+ "would either NPE or return spurious results depending on the backend");
		assertEquals(0, queryStore.getCallCount());
	}

	@Test
	public void build_shouldCallQueryStore_whenInputsAreValid() {
		// Positive control: pins the happy path. Without this, the three null/blank tests
		// would all still pass on a refactor that inverted the guard and short-circuited
		// the VALID path — every test asserts callCount==0 in failure cases, none asserts
		// callCount==1 in the success case.
		builder.build(patient(1), "any allergies?");

		assertEquals(1, queryStore.getCallCount(),
				"valid (patient with uuid, non-blank question) must reach querystore — "
				+ "this is the only test that locks the happy path against guard inversion");
	}

	@Test
	public void build_shouldCallGetPatientChartNotSearchByPatient_whenPreFilterIsFalse() {
		// Decision 15 dispatch contract: preFilter=false routes to getPatientChart for the
		// unfiltered full-chart path that the LLM consumer wants. A regression that always
		// called searchByPatient would still produce *a* chart (ranked top-K), but it would
		// be the wrong shape — relevance-filtered when the consumer explicitly opted out.
		builder.usePreFilter = false;

		builder.build(patient(1), "any allergies?");

		assertEquals(1, queryStore.getPatientChartCalls,
				"preFilter=false must dispatch to getPatientChart — the new querystore Decision 15 path");
		assertEquals(0, queryStore.searchByPatientCalls,
				"preFilter=false must NOT call searchByPatient — that's the ranked path");
	}

	@Test
	public void build_shouldCallSearchByPatientNotGetPatientChart_whenPreFilterIsTrue() {
		// Positive control for the preFilter=true path. Without this counter-assertion, the
		// preFilter=false test above could pass on a regression that inverted the branch and
		// routed BOTH modes to getPatientChart — the preFilter=true path would silently lose
		// its question-conditioned ranking.
		builder.usePreFilter = true;

		builder.build(patient(1), "any allergies?");

		assertEquals(1, queryStore.searchByPatientCalls,
				"preFilter=true must dispatch to searchByPatient — the ranked top-K path");
		assertEquals(0, queryStore.getPatientChartCalls,
				"preFilter=true must NOT call getPatientChart — that's the full-chart path");
	}

	@Test
	public void build_shouldStillCallGetPatientChart_whenQuestionIsBlankAndPreFilterIsFalse() {
		// The blank-question short-circuit only applies in preFilter mode. Full-chart mode
		// ignores the question entirely — Decision 15's getPatientChart has no question
		// parameter — so a blank question must still produce the full chart. This is what
		// makes the warmup re-enablement (separate follow-up) viable: warmup calls
		// buildChart(patient, "") and expects a real chart back when chart bytes are
		// question-independent. A regression that short-circuited blank questions in both
		// modes would silently leave warmup priming an empty prompt.
		builder.usePreFilter = false;

		PatientChart chart = builder.build(patient(1), "   ");

		assertEquals(1, queryStore.getPatientChartCalls,
				"blank question must reach getPatientChart in preFilter=false mode — the question is unused");
		assertEquals(0, queryStore.searchByPatientCalls,
				"blank question must not reach searchByPatient");
		// chart is built from stubChart (empty), so the resulting chart has 0 mappings —
		// the test's claim is about dispatch, not chart contents.
		assertEquals(0, chart.getMappings().size(),
				"empty stubChart produces an empty PatientChart; this just confirms the call returned cleanly");
	}

	@Test
	public void build_shouldStillSkipNullPatient_whenPreFilterIsFalse() {
		// Hard-error guards apply in both modes — null patient can't reach getPatientChart
		// without NPEing on getUuid(). Mirrors the preFilter=true null-patient guard test;
		// this is the symmetric coverage for the new branch.
		builder.usePreFilter = false;

		PatientChart chart = builder.build(null, "any allergies?");

		assertEquals(0, chart.getMappings().size());
		assertEquals(0, queryStore.getPatientChartCalls,
				"null patient must short-circuit in preFilter=false mode too");
	}

	@Test
	public void build_shouldSkipNullAndMalformedHitsAndKeepValidOnes() {
		// Locks the record-conversion loop: a refactor that breaks SerializedRecord field
		// order, drops the null-text guard, or removes a skip clause would silently corrupt
		// the chart sent to the LLM without any test failing.
		QueryDocument valid1 = new QueryDocument();
		valid1.setResourceType("Condition");
		valid1.setResourceUuid("cond-1");
		valid1.setText("Condition: Hypertension. Status: ACTIVE");
		QueryDocument valid2 = new QueryDocument();
		valid2.setResourceType("Obs");
		valid2.setResourceUuid("obs-1");
		valid2.setText("Systolic blood pressure: 140 mmHg");
		QueryDocument malformedNoType = new QueryDocument();
		malformedNoType.setResourceUuid("dropped-no-type");
		// resourceType deliberately left null
		malformedNoType.setText("should be dropped");
		QueryDocument malformedNoUuid = new QueryDocument();
		malformedNoUuid.setResourceType("Condition");
		// resourceUuid deliberately left null
		malformedNoUuid.setText("should be dropped");
		List<QueryDocument> mixed = new ArrayList<>();
		mixed.add(valid1);
		mixed.add(null);                // null doc → skip
		mixed.add(malformedNoType);     // missing type → skip
		mixed.add(valid2);
		mixed.add(malformedNoUuid);     // missing uuid → skip
		queryStore.stubHits = mixed;

		PatientChart chart = builder.build(patient(1), "any allergies?");

		assertEquals(2, chart.getMappings().size(),
				"only the two well-formed QueryDocuments must reach the chart — null/missing-type/"
				+ "missing-uuid docs are logged at WARN and dropped from the conversion loop");
		// Pin the resource uuids that survived so a future serializer change can't silently
		// reshuffle which doc maps to which mapping index.
		assertEquals("cond-1", chart.getMappings().get(0).getResourceUuid(),
				"first surviving record should be valid1 (Condition cond-1) — order preserved from the stub");
		assertEquals("obs-1", chart.getMappings().get(1).getResourceUuid(),
				"second surviving record should be valid2 (Obs obs-1)");
	}

	/**
	 * Subclass that bypasses {@code Context.getService} so this test runs without
	 * a live OpenMRS context. The {@code resolve*} overrides are the seam — every
	 * other code path goes through the real builder.
	 *
	 * <p>Defaults {@code resolveUsePreFilter()} to {@code true} so the legacy tests
	 * (written when {@code searchByPatient} was the only dispatch target) continue to
	 * exercise the preFilter path without explicit per-test wiring. Tests that need
	 * the full-chart branch flip it via {@link #usePreFilter}.
	 */
	private static final class TestableQueryStoreChartBuilder extends QueryStoreChartBuilder {

		private final QueryStoreService stub;

		boolean usePreFilter = true;

		TestableQueryStoreChartBuilder(QueryStoreService stub) {
			this.stub = stub;
		}

		@Override
		protected QueryStoreService resolveQueryStoreService() {
			return stub;
		}

		@Override
		protected int resolveQueryStoreTopK() {
			return 10;
		}

		@Override
		protected boolean resolveUsePreFilter() {
			return usePreFilter;
		}
	}

	private static final class CountingQueryStore implements QueryStoreService {

		int searchByPatientCalls = 0;

		int getPatientChartCalls = 0;

		List<QueryDocument> stubHits = new ArrayList<QueryDocument>();

		List<QueryDocument> stubChart = new ArrayList<QueryDocument>();

		/** Backwards-compatible aggregate counter — pre-Decision-15 tests assert on
		 *  total querystore calls without caring which method. */
		int getCallCount() {
			return searchByPatientCalls + getPatientChartCalls;
		}

		@Override
		public List<QueryDocument> searchByPatient(String patientUuid, String question, int topK) {
			searchByPatientCalls++;
			return stubHits;
		}

		@Override
		public List<QueryDocument> getPatientChart(String patientUuid) {
			getPatientChartCalls++;
			return stubChart;
		}

		@Override
		public List<QueryDocument> search(String question, int topK) {
			throw new UnsupportedOperationException("not used by chartsearchai");
		}

		@Override
		public WriteResult index(QueryDocument doc) {
			throw new UnsupportedOperationException();
		}

		@Override
		public void delete(String resourceType, String resourceUuid) {
			throw new UnsupportedOperationException();
		}

		@Override
		public void bulkDeleteByPatient(String patientUuid) {
			throw new UnsupportedOperationException();
		}

		@Override
		public void onStartup() {
		}

		@Override
		public void onShutdown() {
		}
	}
}
