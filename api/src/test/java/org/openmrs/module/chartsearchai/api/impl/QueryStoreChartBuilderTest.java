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
 * <p>The class is mostly a thin pass-through to {@code QueryStoreService.searchByPatient};
 * these tests pin the contract that callers (especially the warmup path) rely on —
 * specifically, that blank or null questions short-circuit before reaching querystore
 * and return an empty chart, never an exception.
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
		assertEquals(0, queryStore.callCount,
				"blank question must not consume a querystore call — that costs RPC + a Gemma "
				+ "encoder pass for no useful output");
	}

	@Test
	public void build_shouldReturnEmptyChart_whenQuestionIsNull() {
		PatientChart chart = builder.build(patient(1), null);

		assertEquals(0, chart.getMappings().size(),
				"null question must not NPE — caller contract is empty chart");
		assertEquals(0, queryStore.callCount);
	}

	@Test
	public void build_shouldReturnEmptyChart_whenPatientIsNull() {
		PatientChart chart = builder.build(null, "any allergies?");

		assertEquals(0, chart.getMappings().size(),
				"null patient must not NPE — caller contract is empty chart");
		assertEquals(0, queryStore.callCount);
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
		assertEquals(0, queryStore.callCount);
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
		assertEquals(0, queryStore.callCount);
	}

	@Test
	public void build_shouldCallQueryStore_whenInputsAreValid() {
		// Positive control: pins the happy path. Without this, the three null/blank tests
		// would all still pass on a refactor that inverted the guard and short-circuited
		// the VALID path — every test asserts callCount==0 in failure cases, none asserts
		// callCount==1 in the success case.
		builder.build(patient(1), "any allergies?");

		assertEquals(1, queryStore.callCount,
				"valid (patient with uuid, non-blank question) must reach querystore — "
				+ "this is the only test that locks the happy path against guard inversion");
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
	 */
	private static final class TestableQueryStoreChartBuilder extends QueryStoreChartBuilder {

		private final QueryStoreService stub;

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
	}

	private static final class CountingQueryStore implements QueryStoreService {

		int callCount = 0;
		List<QueryDocument> stubHits = new ArrayList<QueryDocument>();

		@Override
		public List<QueryDocument> searchByPatient(String patientUuid, String question, int topK) {
			callCount++;
			return stubHits;
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
