/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;

public class LuceneIndexerTest {

	private LuceneIndexer indexer;

	private StubRecordLoader stubLoader;

	@BeforeEach
	public void setUp() throws Exception {
		indexer = new LuceneIndexer();
		indexer.setDirectory(new ByteBuffersDirectory());
		stubLoader = new StubRecordLoader();
		// Inject the stub record loader via reflection
		Field f = LuceneIndexer.class.getDeclaredField("recordLoader");
		f.setAccessible(true);
		f.set(indexer, stubLoader);
	}

	@AfterEach
	public void tearDown() throws IOException {
		indexer.close();
	}

	@Test
	public void search_shouldReturnConditionRecordsForConditionsQuery() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null),
				new SerializedRecord("condition", 11,
						"Condition: Hypertension. Status: ACTIVE", null),
				new SerializedRecord("obs", 20,
						"Test — Systolic Blood Pressure: 137.0", null),
				new SerializedRecord("obs", 21,
						"Test — Weight (kg): 94.0", null),
				new SerializedRecord("diagnosis", 30,
						"Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary", null));

		indexer.indexPatient(patient);

		List<LuceneIndexer.LuceneSearchResult> results =
				indexer.search(patient, "conditions", 10);

		Set<Integer> ids = new HashSet<Integer>();
		for (LuceneIndexer.LuceneSearchResult r : results) {
			ids.add(r.getResourceId());
		}
		assertTrue(ids.contains(10), "Should find Tuberculosis condition");
		assertTrue(ids.contains(11), "Should find Hypertension condition");
	}

	@Test
	public void search_shouldReturnEmptyForNonMatchingQuery() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null),
				new SerializedRecord("obs", 20,
						"Test — Systolic Blood Pressure: 137.0", null));

		indexer.indexPatient(patient);

		List<LuceneIndexer.LuceneSearchResult> results =
				indexer.search(patient, "diabetes", 10);

		assertTrue(results.isEmpty(), "Should return no results for unmatched query");
	}

	@Test
	public void search_shouldFilterByPatient() {
		Patient patient1 = makePatient(1);
		Patient patient2 = makePatient(2);

		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null));
		indexer.indexPatient(patient1);

		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 20,
						"Condition: Hypertension. Status: ACTIVE", null));
		indexer.indexPatient(patient2);

		List<LuceneIndexer.LuceneSearchResult> results1 =
				indexer.search(patient1, "condition", 10);
		List<LuceneIndexer.LuceneSearchResult> results2 =
				indexer.search(patient2, "condition", 10);

		assertEquals(1, results1.size());
		assertEquals(10, results1.get(0).getResourceId());
		assertEquals(1, results2.size());
		assertEquals(20, results2.get(0).getResourceId());
	}

	@Test
	public void hasIndex_shouldReturnFalseBeforeIndexing() {
		Patient patient = makePatient(1);
		assertFalse(indexer.hasIndex(patient));
	}

	@Test
	public void hasIndex_shouldReturnTrueAfterIndexing() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null));
		indexer.indexPatient(patient);
		assertTrue(indexer.hasIndex(patient));
	}

	@Test
	public void indexPatient_shouldReplaceExistingDocuments() {
		Patient patient = makePatient(1);

		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null));
		indexer.indexPatient(patient);

		// Re-index with different records
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 20,
						"Condition: Hypertension. Status: ACTIVE", null));
		indexer.indexPatient(patient);

		List<LuceneIndexer.LuceneSearchResult> results =
				indexer.search(patient, "condition", 10);

		assertEquals(1, results.size());
		assertEquals(20, results.get(0).getResourceId(),
				"Should only find the re-indexed record");
	}

	@Test
	public void deletePatientIndex_shouldRemoveAllDocuments() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 10,
						"Condition: Tuberculosis. Status: ACTIVE", null));
		indexer.indexPatient(patient);
		assertTrue(indexer.hasIndex(patient));

		indexer.deletePatientIndex(patient);
		assertFalse(indexer.hasIndex(patient));
	}

	@Test
	public void search_shouldReturnBloodPressureForVitalsQuery() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("obs", 1,
						"Test — Systolic Blood Pressure: 137.0", null),
				new SerializedRecord("obs", 2,
						"Test — Diastolic Blood Pressure: 67.0", null),
				new SerializedRecord("obs", 3,
						"Test — Weight (kg): 94.0", null),
				new SerializedRecord("obs", 4,
						"Test — Temperature (C): 36.7", null),
				new SerializedRecord("obs", 5,
						"Test — CD4 Count: 988.0", null),
				new SerializedRecord("condition", 6,
						"Condition: Tuberculosis. Status: ACTIVE", null));

		indexer.indexPatient(patient);

		List<LuceneIndexer.LuceneSearchResult> results =
				indexer.search(patient, "blood pressure", 10);

		Set<Integer> ids = new HashSet<Integer>();
		for (LuceneIndexer.LuceneSearchResult r : results) {
			ids.add(r.getResourceId());
		}
		assertTrue(ids.contains(1), "Should find Systolic BP");
		assertTrue(ids.contains(2), "Should find Diastolic BP");
		assertFalse(ids.contains(6), "Should not include conditions");
	}

	@Test
	public void search_conditionsQuery_secondDataset_shouldReturnAllConditions() {
		Patient patient = makePatient(1);
		stubLoader.records = Arrays.asList(
				new SerializedRecord("condition", 2,
						"Condition: Nonparalytic stroke. Status: ACTIVE", null),
				new SerializedRecord("condition", 3,
						"Condition: Scarring Alopecia. Status: ACTIVE", null),
				new SerializedRecord("diagnosis", 4,
						"Diagnosis: Nonparalytic stroke. Certainty: PROVISIONAL. Rank: Primary", null),
				new SerializedRecord("diagnosis", 5,
						"Diagnosis: Scarring Alopecia. Certainty: CONFIRMED. Rank: Primary", null),
				new SerializedRecord("obs", 6,
						"Finding — Temperature (c)): 38.9 (CRITICALLY_HIGH)", null),
				new SerializedRecord("condition", 17,
						"Condition: Granuloma annulare. Status: ACTIVE", null),
				new SerializedRecord("condition", 18,
						"Condition: Syphilitic Cirrhosis. Status: ACTIVE", null),
				new SerializedRecord("condition", 30,
						"Condition: Atherosclerosis. Status: ACTIVE", null),
				new SerializedRecord("condition", 31,
						"Condition: Complete tear of ligament of ankle or foot. Status: ACTIVE", null),
				new SerializedRecord("condition", 32,
						"Condition: Mild depressive episode. Status: ACTIVE", null),
				new SerializedRecord("condition", 45,
						"Condition: Female infertility. Status: ACTIVE", null),
				new SerializedRecord("condition", 56,
						"Condition: Personal history of blood transfusion. Status: ACTIVE", null),
				new SerializedRecord("condition", 57,
						"Condition: Chronic fatigue. Status: ACTIVE", null));

		indexer.indexPatient(patient);

		List<LuceneIndexer.LuceneSearchResult> results =
				indexer.search(patient, "conditions", 50);

		Set<Integer> ids = new HashSet<Integer>();
		for (LuceneIndexer.LuceneSearchResult r : results) {
			ids.add(r.getResourceId());
		}
		// All 10 condition records should be found — Lucene's text matching
		// finds "condition" in the prefixed text "Medical condition: Condition: ..."
		Set<Integer> expectedConditions = new HashSet<Integer>(
				Arrays.asList(2, 3, 17, 18, 30, 31, 32, 45, 56, 57));
		assertTrue(ids.containsAll(expectedConditions),
				"Should find all 10 conditions, got: " + ids);
	}

	private static Patient makePatient(int id) {
		Patient p = new Patient();
		p.setPatientId(id);
		return p;
	}

	/**
	 * Stub PatientRecordLoader that returns preconfigured records.
	 */
	private static class StubRecordLoader extends PatientRecordLoader {

		List<SerializedRecord> records = Collections.emptyList();

		StubRecordLoader() {
			super();
		}

		@Override
		public List<SerializedRecord> loadAll(Patient patient) {
			return new ArrayList<SerializedRecord>(records);
		}
	}
}
