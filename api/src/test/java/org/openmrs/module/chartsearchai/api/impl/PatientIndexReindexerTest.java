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
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.DaemonToken;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.ReindexStatus;
import org.openmrs.module.querystore.api.QueryStoreService;
import org.openmrs.module.querystore.backend.WriteResult;
import org.openmrs.module.querystore.bootstrap.BootstrapProgress;
import org.openmrs.module.querystore.bootstrap.BootstrapService;
import org.openmrs.module.querystore.model.QueryDocument;

/**
 * Unit tests for {@link PatientIndexReindexer}, the "index all patients" orchestrator.
 *
 * <p>These exercise the real reindex loop (paging, per-patient ordering, status
 * tracking, error isolation) and stub only the external boundaries — the patient
 * source, the querystore write API, the bootstrap projection API, and the embedding
 * indexer — via the same {@code resolve*}/seam pattern used by
 * {@link QueryStoreChartBuilder}. Those services are from an external module and are
 * never live in a unit test, so recording fakes are the established doubles.
 *
 * <p>The querystore contract under test is the per-patient recipe verified on the demo's
 * dump-loaded data: {@link QueryStoreService#bulkDeleteByPatient(String)} THEN
 * {@link BootstrapService#ensureIndexed(String)} — delete-first is required because
 * {@code ensureIndexed} is a no-op on a partially-indexed patient.
 */
public class PatientIndexReindexerTest {

	private static Patient patient(int id) {
		Patient p = new Patient();
		p.setPatientId(id);
		p.setUuid("uuid-" + id);
		return p;
	}

	private static List<Patient> patients(int n) {
		List<Patient> list = new ArrayList<Patient>();
		for (int i = 1; i <= n; i++) {
			list.add(patient(i));
		}
		return list;
	}

	@Test
	public void getStatus_shouldBeIdleBeforeAnyRun() {
		ReindexStatus status = new TestableReindexer(patients(0)).getStatus();
		assertFalse(status.isRunning(), "fresh reindexer must report not-running");
		assertEquals(0, status.getProcessed());
		assertNull(status.getStartedAt(), "no run has started yet");
		assertNull(status.getBackend());
	}

	@Test
	public void start_shouldThrowAndResetRunning_whenDaemonTokenMissing_thenSucceedOnceTokenSet() {
		TestableReindexer reindexer = new TestableReindexer(patients(2));
		reindexer.queryStoreEnabled = true;

		// No DaemonToken yet (module not fully started): start() must reject loudly and
		// NOT spawn anything.
		assertThrows(IllegalStateException.class, reindexer::start);
		assertEquals(0, reindexer.runAsyncCalls, "no async work may be scheduled without a token");

		// The failed start() must have reset the running guard, so a later start() (now
		// with a token) is accepted — proving the guard didn't latch on the throw.
		reindexer.setDaemonToken(new DaemonToken("test"));
		reindexer.runAsyncInline = true;

		assertTrue(reindexer.start(), "start() must succeed once a token is available");
		assertEquals(1, reindexer.runAsyncCalls);
		assertEquals(2, reindexer.getStatus().getProcessed(), "the scheduled run actually reindexed");
	}

	@Test
	public void start_shouldRejectConcurrentRun_whileOneIsAlreadyInFlight() {
		TestableReindexer reindexer = new TestableReindexer(patients(2));
		reindexer.queryStoreEnabled = true;
		reindexer.setDaemonToken(new DaemonToken("test"));
		reindexer.runAsyncInline = false; // simulate an in-flight run that hasn't finished

		assertTrue(reindexer.start(), "first start() wins the guard");
		assertFalse(reindexer.start(), "second start() must be rejected while a run is in flight");
		assertEquals(1, reindexer.runAsyncCalls, "only one async run may be scheduled");
	}

	@Test
	public void start_shouldResetGuard_whenAsyncSpawnFails() {
		TestableReindexer reindexer = new TestableReindexer(patients(2));
		reindexer.queryStoreEnabled = true;
		reindexer.setDaemonToken(new DaemonToken("test"));
		reindexer.runAsyncThrows = true; // the daemon spawn blows up between CAS and the task

		assertThrows(IllegalStateException.class, reindexer::start);

		// The guard must NOT latch true on a spawn failure (the task's finally never ran),
		// otherwise every future reindex is wedged at 409. A clean retry must be accepted.
		reindexer.runAsyncThrows = false;
		reindexer.runAsyncInline = true;
		assertTrue(reindexer.start(), "guard must be released after a failed spawn");
		assertEquals(2, reindexer.getStatus().getProcessed());
	}

	@Test
	public void start_shouldResetGuard_afterRunCompletes_allowingAnotherRun() {
		TestableReindexer reindexer = new TestableReindexer(patients(2));
		reindexer.queryStoreEnabled = true;
		reindexer.setDaemonToken(new DaemonToken("test"));
		reindexer.runAsyncInline = true; // run completes synchronously within start()

		assertTrue(reindexer.start());
		assertFalse(reindexer.getStatus().isRunning(), "guard and status must clear after completion");
		assertTrue(reindexer.start(), "a second run is allowed once the first finished");
		assertEquals(2, reindexer.runAsyncCalls);
	}

	@Test
	public void reindexAll_shouldBulkDeleteThenEnsureIndex_forEachPatient_whenQueryStoreEnabled() {
		TestableReindexer reindexer = new TestableReindexer(patients(3));
		reindexer.queryStoreEnabled = true;

		reindexer.reindexAll();

		// Per-patient recipe and ordering: delete BEFORE ensureIndexed, for every patient,
		// in patient order. ensureIndexed alone is a no-op on a partial index.
		assertEquals(
				Arrays.asList(
						"delete:uuid-1", "ensure:uuid-1",
						"delete:uuid-2", "ensure:uuid-2",
						"delete:uuid-3", "ensure:uuid-3"),
				reindexer.ops,
				"each patient must be bulkDeleteByPatient then ensureIndexed, in order");

		ReindexStatus status = reindexer.getStatus();
		assertFalse(status.isRunning(), "run must be marked finished");
		assertEquals("querystore", status.getBackend());
		assertEquals(3, status.getProcessed());
		assertEquals(3, status.getSucceeded());
		assertEquals(0, status.getFailed());
	}

	@Test
	public void reindexAll_shouldIsolateFailureAndContinue_whenOnePatientThrows() {
		TestableReindexer reindexer = new TestableReindexer(patients(3));
		reindexer.queryStoreEnabled = true;
		reindexer.failEnsureForUuid = "uuid-2"; // bootstrap blows up on the 2nd patient

		reindexer.reindexAll();

		// Patient 2 fails at ensureIndexed but patient 3 is still fully processed —
		// one bad patient must not abort the whole reindex.
		assertEquals(
				Arrays.asList(
						"delete:uuid-1", "ensure:uuid-1",
						"delete:uuid-2", "ensure:uuid-2",
						"delete:uuid-3", "ensure:uuid-3"),
				reindexer.ops,
				"a failure on one patient must not skip the delete/ensure attempt on later patients");

		ReindexStatus status = reindexer.getStatus();
		assertEquals(3, status.getProcessed());
		assertEquals(2, status.getSucceeded());
		assertEquals(1, status.getFailed());
		assertFalse(status.isRunning());
	}

	@Test
	public void reindexAll_shouldPageThroughEveryPatient_acrossMultipleBatches() {
		// More than one BATCH_SIZE (50) page, with a partial final page, to pin the
		// offset-advancing loop: 120 = 50 + 50 + 20.
		TestableReindexer reindexer = new TestableReindexer(patients(120));
		reindexer.queryStoreEnabled = true;

		reindexer.reindexAll();

		assertEquals(240, reindexer.ops.size(), "2 querystore ops per patient × 120 patients");
		assertEquals("ensure:uuid-120", reindexer.ops.get(reindexer.ops.size() - 1),
				"the very last patient on the final partial page must be reached");
		assertTrue(reindexer.batchOffsets.contains(0), "first page requested at offset 0");
		assertTrue(reindexer.batchOffsets.contains(50), "second page requested at offset 50");
		assertTrue(reindexer.batchOffsets.contains(100), "third (partial) page requested at offset 100");

		ReindexStatus status = reindexer.getStatus();
		assertEquals(120, status.getProcessed());
		assertEquals(120, status.getSucceeded());
	}

	@Test
	public void reindexAll_shouldFlushAndClearSessionBetweenBatches() {
		TestableReindexer reindexer = new TestableReindexer(patients(120));
		reindexer.queryStoreEnabled = true;

		reindexer.reindexAll();

		// One flush/clear per page consumed (3 full+partial pages); bounds memory on
		// large patient sets exactly as the embedding backfill task does.
		assertEquals(3, reindexer.flushCount,
				"the Hibernate session must be flushed/cleared once per processed batch");
	}

	@Test
	public void reindexAll_shouldCallIndexPatientForEachPatient_whenQueryStoreDisabled() {
		TestableReindexer reindexer = new TestableReindexer(patients(2));
		reindexer.queryStoreEnabled = false;

		reindexer.reindexAll();

		assertEquals(Arrays.asList("index:1", "index:2"), reindexer.indexedPatientIds,
				"embedding backend must re-index each patient via EmbeddingIndexer.indexPatient");
		assertTrue(reindexer.ops.isEmpty(), "querystore must not be touched on the embedding backend");

		ReindexStatus status = reindexer.getStatus();
		assertEquals("embedding", status.getBackend());
		assertEquals(2, status.getProcessed());
		assertEquals(2, status.getSucceeded());
	}

	// --- Test double: overrides the external-boundary seams only ---

	private static final class TestableReindexer extends PatientIndexReindexer {

		final List<Patient> allPatients;

		boolean queryStoreEnabled;

		String failEnsureForUuid;

		final List<String> ops = new ArrayList<String>();

		final List<String> indexedPatientIds = new ArrayList<String>();

		final List<Integer> batchOffsets = new ArrayList<Integer>();

		int flushCount;

		boolean runAsyncInline;

		boolean runAsyncThrows;

		int runAsyncCalls;

		TestableReindexer(List<Patient> allPatients) {
			this.allPatients = allPatients;
		}

		@Override
		protected void runAsync(Runnable task, DaemonToken token) {
			runAsyncCalls++;
			if (runAsyncThrows) {
				throw new IllegalStateException("simulated daemon spawn failure");
			}
			if (runAsyncInline) {
				task.run(); // run the daemon body synchronously so the test can assert on it
			}
		}

		@Override
		protected boolean isQueryStoreEnabled() {
			return queryStoreEnabled;
		}

		@Override
		protected List<Patient> getPatientBatch(int offset, int batchSize) {
			batchOffsets.add(offset);
			if (offset >= allPatients.size()) {
				return new ArrayList<Patient>();
			}
			int end = Math.min(offset + batchSize, allPatients.size());
			return new ArrayList<Patient>(allPatients.subList(offset, end));
		}

		@Override
		protected void flushAndClear() {
			flushCount++;
		}

		@Override
		protected QueryStoreService resolveQueryStoreService() {
			return new RecordingQueryStore(ops);
		}

		@Override
		protected BootstrapService resolveBootstrapService() {
			return new RecordingBootstrap(ops, failEnsureForUuid);
		}

		@Override
		protected EmbeddingIndexer resolveEmbeddingIndexer() {
			return new RecordingEmbeddingIndexer(indexedPatientIds);
		}
	}

	private static final class RecordingEmbeddingIndexer extends EmbeddingIndexer {

		private final List<String> indexed;

		RecordingEmbeddingIndexer(List<String> indexed) {
			this.indexed = indexed;
		}

		@Override
		public void indexPatient(Patient patient) {
			indexed.add("index:" + patient.getPatientId());
		}
	}

	private static final class RecordingQueryStore implements QueryStoreService {

		private final List<String> ops;

		RecordingQueryStore(List<String> ops) {
			this.ops = ops;
		}

		@Override
		public void bulkDeleteByPatient(String patientUuid) {
			ops.add("delete:" + patientUuid);
		}

		@Override
		public List<QueryDocument> getPatientChart(String patientUuid) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public List<QueryDocument> searchByPatient(String patientUuid, String question, int topK) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public List<QueryDocument> search(String question, int topK) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public WriteResult index(QueryDocument doc) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public void delete(String resourceType, String resourceUuid) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public void onStartup() {
		}

		@Override
		public void onShutdown() {
		}
	}

	private static final class RecordingBootstrap implements BootstrapService {

		private final List<String> ops;

		private final String failForUuid;

		RecordingBootstrap(List<String> ops, String failForUuid) {
			this.ops = ops;
			this.failForUuid = failForUuid;
		}

		@Override
		public void ensureIndexed(String patientUuid) {
			ops.add("ensure:" + patientUuid);
			if (patientUuid.equals(failForUuid)) {
				throw new RuntimeException("simulated ensureIndexed failure for " + patientUuid);
			}
		}

		@Override
		public void reindexPatient(String patientUuid) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public void bootstrap() {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public void bootstrap(String resourceType) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public List<BootstrapProgress> getStatus() {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public BootstrapProgress getStatus(String resourceType) {
			throw new UnsupportedOperationException("not used by reindex");
		}

		@Override
		public void onStartup() {
		}

		@Override
		public void onShutdown() {
		}
	}
}
