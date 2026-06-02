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

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.openmrs.Patient;
import org.openmrs.api.context.Context;
import org.openmrs.api.context.Daemon;
import org.openmrs.module.DaemonToken;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.openmrs.module.chartsearchai.api.EmbeddingIndexer;
import org.openmrs.module.chartsearchai.api.ReindexStatus;
import org.openmrs.module.querystore.api.QueryStoreService;
import org.openmrs.module.querystore.bootstrap.BootstrapService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Reindexes every patient's chart into the active retrieval backend. Backs the
 * admin "index all" REST endpoint.
 *
 * <p>This is the manual, full-rebuild counterpart to the incremental indexing the
 * module normally does on data changes. It exists because a backend index can drift
 * from the live database when changes bypass the {@code saveObs}/{@code saveEncounter}
 * AOP bridge — most notably a SQL-dump load on the demo server, after which the
 * querystore is missing records that were inserted directly.
 *
 * <p>Backend dispatch:
 * <ul>
 *   <li><b>querystore</b> (when {@code chartsearchai.querystore.enabled=true}): for each
 *       patient, {@link QueryStoreService#bulkDeleteByPatient(String)} THEN
 *       {@link BootstrapService#ensureIndexed(String)}. The delete is mandatory and must
 *       come first: {@code ensureIndexed} early-returns on a patient that already has
 *       <em>any</em> document, so on its own it cannot heal a partially-indexed patient —
 *       exactly the drift this endpoint repairs. Deleting first forces a full
 *       re-projection from the current database. (These two long-standing primitives are
 *       used rather than the newer single-call {@code reindexPatient} so the endpoint
 *       works against querystore versions deployed before that method existed.)</li>
 *   <li><b>embedding</b> (otherwise): {@link EmbeddingIndexer#indexPatient(Patient)},
 *       which atomically replaces the patient's embeddings.</li>
 * </ul>
 *
 * <p>{@link #start()} runs the work on an OpenMRS daemon thread (so it has a session
 * and authentication for the DB reads) and returns immediately, because reindexing a
 * whole server can take minutes — far longer than an HTTP request should be held open.
 * Progress is published as an immutable {@link ReindexStatus} for the status endpoint
 * to poll. Patients are paged so a large census never loads into one session, and a
 * failure on one patient is counted and skipped rather than aborting the run.
 *
 * <p>The {@code protected resolve*}/{@code getPatientBatch}/{@code flushAndClear}/
 * {@code isQueryStoreEnabled} methods are test seams (mirroring
 * {@link QueryStoreChartBuilder}), not an extension point. Subclassing this bean
 * outside the test package is not supported.
 */
@Component("chartSearchAi.patientIndexReindexer")
public class PatientIndexReindexer {

	private static final Logger log = LoggerFactory.getLogger(PatientIndexReindexer.class);

	static final String BACKEND_QUERYSTORE = "querystore";

	static final String BACKEND_EMBEDDING = "embedding";

	private static final int BATCH_SIZE = 50;

	/** Guards against a second {@link #start()} while a daemon run is in flight. */
	private final AtomicBoolean running = new AtomicBoolean(false);

	private final AtomicReference<ReindexStatus> status =
			new AtomicReference<ReindexStatus>(ReindexStatus.idle());

	private volatile DaemonToken daemonToken;

	public void setDaemonToken(DaemonToken token) {
		this.daemonToken = token;
	}

	/** The latest published progress snapshot. */
	public ReindexStatus getStatus() {
		return status.get();
	}

	/**
	 * Kicks off a full reindex on a daemon thread and returns immediately.
	 *
	 * @return {@code true} if a run was started; {@code false} if one is already in flight
	 * @throws IllegalStateException if the module's {@link DaemonToken} is not yet available
	 */
	public boolean start() {
		if (!running.compareAndSet(false, true)) {
			return false;
		}
		DaemonToken token = daemonToken;
		if (token == null) {
			running.set(false);
			throw new IllegalStateException(
					"DaemonToken not available; the module is not fully started yet");
		}
		Runnable task = new Runnable() {

			@Override
			public void run() {
				try {
					reindexAll();
				}
				// Throwable, not Exception: when querystore is enabled but its module is
				// absent, resolving the service throws LinkageError/NoClassDefFoundError
				// (an Error), which an Exception catch would let escape to the daemon
				// thread's default handler unlogged.
				catch (Throwable t) {
					log.error("Reindex-all failed", t);
				}
				finally {
					running.set(false);
				}
			}
		};
		// The guard is reset inside the task's finally — but only if the task is actually
		// scheduled. If the spawn itself fails, reset here so the guard can't latch true
		// and wedge every future reindex at 409.
		try {
			runAsync(task, token);
		}
		catch (RuntimeException | Error e) {
			running.set(false);
			throw e;
		}
		return true;
	}

	/**
	 * Synchronously reindexes every patient into the active backend, publishing
	 * progress to {@link #getStatus()} as it goes. This is the work {@link #start()}
	 * runs on its daemon thread; it is package-visible so it can be driven directly
	 * by tests and (if ever needed) a scheduled task.
	 */
	void reindexAll() {
		boolean queryStore = isQueryStoreEnabled();
		String backend = queryStore ? BACKEND_QUERYSTORE : BACKEND_EMBEDDING;
		long startedAt = now();
		int processed = 0;
		int succeeded = 0;
		int failed = 0;

		// Publish the running snapshot before resolving the backend, so that a resolve
		// failure (e.g. querystore enabled but its module absent) still flips the status
		// to finished via the finally block rather than leaving it stuck at idle.
		status.set(new ReindexStatus(true, backend, 0, 0, 0, startedAt, null));
		log.info("Reindex-all started (backend={})", backend);

		try {
			QueryStoreService queryStoreService = null;
			BootstrapService bootstrapService = null;
			EmbeddingIndexer embeddingIndexer = null;
			if (queryStore) {
				queryStoreService = resolveQueryStoreService();
				bootstrapService = resolveBootstrapService();
			} else {
				embeddingIndexer = resolveEmbeddingIndexer();
			}

			int offset = 0;
			List<Patient> batch = getPatientBatch(offset, BATCH_SIZE);
			while (batch != null && !batch.isEmpty()) {
				for (Patient patient : batch) {
					try {
						if (queryStore) {
							String uuid = patient.getUuid();
							// Delete first, then re-project: ensureIndexed is a no-op on a
							// patient that still has any stale document, so the delete is what
							// forces a full re-projection from the current database.
							queryStoreService.bulkDeleteByPatient(uuid);
							bootstrapService.ensureIndexed(uuid);
						} else {
							embeddingIndexer.indexPatient(patient);
						}
						succeeded++;
					}
					catch (Exception e) {
						failed++;
						log.error("Reindex failed for patient [id={}]: {}",
								patient.getPatientId(), e.getMessage());
					}
					processed++;
					status.set(new ReindexStatus(true, backend, processed, succeeded, failed,
							startedAt, null));
				}
				flushAndClear();
				offset += BATCH_SIZE;
				batch = getPatientBatch(offset, BATCH_SIZE);
			}
			log.info("Reindex-all completed (backend={}): {} processed, {} succeeded, {} failed",
					backend, processed, succeeded, failed);
		}
		finally {
			status.set(new ReindexStatus(false, backend, processed, succeeded, failed,
					startedAt, now()));
		}
	}

	// --- Test seams: production reaches the OpenMRS context / external module ---

	/** Production runs the reindex on an OpenMRS daemon thread (which opens a session and
	 *  authenticates for the DB reads). Overridden in tests to run the task inline. */
	protected void runAsync(Runnable task, DaemonToken token) {
		Daemon.runInDaemonThread(task, token);
	}

	protected boolean isQueryStoreEnabled() {
		return ChartSearchAiUtils.isQueryStoreEnabled();
	}

	protected List<Patient> getPatientBatch(int offset, int batchSize) {
		return Context.getPatientService().getPatients(null, offset, batchSize);
	}

	protected void flushAndClear() {
		Context.flushSession();
		Context.clearSession();
	}

	protected QueryStoreService resolveQueryStoreService() {
		return Context.getService(QueryStoreService.class);
	}

	protected BootstrapService resolveBootstrapService() {
		return Context.getService(BootstrapService.class);
	}

	protected EmbeddingIndexer resolveEmbeddingIndexer() {
		return Context.getRegisteredComponent("embeddingIndexer", EmbeddingIndexer.class);
	}

	protected long now() {
		return System.currentTimeMillis();
	}
}
