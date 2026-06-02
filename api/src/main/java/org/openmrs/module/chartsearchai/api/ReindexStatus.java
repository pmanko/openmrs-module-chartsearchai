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

/**
 * Immutable snapshot of an "index all patients" run. Published by
 * {@code PatientIndexReindexer} and surfaced by the REST status endpoint so an
 * administrator can poll progress of a background reindex without holding an
 * HTTP request open for the whole job.
 *
 * <p>Counts are cumulative for the current (or most recent) run: {@code processed}
 * is every patient attempted, {@code succeeded} + {@code failed} = {@code processed}.
 * {@code startedAt} / {@code finishedAt} are epoch milliseconds; {@code finishedAt}
 * is {@code null} while a run is in flight and {@code startedAt} is {@code null}
 * before the first run.
 */
public final class ReindexStatus {

	private final boolean running;

	private final String backend;

	private final int processed;

	private final int succeeded;

	private final int failed;

	private final Long startedAt;

	private final Long finishedAt;

	public ReindexStatus(boolean running, String backend, int processed, int succeeded,
			int failed, Long startedAt, Long finishedAt) {
		this.running = running;
		this.backend = backend;
		this.processed = processed;
		this.succeeded = succeeded;
		this.failed = failed;
		this.startedAt = startedAt;
		this.finishedAt = finishedAt;
	}

	/** The state before any reindex has been triggered. */
	public static ReindexStatus idle() {
		return new ReindexStatus(false, null, 0, 0, 0, null, null);
	}

	public boolean isRunning() {
		return running;
	}

	/** "querystore" or "embedding" — the backend the run targeted, or {@code null} when idle. */
	public String getBackend() {
		return backend;
	}

	public int getProcessed() {
		return processed;
	}

	public int getSucceeded() {
		return succeeded;
	}

	public int getFailed() {
		return failed;
	}

	public Long getStartedAt() {
		return startedAt;
	}

	public Long getFinishedAt() {
		return finishedAt;
	}
}
