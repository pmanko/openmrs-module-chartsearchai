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

/**
 * chartsearchai's seam onto querystore's read API (querystore ADR Decision 16). The default
 * {@link InProcessQueryStoreClient} wraps querystore's in-process {@code QueryStoreService}; a future
 * HTTP implementation can target the {@code /ws/rest/v1/querystore/patientrecord} REST endpoint for the
 * thin-client split without {@code QueryStoreChartBuilder} changing. Returns {@link QueryRecord}s (the
 * embedding-free transport DTO), so chartsearchai is decoupled from querystore's {@code QueryDocument}.
 *
 * <p>Mirrors only the two read methods chartsearchai's chart build uses; resolution/availability
 * handling stays in {@link QueryStoreChartBuilder} (a failure degrades to an empty chart there).
 */
interface QueryStoreClient {

	/** The full patient chart (querystore {@code getPatientChart}), reverse-chronological, no ranking. */
	List<QueryRecord> getPatientChart(String patientUuid);

	/** A hybrid-ranked top-{@code limit} window for the focus hint (querystore {@code searchByPatient}). */
	List<QueryRecord> searchByPatient(String patientUuid, String query, int limit);
}
