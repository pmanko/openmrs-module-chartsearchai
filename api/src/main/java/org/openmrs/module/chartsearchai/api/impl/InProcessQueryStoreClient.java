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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.openmrs.api.context.Context;
import org.openmrs.module.querystore.api.QueryStoreService;
import org.openmrs.module.querystore.model.QueryDocument;
import org.springframework.stereotype.Component;

/**
 * Default {@link QueryStoreClient}: resolves querystore's in-process {@code QueryStoreService} and maps
 * each {@code QueryDocument} to an embedding-free {@link QueryRecord}. Co-located with chartsearchai in
 * the OpenMRS JVM, the in-process path runs the {@code @Authorized} read methods under the clinician's
 * {@code UserContext} (preserving querystore ADR Decision 14's per-caller contract) and keeps the chart
 * bytes byte-identical — no JSON round-trip to drift the LLM prompt-cache (ADR Decision 16). The HTTP
 * impl, when added, slots in behind this same interface for the thin-client split.
 *
 * <p>The {@code QueryStoreService} is resolved lazily per call via {@link Context} (the
 * {@link #queryStore()} seam) rather than injected: this keeps the {@code QueryStoreService.class}
 * linkage lazy, so an availability/linkage failure surfaces to {@link QueryStoreChartBuilder}, which
 * degrades to an empty chart rather than breaking chart assembly.
 */
@Component("chartSearchAi.inProcessQueryStoreClient")
class InProcessQueryStoreClient implements QueryStoreClient {

	@Override
	public List<QueryRecord> getPatientChart(String patientUuid) {
		return map(queryStore().getPatientChart(patientUuid));
	}

	@Override
	public List<QueryRecord> searchByPatient(String patientUuid, String query, int limit) {
		return map(queryStore().searchByPatient(patientUuid, query, limit));
	}

	/** Seam for tests: production resolves querystore's read service via the OpenMRS context. */
	protected QueryStoreService queryStore() {
		return Context.getService(QueryStoreService.class);
	}

	/** Maps querystore documents to the embedding-free DTO, preserving list order and null positions
	 *  (QueryStoreChartBuilder's conversion loop skips null/malformed records and logs them). */
	private static List<QueryRecord> map(List<QueryDocument> docs) {
		if (docs == null || docs.isEmpty()) {
			return Collections.<QueryRecord>emptyList();
		}
		List<QueryRecord> out = new ArrayList<QueryRecord>(docs.size());
		for (QueryDocument doc : docs) {
			out.add(doc == null ? null : toRecord(doc));
		}
		return out;
	}

	private static QueryRecord toRecord(QueryDocument doc) {
		QueryRecord r = new QueryRecord();
		r.setResourceType(doc.getResourceType());
		r.setResourceUuid(doc.getResourceUuid());
		r.setDate(doc.getDate());
		r.setText(doc.getText());
		for (Map.Entry<String, Object> entry : doc.getMetadata().entrySet()) {
			r.putMetadata(entry.getKey(), entry.getValue());
		}
		return r;
	}
}
