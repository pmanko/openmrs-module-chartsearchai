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

import org.openmrs.Patient;
import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.openmrs.module.querystore.api.QueryStoreService;
import org.openmrs.module.querystore.model.QueryDocument;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Bridge to the querystore module's read API. Resolves the service lazily via
 * {@link Context#getService(Class)} so chartsearchai still starts when the
 * querystore omod isn't installed — lookup failure surfaces as an empty chart,
 * same outcome as a search returning no hits.
 *
 * <p>Branches on the {@code chartsearchai.embedding.preFilter} global property:
 * when {@code preFilter=true}, dispatches to
 * {@link QueryStoreService#searchByPatient(String, String, int)} for ranked,
 * question-conditioned retrieval; when {@code preFilter=false}, dispatches to
 * {@link QueryStoreService#getPatientChart(String)} — querystore Decision 15's
 * unfiltered full-chart enumeration. The full-chart path ignores the
 * {@code question} parameter (the LLM does its own reasoning over the whole
 * chart) so a blank/null question is no longer a short-circuit there.
 *
 * <p>The {@code protected resolve*} methods and the package-private
 * {@link #setChartSerializer} are test seams, not an extension point.
 * Subclassing this bean outside the test package is not supported.
 */
@Component("chartSearchAi.queryStoreChartBuilder")
class QueryStoreChartBuilder {

	private static final Logger log = LoggerFactory.getLogger(QueryStoreChartBuilder.class);

	// Mode labels emitted in the [timing] querystoreBuild log lines so ops dashboards can
	// distinguish the two dispatch shapes (their hits and rpcMs distributions differ by
	// 1-2 orders of magnitude). Kept as compile-time constants so a typo on any future log
	// line — there will be more sites as the slice's logging contract gets reused — surfaces
	// at compile time rather than as a silently-dropped grep.
	static final String MODE_PRE_FILTER = "preFilter";

	static final String MODE_FULL_CHART = "fullChart";

	// Mode label for the input-error path (null patient / null uuid), which fires BEFORE
	// resolveUsePreFilter() and so cannot honestly label the dispatch. Kept distinct from
	// the two real modes so dashboards bucket input errors separately.
	static final String MODE_UNKNOWN = "unknown";

	@Autowired
	private PatientChartSerializer chartSerializer;

	/** Test seam: production wires {@link PatientChartSerializer} via {@link Autowired}.
	 *  Package-private so {@code QueryStoreChartBuilderTest} can inject a real serializer
	 *  without bringing up Spring; matches the {@code resolveX()} method-override seam
	 *  pattern used for {@link QueryStoreService} and topK. */
	void setChartSerializer(PatientChartSerializer chartSerializer) {
		this.chartSerializer = chartSerializer;
	}

	PatientChart build(Patient patient, String question) {
		long buildStart = System.currentTimeMillis();
		// Hard-error guards apply in both modes: null patient or null uuid can't reach either
		// querystore method without NPE.
		if (patient == null || patient.getUuid() == null) {
			// mode=unknown — the dispatch isn't determined yet (resolveUsePreFilter() runs
			// below). Emitting an explicit label keeps the timing log shape uniform so a
			// dashboard grepping for mode= doesn't undercount input-error events.
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=skipped",
					patient == null ? null : patient.getPatientId(),
					MODE_UNKNOWN, System.currentTimeMillis() - buildStart);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		boolean usePreFilter = resolveUsePreFilter();
		// `mode` labels each [timing] log line below so operators can distinguish the two
		// dispatch shapes without correlating against the preFilter GP value at the time —
		// Decision 15's full-chart mode has materially different `hits` and `rpcMs`
		// distributions, and a single grep should be enough to tell the modes apart.
		String mode = usePreFilter ? MODE_PRE_FILTER : MODE_FULL_CHART;
		// Blank-question short-circuit only fires in preFilter mode: searchByPatient with an
		// empty query is spurious (no ranking signal, wasted RPC). The full-chart path
		// (getPatientChart) ignores the question entirely — a blank question still produces
		// the patient's full indexed chart, which is exactly what the LLM consumer wants.
		if (usePreFilter && (question == null || question.trim().isEmpty())) {
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=skipped",
					patient.getPatientId(), mode, System.currentTimeMillis() - buildStart);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		QueryStoreService queryStore;
		try {
			queryStore = resolveQueryStoreService();
		}
		catch (APIException | LinkageError e) {
			// LinkageError covers NoClassDefFoundError when the querystore-api jar
			// is absent at runtime — the QueryStoreService.class literal forces
			// JVM linkage, which APIException doesn't catch.
			// WARN (not INFO): default org.openmrs.* log level is WARN, and a misconfigured
			// querystore.enabled=true silently produces empty-chart LLM responses if this
			// fires. Operators need this to surface, with an actionable next step.
			log.warn("chartsearchai.querystore.enabled=true but QueryStoreService is unavailable — "
					+ "install openmrs-module-querystore or set chartsearchai.querystore.enabled=false. "
					+ "Returning empty chart.");
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=unavailable",
					patient.getPatientId(), mode, System.currentTimeMillis() - buildStart);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		long rpcStart = System.currentTimeMillis();
		List<QueryDocument> hits;
		try {
			if (usePreFilter) {
				String preprocessedQuestion = QueryPreprocessor.stripQueryStopwords(question);
				int topK = resolveQueryStoreTopK();
				hits = queryStore.searchByPatient(patient.getUuid(), preprocessedQuestion, topK);
			}
			else {
				// querystore Decision 15: unfiltered full-chart enumeration, ordered by
				// record_date desc with deterministic tie-break. Replaces the legacy
				// ChartCache/PatientRecordLoader full-chart path now that querystore serializes
				// every clinical record into per-type indices that the events-first sync keeps
				// current — no need for chartsearchai to maintain a parallel serialization
				// pipeline.
				hits = queryStore.getPatientChart(patient.getUuid());
			}
		}
		catch (RuntimeException e) {
			log.error("QueryStore retrieval failed for patient [uuid={}]", patient.getUuid(), e);
			long failMs = System.currentTimeMillis() - rpcStart;
			// `errorClass` discriminates between backend-side exceptions
			// (`IllegalStateException` thrown by the per-tier backends on RPC failure,
			// `APIException` from authorization or service-context issues) and code-bug
			// exceptions (`NullPointerException` from a malformed QueryDocument, etc.).
			// Without it, dashboards that bucket "why are charts empty?" cannot distinguish
			// "querystore backend is down" from "chartsearchai has a regression."
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 rpcMs={} serializeMs=0 totalMs={} outcome=error errorClass={}",
					patient.getPatientId(), mode, failMs, System.currentTimeMillis() - buildStart,
					e.getClass().getSimpleName());
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}
		long rpcMs = System.currentTimeMillis() - rpcStart;

		List<SerializedRecord> records;
		if (hits == null || hits.isEmpty()) {
			records = Collections.<SerializedRecord>emptyList();
		}
		else {
			records = new ArrayList<SerializedRecord>(hits.size());
			for (QueryDocument doc : hits) {
				if (doc == null) {
					log.warn("Skipping null QueryDocument");
					continue;
				}
				if (doc.getResourceType() == null || doc.getResourceUuid() == null) {
					log.warn("Skipping malformed QueryDocument: type={} uuid={}",
							doc.getResourceType(), doc.getResourceUuid());
					continue;
				}
				String text = doc.getText() == null ? "" : doc.getText();
				records.add(new SerializedRecord(doc.getResourceType(), doc.getResourceUuid(),
						text, DateFormatUtil.toLegacyDate(doc.getDate())));
			}
		}
		// Single timer + log site for both empty and populated hits: keeps the
		// empty-chart serialization cost visible (Jackson init, first-call warmup)
		// instead of bucketing it as zero. records.size() (not hits.size()) is the
		// count that actually reached the LLM — any null/malformed docs are logged
		// at WARN earlier in the loop and dropped from this count.
		long serializeStart = System.currentTimeMillis();
		PatientChart chart = chartSerializer.serialize(patient, records);
		long serializeMs = System.currentTimeMillis() - serializeStart;
		// totalMs spans the whole successful build (post-blank-guard), so it includes the
		// record-building loop time that rpcMs and serializeMs don't account for. The
		// caller's chartBuildMs in LlmInferenceService.search should approximately equal
		// this number; a large gap means time is going to query preprocessing or service
		// resolution, both pre-rpcStart.
		long totalMs = System.currentTimeMillis() - buildStart;
		log.info("[timing] querystoreBuild patient={} mode={} hits={} rpcMs={} serializeMs={} totalMs={} outcome=ok",
				patient.getPatientId(), mode, records.size(), rpcMs, serializeMs, totalMs);
		return chart;
	}

	/** Seam for tests: production resolves via the OpenMRS context. */
	protected QueryStoreService resolveQueryStoreService() {
		return Context.getService(QueryStoreService.class);
	}

	/** Seam for tests: production reads the global property. */
	protected int resolveQueryStoreTopK() {
		return PipelineSettings.getQueryStoreTopK();
	}

	/** Seam for tests: production reads the global property. */
	protected boolean resolveUsePreFilter() {
		return PipelineSettings.usePreFilter();
	}
}
