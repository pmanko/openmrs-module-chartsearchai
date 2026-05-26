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
 * Bridge to the querystore module's {@code searchByPatient} API. Resolves
 * the service lazily via {@link Context#getService(Class)} so chartsearchai
 * still starts when the querystore omod isn't installed — lookup failure
 * surfaces as an empty chart, same outcome as a search returning no hits.
 *
 * <p>The {@code protected resolve*} methods and the package-private
 * {@link #setChartSerializer} are test seams, not an extension point.
 * Subclassing this bean outside the test package is not supported.
 */
@Component("chartSearchAi.queryStoreChartBuilder")
class QueryStoreChartBuilder {

	private static final Logger log = LoggerFactory.getLogger(QueryStoreChartBuilder.class);

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
		if (patient == null || patient.getUuid() == null
				|| question == null || question.trim().isEmpty()) {
			// outcome=skipped — caller passed invalid inputs; no querystore call, no chart.
			// Logging this so operator-side chartBuildMs has a matching inner line.
			log.info("[timing] querystoreBuild patient={} hits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=skipped",
					patient == null ? null : patient.getPatientId(),
					System.currentTimeMillis() - buildStart);
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
			log.info("[timing] querystoreBuild patient={} hits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=unavailable",
					patient.getPatientId(), System.currentTimeMillis() - buildStart);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		String preprocessedQuestion = QueryPreprocessor.stripQueryStopwords(question);
		int topK = resolveQueryStoreTopK();
		long rpcStart = System.currentTimeMillis();
		List<QueryDocument> hits;
		try {
			hits = queryStore.searchByPatient(patient.getUuid(), preprocessedQuestion, topK);
		}
		catch (RuntimeException e) {
			log.error("QueryStore search failed for patient [uuid={}]", patient.getUuid(), e);
			long failMs = System.currentTimeMillis() - rpcStart;
			log.info("[timing] querystoreBuild patient={} hits=0 rpcMs={} serializeMs=0 totalMs={} outcome=error",
					patient.getPatientId(), failMs, System.currentTimeMillis() - buildStart);
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
		log.info("[timing] querystoreBuild patient={} hits={} rpcMs={} serializeMs={} totalMs={} outcome=ok",
				patient.getPatientId(), records.size(), rpcMs, serializeMs, totalMs);
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
}
