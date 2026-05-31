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

import java.lang.reflect.Method;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.openmrs.Patient;
import org.openmrs.api.APIException;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.openmrs.module.chartsearchai.util.DateFormatUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Bridge to the querystore module's read API. The service and its
 * {@code QueryDocument} payloads are resolved reflectively (via
 * {@link Class#forName(String)} + {@link Method#invoke}) so chartsearchai
 * compiles and starts without the querystore-api jar on the classpath; absence
 * surfaces as an empty chart, same outcome as a search returning no hits.
 *
 * <p>Always fetches the full patient chart via the querystore
 * {@code getPatientChart(String)} method so the chart bytes sent to the LLM are
 * a function of the patient only — that's the property llama-server's KV-cache
 * reuse needs in order to skip ~99% of the prefill on subsequent queries for the
 * same patient. When {@code chartsearchai.embedding.preFilter=true} and the
 * question is non-blank, additionally calls {@code searchByPatient(String,
 * String, int)} to obtain a relevance ranking, then renders those hits as a
 * short "Records ranked by similarity to the query: ..." focus-hint line in the
 * LLM prompt (handled in {@code LlmProvider.buildUserMessage} via the
 * {@link PatientChart#getFocusIndices()} payload). The hint biases the LLM's
 * attention without removing records the LLM needs for negative reasoning.
 *
 * <p>The {@code protected resolve*} methods and the package-private
 * {@link #setChartSerializer} are test seams, not an extension point. The
 * service seam returns {@link Object} because chartsearchai has no compile-time
 * dependency on {@code QueryStoreService}; tests that stub a typed service rely
 * on reflective dispatch finding the stubbed methods. Subclassing this bean
 * outside the test package is not supported.
 */
@Component("chartSearchAi.queryStoreChartBuilder")
class QueryStoreChartBuilder {

	private static final Logger log = LoggerFactory.getLogger(QueryStoreChartBuilder.class);

	private static final String QUERY_STORE_SERVICE_CLASS = "org.openmrs.module.querystore.api.QueryStoreService";

	private static final String QUERY_DOCUMENT_CLASS = "org.openmrs.module.querystore.model.QueryDocument";

	// Mode labels emitted in the [timing] querystoreBuild log lines so ops dashboards can
	// distinguish the two dispatch shapes. preFilter mode does the extra searchByPatient
	// call for the focus hint; fullChart skips it. Kept as compile-time constants so a typo
	// on any future log line surfaces at compile time rather than as a silently-dropped grep.
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
	 *  pattern used for the querystore service and topK. */
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
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 focusHits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=skipped",
					patient == null ? null : patient.getPatientId(),
					MODE_UNKNOWN, System.currentTimeMillis() - buildStart);
			return emptyChart(patient);
		}

		boolean usePreFilter = resolveUsePreFilter();
		// `mode` labels each [timing] log line so operators can tell focus-hint preFilter
		// dispatch (extra searchByPatient call) from plain fullChart dispatch.
		String mode = usePreFilter ? MODE_PRE_FILTER : MODE_FULL_CHART;

		Class<?> serviceClass;
		try {
			serviceClass = Class.forName(QUERY_STORE_SERVICE_CLASS);
		}
		catch (ClassNotFoundException e) {
			log.info("QueryStoreService class not on classpath (querystore omod not installed)");
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 focusHits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=unavailable",
					patient.getPatientId(), mode, System.currentTimeMillis() - buildStart);
			return emptyChart(patient);
		}

		Object queryStore;
		try {
			queryStore = resolveQueryStoreService();
		}
		catch (APIException | LinkageError e) {
			// LinkageError covers NoClassDefFoundError when the querystore-api jar is absent at
			// runtime — resolving the service can force JVM linkage that APIException doesn't catch.
			// WARN (not INFO): default org.openmrs.* log level is WARN, and a misconfigured
			// querystore.enabled=true silently produces empty-chart LLM responses if this
			// fires. Operators need this to surface, with an actionable next step.
			log.warn("chartsearchai.querystore.enabled=true but QueryStoreService is unavailable — "
					+ "install openmrs-module-querystore or set chartsearchai.querystore.enabled=false. "
					+ "Returning empty chart.");
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 focusHits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=unavailable",
					patient.getPatientId(), mode, System.currentTimeMillis() - buildStart);
			return emptyChart(patient);
		}

		// Resolve the QueryDocument accessors once for this build; absence of the querystore-api
		// jar (shape drift) surfaces as an empty chart, same as a search returning no hits.
		QueryDocumentAccess access;
		try {
			access = QueryDocumentAccess.resolve();
		}
		catch (ReflectiveOperationException e) {
			log.error("QueryDocument accessors not resolvable (querystore-api shape changed?)", e);
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 focusHits=0 rpcMs=0 serializeMs=0 totalMs={} outcome=unavailable",
					patient.getPatientId(), mode, System.currentTimeMillis() - buildStart);
			return emptyChart(patient);
		}

		// Full chart first — this is what the LLM sees and what determines the KV-cache
		// prefix. Always called regardless of mode so the chart bytes are a function of
		// the patient only.
		long rpcStart = System.currentTimeMillis();
		List<?> chartDocs;
		try {
			Method getPatientChart = serviceClass.getMethod("getPatientChart", String.class);
			chartDocs = (List<?>) getPatientChart.invoke(queryStore, patient.getUuid());
		}
		catch (ReflectiveOperationException | RuntimeException e) {
			log.error("QueryStore.getPatientChart failed for patient [uuid={}]", patient.getUuid(), e);
			long failMs = System.currentTimeMillis() - rpcStart;
			log.info("[timing] querystoreBuild patient={} mode={} hits=0 focusHits=0 rpcMs={} serializeMs=0 totalMs={} outcome=error errorClass={}",
					patient.getPatientId(), mode, failMs, System.currentTimeMillis() - buildStart,
					e.getClass().getSimpleName());
			return emptyChart(patient);
		}

		// Focus hint: only in preFilter mode, only with a non-blank question (searchByPatient
		// with a blank query is spurious — no ranking signal). The hint is a tiny payload
		// (UUIDs collected here, rendered as 1-based indices in the chart serializer).
		Set<String> focusUuids = Collections.<String>emptySet();
		int focusHits = 0;
		if (usePreFilter && question != null && !question.trim().isEmpty()) {
			try {
				String preprocessedQuestion = QueryPreprocessor.stripQueryStopwords(question);
				int topK = resolveQueryStoreTopK();
				Method searchByPatient = serviceClass.getMethod(
						"searchByPatient", String.class, String.class, int.class);
				List<?> hits = (List<?>) searchByPatient.invoke(queryStore,
						patient.getUuid(), preprocessedQuestion, topK);
				focusUuids = collectFocusUuids(hits, access);
				focusHits = focusUuids.size();
			}
			catch (ReflectiveOperationException | RuntimeException e) {
				// Focus-hint failure must not block the LLM call — the full chart is already
				// fetched and is usable on its own (equivalent to fullChart mode). Log + fall
				// through with empty focus.
				log.warn("QueryStore.searchByPatient failed for patient [uuid={}] — proceeding without focus hint",
						patient.getUuid(), e);
			}
		}
		long rpcMs = System.currentTimeMillis() - rpcStart;

		List<SerializedRecord> records = toSerializedRecords(chartDocs, access);
		long serializeStart = System.currentTimeMillis();
		PatientChart chart = chartSerializer.serialize(patient, records, focusUuids);
		long serializeMs = System.currentTimeMillis() - serializeStart;
		long totalMs = System.currentTimeMillis() - buildStart;
		log.info("[timing] querystoreBuild patient={} mode={} hits={} focusHits={} rpcMs={} serializeMs={} totalMs={} outcome=ok",
				patient.getPatientId(), mode, records.size(), focusHits, rpcMs, serializeMs, totalMs);
		return chart;
	}

	/** Collects {@code resource_uuid}s from a hit list, skipping nulls and malformed docs.
	 *  These uuids are mapped to 1-based chart indices in {@code PatientChartSerializer.serialize}
	 *  to render the LLM-facing focus hint. */
	private Set<String> collectFocusUuids(List<?> hits, QueryDocumentAccess access) {
		if (hits == null || hits.isEmpty()) {
			return Collections.<String>emptySet();
		}
		Set<String> uuids = new HashSet<String>(hits.size() * 2);
		for (Object doc : hits) {
			if (doc == null) {
				continue;
			}
			String uuid = access.readResourceUuid(doc, log);
			if (uuid != null) {
				uuids.add(uuid);
			}
		}
		return uuids;
	}

	/** Converts a querystore hit list into the chartsearchai serializer's input shape,
	 *  dropping null and malformed docs with a WARN so operators can spot upstream
	 *  serialization regressions without losing the rest of the chart. */
	private List<SerializedRecord> toSerializedRecords(List<?> docs, QueryDocumentAccess access) {
		if (docs == null || docs.isEmpty()) {
			return Collections.<SerializedRecord>emptyList();
		}
		List<SerializedRecord> out = new ArrayList<SerializedRecord>(docs.size());
		for (Object doc : docs) {
			if (doc == null) {
				log.warn("Skipping null QueryDocument");
				continue;
			}
			SerializedRecord record = access.toSerializedRecord(doc, log);
			if (record != null) {
				out.add(record);
			}
		}
		return out;
	}

	/** Seam for tests: production resolves the querystore service via the OpenMRS context.
	 *  Returns {@link Object} (not {@code QueryStoreService}) because chartsearchai has no
	 *  compile-time dependency on the querystore-api jar; the service is dispatched
	 *  reflectively in {@link #build}. */
	protected Object resolveQueryStoreService() throws APIException, LinkageError {
		try {
			return Context.getService(Class.forName(QUERY_STORE_SERVICE_CLASS));
		}
		catch (ClassNotFoundException e) {
			throw new APIException("QueryStoreService class not on classpath", e);
		}
	}

	/** Seam for tests: production reads the global property. */
	protected int resolveQueryStoreTopK() {
		return PipelineSettings.getQueryStoreTopK();
	}

	/** Seam for tests: production reads the global property. */
	protected boolean resolveUsePreFilter() {
		return PipelineSettings.usePreFilter();
	}

	private PatientChart emptyChart(Patient patient) {
		return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
	}

	/**
	 * Cached reflective accessors over {@code QueryDocument} methods. Looked
	 * up once per build invocation (the resolution result isn't cached at
	 * class level since the querystore-api jar may not be on the classpath
	 * at chartsearchai start-up).
	 */
	private static final class QueryDocumentAccess {

		private final Method getResourceType;

		private final Method getResourceUuid;

		private final Method getText;

		private final Method getDate;

		private QueryDocumentAccess(Method getResourceType, Method getResourceUuid,
				Method getText, Method getDate) {
			this.getResourceType = getResourceType;
			this.getResourceUuid = getResourceUuid;
			this.getText = getText;
			this.getDate = getDate;
		}

		static QueryDocumentAccess resolve() throws ReflectiveOperationException {
			Class<?> docClass = Class.forName(QUERY_DOCUMENT_CLASS);
			return new QueryDocumentAccess(
					docClass.getMethod("getResourceType"),
					docClass.getMethod("getResourceUuid"),
					docClass.getMethod("getText"),
					docClass.getMethod("getDate"));
		}

		/** Reads just the resource uuid (used by the focus-hint collector); null on a
		 *  reflective failure or a malformed doc with no uuid. */
		String readResourceUuid(Object doc, Logger log) {
			try {
				return (String) getResourceUuid.invoke(doc);
			}
			catch (ReflectiveOperationException e) {
				log.warn("Failed to read QueryDocument resourceUuid for focus hint", e);
				return null;
			}
		}

		SerializedRecord toSerializedRecord(Object doc, Logger log) {
			try {
				String resourceType = (String) getResourceType.invoke(doc);
				String resourceUuid = (String) getResourceUuid.invoke(doc);
				if (resourceType == null || resourceUuid == null) {
					log.warn("Skipping malformed QueryDocument: type={} uuid={}",
							resourceType, resourceUuid);
					return null;
				}
				Object textObj = getText.invoke(doc);
				String text = textObj == null ? "" : (String) textObj;
				LocalDate date = (LocalDate) getDate.invoke(doc);
				return new SerializedRecord(resourceType, resourceUuid, text,
						DateFormatUtil.toLegacyDate(date));
			}
			catch (ReflectiveOperationException e) {
				log.error("Failed to read QueryDocument fields", e);
				return null;
			}
		}
	}
}
