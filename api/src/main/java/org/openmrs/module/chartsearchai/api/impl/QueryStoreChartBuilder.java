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
 */
@Component("chartSearchAi.queryStoreChartBuilder")
class QueryStoreChartBuilder {

	private static final Logger log = LoggerFactory.getLogger(QueryStoreChartBuilder.class);

	@Autowired
	private PatientChartSerializer chartSerializer;

	PatientChart build(Patient patient, String question) {
		if (patient == null || patient.getUuid() == null
				|| question == null || question.trim().isEmpty()) {
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		QueryStoreService queryStore;
		try {
			queryStore = Context.getService(QueryStoreService.class);
		}
		catch (APIException | LinkageError e) {
			// LinkageError covers NoClassDefFoundError when the querystore-api jar
			// is absent at runtime — the QueryStoreService.class literal forces
			// JVM linkage, which APIException doesn't catch.
			log.info("QueryStoreService not available (querystore omod installed?)");
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		List<QueryDocument> hits;
		try {
			hits = queryStore.searchByPatient(patient.getUuid(), question,
					PipelineSettings.getQueryStoreTopK());
		}
		catch (RuntimeException e) {
			log.error("QueryStore search failed for patient [uuid={}]", patient.getUuid(), e);
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		if (hits == null || hits.isEmpty()) {
			return chartSerializer.serialize(patient, Collections.<SerializedRecord>emptyList());
		}

		List<SerializedRecord> records = new ArrayList<SerializedRecord>(hits.size());
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
		return chartSerializer.serialize(patient, records);
	}
}
