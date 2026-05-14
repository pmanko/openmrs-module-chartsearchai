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

import org.openmrs.Patient;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Shared helper used by the AOP advice classes to re-index or delete
 * Lucene and Elasticsearch pipeline data after patient data changes.
 * Each indexer's {@code reindexIfActive}/{@code deleteIfActive} method
 * checks whether its pipeline is currently selected before doing any work.
 */
final class IndexingHelper {

	private static final Logger log = LoggerFactory.getLogger(IndexingHelper.class);

	/** When the querystore migration flag is on, querystore owns indexing
	 *  and chartsearchai's AOP + backfill become no-ops to avoid double
	 *  indexing. */
	static boolean isDisabledByQueryStore() {
		return ChartSearchAiUtils.isQueryStoreEnabled();
	}

	static void reindexOtherPipelines(Patient patient) {
		if (patient == null) {
			return;
		}
		try {
			LuceneIndexer luceneIndexer = Context.getRegisteredComponent(
					"luceneIndexer", LuceneIndexer.class);
			if (luceneIndexer != null) {
				luceneIndexer.reindexIfActive(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to re-index Lucene for patient [id={}]",
					patient.getPatientId(), e);
		}
		try {
			ElasticsearchIndexer esIndexer = Context.getRegisteredComponent(
					"elasticsearchIndexer", ElasticsearchIndexer.class);
			if (esIndexer != null) {
				esIndexer.reindexIfActive(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to re-index Elasticsearch for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	static void deleteOtherPipelineIndexes(Patient patient) {
		if (patient == null) {
			return;
		}
		try {
			LuceneIndexer luceneIndexer = Context.getRegisteredComponent(
					"luceneIndexer", LuceneIndexer.class);
			if (luceneIndexer != null) {
				luceneIndexer.deleteIfActive(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to delete Lucene index for patient [id={}]",
					patient.getPatientId(), e);
		}
		try {
			ElasticsearchIndexer esIndexer = Context.getRegisteredComponent(
					"elasticsearchIndexer", ElasticsearchIndexer.class);
			if (esIndexer != null) {
				esIndexer.deleteIfActive(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to delete Elasticsearch index for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	private IndexingHelper() {
	}
}
