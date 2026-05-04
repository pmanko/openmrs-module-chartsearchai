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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.springframework.stereotype.Component;

/**
 * Caches serialized {@link PatientChart} objects keyed by patient ID, so successive
 * chart-search queries on the same patient skip re-fetching and re-serializing the
 * full chart from the OpenMRS DB. The serialization itself costs ~300–500 ms on a
 * patient with a few hundred observations and is byte-identical across queries
 * (the LLM-side prompt-prefix cache already relies on this).
 *
 * <p>Bounded LRU. When the size exceeds {@link #MAX_ENTRIES}, the least-recently-used
 * entry is dropped.
 *
 * <p>Cache invalidation is event-driven, not TTL-based. The AOP advice classes
 * ({@code ObsIndexingAdvice}, {@code EncounterIndexingAdvice},
 * {@code PatientDataIndexingAdvice}) call {@link #invalidate(Patient)} after every
 * service method that mutates a patient's chart — same coverage the embedding
 * re-index path uses, so nothing missed there is missed here either.
 */
@Component("chartSearchAi.chartCache")
public class ChartCache {

	private static final int MAX_ENTRIES = 50;

	private final Map<Integer, PatientChart> cache = Collections.synchronizedMap(
			new LinkedHashMap<Integer, PatientChart>(MAX_ENTRIES, 0.75f, true) {

				private static final long serialVersionUID = 1L;

				@Override
				protected boolean removeEldestEntry(Map.Entry<Integer, PatientChart> eldest) {
					return size() > MAX_ENTRIES;
				}
			});

	public PatientChart get(Patient patient) {
		if (patient == null || patient.getPatientId() == null) {
			return null;
		}
		return cache.get(patient.getPatientId());
	}

	public void put(Patient patient, PatientChart chart) {
		if (patient == null || patient.getPatientId() == null || chart == null) {
			return;
		}
		cache.put(patient.getPatientId(), chart);
	}

	public void invalidate(Patient patient) {
		if (patient == null || patient.getPatientId() == null) {
			return;
		}
		cache.remove(patient.getPatientId());
	}

	public int size() {
		return cache.size();
	}

	public void clear() {
		cache.clear();
	}
}
