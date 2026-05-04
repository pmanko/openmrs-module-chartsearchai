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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.Collections;

import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;

public class ChartCacheTest {

	private static Patient patient(int id) {
		Patient p = new Patient();
		p.setPatientId(id);
		return p;
	}

	private static PatientChart chart(String text) {
		return new PatientChart(text, Collections.emptyList());
	}

	@Test
	public void get_shouldReturnNullForUncachedPatient() {
		ChartCache cache = new ChartCache();
		assertNull(cache.get(patient(1)));
	}

	@Test
	public void putThenGet_shouldReturnSameInstance() {
		ChartCache cache = new ChartCache();
		PatientChart c = chart("records");

		cache.put(patient(1), c);

		assertSame(c, cache.get(patient(1)),
				"cache must return the exact instance it was given so callers can rely on "
				+ "PatientChart#getMappings() identity for citation extraction");
	}

	@Test
	public void put_shouldKeyByPatientIdNotInstance() {
		ChartCache cache = new ChartCache();
		PatientChart c = chart("records");

		cache.put(patient(1), c);

		// Different Patient instance but same ID — Hibernate may produce different
		// instances across sessions; cache must treat them as the same patient.
		assertSame(c, cache.get(patient(1)));
	}

	@Test
	public void invalidate_shouldRemoveEntry() {
		ChartCache cache = new ChartCache();
		cache.put(patient(1), chart("records"));

		cache.invalidate(patient(1));

		assertNull(cache.get(patient(1)),
				"after invalidate the cache must not return the stale chart");
	}

	@Test
	public void put_shouldKeepDifferentPatientsIndependent() {
		ChartCache cache = new ChartCache();
		PatientChart a = chart("a");
		PatientChart b = chart("b");

		cache.put(patient(1), a);
		cache.put(patient(2), b);

		assertSame(a, cache.get(patient(1)));
		assertSame(b, cache.get(patient(2)));
	}

	@Test
	public void invalidate_shouldNotAffectOtherPatients() {
		ChartCache cache = new ChartCache();
		PatientChart a = chart("a");
		PatientChart b = chart("b");
		cache.put(patient(1), a);
		cache.put(patient(2), b);

		cache.invalidate(patient(1));

		assertNull(cache.get(patient(1)));
		assertSame(b, cache.get(patient(2)),
				"invalidating one patient must not evict another");
	}

	@Test
	public void put_shouldEvictOldestWhenOverLimit() {
		ChartCache cache = new ChartCache();
		// Fill beyond MAX_ENTRIES (50).
		for (int i = 1; i <= 60; i++) {
			cache.put(patient(i), chart("p" + i));
		}

		assertEquals(50, cache.size(),
				"cache must enforce its bound — unbounded growth would leak memory in long-running "
				+ "deployments where many patients are queried");
		assertNull(cache.get(patient(1)),
				"oldest entry must be evicted when capacity is exceeded");
		assertNotNull(cache.get(patient(60)),
				"newest entry must remain");
	}

	@Test
	public void get_shouldRefreshLruRecency() {
		ChartCache cache = new ChartCache();
		for (int i = 1; i <= 50; i++) {
			cache.put(patient(i), chart("p" + i));
		}

		// Touch patient 1 — moves it to most-recently-used.
		cache.get(patient(1));

		// Add one more, forcing one eviction.
		cache.put(patient(51), chart("p51"));

		assertNotNull(cache.get(patient(1)),
				"recently-accessed entries must survive eviction; otherwise active patients "
				+ "would get evicted while idle ones linger");
		assertNull(cache.get(patient(2)),
				"the actually-oldest entry should have been evicted");
	}

	@Test
	public void put_shouldSilentlyIgnoreNullPatient() {
		ChartCache cache = new ChartCache();
		cache.put(null, chart("x"));
		assertEquals(0, cache.size());
	}

	@Test
	public void put_shouldSilentlyIgnoreNullChart() {
		ChartCache cache = new ChartCache();
		cache.put(patient(1), null);
		assertNull(cache.get(patient(1)));
	}

	@Test
	public void put_shouldSilentlyIgnorePatientWithoutId() {
		ChartCache cache = new ChartCache();
		Patient transientPatient = new Patient();
		cache.put(transientPatient, chart("x"));
		assertEquals(0, cache.size());
	}

	@Test
	public void clear_shouldEmptyCache() {
		ChartCache cache = new ChartCache();
		cache.put(patient(1), chart("a"));
		cache.put(patient(2), chart("b"));

		cache.clear();

		assertEquals(0, cache.size());
		assertNull(cache.get(patient(1)));
	}
}
