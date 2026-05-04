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
import org.openmrs.module.chartsearchai.api.impl.ChartCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Static helper used by the AOP advice classes to drop a patient's cached chart
 * from {@link ChartCache} after their data changes. Kept separate from the cache
 * itself because the advice classes are instantiated by Spring AOP and use the
 * service-locator pattern ({@link Context#getRegisteredComponent}) instead of
 * field injection — this just avoids repeating the lookup-and-catch boilerplate
 * across three advice classes.
 */
public final class ChartCacheInvalidator {

	private static final Logger log = LoggerFactory.getLogger(ChartCacheInvalidator.class);

	private ChartCacheInvalidator() {
	}

	public static void invalidate(Patient patient) {
		if (patient == null || patient.getPatientId() == null) {
			return;
		}
		try {
			ChartCache cache = Context.getRegisteredComponent(
					"chartSearchAi.chartCache", ChartCache.class);
			cache.invalidate(patient);
		}
		catch (Exception e) {
			log.warn("Failed to invalidate chart cache for patient [id={}]",
					patient.getPatientId(), e);
		}
	}
}
