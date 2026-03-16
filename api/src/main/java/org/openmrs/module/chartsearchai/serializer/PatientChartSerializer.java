/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.serializer;

import java.util.List;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.DateFormatUtil;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Serializes an entire patient chart into labeled records for direct LLM inference.
 * Each record is prefixed with a typed label using the resource ID (e.g. [Obs #456],
 * [Allergy #789]) so that citations in the LLM response directly identify the source
 * record in OpenMRS without a separate mapping.
 */
@Component
public class PatientChartSerializer {

	@Autowired
	private PatientRecordLoader recordLoader;

	/**
	 * Serialize all clinical records for a patient into labeled text lines.
	 *
	 * @param patient the patient whose chart to serialize
	 * @return the serialized chart with labeled records
	 */
	public PatientChart serialize(Patient patient) {
		List<SerializedRecord> records = recordLoader.loadAll(patient);
		StringBuilder sb = new StringBuilder();

		for (SerializedRecord record : records) {
			String displayType = toDisplayType(record.getResourceType());
			sb.append("[").append(displayType).append(" #")
					.append(record.getResourceId());
			if (record.getDate() != null) {
				sb.append(", ").append(DateFormatUtil.formatDate(record.getDate()));
			}
			sb.append("] ").append(record.getText()).append("\n");
		}

		return new PatientChart(sb.toString());
	}

	public static String toDisplayType(String resourceType) {
		if (resourceType == null || resourceType.isEmpty()) {
			return "Record";
		}
		return resourceType.substring(0, 1).toUpperCase() + resourceType.substring(1);
	}

	/**
	 * The serialized patient chart with labeled records.
	 */
	public static class PatientChart {

		private final String text;

		public PatientChart(String text) {
			this.text = text;
		}

		public String getText() {
			return text;
		}
	}
}
