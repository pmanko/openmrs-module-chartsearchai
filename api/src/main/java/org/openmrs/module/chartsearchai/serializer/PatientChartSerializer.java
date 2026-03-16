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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Serializes an entire patient chart into numbered records for direct LLM inference.
 * Each record is prefixed with a sequential number (e.g. [1]) to minimize
 * token usage. The mapping from number back to resource type and ID is returned
 * alongside the text.
 */
@Component
public class PatientChartSerializer {

	@Autowired
	private PatientRecordLoader recordLoader;

	/**
	 * Serialize all clinical records for a patient into numbered text lines.
	 *
	 * @param patient the patient whose chart to serialize
	 * @return the serialized chart with numbered records and index mapping
	 */
	public PatientChart serialize(Patient patient) {
		return serialize(recordLoader.loadAll(patient));
	}

	/**
	 * Serialize a pre-filtered list of records into numbered text lines.
	 *
	 * @param records the records to serialize
	 * @return the serialized chart with numbered records and index mapping
	 */
	public PatientChart serialize(List<SerializedRecord> records) {
		StringBuilder sb = new StringBuilder();
		List<RecordMapping> mappings = new ArrayList<RecordMapping>();

		for (int i = 0; i < records.size(); i++) {
			SerializedRecord record = records.get(i);
			int index = i + 1;
			mappings.add(new RecordMapping(index, record.getResourceType(), record.getResourceId(), record.getDate()));

			sb.append("[").append(index).append("] ").append(record.getText()).append("\n");
		}

		return new PatientChart(sb.toString(), Collections.unmodifiableList(mappings));
	}

	/**
	 * The serialized patient chart with numbered records and index mapping.
	 */
	public static class PatientChart {

		private final String text;

		private final List<RecordMapping> mappings;

		public PatientChart(String text, List<RecordMapping> mappings) {
			this.text = text;
			this.mappings = mappings;
		}

		public String getText() {
			return text;
		}

		public List<RecordMapping> getMappings() {
			return mappings;
		}
	}

	/**
	 * Maps a sequential index used in the LLM prompt back to the OpenMRS resource.
	 */
	public static class RecordMapping {

		private final int index;

		private final String resourceType;

		private final Integer resourceId;

		private final Date date;

		public RecordMapping(int index, String resourceType, Integer resourceId, Date date) {
			this.index = index;
			this.resourceType = resourceType;
			this.resourceId = resourceId;
			this.date = date;
		}

		public int getIndex() {
			return index;
		}

		public String getResourceType() {
			return resourceType;
		}

		public Integer getResourceId() {
			return resourceId;
		}

		public Date getDate() {
			return date;
		}
	}
}
