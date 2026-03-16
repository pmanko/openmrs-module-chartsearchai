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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.GregorianCalendar;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.springframework.test.util.ReflectionTestUtils;

public class PatientChartSerializerTest {

	@Test
	public void serialize_shouldLabelRecordsWithResourceId() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 101, "Blood pressure 120/80", null),
				new SerializedRecord("obs", 102, "Weight 72 kg", null));

		PatientChart chart = serializer.serialize(new Patient());

		assertTrue(chart.getText().contains("[Obs #101] Blood pressure 120/80\n"));
		assertTrue(chart.getText().contains("[Obs #102] Weight 72 kg\n"));
	}

	@Test
	public void serialize_shouldLabelDifferentTypesWithResourceId() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 456, "Blood pressure", null),
				new SerializedRecord("order", 201, "Metformin", null));

		PatientChart chart = serializer.serialize(new Patient());

		assertTrue(chart.getText().contains("[Obs #456] Blood pressure\n"));
		assertTrue(chart.getText().contains("[Order #201] Metformin\n"));
	}

	@Test
	public void serialize_shouldIncludeDateInLabelWhenPresent() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("condition", 191, "Female infertility",
						new GregorianCalendar(2024, 0, 15).getTime()));

		PatientChart chart = serializer.serialize(new Patient());

		assertTrue(chart.getText().contains("[Condition #191, 2024-01-15] Female infertility\n"));
	}

	@Test
	public void serialize_shouldReturnEmptyChartForPatientWithNoRecords() {
		PatientChartSerializer serializer = createSerializer();

		PatientChart chart = serializer.serialize(new Patient());

		assertEquals("", chart.getText());
	}

	private PatientChartSerializer createSerializer(final SerializedRecord... records) {
		PatientRecordLoader stubLoader = new PatientRecordLoader() {
			@Override
			public List<SerializedRecord> loadAll(Patient patient) {
				List<SerializedRecord> list = new ArrayList<SerializedRecord>();
				for (SerializedRecord r : records) {
					list.add(r);
				}
				return list;
			}
		};

		PatientChartSerializer serializer = new PatientChartSerializer();
		ReflectionTestUtils.setField(serializer, "recordLoader", stubLoader);
		return serializer;
	}
}
