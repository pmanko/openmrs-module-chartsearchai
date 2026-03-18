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
import java.util.Calendar;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.PatientChart;
import org.openmrs.module.chartsearchai.serializer.PatientChartSerializer.RecordMapping;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.springframework.test.util.ReflectionTestUtils;

public class PatientChartSerializerTest {

	@Test
	public void serialize_shouldLabelRecordsWithSequentialNumbers() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 101, "Blood pressure 120/80", null),
				new SerializedRecord("obs", 102, "Weight 72 kg", null));

		PatientChart chart = serializer.serialize(new Patient());

		assertTrue(chart.getText().contains("[1] Blood pressure 120/80\n"));
		assertTrue(chart.getText().contains("[2] Weight 72 kg\n"));
	}

	@Test
	public void serialize_shouldReturnMappingsFromIndexToResource() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 456, "Blood pressure", null),
				new SerializedRecord("order", 201, "Metformin", null));

		PatientChart chart = serializer.serialize(new Patient());

		List<RecordMapping> mappings = chart.getMappings();
		assertEquals(2, mappings.size());
		assertEquals(1, mappings.get(0).getIndex());
		assertEquals("obs", mappings.get(0).getResourceType());
		assertEquals(Integer.valueOf(456), mappings.get(0).getResourceId());
		assertEquals(2, mappings.get(1).getIndex());
		assertEquals("order", mappings.get(1).getResourceType());
		assertEquals(Integer.valueOf(201), mappings.get(1).getResourceId());
	}

	@Test
	public void serialize_shouldIncludeDemographicsHeader() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 101, "Blood pressure 120/80", null));

		Patient patient = new Patient();
		patient.setGender("M");
		Calendar cal = Calendar.getInstance();
		cal.add(Calendar.YEAR, -45);
		patient.setBirthdate(cal.getTime());

		PatientChart chart = serializer.serialize(patient);

		assertTrue(chart.getText().startsWith("Patient: 45-year-old Male\n\n"));
		assertTrue(chart.getText().contains("[1] Blood pressure 120/80\n"));
	}

	@Test
	public void serialize_shouldIncludeGenderOnlyWhenAgeUnknown() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 101, "Weight 72 kg", null));

		Patient patient = new Patient();
		patient.setGender("F");

		PatientChart chart = serializer.serialize(patient);

		assertTrue(chart.getText().startsWith("Patient: Female\n\n"));
	}

	@Test
	public void serialize_shouldOmitDemographicsWhenBothNull() {
		PatientChartSerializer serializer = createSerializer(
				new SerializedRecord("obs", 101, "Weight 72 kg", null));

		PatientChart chart = serializer.serialize(new Patient());

		assertTrue(chart.getText().startsWith("[1]"));
	}

	@Test
	public void serialize_shouldReturnEmptyChartForPatientWithNoRecords() {
		PatientChartSerializer serializer = createSerializer();

		PatientChart chart = serializer.serialize(new Patient());

		assertEquals("", chart.getText());
		assertTrue(chart.getMappings().isEmpty());
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
