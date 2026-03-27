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
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.openmrs.Allergy;
import org.openmrs.Condition;
import org.openmrs.Diagnosis;
import org.openmrs.MedicationDispense;
import org.openmrs.Obs;
import org.openmrs.Order;
import org.openmrs.Patient;
import org.openmrs.PatientProgram;
import org.openmrs.api.context.Context;
import org.openmrs.parameter.MedicationDispenseCriteria;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Loads and serializes all clinical records for a patient. Provides a single point for
 * record iteration logic shared by {@code PatientChartSerializer} and
 * {@code EmbeddingIndexer}.
 */
@Component
public class PatientRecordLoader {

	@Autowired
	private ObsTextSerializer obsSerializer;

	@Autowired
	private ConditionTextSerializer conditionSerializer;

	@Autowired
	private AllergyTextSerializer allergySerializer;

	@Autowired
	private OrderTextSerializer orderSerializer;

	@Autowired
	private DiagnosisTextSerializer diagnosisSerializer;

	@Autowired
	private PatientProgramTextSerializer programSerializer;

	@Autowired
	private MedicationDispenseTextSerializer medicationDispenseSerializer;

	/**
	 * Load all clinical records for a patient and serialize each to text.
	 *
	 * @param patient the patient whose records to load
	 * @return serialized records with resource type and ID for each
	 */
	public List<SerializedRecord> loadAll(Patient patient) {
		List<SerializedRecord> records = new ArrayList<SerializedRecord>();
		// Deduplicate by resource key (type:id) to avoid re-processing the same record,
		// but allow distinct records with identical text (e.g. two separate BP readings
		// that happen to have the same value).
		Set<String> seenKeys = new HashSet<String>();

		// Observations (top-level only — group members are inlined by serializer)
		for (Obs obs : Context.getObsService().getObservationsByPerson(patient)) {
			if (obs.getObsGroup() != null) {
				continue;
			}
			String text = obsSerializer.toText(obs);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_OBS, obs.getObsId(), seenKeys)) {
				Date date = obs.getObsDatetime() != null ? obs.getObsDatetime() : obs.getDateCreated();
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_OBS, obs.getObsId(), text, date));
			}
		}

		// Conditions
		for (Condition condition : Context.getConditionService().getAllConditions(patient)) {
			String text = conditionSerializer.toText(condition);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_CONDITION, condition.getConditionId(), seenKeys)) {
				Date date = condition.getOnsetDate() != null ? condition.getOnsetDate() : condition.getDateCreated();
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_CONDITION, condition.getConditionId(), text, date));
			}
		}

		// Allergies
		for (Allergy allergy : Context.getPatientService().getAllergies(patient)) {
			String text = allergySerializer.toText(allergy);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY, allergy.getAllergyId(), seenKeys)) {
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY, allergy.getAllergyId(), text, allergy.getDateCreated()));
			}
		}

		// Diagnoses
		for (Diagnosis diagnosis : Context.getDiagnosisService().getDiagnoses(patient, null)) {
			String text = diagnosisSerializer.toText(diagnosis);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS, diagnosis.getDiagnosisId(), seenKeys)) {
				Date date = diagnosis.getEncounter() != null
						? diagnosis.getEncounter().getEncounterDatetime() : diagnosis.getDateCreated();
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS, diagnosis.getDiagnosisId(), text, date));
			}
		}

		// Orders (skip voided — getAllOrdersByPatient includes voided orders)
		for (Order order : Context.getOrderService().getAllOrdersByPatient(patient)) {
			if (Boolean.TRUE.equals(order.getVoided())) {
				continue;
			}
			String text = orderSerializer.toText(order);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_ORDER, order.getOrderId(), seenKeys)) {
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_ORDER, order.getOrderId(), text, order.getDateActivated()));
			}
		}

		// Program enrollments
		for (PatientProgram pp : Context.getProgramWorkflowService()
				.getPatientPrograms(patient, null, null, null, null, null, false)) {
			String text = programSerializer.toText(pp);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM, pp.getPatientProgramId(), seenKeys)) {
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM, pp.getPatientProgramId(), text, pp.getDateEnrolled()));
			}
		}

		// Medication dispenses
		MedicationDispenseCriteria dispenseCriteria = new MedicationDispenseCriteria();
		dispenseCriteria.setPatient(patient);
		for (MedicationDispense dispense : Context.getMedicationDispenseService()
				.getMedicationDispenseByCriteria(dispenseCriteria)) {
			String text = medicationDispenseSerializer.toText(dispense);
			if (addIfValid(text, ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE,
					dispense.getMedicationDispenseId(), seenKeys)) {
				Date date = dispense.getDateHandedOver() != null
						? dispense.getDateHandedOver() : dispense.getDateCreated();
				records.add(new SerializedRecord(ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE,
						dispense.getMedicationDispenseId(), text, date));
			}
		}

		Collections.sort(records, Comparator.comparing(SerializedRecord::getDate,
				Comparator.nullsLast(Comparator.reverseOrder())));

		return records;
	}

	private static boolean addIfValid(String text, String resourceType, Integer resourceId,
			Set<String> seenKeys) {
		if (text == null || text.trim().isEmpty()) {
			return false;
		}
		return seenKeys.add(resourceType + ":" + resourceId);
	}

	/**
	 * A serialized clinical record with its resource type and ID.
	 */
	public static class SerializedRecord {

		private final String resourceType;

		private final Integer resourceId;

		private final String text;

		private final Date date;

		public SerializedRecord(String resourceType, Integer resourceId, String text, Date date) {
			this.resourceType = resourceType;
			this.resourceId = resourceId;
			this.text = text;
			this.date = date;
		}

		public String getResourceType() {
			return resourceType;
		}

		public Integer getResourceId() {
			return resourceId;
		}

		public String getText() {
			return text;
		}

		public Date getDate() {
			return date;
		}
	}
}
