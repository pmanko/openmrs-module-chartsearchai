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

import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.openmrs.Diagnosis;
import org.openmrs.Encounter;
import org.openmrs.Obs;
import org.openmrs.Patient;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.api.db.ChartSearchAiDAO;
import org.openmrs.module.chartsearchai.embedding.EmbeddingProvider;
import org.openmrs.module.chartsearchai.model.ChartEmbedding;
import org.openmrs.module.chartsearchai.serializer.DiagnosisTextSerializer;
import org.openmrs.module.chartsearchai.serializer.ObsTextSerializer;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

/**
 * Indexes patient clinical data by converting records to text via serializers, computing
 * vector embeddings, and persisting them to the {@code chartsearchai_embedding} table.
 * Supports both full patient re-indexing and incremental encounter indexing.
 */
@Component
@Transactional
public class EmbeddingIndexer {

	private static final Logger log = LoggerFactory.getLogger(EmbeddingIndexer.class);

	@Autowired
	private PatientRecordLoader recordLoader;

	@Autowired
	private ObsTextSerializer obsSerializer;

	@Autowired
	private DiagnosisTextSerializer diagnosisSerializer;

	@Autowired
	@Qualifier("chartSearchAi.embeddingProvider")
	private EmbeddingProvider embeddingProvider;

	@Autowired
	private ChartSearchAiDAO dao;

	/**
	 * Full index of a patient's chart. Deletes existing embeddings and re-indexes all
	 * clinical data. Used on first query or as a nightly batch job.
	 *
	 * @param patient the patient to index
	 */
	public void indexPatient(Patient patient) {
		log.info("Indexing patient [id={}]", patient.getPatientId());
		Date now = new Date();

		// Build all embeddings first so the patient is never without data
		List<SerializedRecord> records = recordLoader.loadAll(patient);
		List<ChartEmbedding> newEmbeddings = new java.util.ArrayList<ChartEmbedding>(records.size());
		int failed = 0;
		for (SerializedRecord record : records) {
			try {
				ChartEmbedding ce = new ChartEmbedding();
				ce.setPatient(patient);
				ce.setResourceType(record.getResourceType());
				ce.setResourceId(record.getResourceId());
				ce.setTextContent(record.getText());
				String embeddingText = ChartSearchAiConstants.getEmbeddingPrefix(
						record.getResourceType(), record.getText()) + record.getText();
				ce.setEmbeddingVector(embeddingProvider.embed(embeddingText));
				ce.setDateCreated(now);
				newEmbeddings.add(ce);
			}
			catch (Exception e) {
				failed++;
				log.warn("Failed to embed {} [id={}] for patient [id={}]: {}",
						record.getResourceType(), record.getResourceId(),
						patient.getPatientId(), e.getMessage());
			}
		}
		if (failed > 0) {
			log.warn("Skipped {} of {} records during indexing of patient [id={}]",
					failed, records.size(), patient.getPatientId());
		}

		// Delete old and insert new within the same transaction
		dao.deleteByPatient(patient);
		for (ChartEmbedding ce : newEmbeddings) {
			dao.saveChartEmbedding(ce);
		}

		log.info("Finished indexing patient [id={}] ({} of {} records)",
				patient.getPatientId(), newEmbeddings.size(), records.size());
	}

	/**
	 * Deletes all embeddings for a patient. Used after a patient merge to clean up
	 * the non-preferred patient's stale embeddings.
	 *
	 * @param patient the patient whose embeddings to delete
	 */
	public void deletePatientEmbeddings(Patient patient) {
		log.info("Deleting all embeddings for patient [id={}]", patient.getPatientId());
		dao.deleteByPatient(patient);
	}

	/**
	 * Incremental index of a single encounter. Upserts embeddings for the encounter's
	 * observations and diagnoses without re-indexing the entire patient.
	 *
	 * @param encounter the encounter to index
	 */
	public void indexEncounter(Encounter encounter) {
		Patient patient = encounter.getPatient();
		if (patient == null) {
			log.warn("Skipping indexing for encounter [id={}] — no patient", encounter.getEncounterId());
			return;
		}
		log.info("Incrementally indexing encounter [id={}] for patient [id={}]",
				encounter.getEncounterId(), patient.getPatientId());
		Date now = new Date();
		Set<String> seenText = new HashSet<String>();

		Set<Obs> allObs = encounter.getAllObs();
		if (allObs != null) {
			for (Obs obs : allObs) {
				if (obs.getObsGroup() != null) {
					continue;
				}
				String text = obsSerializer.toText(obs);
				if (text != null && !text.trim().isEmpty() && seenText.add(text)) {
					upsertEmbedding(patient, ChartSearchAiConstants.RESOURCE_TYPE_OBS, obs.getObsId(), text, now);
				}
			}
		}

		if (encounter.getDiagnoses() != null) {
			for (Diagnosis diagnosis : encounter.getDiagnoses()) {
				String text = diagnosisSerializer.toText(diagnosis);
				if (text != null && !text.trim().isEmpty() && seenText.add(text)) {
					upsertEmbedding(patient, ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS, diagnosis.getDiagnosisId(), text, now);
				}
			}
		}

		log.info("Finished indexing encounter [id={}]", encounter.getEncounterId());
	}

	private void saveEmbedding(Patient patient, String resourceType, Integer resourceId,
			String text, Date now) {
		ChartEmbedding ce = new ChartEmbedding();
		ce.setPatient(patient);
		ce.setResourceType(resourceType);
		ce.setResourceId(resourceId);
		ce.setTextContent(text);
		ce.setEmbeddingVector(embeddingProvider.embed(
				ChartSearchAiConstants.getEmbeddingPrefix(resourceType, text) + text));
		ce.setDateCreated(now);
		dao.saveChartEmbedding(ce);
	}

	private void upsertEmbedding(Patient patient, String resourceType, Integer resourceId,
			String text, Date now) {
		ChartEmbedding existing = dao.getByResource(resourceType, resourceId);
		if (existing != null) {
			existing.setTextContent(text);
			existing.setEmbeddingVector(embeddingProvider.embed(
					ChartSearchAiConstants.getEmbeddingPrefix(resourceType, text) + text));
			existing.setDateCreated(now);
			dao.saveChartEmbedding(existing);
		} else {
			saveEmbedding(patient, resourceType, resourceId, text, now);
		}
	}

	/**
	 * Extracts the first sentence from serialized record text (up to the first
	 * ". "). Retained for backward compatibility and testing; embeddings now use
	 * the full serialized text to capture dosing, severity, reactions, and other
	 * details beyond the opening sentence.
	 */
	static String firstSentence(String text) {
		if (text == null) {
			return "";
		}
		int idx = text.indexOf(". ");
		return idx > 0 ? text.substring(0, idx) : text;
	}

}
