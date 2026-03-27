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

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.openmrs.Allergy;
import org.openmrs.Condition;
import org.openmrs.Diagnosis;
import org.openmrs.MedicationDispense;
import org.openmrs.Order;
import org.openmrs.Patient;
import org.openmrs.PatientProgram;
import org.openmrs.api.context.Context;
import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.aop.AfterReturningAdvice;

/**
 * AOP advice that triggers a full patient re-index when conditions, diagnoses, allergies,
 * or orders are modified outside of encounter saves. Also handles patient merges by
 * re-indexing the preferred patient and deleting the non-preferred patient's embeddings.
 * Complements the {@link EncounterIndexingAdvice} which handles the more common encounter path.
 *
 * <p>Only active when {@code chartsearchai.embedding.preFilter} is {@code true}. Registered
 * in config.xml as advice on {@code ConditionService}, {@code DiagnosisService},
 * {@code PatientService}, {@code OrderService}, {@code ProgramWorkflowService},
 * and {@code MedicationDispenseService}.</p>
 */
public class PatientDataIndexingAdvice implements AfterReturningAdvice {

	private static final Logger log = LoggerFactory.getLogger(PatientDataIndexingAdvice.class);

	private static final Set<String> CONDITION_METHODS = new HashSet<String>(
			Arrays.asList("saveCondition", "voidCondition", "unvoidCondition", "purgeCondition"));

	private static final Set<String> DIAGNOSIS_METHODS = new HashSet<String>(
			Arrays.asList("save", "voidDiagnosis", "unvoidDiagnosis", "purgeDiagnosis"));

	private static final Set<String> ALLERGY_METHODS = new HashSet<String>(
			Arrays.asList("saveAllergy", "setAllergies", "removeAllergy", "voidAllergy"));

	private static final Set<String> ORDER_METHODS = new HashSet<String>(
			Arrays.asList("saveOrder", "saveRetrospectiveOrder", "voidOrder", "unvoidOrder",
					"purgeOrder", "discontinueOrder"));

	private static final Set<String> PROGRAM_METHODS = new HashSet<String>(
			Arrays.asList("savePatientProgram", "voidPatientProgram", "unvoidPatientProgram",
					"purgePatientProgram"));

	private static final Set<String> MEDICATION_DISPENSE_METHODS = new HashSet<String>(
			Arrays.asList("saveMedicationDispense", "voidMedicationDispense",
					"unvoidMedicationDispense", "deleteMedicationDispense"));

	@Override
	public void afterReturning(Object returnValue, Method method, Object[] args, Object target) {
		String methodName = method.getName();

		boolean isMerge = "mergePatients".equals(methodName);
		Patient patient = isMerge ? null : extractPatient(methodName, args);
		if (!isMerge && patient == null) {
			return;
		}

		String preFilter = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_EMBEDDING_PRE_FILTER, "true");
		if ("false".equalsIgnoreCase(preFilter.trim())) {
			return;
		}

		if (isMerge) {
			handleMergePatients(args);
			return;
		}

		try {
			EmbeddingIndexer indexer = Context.getRegisteredComponent(
					"embeddingIndexer", EmbeddingIndexer.class);
			indexer.indexPatient(patient);
		}
		catch (Exception e) {
			log.error("Failed to re-index patient [id={}] after {} call",
					patient.getPatientId(), methodName, e);
		}

		reindexLucene(patient);
		reindexElasticsearch(patient);
	}

	private void handleMergePatients(Object[] args) {
		if (args.length < 2 || !(args[0] instanceof Patient) || !(args[1] instanceof Patient)) {
			return;
		}
		Patient preferred = (Patient) args[0];
		Patient notPreferred = (Patient) args[1];
		try {
			EmbeddingIndexer indexer = Context.getRegisteredComponent(
					"embeddingIndexer", EmbeddingIndexer.class);
			indexer.indexPatient(preferred);
			indexer.deletePatientEmbeddings(notPreferred);
		}
		catch (Exception e) {
			log.error("Failed to re-index after mergePatients [preferred={}, notPreferred={}]",
					preferred.getPatientId(), notPreferred.getPatientId(), e);
		}

		reindexLucene(preferred);
		deleteLuceneIndex(notPreferred);
		reindexElasticsearch(preferred);
		deleteElasticsearchIndex(notPreferred);
	}

	private void reindexLucene(Patient patient) {
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_LUCENE.equalsIgnoreCase(pipeline.trim())) {
			return;
		}
		try {
			LuceneIndexer luceneIndexer = Context.getRegisteredComponent(
					"luceneIndexer", LuceneIndexer.class);
			if (luceneIndexer != null && luceneIndexer.hasIndex(patient)) {
				luceneIndexer.indexPatient(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to re-index Lucene for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	private void deleteLuceneIndex(Patient patient) {
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_LUCENE.equalsIgnoreCase(pipeline.trim())) {
			return;
		}
		try {
			LuceneIndexer luceneIndexer = Context.getRegisteredComponent(
					"luceneIndexer", LuceneIndexer.class);
			if (luceneIndexer != null) {
				luceneIndexer.deletePatientIndex(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to delete Lucene index for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	private void reindexElasticsearch(Patient patient) {
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_ELASTICSEARCH.equalsIgnoreCase(pipeline.trim())) {
			return;
		}
		try {
			ElasticsearchIndexer esIndexer = Context.getRegisteredComponent(
					"elasticsearchIndexer", ElasticsearchIndexer.class);
			if (esIndexer != null && esIndexer.hasIndex(patient)) {
				esIndexer.indexPatient(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to re-index Elasticsearch for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	private void deleteElasticsearchIndex(Patient patient) {
		String pipeline = Context.getAdministrationService()
				.getGlobalProperty(ChartSearchAiConstants.GP_RETRIEVAL_PIPELINE, "");
		if (!ChartSearchAiConstants.PIPELINE_ELASTICSEARCH.equalsIgnoreCase(pipeline.trim())) {
			return;
		}
		try {
			ElasticsearchIndexer esIndexer = Context.getRegisteredComponent(
					"elasticsearchIndexer", ElasticsearchIndexer.class);
			if (esIndexer != null) {
				esIndexer.deletePatientIndex(patient);
			}
		}
		catch (Exception e) {
			log.error("Failed to delete Elasticsearch index for patient [id={}]",
					patient.getPatientId(), e);
		}
	}

	Patient extractPatient(String methodName, Object[] args) {
		if (CONDITION_METHODS.contains(methodName) && args.length > 0 && args[0] instanceof Condition) {
			return ((Condition) args[0]).getPatient();
		} else if (DIAGNOSIS_METHODS.contains(methodName) && args.length > 0 && args[0] instanceof Diagnosis) {
			return ((Diagnosis) args[0]).getPatient();
		} else if (ALLERGY_METHODS.contains(methodName)) {
			if (args.length > 0 && args[0] instanceof Patient) {
				return (Patient) args[0];
			} else if (args.length > 0 && args[0] instanceof Allergy) {
				return ((Allergy) args[0]).getPatient();
			}
		} else if (ORDER_METHODS.contains(methodName) && args.length > 0 && args[0] instanceof Order) {
			return ((Order) args[0]).getPatient();
		} else if (PROGRAM_METHODS.contains(methodName) && args.length > 0 && args[0] instanceof PatientProgram) {
			return ((PatientProgram) args[0]).getPatient();
		} else if (MEDICATION_DISPENSE_METHODS.contains(methodName) && args.length > 0
				&& args[0] instanceof MedicationDispense) {
			return ((MedicationDispense) args[0]).getPatient();
		}
		return null;
	}
}
