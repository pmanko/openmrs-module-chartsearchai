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

/**
 * Converts an OpenMRS clinical record into a concise, human-readable text string suitable for
 * vector embedding. Each implementation handles a specific resource type (Obs, Condition,
 * Allergy, etc.) and produces text that captures the clinically meaningful content while
 * discarding structural metadata that adds no semantic value.
 *
 * <h3>Date handling</h3>
 *
 * <p>Serializers must distinguish between <b>record timestamps</b> and
 * <b>clinically significant dates</b> (see ADR Decision 16 in {@code docs/adr.md}).
 *
 * <ul>
 *   <li><b>Record timestamps</b> (obs datetime, encounter date, creation date) are
 *       <em>excluded</em> from the serialized text. They are collected separately by
 *       {@link PatientRecordLoader} and added as a parenthetical citation label
 *       (e.g. {@code "(2024-01-15)"}) by {@link PatientChartSerializer} at prompt
 *       assembly time. Including them in the embedding text would pollute semantic
 *       similarity — two records from the same date would appear more similar than
 *       they actually are.</li>
 *   <li><b>Clinically significant dates</b> (resolution date, stop date, enrollment
 *       date, date-typed obs values) are <em>included</em> in the serialized text
 *       because they carry clinical meaning distinct from the record timestamp.
 *       Examples: {@code "Resolved: 2024-03-01"} in a condition,
 *       {@code "Stopped: 2024-06-20"} in an order,
 *       {@code "Date of Symptom Onset: 2024-02-10"} in an obs.</li>
 * </ul>
 *
 * @param <T> the OpenMRS domain type to serialize
 */
public interface ClinicalTextSerializer<T> {

	/**
	 * Convert a clinical record to a text representation for embedding.
	 *
	 * @param record the clinical record to serialize
	 * @return a concise text string capturing the clinical meaning of the record
	 */
	String toText(T record);
}
