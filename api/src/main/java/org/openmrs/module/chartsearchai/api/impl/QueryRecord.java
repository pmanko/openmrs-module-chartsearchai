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

import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * chartsearchai's transport-agnostic view of a querystore read record — the embedding-free DTO returned
 * by {@link QueryStoreClient}. Mirrors the {@code /querystore/patientrecord} REST representation
 * (querystore ADR Decision 16): the in-process client maps querystore's {@code QueryDocument} to this,
 * dropping the embedding vector (backend infrastructure) and {@code lastModified}. Carrying only what the
 * chart build consumes — {@code resourceType}, {@code resourceUuid}, {@code date}, {@code text}, and the
 * obs-group {@code metadata} — decouples chartsearchai from querystore's model type so an HTTP-backed
 * client can later return the same shape without chartsearchai changing.
 */
final class QueryRecord {

	private String resourceType;

	private String resourceUuid;

	private LocalDate date;

	private String text;

	private final Map<String, Object> metadata = new LinkedHashMap<String, Object>();

	String getResourceType() {
		return resourceType;
	}

	void setResourceType(String resourceType) {
		this.resourceType = resourceType;
	}

	String getResourceUuid() {
		return resourceUuid;
	}

	void setResourceUuid(String resourceUuid) {
		this.resourceUuid = resourceUuid;
	}

	LocalDate getDate() {
		return date;
	}

	void setDate(LocalDate date) {
		this.date = date;
	}

	String getText() {
		return text;
	}

	void setText(String text) {
		this.text = text;
	}

	/** Never null — initialized to an empty map; {@code QueryStoreChartBuilder.metadataString} relies on
	 *  this (mirrors {@code QueryDocument.getMetadata()}'s non-null contract). */
	Map<String, Object> getMetadata() {
		return metadata;
	}

	QueryRecord putMetadata(String key, Object value) {
		metadata.put(key, value);
		return this;
	}
}
