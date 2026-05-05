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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/** Strict {answer, citations} schema — forces close-brace stop and avoids Anthropic's `json_object` 400. */
final class ChartAnswerResponseFormat {

	private ChartAnswerResponseFormat() {
	}

	static ObjectNode build(ObjectMapper mapper) {
		ObjectNode answerProp = mapper.createObjectNode();
		answerProp.put("type", "string");

		ObjectNode citationItem = mapper.createObjectNode();
		citationItem.put("type", "integer");
		ObjectNode citationsProp = mapper.createObjectNode();
		citationsProp.put("type", "array");
		citationsProp.set("items", citationItem);

		ObjectNode properties = mapper.createObjectNode();
		properties.set("answer", answerProp);
		properties.set("citations", citationsProp);

		ArrayNode required = mapper.createArrayNode();
		required.add("answer");
		required.add("citations");

		ObjectNode schema = mapper.createObjectNode();
		schema.put("type", "object");
		schema.set("properties", properties);
		schema.set("required", required);
		schema.put("additionalProperties", false);

		ObjectNode jsonSchema = mapper.createObjectNode();
		jsonSchema.put("name", "chart_answer");
		jsonSchema.put("strict", true);
		jsonSchema.set("schema", schema);

		ObjectNode responseFormat = mapper.createObjectNode();
		responseFormat.put("type", "json_schema");
		responseFormat.set("json_schema", jsonSchema);
		return responseFormat;
	}
}
