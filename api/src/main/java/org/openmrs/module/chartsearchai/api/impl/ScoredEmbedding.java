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

import org.openmrs.module.chartsearchai.model.ChartEmbedding;

/**
 * A {@link ChartEmbedding} paired with its blended score and the two
 * component scores (keyword and semantic) that produced it. Carried
 * through the ranking pipeline so downstream gates can reason about
 * the score breakdown rather than only the blended value.
 */
class ScoredEmbedding {

	final ChartEmbedding embedding;

	final double score;

	final double keywordScore;

	final double semanticScore;

	ScoredEmbedding(ChartEmbedding embedding, double score, double keywordScore,
			double semanticScore) {
		this.embedding = embedding;
		this.score = score;
		this.keywordScore = keywordScore;
		this.semanticScore = semanticScore;
	}
}
