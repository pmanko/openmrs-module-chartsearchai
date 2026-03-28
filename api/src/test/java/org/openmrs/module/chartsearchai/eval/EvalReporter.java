/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.eval;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Instant;
import java.util.Map;

/**
 * Utility that appends per-case and summary metrics to
 * {@code target/eval-results.csv} so that eval runs produce a
 * machine-readable artifact alongside the normal test output.
 */
public final class EvalReporter {

	private static final String CSV_PATH = "target/eval-results.csv";

	private static final String HEADER = "timestamp,suite,caseId,metric_name,metric_value";

	private EvalReporter() {
	}

	/**
	 * Appends one line per metric entry for the given eval case.
	 *
	 * @param suite   the eval suite name (e.g. "citation", "retrieval")
	 * @param caseId  the individual case identifier
	 * @param metrics map of metric name to metric value
	 */
	public static void appendResult(String suite, String caseId, Map<String, Object> metrics) {
		ensureFileWithHeader();
		String timestamp = Instant.now().toString();
		try (PrintWriter pw = new PrintWriter(new FileWriter(CSV_PATH, true))) {
			for (Map.Entry<String, Object> entry : metrics.entrySet()) {
				pw.println(escapeCsv(timestamp) + ","
						+ escapeCsv(suite) + ","
						+ escapeCsv(caseId) + ","
						+ escapeCsv(entry.getKey()) + ","
						+ escapeCsv(String.valueOf(entry.getValue())));
			}
		}
		catch (IOException e) {
			// Best-effort reporting; do not fail the test
			System.err.println("EvalReporter: failed to write result: " + e.getMessage());
		}
	}

	/**
	 * Appends summary-level metrics for the given suite. The caseId column
	 * is set to {@code _summary}.
	 *
	 * @param suite          the eval suite name
	 * @param summaryMetrics map of metric name to metric value
	 */
	public static void appendSummary(String suite, Map<String, Object> summaryMetrics) {
		appendResult(suite, "_summary", summaryMetrics);
	}

	private static void ensureFileWithHeader() {
		File file = new File(CSV_PATH);
		if (!file.exists()) {
			file.getParentFile().mkdirs();
			try (PrintWriter pw = new PrintWriter(new FileWriter(file))) {
				pw.println(HEADER);
			}
			catch (IOException e) {
				System.err.println("EvalReporter: failed to create CSV file: " + e.getMessage());
			}
		}
	}

	private static String escapeCsv(String value) {
		if (value == null) {
			return "";
		}
		if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
			return "\"" + value.replace("\"", "\"\"") + "\"";
		}
		return value;
	}
}
