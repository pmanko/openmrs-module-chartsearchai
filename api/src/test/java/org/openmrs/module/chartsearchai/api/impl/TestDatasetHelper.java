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

import java.util.ArrayList;
import java.util.List;

import org.openmrs.module.chartsearchai.ChartSearchAiConstants;
import org.openmrs.module.chartsearchai.serializer.PatientRecordLoader.SerializedRecord;

/**
 * Shared helpers for tests that use the {@code FULL_PATIENT_DATASET} test
 * dataset. Consolidates the dataset-parsing logic that was previously
 * duplicated across multiple test files.
 */
final class TestDatasetHelper {

	static final String MODEL_DIR = System.getProperty(
			"chartsearchai.embedding.model.dir", "../models/all-MiniLM-L6-v2");

	static final String MODEL_PATH = MODEL_DIR + "/model.onnx";

	static final String VOCAB_PATH = MODEL_DIR + "/vocab.txt";

	static boolean modelFilesExist() {
		return new java.io.File(MODEL_PATH).exists()
				&& new java.io.File(VOCAB_PATH).exists();
	}

	// Full 153-record dataset from a real 16-year-old Male patient chart.
	static final String[] FULL_PATIENT_DATASET = {
			/* [  1] */ "Medication prescription: (2026-03-18) Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Every six hours. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: REVISE. Urgency: ROUTINE. Reason: Spectrum",
			/* [  2] */ "Medication prescription: (2026-03-18) Drug order: Azithromycin. Dose: 2.0 Tablet) Intravenous Thrice daily. Duration: 5 Days. Quantity: 4.0 Tablet). As needed (subject to heart attack). Dosing: Take after eating. Action: NEW. Urgency: ROUTINE. Reason: Spectrum. Stopped: 2026-03-18",
			/* [  3] */ "Program enrollment: (2026-03-18) Program: PMTCT. Enrolled: 2026-03-18. Status: Active",
			/* [  4] */ "Clinical observation: (2026-03-18) Finding — Immunization history: Immunizations: Polio vaccination, oral, Oral polio vacc, Polio immunization, oral); Vaccination date: 2026-03-17; Immunization sequence number: 1.0",
			/* [  5] */ "Patient allergy: (2026-03-18) Allergy: Beef (food allergen). Severity: Severe. Reactions: Diarrhea, Itching. Comments: Happens during pregnancy",
			/* [  6] */ "Clinical observation: (2026-03-05) Assessment — Method of family planning: Condoms",
			/* [  7] */ "Clinical observation: (2026-03-05) Assessment — Method of family planning: Diaphragm. Note: in bathroom",
			/* [  8] */ "Medical condition: (2025-11-12) Condition: Tuberculosis. Status: ACTIVE",
			/* [  9] */ "Clinical observation: (2025-10-30) Test — CD4 Count: 988.0 cells/mmL",
			/* [ 10] */ "Clinical observation: (2025-10-30) Drug — Pyrimethamine / sulfadoxine: 11.58",
			/* [ 11] */ "Clinical observation: (2025-10-30) Units of Measure — Syringe): 65.0",
			/* [ 12] */ "Clinical observation: (2025-10-30) Diagnosis — Kaposi sarcoma oral: 3.91",
			/* [ 13] */ "Clinical observation: (2025-10-30) Assessment — Primary Diagnosis: Tuberculosis",
			/* [ 14] */ "Clinical observation: (2025-10-30) Frequency — Every twenty-four hours: Every four hours",
			/* [ 15] */ "Clinical observation: (2025-10-30) Test — Height (cm): 131.0 cm",
			/* [ 16] */ "Clinical observation: (2025-10-30) Test — Respiratory Rate: 18.0 breaths/min",
			/* [ 17] */ "Clinical observation: (2025-10-30) Test — Pulse: 95.0 beats/min",
			/* [ 18] */ "Clinical observation: (2025-10-30) Test — Temperature (C): 36.7 DEG C",
			/* [ 19] */ "Clinical observation: (2025-10-30) Test — Weight (kg): 94.0 kg",
			/* [ 20] */ "Clinical observation: (2025-09-17) Diagnosis — Fetishism: Patient presents with mild symptoms. Advised rest and fluids.",
			/* [ 21] */ "Clinical observation: (2025-09-17) Frequency — Every twenty-four hours: Every eight hours",
			/* [ 22] */ "Clinical diagnosis: (2025-09-17) Diagnosis: Gastroenteritis. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 23] */ "Clinical observation: (2025-09-17) Test — Systolic Blood Pressure: 97.0 mmHg",
			/* [ 24] */ "Clinical observation: (2025-09-17) Test — Diastolic Blood Pressure: 99.0 mmHg",
			/* [ 25] */ "Clinical observation: (2025-09-17) Test — Pulse: 62.0 beats/min",
			/* [ 26] */ "Clinical observation: (2025-09-17) Test — Temperature (C): 37.7 DEG C",
			/* [ 27] */ "Clinical observation: (2025-09-17) Test — Weight (kg): 107.0 kg",
			/* [ 28] */ "Clinical observation: (2025-09-17) Test — Height (cm): 137.0 cm",
			/* [ 29] */ "Clinical observation: (2025-09-17) Test — Respiratory Rate: 24.0 breaths/min",
			/* [ 30] */ "Clinical observation: (2025-09-17) Assessment — Primary Diagnosis: Anemia",
			/* [ 31] */ "Clinical observation: (2025-09-17) Frequency — Every twenty-four hours: Every five hours",
			/* [ 32] */ "Clinical observation: (2025-09-17) Test — Diastolic Blood Pressure: 92.0 mmHg",
			/* [ 33] */ "Clinical observation: (2025-09-17) Test — Respiratory Rate: 32.0 breaths/min",
			/* [ 34] */ "Clinical observation: (2025-09-17) Test — Weight (kg): 139.0 kg",
			/* [ 35] */ "Clinical observation: (2025-09-17) Test — Blood Oxygen Saturation: 88.0 %",
			/* [ 36] */ "Clinical observation: (2025-06-29) Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [ 37] */ "Clinical observation: (2025-06-29) Test — Systolic Blood Pressure: 122.0 mmHg",
			/* [ 38] */ "Clinical observation: (2025-06-29) Test — Height (cm): 103.0 cm",
			/* [ 39] */ "Clinical observation: (2025-06-29) Test — Temperature (C): 40.3 DEG C",
			/* [ 40] */ "Clinical diagnosis: (2025-06-29) Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 41] */ "Clinical observation: (2025-06-28) Assessment — Primary Diagnosis: HIV Disease",
			/* [ 42] */ "Clinical observation: (2025-06-28) Diagnosis — Fetishism: Patient stable on current regimen.",
			/* [ 43] */ "Clinical observation: (2025-06-28) Frequency — Every twenty-four hours: Married",
			/* [ 44] */ "Clinical observation: (2025-06-28) Test — Pulse: 62.0 beats/min",
			/* [ 45] */ "Clinical observation: (2025-06-28) Test — Respiratory Rate: 23.0 breaths/min",
			/* [ 46] */ "Clinical observation: (2025-06-28) Test — Blood Oxygen Saturation: 92.0 %",
			/* [ 47] */ "Clinical observation: (2025-06-28) Test — Height (cm): 126.0 cm",
			/* [ 48] */ "Clinical observation: (2025-06-28) Test — Systolic Blood Pressure: 101.0 mmHg",
			/* [ 49] */ "Clinical observation: (2025-06-28) Test — Diastolic Blood Pressure: 99.0 mmHg",
			/* [ 50] */ "Clinical observation: (2025-06-28) Assessment — Primary Diagnosis: Diabetes Mellitus",
			/* [ 51] */ "Clinical observation: (2025-06-28) Diagnosis — Fetishism: New complaint of persistent cough for 2 weeks.",
			/* [ 52] */ "Clinical diagnosis: (2025-06-28) Diagnosis: Urinary Tract Infection. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 53] */ "Clinical diagnosis: (2025-06-28) Diagnosis: Tuberculosis. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 54] */ "Patient allergy: (2024-12-29) Allergy: Fomepizole (drug allergen)",
			/* [ 55] */ "Medical condition: (2024-09-08) Condition: Hypertension. Status: ACTIVE",
			/* [ 56] */ "Clinical observation: (2023-04-30) Assessment — Primary Diagnosis: Anemia",
			/* [ 57] */ "Clinical observation: (2023-04-30) Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 58] */ "Clinical observation: (2023-04-30) Test — Height (cm): 101.0 cm",
			/* [ 59] */ "Clinical observation: (2023-04-30) Test — Systolic Blood Pressure: 123.0 mmHg",
			/* [ 60] */ "Clinical observation: (2023-04-30) Test — Blood Oxygen Saturation: 86.0 %",
			/* [ 61] */ "Clinical observation: (2023-04-30) Test — Pulse: 51.0 beats/min",
			/* [ 62] */ "Clinical diagnosis: (2023-04-30) Diagnosis: Skin Infection. Certainty: CONFIRMED. Rank: Primary",
			/* [ 63] */ "Clinical observation: (2023-04-29) Assessment — Primary Diagnosis: Pneumonia",
			/* [ 64] */ "Clinical observation: (2023-04-29) Test — Weight (kg): 38.0 kg",
			/* [ 65] */ "Clinical observation: (2023-04-29) Test — Diastolic Blood Pressure: 71.0 mmHg",
			/* [ 66] */ "Clinical observation: (2023-04-29) Test — Respiratory Rate: 16.0 breaths/min",
			/* [ 67] */ "Clinical diagnosis: (2023-04-29) Diagnosis: Malaria. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 68] */ "Clinical diagnosis: (2023-04-29) Diagnosis: Diabetes Mellitus. Certainty: CONFIRMED. Rank: Secondary",
			/* [ 69] */ "Clinical observation: (2022-11-09) Assessment — Primary Diagnosis: HIV Disease",
			/* [ 70] */ "Clinical diagnosis: (2022-11-09) Diagnosis: HIV Disease. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 71] */ "Clinical observation: (2022-10-07) Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [ 72] */ "Clinical diagnosis: (2022-10-07) Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [ 73] */ "Clinical diagnosis: (2022-10-07) Diagnosis: Anemia. Certainty: PROVISIONAL. Rank: Secondary",
			/* [ 74] */ "Clinical observation: (2022-10-07) Test — Systolic Blood Pressure: 137.0 mmHg",
			/* [ 75] */ "Clinical observation: (2022-10-07) Test — Diastolic Blood Pressure: 67.0 mmHg",
			/* [ 76] */ "Clinical observation: (2022-10-07) Test — Pulse: 86.0 beats/min",
			/* [ 77] */ "Clinical observation: (2022-10-07) Test — Temperature (C): 39.3 DEG C",
			/* [ 78] */ "Clinical observation: (2022-10-07) Test — Weight (kg): 146.0 kg",
			/* [ 79] */ "Clinical observation: (2022-10-07) Test — Height (cm): 107.0 cm",
			/* [ 80] */ "Clinical observation: (2022-10-07) Test — Blood Oxygen Saturation: 86.0 %",
			/* [ 81] */ "Clinical observation: (2022-10-07) Test — Respiratory Rate: 30.0 breaths/min",
			/* [ 82] */ "Clinical observation: (2022-04-27) Test — Diastolic Blood Pressure: 105.0 mmHg",
			/* [ 83] */ "Clinical observation: (2022-04-27) Test — Pulse: 70.0 beats/min",
			/* [ 84] */ "Clinical observation: (2022-04-27) Test — Blood Oxygen Saturation: 100.0 %",
			/* [ 85] */ "Clinical observation: (2022-04-27) Test — Respiratory Rate: 40.0 breaths/min",
			/* [ 86] */ "Clinical observation: (2022-04-27) Test — CD4 Count: 1191.0 cells/mmL",
			/* [ 87] */ "Clinical observation: (2022-04-27) Units of Measure — Syringe): 339.0",
			/* [ 88] */ "Clinical observation: (2022-04-27) Misc — Milligram per meter squared: 47.0",
			/* [ 89] */ "Clinical observation: (2022-04-27) Diagnosis — Kaposi sarcoma oral: 3.5",
			/* [ 90] */ "Clinical observation: (2022-04-27) Diagnosis — Photoallergy: 9.93",
			/* [ 91] */ "Clinical observation: (2022-01-16) Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [ 92] */ "Clinical observation: (2022-01-16) Diagnosis — Fetishism: Chronic disease management visit. Medication adjusted.",
			/* [ 93] */ "Clinical diagnosis: (2022-01-16) Diagnosis: Hypertension. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 94] */ "Clinical observation: (2022-01-16) Test — Systolic Blood Pressure: 147.0 mmHg",
			/* [ 95] */ "Clinical observation: (2022-01-16) Test — Diastolic Blood Pressure: 58.0 mmHg",
			/* [ 96] */ "Clinical observation: (2022-01-16) Test — Pulse: 53.0 beats/min",
			/* [ 97] */ "Clinical observation: (2022-01-16) Test — Temperature (C): 36.4 DEG C",
			/* [ 98] */ "Clinical observation: (2022-01-16) Test — Height (cm): 163.0 cm",
			/* [ 99] */ "Clinical observation: (2022-01-16) Test — Respiratory Rate: 19.0 breaths/min",
			/* [100] */ "Clinical observation: (2022-01-16) Assessment — Primary Diagnosis: Headache",
			/* [101] */ "Clinical observation: (2022-01-16) Test — Pulse: 83.0 beats/min",
			/* [102] */ "Clinical observation: (2022-01-16) Test — Weight (kg): 68.0 kg",
			/* [103] */ "Clinical observation: (2022-01-16) Test — Diastolic Blood Pressure: 50.0 mmHg",
			/* [104] */ "Clinical observation: (2022-01-16) Test — Blood Oxygen Saturation: 94.0 %",
			/* [105] */ "Clinical observation: (2022-01-16) Test — Respiratory Rate: 40.0 breaths/min",
			/* [106] */ "Clinical observation: (2021-11-30) Assessment — Primary Diagnosis: Malaria",
			/* [107] */ "Clinical observation: (2021-11-30) Diagnosis — Fetishism: Well-child visit. Growth and development normal.",
			/* [108] */ "Clinical observation: (2021-11-30) Test — Diastolic Blood Pressure: 93.0 mmHg",
			/* [109] */ "Clinical observation: (2021-11-30) Test — Blood Oxygen Saturation: 95.0 %",
			/* [110] */ "Clinical observation: (2021-11-30) Test — Respiratory Rate: 36.0 breaths/min",
			/* [111] */ "Clinical diagnosis: (2021-11-30) Diagnosis: HIV Disease. Certainty: CONFIRMED. Rank: Primary",
			/* [112] */ "Clinical observation: (2021-11-30) Test — Systolic Blood Pressure: 98.0 mmHg",
			/* [113] */ "Clinical observation: (2021-11-30) Test — Pulse: 66.0 beats/min",
			/* [114] */ "Clinical observation: (2021-11-30) Test — Temperature (C): 37.8 DEG C",
			/* [115] */ "Clinical observation: (2021-11-30) Test — Weight (kg): 121.0 kg",
			/* [116] */ "Clinical observation: (2021-11-30) Test — Height (cm): 173.0 cm",
			/* [117] */ "Clinical observation: (2021-11-30) Test — Blood Oxygen Saturation: 94.0 %",
			/* [118] */ "Clinical observation: (2021-11-30) Test — Respiratory Rate: 29.0 breaths/min",
			/* [119] */ "Clinical observation: (2021-11-30) Assessment — Primary Diagnosis: Urinary Tract Infection",
			/* [120] */ "Clinical observation: (2021-11-30) Diagnosis — Fetishism: Presenting with fever and body aches for 3 days.",
			/* [121] */ "Clinical observation: (2021-09-16) Diagnosis — Fetishism: Patient counseled on lifestyle modifications.",
			/* [122] */ "Clinical observation: (2021-09-16) Frequency — Every twenty-four hours: Every four hours",
			/* [123] */ "Clinical observation: (2021-09-16) Assessment — Primary Diagnosis: Skin Infection",
			/* [124] */ "Clinical observation: (2021-09-16) Diagnosis — Fetishism: Annual physical examination. Labs ordered.",
			/* [125] */ "Clinical observation: (2021-09-16) Test — Height (cm): 137.0 cm",
			/* [126] */ "Clinical observation: (2021-09-16) Test — Respiratory Rate: 28.0 breaths/min",
			/* [127] */ "Clinical observation: (2021-09-16) Test — Systolic Blood Pressure: 134.0 mmHg",
			/* [128] */ "Clinical diagnosis: (2021-09-16) Diagnosis: Asthma. Certainty: CONFIRMED. Rank: Primary",
			/* [129] */ "Clinical observation: (2021-09-16) Test — Systolic Blood Pressure: 117.0 mmHg",
			/* [130] */ "Clinical observation: (2021-09-16) Test — Diastolic Blood Pressure: 70.0 mmHg",
			/* [131] */ "Clinical observation: (2021-09-16) Test — Pulse: 115.0 beats/min",
			/* [132] */ "Clinical observation: (2021-09-16) Test — Temperature (C): 40.1 DEG C",
			/* [133] */ "Clinical observation: (2021-09-16) Test — Height (cm): 186.0 cm",
			/* [134] */ "Clinical observation: (2021-09-16) Test — Respiratory Rate: 22.0 breaths/min",
			/* [135] */ "Clinical observation: (2021-07-20) Assessment — Primary Diagnosis: Tuberculosis",
			/* [136] */ "Clinical observation: (2021-07-20) Assessment — Primary Diagnosis: Tuberculosis",
			/* [137] */ "Clinical observation: (2021-07-20) Test — Respiratory Rate: 28.0 breaths/min",
			/* [138] */ "Clinical observation: (2021-07-20) Test — Diastolic Blood Pressure: 76.0 mmHg",
			/* [139] */ "Clinical observation: (2021-07-20) Test — Systolic Blood Pressure: 102.0 mmHg",
			/* [140] */ "Clinical observation: (2021-07-20) Units of Measure — Syringe): 237.0",
			/* [141] */ "Clinical observation: (2021-07-20) Misc — Milligram per meter squared: 17.0",
			/* [142] */ "Clinical observation: (2021-07-20) Diagnosis — Photoallergy: 8.27",
			/* [143] */ "Clinical observation: (2020-09-26) Diagnosis — Fetishism: Routine checkup. No significant findings.",
			/* [144] */ "Clinical observation: (2020-09-26) Test — Temperature (C): 39.3 DEG C",
			/* [145] */ "Clinical observation: (2020-09-26) Test — Diastolic Blood Pressure: 78.0 mmHg",
			/* [146] */ "Clinical observation: (2020-09-26) Test — Blood Oxygen Saturation: 88.0 %",
			/* [147] */ "Clinical diagnosis: (2020-09-26) Diagnosis: Diabetes Mellitus. Certainty: PROVISIONAL. Rank: Primary",
			/* [148] */ "Clinical observation: (2020-09-26) Test — Systolic Blood Pressure: 151.0 mmHg",
			/* [149] */ "Clinical observation: (2020-09-26) Test — Diastolic Blood Pressure: 53.0 mmHg",
			/* [150] */ "Clinical observation: (2020-09-26) Test — Pulse: 117.0 beats/min",
			/* [151] */ "Clinical observation: (2020-09-26) Test — Temperature (C): 39.4 DEG C",
			/* [152] */ "Clinical observation: (2020-09-26) Test — Blood Oxygen Saturation: 88.0 %",
			/* [153] */ "Clinical observation: (2020-09-26) Test — Respiratory Rate: 15.0 breaths/min",
	};

	// Second 67-record dataset from a 4-year-old Male patient chart.
	static final String[] SECOND_PATIENT_DATASET = {
			/* [ 1] */ "Clinical observation: Assessment — Text of encounter note: Est sit amet facilisis magna etiam tempor. Sed vulputate odio ut enim blandit volutpat. Eget est lorem ipsum dolor sit amet. Arcu vitae elementum curabitur vitae nunc sed velit. Venenatis cras sed felis eget velit.",
			/* [ 2] */ "Medical condition: Condition: Nonparalytic stroke. Status: ACTIVE",
			/* [ 3] */ "Medical condition: Condition: Scarring Alopecia. Status: ACTIVE",
			/* [ 4] */ "Clinical diagnosis: Diagnosis: Nonparalytic stroke. Certainty: PROVISIONAL. Rank: Primary",
			/* [ 5] */ "Clinical diagnosis: Diagnosis: Scarring Alopecia. Certainty: CONFIRMED. Rank: Primary",
			/* [ 6] */ "Clinical observation: Finding — Temperature (c)): 38.9 (CRITICALLY_HIGH)",
			/* [ 7] */ "Clinical observation: Finding — Systolic blood pressure: 133.0 (HIGH)",
			/* [ 8] */ "Clinical observation: Finding — Diastolic blood pressure: 57.0 (NORMAL)",
			/* [ 9] */ "Clinical observation: Finding — Pulse: 100.0 (NORMAL)",
			/* [10] */ "Clinical observation: Finding — Arterial blood oxygen saturation (pulse oximeter): 96.6",
			/* [11] */ "Clinical observation: Finding — Weight (kg): 68.0",
			/* [12] */ "Clinical observation: Finding — Respiratory rate: 17.0 (CRITICALLY_LOW)",
			/* [13] */ "Clinical observation: Finding — Height (cm): 168.0",
			/* [14] */ "Clinical observation: Test — High-density lipoprotein cholesterol measurement (mmol/L), High-density lipoprotein cholesterol): 96.0",
			/* [15] */ "Lab test order: Test order: High-density lipoprotein cholesterol measurement (mmol/L), High-density lipoprotein cholesterol). Action: NEW. Urgency: ROUTINE",
			/* [16] */ "Clinical observation: Assessment — Text of encounter note: Neque convallis a cras semper auctor neque vitae. Proin nibh nisl condimentum id venenatis a condimentum vitae. Viverra accumsan in nisl nisi scelerisque eu ultrices vitae. Ac tincidunt vitae semper quis lectus nulla.",
			/* [17] */ "Medical condition: Condition: Granuloma annulare. Status: ACTIVE",
			/* [18] */ "Medical condition: Condition: Syphilitic Cirrhosis. Status: ACTIVE",
			/* [19] */ "Clinical diagnosis: Diagnosis: Granuloma annulare. Certainty: PROVISIONAL. Rank: Primary",
			/* [20] */ "Clinical diagnosis: Diagnosis: Syphilitic Cirrhosis. Certainty: CONFIRMED. Rank: Primary",
			/* [21] */ "Clinical observation: Finding — Temperature (c)): 37.2 (NORMAL)",
			/* [22] */ "Clinical observation: Finding — Systolic blood pressure: 104.0 (NORMAL)",
			/* [23] */ "Clinical observation: Finding — Diastolic blood pressure: 52.0 (NORMAL)",
			/* [24] */ "Clinical observation: Finding — Pulse: 108.0 (NORMAL)",
			/* [25] */ "Clinical observation: Finding — Arterial blood oxygen saturation (pulse oximeter): 93.0 (LOW)",
			/* [26] */ "Clinical observation: Finding — Weight (kg): 49.0",
			/* [27] */ "Clinical observation: Finding — Respiratory rate: 22.0 (NORMAL)",
			/* [28] */ "Clinical observation: Finding — Height (cm): 186.0",
			/* [29] */ "Clinical observation: Assessment — Text of encounter note: Faucibus et molestie ac feugiat sed lectus. Condimentum lacinia quis vel eros donec ac. Urna porttitor rhoncus dolor purus.",
			/* [30] */ "Medical condition: Condition: Atherosclerosis. Status: ACTIVE",
			/* [31] */ "Medical condition: Condition: Complete tear of ligament of ankle or foot. Status: ACTIVE",
			/* [32] */ "Medical condition: Condition: Mild depressive episode. Status: ACTIVE",
			/* [33] */ "Clinical diagnosis: Diagnosis: Atherosclerosis. Certainty: PROVISIONAL. Rank: Primary",
			/* [34] */ "Clinical diagnosis: Diagnosis: Complete tear of ligament of ankle or foot. Certainty: CONFIRMED. Rank: Primary",
			/* [35] */ "Clinical diagnosis: Diagnosis: Mild depressive episode. Certainty: CONFIRMED. Rank: Primary",
			/* [36] */ "Clinical observation: Finding — Temperature (c)): 38.7 (CRITICALLY_HIGH)",
			/* [37] */ "Clinical observation: Finding — Systolic blood pressure: 105.0 (NORMAL)",
			/* [38] */ "Clinical observation: Finding — Diastolic blood pressure: 60.0 (NORMAL)",
			/* [39] */ "Clinical observation: Finding — Pulse: 65.0 (CRITICALLY_LOW)",
			/* [40] */ "Clinical observation: Finding — Arterial blood oxygen saturation (pulse oximeter): 99.6",
			/* [41] */ "Clinical observation: Finding — Weight (kg): 72.0",
			/* [42] */ "Clinical observation: Finding — Respiratory rate: 9.0 (CRITICALLY_LOW)",
			/* [43] */ "Clinical observation: Finding — Height (cm): 186.0",
			/* [44] */ "Clinical observation: Assessment — Text of encounter note: Est sit amet facilisis magna etiam tempor. Sed vulputate odio ut enim blandit volutpat. Eget est lorem ipsum dolor sit amet. Arcu vitae elementum curabitur vitae nunc sed velit. Venenatis cras sed felis eget velit.",
			/* [45] */ "Medical condition: Condition: Female infertility. Status: ACTIVE",
			/* [46] */ "Clinical diagnosis: Diagnosis: Female infertility. Certainty: PROVISIONAL. Rank: Primary",
			/* [47] */ "Clinical observation: Finding — Temperature (c)): 36.0 (NORMAL)",
			/* [48] */ "Clinical observation: Finding — Systolic blood pressure: 128.0 (HIGH)",
			/* [49] */ "Clinical observation: Finding — Diastolic blood pressure: 86.0 (HIGH)",
			/* [50] */ "Clinical observation: Finding — Pulse: 70.0 (LOW)",
			/* [51] */ "Clinical observation: Finding — Arterial blood oxygen saturation (pulse oximeter): 89.3 (CRITICALLY_LOW)",
			/* [52] */ "Clinical observation: Finding — Weight (kg): 50.0",
			/* [53] */ "Clinical observation: Finding — Respiratory rate: 16.0 (CRITICALLY_LOW)",
			/* [54] */ "Clinical observation: Finding — Height (cm): 185.0",
			/* [55] */ "Clinical observation: Assessment — Text of encounter note: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Vulputate enim nulla aliquet porttitor. Fermentum iaculis eu non diam phasellus vestibulum lorem sed. Orci ac auctor augue mauris augue neque. Fames ac turpis egestas sed tempus urna. Sit amet justo donec enim diam vulputate. Tortor aliquam nulla facilisi cras fermentum. Aliquet eget sit amet tellus. Elit ullamcorper dignissim cras tincidunt lobortis feugiat. Nisl tincidunt eget nullam non nisi est. Volutpat maecenas volutpat blandit aliquam etiam erat.",
			/* [56] */ "Medical condition: Condition: Personal history of blood transfusion. Status: ACTIVE",
			/* [57] */ "Medical condition: Condition: Chronic fatigue. Status: ACTIVE",
			/* [58] */ "Clinical diagnosis: Diagnosis: Personal history of blood transfusion. Certainty: PROVISIONAL. Rank: Primary",
			/* [59] */ "Clinical diagnosis: Diagnosis: Chronic fatigue. Certainty: CONFIRMED. Rank: Primary",
			/* [60] */ "Clinical observation: Finding — Temperature (c)): 36.5 (NORMAL)",
			/* [61] */ "Clinical observation: Finding — Systolic blood pressure: 131.0 (HIGH)",
			/* [62] */ "Clinical observation: Finding — Diastolic blood pressure: 60.0 (NORMAL)",
			/* [63] */ "Clinical observation: Finding — Pulse: 63.0 (CRITICALLY_LOW)",
			/* [64] */ "Clinical observation: Finding — Arterial blood oxygen saturation (pulse oximeter): 89.7 (CRITICALLY_LOW)",
			/* [65] */ "Clinical observation: Finding — Weight (kg): 50.0",
			/* [66] */ "Clinical observation: Finding — Respiratory rate: 8.0 (CRITICALLY_LOW)",
			/* [67] */ "Clinical observation: Finding — Height (cm): 196.0",
	};

	private static final String[] DATASET_PREFIXES = {
			"Medication prescription: ",
			"Lab test order: ",
			"Clinical observation: ",
			"Clinical diagnosis: ",
			"Medical condition: ",
			"Patient allergy: ",
			"Program enrollment: ",
			"Medication dispensed: "
	};

	private static final java.util.regex.Pattern DATE_PREFIX_PATTERN =
			java.util.regex.Pattern.compile("^\\(\\d{4}-\\d{2}-\\d{2}\\)\\s*");

	private TestDatasetHelper() {
	}

	/**
	 * Strips the dataset-format prefix and optional date to recover the raw
	 * serializer output that production stores as
	 * {@link org.openmrs.module.chartsearchai.model.ChartEmbedding#getTextContent()}.
	 *
	 * <p>Dataset format: {@code "Clinical observation: (2025-10-30) Test — Weight (kg): 94.0 kg"}
	 * <br>Production textContent: {@code "Test — Weight (kg): 94.0 kg"}
	 */
	static String stripDatasetPrefixAndDate(String datasetText) {
		String text = datasetText;
		for (String prefix : DATASET_PREFIXES) {
			if (text.startsWith(prefix)) {
				text = text.substring(prefix.length());
				break;
			}
		}
		java.util.regex.Matcher m = DATE_PREFIX_PATTERN.matcher(text);
		if (m.find()) {
			text = text.substring(m.end());
		}
		return text;
	}

	/**
	 * Infers resource type from the serialized text prefix, matching the
	 * resource types assigned by
	 * {@link org.openmrs.module.chartsearchai.serializer.PatientRecordLoader}.
	 */
	static String inferResourceType(String text) {
		if (text.startsWith("Medication prescription:")
				|| text.startsWith("Lab test order:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_ORDER;
		}
		if (text.startsWith("Medical condition:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_CONDITION;
		}
		if (text.startsWith("Clinical diagnosis:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_DIAGNOSIS;
		}
		if (text.startsWith("Patient allergy:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_ALLERGY;
		}
		if (text.startsWith("Program enrollment:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_PROGRAM;
		}
		if (text.startsWith("Medication dispensed:")) {
			return ChartSearchAiConstants.RESOURCE_TYPE_MEDICATION_DISPENSE;
		}
		return ChartSearchAiConstants.RESOURCE_TYPE_OBS;
	}

	/**
	 * Converts a raw dataset array into {@link SerializedRecord} objects
	 * suitable for passing to
	 * {@link org.openmrs.module.chartsearchai.api.EmbeddingIndexer#buildEmbeddings}.
	 */
	static List<SerializedRecord> toSerializedRecords(String[] dataset) {
		List<SerializedRecord> records = new ArrayList<>();
		for (int i = 0; i < dataset.length; i++) {
			String resourceType = inferResourceType(dataset[i]);
			String textContent = stripDatasetPrefixAndDate(dataset[i]);
			records.add(new SerializedRecord(resourceType, i, textContent, null));
		}
		return records;
	}
}
