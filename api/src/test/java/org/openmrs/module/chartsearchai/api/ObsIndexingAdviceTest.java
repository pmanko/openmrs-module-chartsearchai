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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;
import org.openmrs.Obs;
import org.openmrs.Patient;
import org.openmrs.Person;

/**
 * Pure unit tests for {@link ObsIndexingAdvice} patient extraction logic.
 */
public class ObsIndexingAdviceTest {

	private final ObsIndexingAdvice advice = new ObsIndexingAdvice();

	@Test
	public void getPatientFromArgs_shouldExtractPatientFromReturnValue() {
		Patient patient = new Patient(1);
		Obs obs = new Obs();
		obs.setPerson(patient);

		Patient result = advice.getPatientFromArgs(obs, new Object[] {});
		assertEquals(patient, result);
	}

	@Test
	public void getPatientFromArgs_shouldExtractPatientFromFirstArg() {
		Patient patient = new Patient(2);
		Obs obs = new Obs();
		obs.setPerson(patient);

		Patient result = advice.getPatientFromArgs(null, new Object[] { obs });
		assertEquals(patient, result);
	}

	@Test
	public void getPatientFromArgs_shouldResolvePatientWhenObsCarriesPlainPerson() {
		// A service/REST-loaded obs has getPerson() return a plain Person (not a Patient) even though
		// it belongs to a patient. The advice must resolve the Patient by id, or obs writes silently
		// fail to invalidate the answer cache (and re-index) — the production bug this guards against.
		Person person = new Person(7);
		Obs obs = new Obs();
		obs.setPerson(person);
		final Patient resolved = new Patient(7);
		ObsIndexingAdvice withSeam = new ObsIndexingAdvice() {

			@Override
			Patient resolvePatient(Person p) {
				return p != null && Integer.valueOf(7).equals(p.getPersonId()) ? resolved : null;
			}
		};
		assertEquals(resolved, withSeam.getPatientFromArgs(obs, new Object[] {}),
				"a loaded obs whose person is a plain Person must resolve to its Patient (returnValue path)");
		assertEquals(resolved, withSeam.getPatientFromArgs(null, new Object[] { obs }),
				"...and via the saveObs/voidObs argument path too");
	}

	@Test
	public void getPatientFromArgs_shouldReturnNullWhenPersonIsNotPatient() {
		Person person = new Person();
		Obs obs = new Obs();
		obs.setPerson(person);

		Patient result = advice.getPatientFromArgs(obs, new Object[] {});
		assertNull(result);
	}

	@Test
	public void resolvePatient_shouldReturnNullForNullOrUnsavedPersonWithoutTouchingContext() {
		// The null/null-id guard must short-circuit before Context.getPatientService(), so a bare
		// (unsaved) Person never NPEs in a context-free unit test or in production. Removing it would
		// break every loaded-obs write path.
		assertNull(advice.resolvePatient(null));
		assertNull(advice.resolvePatient(new Person()));
	}

	@Test
	public void getPatientFromArgs_shouldReturnNullWhenArgsAreEmpty() {
		Patient result = advice.getPatientFromArgs(null, new Object[] {});
		assertNull(result);
	}

	@Test
	public void getPatientFromArgs_shouldReturnNullWhenArgsAreNull() {
		Patient result = advice.getPatientFromArgs(null, null);
		assertNull(result);
	}

	@Test
	public void getPatientFromArgs_shouldReturnNullWhenReturnValueIsNotObs() {
		Patient result = advice.getPatientFromArgs("not an obs", new Object[] {});
		assertNull(result);
	}
}
