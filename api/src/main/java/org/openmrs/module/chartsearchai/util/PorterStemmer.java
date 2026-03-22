/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.util;

/**
 * Implementation of the Porter Stemming Algorithm (1980) for reducing English words
 * to their root form. Used to normalize clinical queries so that word variants like
 * "allergic", "allergies", and "allergy" produce the same embedding vector.
 *
 * <p>Based on: Porter, M.F., "An algorithm for suffix stripping",
 * Program, 14(3), pp 130-137, July 1980.</p>
 *
 * @see <a href="https://tartarus.org/martin/PorterStemmer/">Reference implementation</a>
 */
public final class PorterStemmer {

	private char[] b; // buffer for the word
	private int k; // offset to the end of the string
	private int j; // general offset into the string

	private PorterStemmer() {
	}

	/**
	 * Stems the given word using the Porter algorithm.
	 *
	 * @param word the word to stem (case-insensitive)
	 * @return the stemmed form, or the original word if it is too short to stem
	 */
	public static String stem(String word) {
		if (word == null || word.length() < 3) {
			return word;
		}
		PorterStemmer s = new PorterStemmer();
		s.b = word.toLowerCase().toCharArray();
		s.k = s.b.length - 1;
		s.step1ab();
		s.step1c();
		s.step2();
		s.step3();
		s.step4();
		s.step5();
		return new String(s.b, 0, s.k + 1);
	}

	private boolean cons(int i) {
		switch (b[i]) {
			case 'a': case 'e': case 'i': case 'o': case 'u':
				return false;
			case 'y':
				return i == 0 || !cons(i - 1);
			default:
				return true;
		}
	}

	/* m() measures the number of consonant sequences between 0 and j.
	   Matches the reference at https://tartarus.org/martin/PorterStemmer/ */
	private int m() {
		int n = 0;
		int i = 0;
		// Skip leading consonants
		while (true) {
			if (i > j) return n;
			if (!cons(i)) break;
			i++;
		}
		i++;
		while (true) {
			// Skip vowels, find next consonant
			while (true) {
				if (i > j) return n;
				if (cons(i)) break;
				i++;
			}
			i++;
			n++;
			// Skip consonants, find next vowel
			while (true) {
				if (i > j) return n;
				if (!cons(i)) break;
				i++;
			}
			i++;
		}
	}

	/* vowelinstem() is true if 0,...j contains a vowel */
	private boolean vowelinstem() {
		for (int i = 0; i <= j; i++) {
			if (!cons(i)) return true;
		}
		return false;
	}

	/* doublec(j) is true if j,(j-1) contain a double consonant. */
	private boolean doublec(int jj) {
		if (jj < 1) return false;
		if (b[jj] != b[jj - 1]) return false;
		return cons(jj);
	}

	/* cvc(i) is true if i-2,i-1,i has the form consonant - vowel - consonant
	   and also if the second c is not w,x or y. */
	private boolean cvc(int i) {
		if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
		int ch = b[i];
		return ch != 'w' && ch != 'x' && ch != 'y';
	}

	private boolean ends(String s) {
		int l = s.length();
		int o = k - l + 1;
		if (o < 0) return false;
		for (int i = 0; i < l; i++) {
			if (b[o + i] != s.charAt(i)) return false;
		}
		j = k - l;
		return true;
	}

	/* setto(s) sets (j+1),...k to the characters in the string s, readjusting k. */
	private void setto(String s) {
		int l = s.length();
		int o = j + 1;
		for (int i = 0; i < l; i++) {
			b[o + i] = s.charAt(i);
		}
		k = j + l;
	}

	private void r(String s) {
		if (m() > 0) setto(s);
	}

	/* step1ab() gets rid of plurals and -ed or -ing. */
	private void step1ab() {
		if (b[k] == 's') {
			if (ends("sses")) k -= 2;
			else if (ends("ies")) {
				// Allocate extra space if needed for potential growth in later steps
				if (k + 3 >= b.length) {
					char[] nb = new char[b.length + 10];
					System.arraycopy(b, 0, nb, 0, b.length);
					b = nb;
				}
				setto("i");
			}
			else if (b[k - 1] != 's') k--;
		}
		if (ends("eed")) {
			if (m() > 0) k--;
		} else if ((ends("ed") || ends("ing")) && vowelinstem()) {
			k = j;
			if (ends("at")) setto("ate");
			else if (ends("bl")) setto("ble");
			else if (ends("iz")) setto("ize");
			else if (doublec(k)) {
				k--;
				int ch = b[k];
				if (ch == 'l' || ch == 's' || ch == 'z') k++;
			}
			else if (m() == 1 && cvc(k)) {
				// Need space for the extra 'e'
				if (k + 1 >= b.length) {
					char[] nb = new char[b.length + 10];
					System.arraycopy(b, 0, nb, 0, b.length);
					b = nb;
				}
				setto("e");
			}
		}
	}

	/* step1c() turns terminal y to i when there is another vowel in the stem. */
	private void step1c() {
		if (ends("y") && vowelinstem()) {
			b[k] = 'i';
		}
	}

	/* step2() maps double suffices to single ones. */
	private void step2() {
		if (k < 1) return;
		switch (b[k - 1]) {
			case 'a':
				if (ends("ational")) { r("ate"); break; }
				if (ends("tional")) { r("tion"); break; }
				break;
			case 'c':
				if (ends("enci")) { r("ence"); break; }
				if (ends("anci")) { r("ance"); break; }
				break;
			case 'e':
				if (ends("izer")) { r("ize"); break; }
				break;
			case 'l':
				if (ends("bli")) { r("ble"); break; }
				if (ends("alli")) { r("al"); break; }
				if (ends("entli")) { r("ent"); break; }
				if (ends("eli")) { r("e"); break; }
				if (ends("ousli")) { r("ous"); break; }
				break;
			case 'o':
				if (ends("ization")) { r("ize"); break; }
				if (ends("ation")) { r("ate"); break; }
				if (ends("ator")) { r("ate"); break; }
				break;
			case 's':
				if (ends("alism")) { r("al"); break; }
				if (ends("iveness")) { r("ive"); break; }
				if (ends("fulness")) { r("ful"); break; }
				if (ends("ousness")) { r("ous"); break; }
				break;
			case 't':
				if (ends("aliti")) { r("al"); break; }
				if (ends("iviti")) { r("ive"); break; }
				if (ends("biliti")) { r("ble"); break; }
				break;
			case 'g':
				if (ends("logi")) { r("log"); break; }
				break;
		}
	}

	/* step3() deals with -ic-, -full, -ness etc. */
	private void step3() {
		switch (b[k]) {
			case 'e':
				if (ends("icate")) { r("ic"); break; }
				if (ends("ative")) { r(""); break; }
				if (ends("alize")) { r("al"); break; }
				break;
			case 'i':
				if (ends("iciti")) { r("ic"); break; }
				break;
			case 'l':
				if (ends("ical")) { r("ic"); break; }
				if (ends("ful")) { r(""); break; }
				break;
			case 's':
				if (ends("ness")) { r(""); break; }
				break;
		}
	}

	/* step4() takes off -ant, -ence etc., in context <c>vcvc<v>. */
	private void step4() {
		if (k < 1) return;
		switch (b[k - 1]) {
			case 'a':
				if (ends("al")) break; return;
			case 'c':
				if (ends("ance")) break;
				if (ends("ence")) break; return;
			case 'e':
				if (ends("er")) break; return;
			case 'i':
				if (ends("ic")) break; return;
			case 'l':
				if (ends("able")) break;
				if (ends("ible")) break; return;
			case 'n':
				if (ends("ant")) break;
				if (ends("ement")) break;
				if (ends("ment")) break;
				if (ends("ent")) break; return;
			case 'o':
				if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
				if (ends("ou")) break; return;
			case 's':
				if (ends("ism")) break; return;
			case 't':
				if (ends("ate")) break;
				if (ends("iti")) break; return;
			case 'u':
				if (ends("ous")) break; return;
			case 'v':
				if (ends("ive")) break; return;
			case 'z':
				if (ends("ize")) break; return;
			default:
				return;
		}
		if (m() > 1) k = j;
	}

	/* step5() removes a final -e if m() > 1, and changes -ll to -l if m() > 1. */
	private void step5() {
		j = k;
		if (b[k] == 'e') {
			int a = m();
			if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
		}
		if (b[k] == 'l' && doublec(k) && m() > 1) k--;
	}
}
