/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.embedding;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * WordPiece tokenizer compatible with BERT-based models (e.g. all-MiniLM-L6-v2).
 * Loads a vocab.txt file and tokenizes text using the WordPiece algorithm:
 * split into words, then greedily match the longest subword tokens from the vocabulary.
 */
public class WordPieceTokenizer {

	private static final Logger log = LoggerFactory.getLogger(WordPieceTokenizer.class);

	private static final String CLS_TOKEN = "[CLS]";

	private static final String SEP_TOKEN = "[SEP]";

	private static final String UNK_TOKEN = "[UNK]";

	private static final String SUBWORD_PREFIX = "##";

	private static final int MAX_WORD_LENGTH = 200;

	private static final Pattern PUNCTUATION = Pattern.compile(
			"([\\p{Punct}\\u2000-\\u206F\\u2E00-\\u2E7F\\\\'\"`])");

	private static final Pattern WHITESPACE = Pattern.compile("\\s+");

	private final Map<String, Integer> vocab;

	private final int maxSequenceLength;

	private final int clsTokenId;

	private final int sepTokenId;

	private final int unkTokenId;

	public WordPieceTokenizer(String vocabFilePath, int maxSequenceLength) throws IOException {
		this.maxSequenceLength = maxSequenceLength;
		this.vocab = loadVocab(vocabFilePath);
		this.clsTokenId = lookupRequired(CLS_TOKEN);
		this.sepTokenId = lookupRequired(SEP_TOKEN);
		this.unkTokenId = lookupRequired(UNK_TOKEN);
	}

	private int lookupRequired(String token) {
		Integer id = vocab.get(token);
		if (id == null) {
			throw new IllegalStateException("Vocabulary is missing required token: " + token);
		}
		return id;
	}

	private List<Integer> tokenizeToIds(String text) {
		List<Integer> tokenIds = new ArrayList<Integer>();
		String normalized = text.toLowerCase().trim();
		normalized = PUNCTUATION.matcher(normalized).replaceAll(" $1 ");
		String[] words = WHITESPACE.split(normalized);
		for (String word : words) {
			if (word.isEmpty()) {
				continue;
			}
			tokenizeWord(word, tokenIds);
		}
		return tokenIds;
	}

	/**
	 * Tokenizes text into input IDs, attention mask, and token type IDs
	 * suitable for BERT model input.
	 *
	 * @return a {@link TokenizedInput} containing the three arrays
	 */
	public TokenizedInput tokenize(String text) {
		List<Integer> tokenIds = new ArrayList<Integer>();
		tokenIds.add(clsTokenId);
		tokenIds.addAll(tokenizeToIds(text));

		// Truncate to leave room for [SEP]
		if (tokenIds.size() > maxSequenceLength - 1) {
			int originalSize = tokenIds.size();
			tokenIds = new ArrayList<Integer>(tokenIds.subList(0, maxSequenceLength - 1));
			log.debug("Truncated input from {} to {} tokens "
					+ "(maxSequenceLength={}, text length={})",
					originalSize, maxSequenceLength - 1,
					maxSequenceLength, text.length());
		}
		tokenIds.add(sepTokenId);

		int seqLen = tokenIds.size();
		long[] inputIds = new long[seqLen];
		long[] attentionMask = new long[seqLen];
		long[] tokenTypeIds = new long[seqLen];

		for (int i = 0; i < seqLen; i++) {
			inputIds[i] = tokenIds.get(i);
			attentionMask[i] = 1;
			tokenTypeIds[i] = 0;
		}

		return new TokenizedInput(inputIds, attentionMask, tokenTypeIds);
	}

	/**
	 * Tokenizes a query-document pair for cross-encoder input.
	 * Format: [CLS] query_tokens [SEP] doc_tokens [SEP]
	 * token_type_ids: 0 for query, 1 for document.
	 */
	public TokenizedInput tokenizePair(String query, String document) {
		List<Integer> queryIds = tokenizeToIds(query);
		List<Integer> docIds = tokenizeToIds(document);

		// Budget: [CLS] + query + [SEP] + doc + [SEP] = 3 special tokens
		int maxContentLen = maxSequenceLength - 3;
		// Give query up to 1/4 of budget, rest to document
		int maxQueryLen = Math.max(16, maxContentLen / 4);
		if (queryIds.size() > maxQueryLen) {
			queryIds = queryIds.subList(0, maxQueryLen);
		}
		int maxDocLen = maxContentLen - queryIds.size();
		if (docIds.size() > maxDocLen) {
			docIds = docIds.subList(0, maxDocLen);
		}

		int seqLen = 1 + queryIds.size() + 1 + docIds.size() + 1;
		long[] inputIds = new long[seqLen];
		long[] attentionMask = new long[seqLen];
		long[] tokenTypeIds = new long[seqLen];
		Arrays.fill(attentionMask, 1);

		int pos = 0;
		inputIds[pos] = clsTokenId;
		pos++;

		for (int id : queryIds) {
			inputIds[pos] = id;
			pos++;
		}

		inputIds[pos] = sepTokenId;
		pos++;

		for (int id : docIds) {
			inputIds[pos] = id;
			tokenTypeIds[pos] = 1;
			pos++;
		}

		inputIds[pos] = sepTokenId;
		tokenTypeIds[pos] = 1;

		return new TokenizedInput(inputIds, attentionMask, tokenTypeIds);
	}

	/**
	 * Returns true if the word would be split into multiple subword
	 * tokens. A single-token word means the vocabulary contains it
	 * as a whole unit; a multi-token word means the model sees it
	 * as fragments.
	 */
	public boolean isSplitWord(String word) {
		if (word == null || word.isEmpty() || word.length() > MAX_WORD_LENGTH) {
			return true;
		}
		String lower = word.toLowerCase();
		return vocab.get(lower) == null;
	}

	private void tokenizeWord(String word, List<Integer> tokenIds) {
		if (word.length() > MAX_WORD_LENGTH) {
			tokenIds.add(unkTokenId);
			return;
		}

		int start = 0;
		boolean isBad = false;
		List<Integer> subTokens = new ArrayList<Integer>();

		while (start < word.length()) {
			int end = word.length();
			Integer matchedId = null;

			while (start < end) {
				String substr = word.substring(start, end);
				if (start > 0) {
					substr = SUBWORD_PREFIX + substr;
				}
				Integer id = vocab.get(substr);
				if (id != null) {
					matchedId = id;
					break;
				}
				end--;
			}

			if (matchedId == null) {
				isBad = true;
				break;
			}

			subTokens.add(matchedId);
			start = end;
		}

		if (isBad) {
			tokenIds.add(unkTokenId);
		} else {
			tokenIds.addAll(subTokens);
		}
	}

	private Map<String, Integer> loadVocab(String filePath) throws IOException {
		Map<String, Integer> vocabMap = new HashMap<String, Integer>();
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8));
		try {
			String line;
			int index = 0;
			while ((line = reader.readLine()) != null) {
				vocabMap.put(line.replace("\r", ""), index);
				index++;
			}
		}
		finally {
			reader.close();
		}
		return vocabMap;
	}

	public static class TokenizedInput {

		private final long[] inputIds;

		private final long[] attentionMask;

		private final long[] tokenTypeIds;

		public TokenizedInput(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
			this.inputIds = inputIds;
			this.attentionMask = attentionMask;
			this.tokenTypeIds = tokenTypeIds;
		}

		public long[] getInputIds() {
			return inputIds;
		}

		public long[] getAttentionMask() {
			return attentionMask;
		}

		public long[] getTokenTypeIds() {
			return tokenTypeIds;
		}

		public int getLength() {
			return inputIds.length;
		}
	}
}
