# Semantic Similarity Search — Improvement Plan

The embedding pre-filter in `LlmInferenceService.findSimilar()` returns inconsistent results: sometimes too many irrelevant records, sometimes missing relevant ones. This document analyzes the root causes and proposes improvements.

## Root Cause Analysis

After deep analysis of the full retrieval pipeline, 9 root causes were identified across three categories: what gets embedded, how queries are processed, and how results are selected.

---

## Category A: What Gets Embedded (Indexing-Side Issues)

### A1. First-Sentence Truncation Discards Semantically Rich Content

**File:** `EmbeddingIndexer.java` (`firstSentence()`)

The `firstSentence()` method truncates text at the first `. `, embedding only the opening fragment:

```
Full text:  "Drug order: Metformin 500mg. Dose: 1.0 Tablet(s) Oral twice daily. Duration: 30 Day(s)"
Embedded:   "Drug order: Metformin 500mg"
```

This means queries like *"oral medications"*, *"twice daily dosing"*, or *"30-day prescriptions"* cannot match because those tokens were never embedded. For obs with group members, the first sentence is just the group concept name (e.g., `"Blood Pressure"`), losing the actual values entirely.

**Impact:** Missing relevant records for queries about dosing, route, frequency, duration, severity, reactions, status — anything after the first sentence.

### A2. Semantic Prefix + First Sentence Creates a Narrow, Rigid Embedding

**File:** `EmbeddingIndexer.java`, `ChartSearchAiConstants.java` (`getEmbeddingPrefix`)

The embedded text is: `prefix + firstSentence(text)`. For example:
```
"Medication prescription: Drug order: Metformin 500mg"
"Patient allergy: Allergy: Penicillin"
```

This creates redundancy — "Medication prescription: Drug order:" is doubly-typed — and the prefix dominates the embedding vector because the first sentence is already short. The embedding represents the *category* more than the *content*.

**Impact:** Records of the same type cluster together regardless of content, reducing intra-type discrimination. A query for "Penicillin" sees all allergies ranked similarly because the prefix dominates.

### A3. No Synonym Expansion in Indexed Text

Clinical records use specific concept names from the OpenMRS dictionary ("Systolic Blood Pressure", "Diastolic Blood Pressure"). The embedding model has some synonym awareness, but clinical abbreviations (BP, Hb, WBC, ARV) and lay terms ("sugar level" for glucose) are poorly represented in all-MiniLM-L6-v2's general-purpose vocabulary.

**Impact:** Queries using informal/abbreviated clinical language get low similarity scores to formally-named records.

---

## Category B: How Queries Are Processed (Query-Side Issues)

### B1. Query-Document Asymmetry

**File:** `LlmInferenceService.java` (`stripQueryStopwords`, `findSimilar`)

The query is: `lowercased, stopword-stripped, bare text`
The document is: `"Semantic Prefix: First sentence of record"`

This structural mismatch hurts cosine similarity. The query `"medications"` is compared against `"Medication prescription: Drug order: Metformin 500mg"` — the query has no prefix, no type label, no structure. The embedding model must bridge this asymmetry through learned semantic understanding alone, which is unreliable for a general-purpose model on domain-specific text.

### B2. No Query Classification or Intent Detection

All queries are processed identically. But clinically, queries fall into distinct retrieval patterns:

| Query Type | Example | Ideal Retrieval |
|---|---|---|
| Category lookup | "any medications?" | All records of type `order` (drug) + `medication_dispense` |
| Specific concept | "Metformin dose?" | Records mentioning Metformin |
| Temporal | "recent lab results?" | Records of type `obs` with test class, sorted by date |
| Cross-record synthesis | "is diabetes controlled?" | HbA1c obs + diabetes condition + medication orders |
| Negation check | "any allergies to penicillin?" | Allergy records + any penicillin-class drugs |

A category lookup like "any medications?" should retrieve ALL drug orders and dispenses, not just the top 10 by similarity. The current pipeline treats this the same as a specific-concept query.

### B3. Aggressive Stopword Removal Can Strip Meaning

**File:** `query-stopwords.txt`

The stopword list includes `"current"`, `"latest"`, and `"known"`. So:
- `"current medications"` → `"medications"` (loses temporal qualifier)
- `"latest lab results"` → `"lab results"` (loses recency intent)
- `"known allergies"` → `"allergies"` (loses established/confirmed intent)

While these are noted as "clinical query fillers," they carry semantic weight that affects what records are relevant.

---

## Category C: How Results Are Selected (Scoring & Filtering Issues)

### C1. Compressed Score Distribution Makes Thresholding Fragile

**Acknowledged in ADR:** all-MiniLM-L6-v2 produces similarity scores in narrow ranges (e.g., 0.20–0.31) for clinical text. With only 0.11 separating the most relevant from the least relevant record, tiny floating-point differences determine inclusion/exclusion. The dual-floor threshold (ratio-based + range-based) and gap detection try to adapt, but they operate on a score distribution that has too little signal.

**Impact:** Small changes in query phrasing cause large changes in which records pass the threshold, explaining the "sometimes more, sometimes less" behavior.

### C2. No Resource-Type Boosting

When a clinician asks "any medications?", drug orders and medication dispenses should be strongly boosted. The current system relies entirely on cosine similarity, which treats all resource types equally. The semantic prefix helps somewhat, but:
- The query has no prefix (asymmetry from B1)
- The prefix effect is diluted by the first-sentence content

### C3. The Adaptive Gap Detection Has Edge Cases

**File:** `LlmInferenceService.java` (`findAdaptiveCutoff`)

The gap detection requires at least 2 prior gaps to compute a meaningful average. For patients with few records or when scores are tightly clustered, this can:
- Cut too aggressively (2 records returned when 5 are relevant)
- Not cut at all (all records above the floor returned when only 3 are relevant)

The `ADAPTIVE_MIN_RECORDS = 2` floor means gap detection never fires before position 2, so it can't cut a clearly irrelevant second result.

---

## Recommended Improvements (Ordered by Impact × Feasibility)

### 1. ★★★ Embed Full Serialized Text Instead of First Sentence Only

**Change:** Remove the `firstSentence()` truncation. Embed the full `prefix + text` instead of `prefix + firstSentence(text)`.

**Why:** The 256-token limit on the WordPiece tokenizer already provides natural truncation for long records. Most serialized records are 1-3 sentences (20-60 tokens), well within the limit. The first-sentence heuristic discards semantically important content (dosing, severity, reactions) that distinguishes records within the same type.

**Risk:** Embedding vectors become more "noisy" for long records. Mitigated by the 256-token truncation and the fact that serializers already produce concise text.

**Files to change:** `EmbeddingIndexer.java` (remove `firstSentence()` calls)

### 2. ★★★ Add Query-Side Prefixing to Match Document Structure

**Change:** Apply the same type of semantic prefixing to the query. Use a lightweight query classifier to prepend context:

```java
// Before embedding, classify and prefix the query
String prefixedQuery = "Clinical question: " + normalizedQuery;
float[] queryVector = embeddingProvider.embed(prefixedQuery);
```

Better yet, use **instruction-prefixed queries** that match the document prefix style. For models like all-MiniLM-L6-v2, a simple `"search_query: "` prefix (which the model was partially trained on) normalizes the query-document gap.

**Why:** Reduces the asymmetry between how queries and documents are embedded, improving cosine similarity accuracy.

**Files to change:** `LlmInferenceService.java` (`findSimilar()` method)

### 3. ★★★ Implement Hybrid Retrieval: Keyword + Semantic

**Change:** Add a BM25-style keyword match alongside embedding similarity. Combine scores:

```
finalScore = α * semanticScore + (1 - α) * keywordScore
```

The keyword score is easy to compute on the already-stored `textContent` field — simple TF-IDF or even substring matching on the query terms against each record's text.

**Why:** This is the single most impactful change for retrieval quality. Pure semantic search misses exact matches (a query for "Metformin" should find all Metformin records regardless of embedding similarity). Pure keyword search misses synonyms. Hybrid catches both.

**Implementation approach:**
- After `stripQueryStopwords()`, extract content keywords
- For each `ChartEmbedding`, compute a keyword overlap score against `textContent`
- Linearly combine with cosine similarity (α = 0.7 semantic, 0.3 keyword is a good starting point)
- Apply the existing threshold/gap detection on the combined score

**Files to change:** `LlmInferenceService.java` (add keyword scoring in `findSimilar()`)

### 4. ★★☆ Add Resource-Type Boosting via Query Classification

**Change:** Add a lightweight keyword-based query classifier that detects the likely target resource types:

```java
Map<String, Set<String>> QUERY_TYPE_MAP = Map.of(
    "medication|drug|prescription|med|rx|dose|dosing|pill|tablet",
        Set.of("order", "medication_dispense"),
    "allergy|allergic|allergen|reaction|anaphylaxis",
        Set.of("allergy"),
    "lab|test|result|level|count|panel|blood work",
        Set.of("obs"),
    "condition|diagnosis|disease|illness|problem",
        Set.of("condition", "diagnosis"),
    "program|enrolled|enrollment|treatment program",
        Set.of("program")
);
```

When a query matches a type pattern, boost records of that type by a configurable factor (e.g., 1.3x their similarity score).

**Why:** A query like "any medications?" should strongly prefer drug orders over allergy records that happen to contain the word "drug allergen." This directly addresses the allergy-over-medication ranking issue mentioned in the ADR.

**Files to change:** `LlmInferenceService.java`, possibly new `QueryClassifier` utility class

### 5. ★★☆ Relax the "Sometimes Too Few" Problem with Dynamic Top-K

**Change:** When the gap detection or ratio threshold reduces results below a configurable minimum (e.g., 3), AND the query appears to be a broad category query ("any medications?", "all conditions"), increase topK to include all records of the matched type.

**Why:** Category queries expect exhaustive results. "What medications is this patient on?" should return ALL medications, not just the top 10 most semantically similar.

**Implementation:** When query classification (from #4) detects a category query, bypass the similarity threshold for records matching the target resource type and include all of them.

**Files to change:** `LlmInferenceService.java` (`findSimilar()`)

### 6. ★★☆ Switch to a Clinical-Domain Embedding Model

**Change:** Replace all-MiniLM-L6-v2 with a model that has wider score separation on clinical text. Candidates:

1. **`BAAI/bge-base-en-v1.5`** (768 dims, ~440MB) — Trained with instruction-aware queries, supports query prefixing natively (`"Represent this sentence for searching: "`). Significantly better retrieval on specialized domains.
2. **`NeuML/pubmedbert-base-embeddings`** — Already tested (per ADR) but rejected because it ranked worse. However, the ADR tested it *with the first-sentence truncation*; re-testing with full text embedding (fix #1) may yield different results.
3. **`sentence-transformers/all-mpnet-base-v2`** (768 dims) — Better than MiniLM on retrieval benchmarks while still general-purpose.

**Why:** Wider score separation means the threshold logic works more reliably. The difference between relevant and irrelevant records should be 0.2-0.3, not 0.05-0.10.

**Risk:** Requires re-indexing all patients (backfill task already supports this). Larger model file (440MB vs 90MB).

### 7. ★☆☆ Add a Cross-Encoder Re-Ranking Stage

**Change:** After the initial cosine similarity retrieval returns the top-K candidates, run a cross-encoder model that scores (query, record) pairs jointly. Re-rank by cross-encoder score and apply the threshold on those scores instead.

**Why:** Cross-encoders are dramatically more accurate than bi-encoders for fine-grained relevance scoring. They consider the full interaction between query and document tokens, catching relationships that independent embedding cannot.

**Implementation:** Load a small cross-encoder ONNX model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90MB). After retrieving the top-20 by cosine similarity, re-score each with the cross-encoder and return the top results.

**Risk:** Adds ~50-200ms per query for re-scoring 20 candidates. Requires a second ONNX model. May be overkill if fixes #1-#4 resolve the issue.

### 8. ★☆☆ Multi-Vector Embedding per Record

**Change:** Instead of one embedding per record, generate 2-3 embeddings capturing different aspects:
1. **Concept embedding**: Just the concept name/type (what it IS)
2. **Value embedding**: The values/details (what it SAYS)
3. **Full embedding**: The complete serialized text

Query similarity is the max across all vectors for a record.

**Why:** Separating "what" from "content" allows a query for "blood pressure" to match the concept vector perfectly even if the values (120/80) would dilute it.

**Risk:** 2-3x storage and indexing time. Significant code complexity increase.

### 9. ★☆☆ LLM-Assisted Query Expansion

**Change:** Before embedding the query, use the local LLM to expand it:

```
Input:    "any medications?"
Expanded: "medications drugs prescriptions drug orders pills tablets dispensed"
```

**Why:** Bridges the vocabulary gap between how clinicians ask questions and how records are stored.

**Risk:** Adds LLM inference latency (~2-5 seconds) to the retrieval step. The LLM is already the bottleneck. Consider using a smaller/faster model or caching expansions.

---

## Priority Recommendation

Start with **#1 + #2 + #3** (full text embedding, query prefixing, hybrid retrieval). These three changes together address the three biggest root causes (truncation, asymmetry, and keyword blindness) with moderate implementation effort and no new model files needed. They can be implemented and tested independently.

If those are insufficient, add **#4** (resource-type boosting) and **#5** (dynamic top-K for category queries).

Changes #6-#9 are more invasive and should only be pursued if the first tier doesn't sufficiently improve quality.
