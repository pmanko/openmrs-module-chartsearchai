# Chart Search AI - Architectural Decisions

This document captures the architectural decisions made for the Chart Search AI module, including alternatives evaluated and the reasoning behind the chosen approaches.

## Problem Statement

Clinicians using OpenMRS often see hundreds of patients daily with limited time per encounter. Finding specific information in a patient's chart — especially across years of records, unstructured notes, and multiple widget pages — is slow and error-prone. A Chart Search feature should help clinicians quickly find what they need by asking natural language questions about a patient's chart.

## Decision 1: What value does an LLM add?

### Analysis

For most chart search queries (~80%), the question maps directly to structured data lookups (e.g., "What are her current medications?" is just a database query). An LLM adds genuine value only for:

- **Natural language query parsing** (~80% of value): Translating "Has she ever had a bad reaction to penicillin?" into a search for allergy records related to penicillin-class drugs.
- **Unstructured text search** (~15%): Finding information in free-text notes that requires language comprehension, not just keyword matching.
- **Synthesis across records** (~5%): Interpreting trends across multiple values over time (e.g., "Is her diabetes getting better or worse?").

An LLM adds no value for structured data lookup, concept synonym matching (OpenMRS `ConceptService` already handles this), filtering/sorting, or display formatting.

### Decision

Use an LLM only as a fallback for hard cases. The primary search path should be deterministic and fast. The LLM earns its compute cost only when simpler approaches cannot solve the problem.

## Decision 2: Overall architecture — RAG vs. alternatives

### Options evaluated

#### Option A: Full FHIR bundle to LLM
Send a complete FHIR bundle with all patient resources to an LLM for processing.

**Rejected because:**
- A patient with 5 years of visits could have thousands of resources, producing 500K-2M tokens — far exceeding any local model's context window (4-8K typical).
- LLMs lose information buried in the middle of long contexts ("lost in the middle" problem).
- Processing a massive bundle on a local model would take minutes, not seconds.
- Maximizes hallucination risk — the model sees lots of clinical terminology and confidently connects dots that don't exist.

#### Option B: Fine-tuned local model (no retrieval)
Fine-tune a small model to generate SQL or API calls from natural language.

**Rejected because:**
- Requires substantial labeled training data (question → correct API call pairs) that doesn't exist yet.
- SQL/API generation errors are silent and dangerous.
- Must re-fine-tune when schema or forms change.
- Training data problem alone makes this impractical for v1.

#### Option C: Traditional search (no LLM at all)
Full-text search index (Lucene/Solr/Elasticsearch) over patient data.

**Kept as fallback layer.** Solves 70% of the problem with 20% of the complexity. No hallucination risk, no compute requirements, works offline. Weakness: no natural language understanding or synthesis.

#### Option D: Agent/tool-use pattern
Give the LLM access to OpenMRS APIs as tools and let it autonomously decide what to call.

**Deferred to v2+.** Architecturally elegant but demands more capable models than the deployment environment can support. Small local models (2-8B) are weak at tool use and multi-step reasoning. Latency from multiple sequential LLM → API → LLM loops is problematic in a 90-second encounter.

#### Option E: Pre-computed summaries (batch processing)
Generate patient summaries offline ahead of time. Search summaries at query time.

**Kept as complement.** Good for common queries (active meds, allergies, problem list). Weakness: stale data, doesn't handle unexpected/novel questions. Best combined with real-time retrieval.

#### Option F: RAG (Retrieval Augmented Generation) — CHOSEN
Retrieve relevant records first using deterministic search, then use the LLM only for query understanding and response formatting.

**Chosen because:**
- Retrieval is deterministic and auditable — every piece of data has a traceable source.
- LLM never invents facts; it only works with data explicitly provided.
- Minimal hallucination surface area.
- Works with small local models since query parsing and response synthesis are short-context tasks.

### Decision

Use RAG with a two-step approach:
1. **Deterministic retrieval**: Java code queries OpenMRS via existing services + embedding similarity search to find the most relevant records.
2. **Response synthesis**: LLM receives the retrieved records and the clinician's question, then produces a concise answer with citations.

## Decision 3: Embedding approach — semantic search index

### Options evaluated for retrieval

#### Option A: Targeted queries with manual concept mapping
Map each query type to specific OpenMRS resource types and concept codes.

**Weakness:** Requires manually mapping every possible query pattern to the right resources. Misses things you wouldn't think to query — e.g., a free-text visit note mentioning "mother had breast cancer."

#### Option B: Concept graph traversal
Use the OpenMRS concept dictionary as a knowledge graph. Map query terms to SNOMED concepts, traverse the hierarchy, query matching records.

**Kept as complement.** Fast (milliseconds), deterministic, leverages existing concept dictionary. Weakness: only works for structured/coded data, misses free-text entirely.

#### Option C: Semantic search index with embeddings — CHOSEN
Pre-index all patient data with vector embeddings. At query time, find relevant records by embedding similarity.

**Chosen because:**
- No manual mapping needed — similarity search catches things you wouldn't think to query.
- Works with both structured and unstructured data.
- The embedding model is tiny (~80MB, runs on CPU in milliseconds).
- Query-time cost is just a vector similarity search — very fast.
- Per-patient search space is small enough (typically <2000 records) for brute-force in-memory cosine similarity.

#### Option D: Clinical concept extraction pipeline (NLP at write time)
Use rule-based NLP (cTAKES, MedSpaCy) to extract structured facts from all data at write time.

**Deferred.** Zero query-time AI cost, works on unstructured text. Weakness: extraction pipeline needs tuning per site, adds processing to the write path.

#### Option E: Map-reduce over chart segments
Split patient chart into time-based segments, classify each for relevance, only send relevant segments to LLM.

**Deferred to v2+.** Handles arbitrarily large charts but adds infrastructure complexity.

### Decision

Semantic search index as the primary retrieval mechanism, with concept graph traversal as a complement for structured data.

## Decision 4: Concise text as LLM input format

### Analysis

Standard serialization formats (FHIR JSON, full OpenMRS domain objects) are poor formats for LLM context windows:

- **Extremely verbose**: A single blood pressure observation is ~800 tokens in FHIR JSON vs. ~15 tokens in compressed form. On a small model with 4-8K context, this matters enormously.
- **Deeply nested**: `coding` inside `code` inside `component` inside `Observation`. Small LLMs are worse at extracting information from nested structures.
- **Redundant metadata**: System URIs, references, profiles waste context tokens.

### Decision

Retrieve data via OpenMRS service APIs (ObsService, ConditionService, PatientService, OrderService, DiagnosisService) and convert records into flat, concise clinical text using `ClinicalTextSerializer` implementations. This gives ~10x token efficiency while preserving clinical meaning.

Example:
```
FHIR JSON: ~800 tokens
Serialized: "Systolic Blood Pressure: 120 mmHg (ABNORMAL)"  ~10 tokens
```

## Decision 5: Embedding granularity

### Options

| Granularity | Pros | Cons |
|---|---|---|
| Individual record | Precise retrieval, fine-grained citations | Many embeddings per patient, records in isolation lose context |
| Per encounter | Groups related data naturally, fewer embeddings | Large encounters produce long text, less precise |
| Per clinical category | Matches how clinicians think | Arbitrary groupings, large text chunks |

### Decision

Embed at the **individual record level**. Each record is serialized to concise clinical text (e.g., `"Systolic Blood Pressure: 120 mmHg (ABNORMAL)"`). This keeps embeddings small and precise while giving the similarity search enough context to work with.

## Decision 6: Embedding model

### Decision

Semantic vectors via **all-MiniLM-L6-v2** running in-process through ONNX Runtime. ~90MB model file, runs on CPU, no GPU needed. Produces 384-dimensional vectors. Requires two files configured via `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`. Captures semantic meaning — effective for clinical queries where synonyms and related concepts matter (e.g., "hypertension" and "high blood pressure" are recognized as related).

A term-frequency hashing approach was considered as a simpler alternative (no model file needed, keyword-overlap retrieval). It was rejected because it cannot capture semantic similarity — for a clinical question like "any infections?", it would find records containing the word "infection" but miss "tuberculosis", "malaria", or "UTI". This defeats the purpose of pre-filtering, since the LLM with the full chart would catch all of these.

## Decision 7: Vector storage — MySQL, not a vector database

### Analysis

MySQL does not natively support vector embeddings (native `VECTOR` type was added in MySQL 9.0+, but OpenMRS deployments typically run MySQL 5.7 or 8.x).

However, a vector database is unnecessary for this use case because:
- Search is **per-patient**, not across all patients
- A patient with 2000 records means 2000 vector comparisons — trivial in Java (microseconds)
- Embeddings are stored as BLOBs (~1.5KB per record for 384 dimensions)

### Decision

Store embeddings as `MEDIUMBLOB` in a regular MySQL table (`chartsearchai_embedding`), indexed by `patient_id`. Load a patient's embeddings into memory and compute cosine similarity in Java. Zero new infrastructure.

The `UNIQUE KEY (resource_type, resource_id)` constraint prevents duplicate embeddings and enables upsert on re-index.

## Decision 8: Index population strategy

### Decision

Three complementary strategies ensure embeddings stay current:

- **On-demand**: When a clinician queries a patient's chart for the first time and no embeddings exist, `LlmInferenceService` triggers `EmbeddingIndexer.indexPatient()` before running the similarity search. This means embeddings are created lazily — no setup required.
- **Incremental via AOP**: After-returning advice on six OpenMRS services triggers re-indexing when clinical data changes:
  - `EncounterService` — incremental encounter indexing (upserts only the encounter's obs and diagnoses)
  - `ObsService` — full patient re-index on save/void/unvoid/purge of standalone observations
  - `ConditionService` — full patient re-index on save/void/unvoid/purge
  - `DiagnosisService` — full patient re-index on save/void/unvoid/purge
  - `PatientService` — full patient re-index on allergy changes; on `mergePatients`, re-indexes the preferred patient and deletes the non-preferred patient's embeddings
  - `OrderService` — full patient re-index on save/void/unvoid/purge/discontinue
- **Backfill**: A one-time scheduled task (`EmbeddingIndexTask`) indexes all patients that don't yet have embeddings. Handles initial population when the module is installed on a system with existing data. Skips already-indexed patients, so it is safe to re-run and picks up where it left off if stopped. Admins trigger it from the scheduler UI; it does not run automatically.

## Decision 9: Text serialization — ClinicalTextSerializer pattern

### Decision

A generic `ClinicalTextSerializer<T>` interface with one implementation per OpenMRS resource type:

| Serializer | Output example |
|---|---|
| `ObsTextSerializer` | `"Systolic Blood Pressure: 120 mmHg (ABNORMAL). Note: Taken after exercise"` |
| `ConditionTextSerializer` | `"Condition: Type 2 Diabetes Mellitus. Status: ACTIVE. Verification: CONFIRMED"` |
| `AllergyTextSerializer` | `"Allergy: Penicillin (DRUG). Severity: Severe. Reactions: Anaphylaxis, Rash"` |
| `DiagnosisTextSerializer` | `"Diagnosis: Malaria. Certainty: CONFIRMED. Rank: Primary"` |
| `OrderTextSerializer` | `"Order: Complete Blood Count. Action: NEW. Urgency: STAT. Reason: Suspected anemia"` |

Key design choices:
- Serializers produce concise text without dates (dates are redundant with the record's sort order in the prompt).
- `ObsTextSerializer` flattens group members into the parent obs text.
- Obs interpretation (`NORMAL`, `ABNORMAL`, `CRITICALLY_ABNORMAL`) and comments are included.
- Units are extracted from `ConceptNumeric`, not `Concept` (which has no `getUnits()` in OpenMRS 2.6.x).

## Decision 10: Single LLM architecture with optional embedding pre-filter

### Context

The current architecture (Decisions 3–9) uses a two-model pipeline: an embedding model for semantic search retrieval, plus a generative LLM for query understanding and response synthesis. This requires vector storage, cosine similarity search, and an embedding indexing strategy.

However, if two conditions are met, this complexity can be eliminated entirely:

1. **The full patient chart fits within the LLM's context window.** A patient with 2000 records, each serialized to ~15 tokens by the `ClinicalTextSerializer`, produces ~30K tokens. Models like Mistral 7B (32K context) and Llama 3.2 3B (128K context) can accommodate this.
2. **A local LLM is available with acceptable latency.** Small quantized models (1.5B–3.8B parameters) can run on CPU via [java-llama.cpp](https://github.com/kherud/java-llama.cpp), which provides Java JNI bindings to llama.cpp and is available on Maven (`de.kherud:llama`). This keeps the module self-contained with no external service dependency.

### Simplified architecture

```
Patient records → ClinicalTextSerializers → All clinical text → LLM → Answer
```

No embedding model, no vector storage, no cosine similarity search, no indexing strategy. The LLM receives all serialized patient records and answers the query directly.

### Advantages over the embedding-based approach

- **Simpler architecture**: One model, no vector storage or indexing infrastructure.
- **More accurate**: The LLM sees the full patient chart and can reason across all records. It understands clinical context, reasoning, and nuance far better than cosine similarity on vectors. For example:
  - *"Has the patient's blood pressure been improving?"* — The LLM can reason over trends across multiple observations over time, comparing values and dates. Vector search just returns individual records that mention blood pressure, with no understanding of whether the numbers are going up or down.
  - *"Any contraindications for prescribing ibuprofen?"* — The LLM can connect an NSAID allergy, a GI bleeding history, and a kidney condition to flag the risk. Vector search might miss records that don't lexically match "ibuprofen" — a GI bleeding episode recorded as `"Condition: Peptic Ulcer. Status: RESOLVED"` has low cosine similarity to the query but is clinically critical.
  - *"Is this patient a fall risk?"* — The LLM can synthesize age, medications with dizziness side effects, a prior hip fracture, and low blood pressure readings into a clinical assessment. Vector search would only find records that happen to contain words similar to "fall risk."
  - *"Should we be concerned about her liver?"* — The LLM can correlate elevated ALT/AST lab results, a hepatotoxic medication history, and alcohol use documented in a social history note. Vector search treats each record independently and cannot connect these dots.
  - *"Is this patient adherent to their TB treatment?"* — The LLM can compare the expected treatment timeline against actual dispensing dates, missed appointment records, and clinician notes about adherence counseling. Vector search would return records mentioning "TB" but cannot evaluate whether the treatment schedule was followed.
  - *"What might be causing her recurrent headaches?"* — The LLM can cross-reference the headache obs with a recent hypertension diagnosis, a new medication with headache as a known side effect, and elevated stress noted in a mental health screening. Vector search finds records containing "headache" but cannot reason about causality across unrelated record types.
  - *"Is it safe to give this child the measles vaccine today?"* — The LLM can check the immunization history for prior doses, look for active febrile illness in today's vitals, review allergy records for egg or neomycin sensitivity, and check for immunosuppressive conditions. Vector search would match on "measles" or "vaccine" but cannot perform the multi-factor safety assessment.
  - *"Summarize this patient's pregnancy history"* — The LLM can piece together gravidity/parity obs, antenatal visit encounters, delivery records, complication diagnoses, and neonatal outcomes across multiple pregnancies spanning years. Vector search returns individual records but cannot weave them into a coherent narrative.
  - *"Why was this patient's ARV regimen changed?"* — The LLM can correlate the regimen change order with a recent viral load result showing treatment failure, a drug resistance test, and a clinician's note about side effects. Vector search finds records mentioning "ARV" but cannot infer the clinical reasoning behind the switch.
  - *"Does this patient need a referral to a specialist?"* — The LLM can evaluate persistent abnormal lab trends despite treatment, a worsening condition status, and failed interventions to suggest that the current care level may be insufficient. Vector search has no concept of "enough has been tried" — it merely retrieves similar-sounding records.
  - *"Is this patient at risk for diabetes complications?"* — The LLM can connect an HbA1c trend showing poor glycemic control, a recent retinal screening referral, peripheral neuropathy symptoms in the review of systems, and a microalbuminuria lab result. Vector search for "diabetes complications" would miss the neuropathy symptoms recorded as `"Obs: Tingling in feet"` and the kidney marker recorded as `"Order: Urine Albumin-Creatinine Ratio"`.
  - *"What happened during the patient's last admission?"* — The LLM can reconstruct a timeline from the admission encounter, daily vitals, medication orders, procedure notes, consultant diagnoses, and discharge summary across dozens of records. Vector search returns fragments but cannot sequence them into a coherent clinical story.
- **No retrieval errors**: Embedding-based retrieval can miss relevant records if the query and record text are semantically distant. Direct LLM inference eliminates this failure mode.
- **No index staleness**: No need for batch or incremental indexing. Every query sees the current chart state.

### Comparison with a knowledge graph approach

A knowledge graph represents clinical data as entities and explicit relationships (e.g., `Patient → has_condition → Diabetes`, `Metformin → treats → Diabetes`, `Metformin → contraindicated_with → Renal Failure`). This is a fundamentally different approach from direct LLM inference, with distinct tradeoffs.

#### Where a knowledge graph is stronger

- **Deterministic and auditable**: Every answer traces to explicit relationships in the graph. There is no hallucination risk — if a contraindication edge exists, it is reported; if it does not, it is not invented.
- **Fast**: Graph traversal completes in milliseconds with no heavy compute requirements.
- **Structured reasoning**: Queries like "What drugs interact with this patient's current medications?" follow explicit edges rather than relying on a model's probabilistic understanding.

#### Where direct LLM inference is stronger

- **No schema to build or maintain**: A clinical knowledge graph requires someone to model every entity type, relationship, and rule upfront. Who defines the `contraindicated_with` relationships for every drug-condition pair? Who adds new relationships when clinical guidelines change? Who maintains the graph when the concept dictionary evolves? This is a massive ongoing investment.
- **Handles unstructured data**: A clinician's note saying "patient reports tingling in feet" is invisible to a knowledge graph unless someone runs NLP extraction to create structured entities first. The LLM reads and understands it directly.
- **Handles novel queries**: The graph can only answer questions about relationships someone thought to model. A query like "Is this patient isolated and at risk for depression?" drawing from social history notes, missed appointments, and living situation obs is impossible in a graph that does not have these relationship types defined.
- **Implicit reasoning**: "Why was this patient's ARV regimen changed?" requires inferring causality from the temporal proximity of a viral load result, a resistance test, and a regimen change order. A knowledge graph would need an explicit `caused_by` edge that no one created.
- **Built-in clinical knowledge**: The LLM brings clinical knowledge from training — it knows that ibuprofen is an NSAID, that NSAIDs are risky with peptic ulcers, and that metformin requires renal monitoring. A knowledge graph only knows what has been explicitly encoded into it.
- **Natural language interface**: The LLM natively understands "Is her sugar under control?" as a question about glycemic management. A knowledge graph requires the query to be translated into a structured graph traversal.

#### Practical considerations for OpenMRS deployments

OpenMRS already has a concept dictionary with some relationship structure (concept classes, concept mappings, drug-concept associations), but this is far from a full clinical knowledge graph with drug interactions, contraindications, risk factor models, and causal relationships. Building and maintaining such a graph requires dedicated clinical informatics expertise that low-resource settings typically lack.

The direct LLM inference approach is more practical for these settings: deploy a single model file and the module works out of the box with clinical reasoning capabilities, no graph construction or maintenance required. The tradeoff is accepting probabilistic answers (with hallucination risk) instead of deterministic graph traversal.

### Hallucination risk comparison

Both the direct LLM inference and embedding-based RAG approaches carry hallucination risk, but the failure modes are different.

#### Embedding-based RAG hallucinations

With RAG, the LLM only sees the top-K retrieved records (default 15, configurable via `chartsearchai.embedding.topK`). This limits how much it can hallucinate *about*, but it introduces a different risk: hallucinating from *missing context*. Examples:

- The retrieval step misses a relevant record (e.g., a resolved penicillin allergy) because the query and record text are semantically distant. The LLM confidently says "no known drug allergies" based on the records it received.
- The LLM sees a single elevated blood pressure reading without the surrounding context of the patient exercising beforehand (that context is in a different obs comment that was not retrieved). It may overstate the clinical significance.
- The LLM receives a medication order and a lab result but not the clinician's note explaining why the medication was started. It invents a plausible but incorrect reason.

#### Direct LLM inference hallucinations

With direct inference, the LLM sees the full patient chart. It will not miss relevant records, but more input means more opportunity to hallucinate from *over-interpreting context*. Examples:

- The LLM sees a headache obs and a new hypertension medication started the same week. It infers the medication caused the headache, when the timing was coincidental.
- The LLM notices elevated liver enzymes and a hepatitis B diagnosis. It concludes the hepatitis is active and causing the elevation, when the enzymes were actually elevated due to a statin started recently.
- The LLM sees multiple records mentioning fatigue across several visits and synthesizes a narrative about chronic fatigue syndrome, when each instance had a different, resolved cause.

#### Mitigation

The mitigation is the same for both approaches: **never present LLM output as clinical fact**. The module should always show the source records alongside the LLM's answer so the clinician can verify. The direct inference approach actually makes this easier — since there is no retrieval step, every record the LLM saw is known and can be cited. With RAG, the clinician must additionally trust that the retrieval step found the right records.

### Source citations

Source citations are straightforward because we control exactly what the LLM sees. Each serialized record is numbered sequentially before being included in the prompt (sorted most recent first):

```
[1] Systolic Blood Pressure: 120 mmHg (ABNORMAL)
[2] Condition: Type 2 Diabetes Mellitus. Status: ACTIVE
[3] Order: Metformin. Action: NEW
[4] HbA1c: 8.2%
```

The system prompt instructs the LLM to cite record numbers in brackets and respond with a JSON object. A GBNF grammar (`json-answer.gbnf`) constrains the LLM output to the exact format `{"answer": "...", "citations": [1, 2]}`, making it structurally impossible for the LLM to produce malformed citations. The `citations` array is parsed directly as structured data — no regex parsing of free text is needed.

Example LLM output:
```json
{"answer": "The patient's diabetes appears poorly controlled. Their most recent HbA1c was 8.2% [4], above the target of 7%, despite being on Metformin [3].", "citations": [4, 3]}
```

On the Java side, each citation number maps back to a `resource_type` + `resource_id` pair maintained in an ordered list (`RecordMapping`) during prompt construction. The UI can then link each citation directly to the source record in OpenMRS, allowing the clinician to verify every claim with one click.

As a safety net, any slash-separated citations that small LLMs occasionally produce in the answer text (e.g., `[5/12]`) are normalized to `[5], [12]` before returning to the user. This is cosmetic only — the authoritative citations come from the structured `citations` array.

### Candidate models

| Model | Quantized Size (Q4_K_M) | RAM | Context Window | CPU Speed (approx.) |
|-------|------------------------|-----|----------------|---------------------|
| Qwen 2.5 1.5B | ~1GB | ~2GB | 32K tokens | ~40–50 tok/s |
| Llama 3.2 3B | ~2GB | ~4GB | 128K tokens | ~20–30 tok/s |
| Phi-3 Mini 3.8B | ~2GB | ~4GB | 4K tokens (128K variant available) | ~15–25 tok/s |
| Mistral 7B | ~4GB | ~8GB | 32K tokens | ~10–15 tok/s |
| Qwen 2.5 7B | ~4GB | ~8GB | 128K tokens | ~8–12 tok/s |
| Gemma 2 9B Instruct | ~5GB | ~10GB | 8K tokens | ~5–10 tok/s |
| Qwen 2.5 14B | ~8GB | ~14GB | 128K tokens | ~3–6 tok/s |

### Recommended model: Llama 3.2 3B

Llama 3.2 3B is the best fit for this use case. Its 128K token context window can hold approximately 6,000 serialized patient records (~15 tokens each), which comfortably accommodates even the largest patient charts. At 3B parameters with Q4_K_M quantization, it is ~2GB on disk and runs at ~20–30 tokens per second on CPU — fast enough for acceptable clinical use.

The alternatives each have a significant limitation for this use case:

- **Qwen 2.5 1.5B** is faster and smaller but its 32K context window limits it to ~2,000 records, and its reasoning capability is weaker at 1.5B parameters.
- **Phi-3 Mini 3.8B** has slightly better reasoning per parameter than Llama 3.2 3B, but its default 4K context window is far too small for full patient charts. The 128K variant exists but is slower on CPU due to the longer context handling.
- **Mistral 7B** has strong reasoning but at 7B parameters it is noticeably slower on CPU (~10–15 tok/s) and requires ~8GB RAM, which may be prohibitive in low-resource deployments.
- **Qwen 2.5 7B** offers strong instruction following and a 128K context window at 7B parameters. A good upgrade path from 3B models for servers with 8–10GB RAM.
- **Gemma 2 9B Instruct** has excellent reasoning and instruction following at 9B parameters, but its 8K context window limits it to ~500 records without embedding pre-filtering. Requires ~10GB RAM.
- **Qwen 2.5 14B** provides the best reasoning of the CPU-viable options with a 128K context window, but requires ~14GB RAM and runs at ~3–6 tok/s on CPU. Best for well-resourced deployments.

All models run via java-llama.cpp with Q4_K_M quantization in GGUF format.

### Licensing

Llama 3.2 3B is free for both research and commercial use under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). No fees or royalties apply. The license permits use, modification, and distribution.

The model is not technically "open source" by the [Open Source Initiative's definition](https://opensource.org/blog/metas-llama-license-is-still-not-open-source) — Meta's license includes restrictions that do not meet OSI criteria. However, for practical purposes the only meaningful restriction is that products with over 700 million monthly active users require a separate license from Meta, which is not a concern for OpenMRS.

The license requires the following attribution to be included in all distributed copies: *"Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (C) Meta Platforms, Inc. All Rights Reserved."*

### Deployment and memory requirements

The model runs **in-process** inside the OpenMRS JVM via [java-llama.cpp](https://github.com/kherud/java-llama.cpp) JNI bindings. No separate web server, process, or HTTP service is needed. The deployment consists of two files:

1. The `.omod` module file (includes the java-llama.cpp dependency)
2. The `.gguf` model file (placed in the OpenMRS application data directory)

The model path is configured via the `chartsearchai.llm.modelFilePath` global property. The model loads into memory on first query and stays resident for subsequent requests. If the model path global property is changed, the model is automatically unloaded and the new model is loaded on the next query — no restart required.

Inference uses temperature 0.0 and a fixed seed for deterministic output. This ensures that identical inputs produce consistent answers, which is important for clinical trust and for the answer cache to be meaningful.

```
OpenMRS JVM
  └── chartsearchai module
        └── LlmInferenceService
              └── java-llama.cpp (JNI)
                    └── llama.cpp (native C++)
                          └── loads GGUF file from disk
```

### Model size trade-offs

The module works with any GGUF-format model. Larger models produce better responses (more accurate, better instruction following, fewer hallucinations) but require more RAM and are slower on CPU. All figures below are for Q4_K_M quantization.

| Model | File Size | RAM (model + KV cache) | Total with OpenMRS JVM | CPU Inference Speed |
|-------|-----------|------------------------|------------------------|---------------------|
| **3B** (e.g. Llama 3.2 3B) | ~2GB | ~3–4GB | ~5–6GB | ~5–15 tokens/sec |
| **7B** (e.g. Qwen 2.5 7B, Mistral 7B) | ~4GB | ~6–8GB | ~8–10GB | ~3–8 tokens/sec |
| **9B** (e.g. Gemma 2 9B Instruct) | ~5GB | ~8–10GB | ~10–12GB | ~3–6 tokens/sec |
| **14B** (e.g. Qwen 2.5 14B) | ~8GB | ~12–14GB | ~14–16GB | ~2–4 tokens/sec |

**3B models** are the most deployable in low-resource settings but struggle with strict instruction following — they tend to produce verbose responses, add unsolicited commentary, and hedge when they should give a direct "not found" answer. Few-shot examples in the system prompt help but do not fully solve this.

**7B models** are the recommended middle ground — significantly better instruction following and clinical reasoning, while still feasible on a server with 10GB RAM.

**9B models** (e.g. Gemma 2 9B Instruct) offer excellent reasoning and instruction following. Note that Gemma 2's 8K context window is smaller than Llama or Qwen models, so embedding pre-filtering is strongly recommended.

**14B models** (e.g. Qwen 2.5 14B) provide the best response quality among CPU-viable options, with strong reasoning and a 128K context window. They require 14–16GB total RAM and produce slower inference (~2–4 tok/s). Suitable for well-resourced deployments where response quality is prioritized over speed.

A server running OpenMRS typically uses 1–2GB for the JVM heap. A 4GB machine is insufficient to run this module — the LLM alone requires at least 3–4GB for the smallest viable model.

### Hardware requirements

The module requires sufficient RAM for both the OpenMRS JVM and the LLM model:
- **Minimum**: ~6GB total (1–2GB JVM + 3–4GB for a 3B model). Latency of a few seconds per query is expected with CPU inference.
- **Recommended**: ~10GB total for a 7B model, which provides significantly better response quality.
- The embedding pre-filter (default: enabled) reduces the number of tokens sent to the LLM, which improves both response quality and latency for large patient charts.

### Decision

A single architecture is used: all queries go through the LLM for reasoning and synthesis. An embedding pre-filter (`chartsearchai.embedding.preFilter`, default `true`) narrows the patient chart to the most relevant records before sending them to the LLM (default top 15, configurable via `chartsearchai.embedding.topK`). This solves the "lost in the middle" problem where small LLMs struggle to find relevant information in large contexts. Set to `false` to send the full chart instead.

Embeddings are indexed on first patient chart access and kept up to date automatically via AOP hooks on data changes. A bulk backfill task is also available for pre-indexing all patients.

Embeddings use all-MiniLM-L6-v2 via ONNX Runtime (~90MB model file, configured via `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`).

### Medical imaging data (X-rays, scans, etc.)

The recommended Llama 3.2 3B model is text-only. Multimodal variants that can interpret images directly (Llama 3.2 11B and 90B) are too large for CPU inference in low-resource settings.

#### Current approach: rely on text reports

For v1, the module relies on the text reports that accompany imaging studies. In OpenMRS, imaging results typically have an associated obs with the radiologist's interpretation (e.g., `"Obs: Chest X-ray findings: bilateral infiltrates consistent with pneumonia"`). This text is already captured by `ObsTextSerializer` and flows through the existing pipeline — the embedding pre-filter can match it by similarity, and the LLM can reason over it.

#### Future options for direct image interpretation

These require either hardware beyond current low-resource constraints or an external service:

- **Multimodal LLM (Llama 3.2 11B/90B)**: Can interpret images alongside text but requires GPU or significantly more RAM than available in target deployments.
- **Specialized medical imaging models**: Small models trained for specific tasks (e.g., CheXNet for chest X-ray classification). Each covers only one type of image, so multiple models would be needed, adding significant complexity and storage requirements.
- **Cloud API**: Offload image interpretation to a cloud-hosted multimodal model. Introduces external dependency, latency, cost, and data privacy concerns that conflict with the self-contained, offline-capable design goals.
- **OCR for paper forms**: Convert photos of handwritten or printed paper forms to text at write time. The extracted text then flows through the existing serializer pipeline. This is more feasible than general medical image interpretation and addresses a common need in low-resource settings where paper forms are digitized by photographing them.

Direct image interpretation is deferred to future work, pending either capable multimodal models that run on CPU within low-resource constraints, or a decision to support optional cloud API integration for sites that have connectivity and consent to external processing.

## Decision 11: REST API and guardrails

### REST endpoints

The module exposes two REST endpoints for chart search queries. Both require the `AI Query Patient Data` privilege and are registered under the OpenMRS `webservices.rest` module namespace.

#### Synchronous endpoint

```
POST /ws/rest/v1/chartsearchai/search
{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?"
}
```

Response:
```json
{
  "answer": "The patient is currently on...[1]...[3]",
  "disclaimer": "This response is AI-generated and may not be accurate...",
  "references": [
    { "index": 3, "resourceType": "order", "resourceId": 789, "date": "2025-03-15" },
    { "index": 1, "resourceType": "obs", "resourceId": 456, "date": "2025-01-10" }
  ]
}
```

#### Streaming endpoint (SSE)

```
POST /ws/rest/v1/chartsearchai/search/stream
{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?"
}
```

Returns a `text/event-stream` with three event types:
- `token` — a chunk of the answer text, streamed as generated
- `done` — final JSON with the complete answer, references (with `index`, `resourceType`, `resourceId`, `date`), and disclaimer
- `error` — an error message if something goes wrong

#### Audit log endpoint

```
GET /ws/rest/v1/chartsearchai/auditlog?patient=...&user=...&fromDate=...&toDate=...&startIndex=0&limit=50
```

Requires the `View AI Audit Logs` privilege. All query parameters are optional. `fromDate` and `toDate` are epoch milliseconds. Returns paginated results ordered by most recent first, with a `totalCount` for pagination.

### Guardrails

- **Input validation**: Patient UUID and question are required. Questions are limited to 1000 characters.
- **Prompt injection detection**: Questions are checked against a regex pattern that rejects common prompt injection phrases (e.g., "ignore previous instructions", "you are now", "system prompt:"). Rejected questions return a 400 error without reaching the LLM.
- **AI disclaimer**: Every response includes a disclaimer stating the output is AI-generated and not a substitute for clinical judgment.
- **Answer caching**: An in-memory LRU cache stores recent answers keyed by (patient UUID, question, pre-filter mode). Configurable TTL via `chartsearchai.cacheTtlMinutes` (default 0 = disabled). When enabled, identical queries within the TTL window return the cached answer without invoking the LLM. The cache key includes the pre-filter setting to prevent cross-contamination between pre-filtered and full-chart results.
- **Rate limiting**: Configurable per-user rate limit (`chartsearchai.rateLimitPerMinute`, default 10). Set to 0 to disable.
- **Database audit logging**: Every query is recorded in the `chartsearchai_audit_log` table with:
  - The authenticated user and patient
  - The question asked and the LLM's response
  - The number of source references returned
  - The search mode used (`pre-filter` or `full-chart`)
  - Response time in milliseconds
  - Timestamp

  This audit trail supports compliance review (who queried which patient's data and what the AI responded) and performance analysis. A scheduled task purges entries older than `chartsearchai.auditLogRetentionDays` (default 90 days, set to 0 to retain all).

## Planned future work

- Add concept graph traversal as a complement to embedding search
- Add pre-computed summaries for common queries
- Agent/tool-use pattern for complex multi-step questions (when better local models are available)
- Unstructured data / image OCR (photos of paper forms)
