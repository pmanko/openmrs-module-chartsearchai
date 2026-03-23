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

This analysis initially suggested using the LLM only as a fallback, with a deterministic primary search path. In practice, the single-LLM approach (Decision 10) proved simpler and more effective — all queries go through the LLM, which handles both the "easy" structured lookups and the "hard" synthesis cases uniformly. The embedding pre-filter narrows the input, and the LLM does the rest. The value analysis above still holds: the LLM earns its cost primarily through natural language understanding and cross-record synthesis.

## Decision 2: Overall architecture — RAG vs. alternatives

### Options evaluated

#### Option A: Full FHIR bundle to LLM
Send a complete FHIR bundle with all patient resources to an LLM for processing.

**Rejected because:**
- A patient with 5 years of visits could have thousands of resources, producing 500K-2M tokens in FHIR JSON — far exceeding even 128K-token context windows, and massively wasteful due to FHIR's verbose structure (see Decision 4).
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

**Deferred.** Solves 70% of the problem with 20% of the complexity. No hallucination risk, no compute requirements, works offline. Weakness: no natural language understanding or synthesis. Could serve as a fallback for environments that cannot run the LLM.

#### Option D: Agent/tool-use pattern
Give the LLM access to OpenMRS APIs as tools and let it autonomously decide what to call.

**Deferred to v2+.** Architecturally elegant but demands more capable models than the deployment environment can support. Small local models (2-8B) are weak at tool use and multi-step reasoning. Latency from multiple sequential LLM → API → LLM loops is problematic in a 90-second encounter.

#### Option E: Pre-computed summaries (batch processing)
Generate patient summaries offline ahead of time. Search summaries at query time.

**Deferred.** Good for common queries (active meds, allergies, problem list). Weakness: stale data, doesn't handle unexpected/novel questions. Best combined with real-time retrieval.

#### Option F: RAG (Retrieval Augmented Generation)
Retrieve relevant records first using deterministic search, then use the LLM only for query understanding and response formatting.

**Strengths:**
- Retrieval is deterministic and auditable — every piece of data has a traceable source.
- The LLM only sees data explicitly provided — it cannot hallucinate about records it was never given. It can still misinterpret or over-infer from the provided data (see Hallucination risk comparison in Decision 10), but the hallucination surface area is smaller than giving it the full chart.
- Works with small local models since query parsing and response synthesis are short-context tasks.

### Decision

The initial plan was full RAG with a deterministic retrieval layer feeding a small subset of records to the LLM. Decision 10 refined this into the current architecture: a **single LLM approach** where all patient records are sent to the LLM (or narrowed by an optional embedding pre-filter). The two-step structure remains — retrieval then synthesis — but the LLM handles all queries, not just "hard" ones. See Decision 10 for the full rationale.

## Decision 3: Embedding approach — semantic search index

### Options evaluated for retrieval

#### Option A: Targeted queries with manual concept mapping
Map each query type to specific OpenMRS resource types and concept codes.

**Weakness:** Requires manually mapping every possible query pattern to the right resources. Misses things you wouldn't think to query — e.g., a free-text visit note mentioning "mother had breast cancer."

#### Option B: Concept graph traversal
Use the OpenMRS concept dictionary as a knowledge graph. Map query terms to SNOMED concepts, traverse the hierarchy, query matching records.

**Deferred.** Fast (milliseconds), deterministic, leverages existing concept dictionary. Weakness: only works for structured/coded data, misses free-text entirely. Could complement embedding search in a future version.

#### Option C: Semantic search index with embeddings — CHOSEN
Pre-index all patient data with vector embeddings. At query time, find relevant records by embedding similarity.

**Chosen because:**
- No manual mapping needed — similarity search catches things you wouldn't think to query.
- Works with both structured and unstructured data.
- The embedding model is tiny (~90MB, runs on CPU in milliseconds).
- Query-time cost is just a vector similarity search — very fast.
- Per-patient search space is small enough (typically <2000 records) for brute-force in-memory cosine similarity.

#### Option D: Clinical concept extraction pipeline (NLP at write time)
Use rule-based NLP (cTAKES, MedSpaCy) to extract structured facts from all data at write time.

**Deferred.** Zero query-time AI cost, works on unstructured text. Weakness: extraction pipeline needs tuning per site, adds processing to the write path.

#### Option E: Map-reduce over chart segments
Split patient chart into time-based segments, classify each for relevance, only send relevant segments to LLM.

**Deferred to v2+.** Handles arbitrarily large charts but adds infrastructure complexity.

### Decision

Semantic search index as the primary retrieval mechanism. Concept graph traversal is deferred to future work as a potential complement for structured data lookups.

## Decision 4: Concise text as LLM input format

### Analysis

Standard serialization formats (FHIR JSON, full OpenMRS domain objects) are poor formats for LLM context windows:

- **Extremely verbose**: A single blood pressure observation is ~800 tokens in FHIR JSON vs. ~15 tokens in compressed form. On a small model with 4-8K context, this matters enormously.
- **Deeply nested**: `coding` inside `code` inside `component` inside `Observation`. Small LLMs are worse at extracting information from nested structures.
- **Redundant metadata**: System URIs, references, profiles waste context tokens.

### Decision

Retrieve data via OpenMRS service APIs (ObsService, ConditionService, PatientService, OrderService, DiagnosisService, ProgramWorkflowService, MedicationDispenseService) and convert records into flat, concise clinical text using `ClinicalTextSerializer` implementations. This gives ~10x token efficiency while preserving clinical meaning.

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

Semantic vectors via **all-MiniLM-L6-v2** running in-process through ONNX Runtime. ~90MB model file, runs on CPU, no GPU needed. Produces 384-dimensional vectors. Requires two files configured via `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`. Captures semantic meaning — effective for clinical queries where synonyms and related concepts matter (e.g., "hypertension" and "high blood pressure" are recognized as related). Embedding dimensions are auto-detected from the model output on first use, so models with different dimensions (e.g., 768-dim pubmedbert-base-embeddings) work without code changes.

A term-frequency hashing approach was considered as a simpler alternative (no model file needed, keyword-overlap retrieval). It was rejected because it cannot capture semantic similarity — for a clinical question like "any infections?", it would find records containing the word "infection" but miss "tuberculosis", "malaria", or "UTI". This defeats the purpose of pre-filtering, since the LLM with the full chart would catch all of these.

## Decision 7: Vector storage — MySQL, not a vector database

### Analysis

MySQL does not natively support vector embeddings (native `VECTOR` type was added in MySQL 9.0+, but OpenMRS deployments typically run MySQL 5.7 or 8.x).

However, a vector database is unnecessary for this use case because:
- Search is **per-patient**, not across all patients
- A patient with 2000 records means 2000 vector comparisons — trivial in Java (microseconds)
- Embeddings are stored as BLOBs (~1.5KB per record for 384 dimensions, ~3KB for 768 dimensions)

### Why not a vector database?

Vector databases (pgvector, Milvus, Pinecone, etc.) use approximate nearest neighbor (ANN) algorithms to efficiently search across millions or billions of vectors. This module searches at most a few thousand vectors per patient — brute-force cosine similarity in Java completes in microseconds, so there is no scale problem for ANN to solve.

Adding a vector database would introduce extra infrastructure to install, configure, and maintain in low-resource settings that already struggle with MySQL + Tomcat — with no performance benefit. It would also return approximate results instead of exact ones, adding a source of retrieval error for no gain. The LLM inference step takes 15–45 seconds; the similarity search is never the bottleneck.

### Decision

Store embeddings as `MEDIUMBLOB` in a regular MySQL table (`chartsearchai_embedding`), indexed by `patient_id`. Load a patient's embeddings into memory and compute cosine similarity in Java. Zero new infrastructure.

The `UNIQUE KEY (resource_type, resource_id)` constraint prevents duplicate embeddings and enables upsert on re-index.

## Decision 8: Index population strategy

### Decision

Three complementary strategies ensure embeddings stay current:

- **On-demand**: When a clinician queries a patient's chart for the first time and no embeddings exist, `LlmInferenceService` triggers `EmbeddingIndexer.indexPatient()` before running the similarity search. This means embeddings are created lazily — no setup required.
- **Incremental via AOP**: After-returning advice on eight OpenMRS services triggers re-indexing when clinical data changes. Most services use a **delete-and-recompute** strategy: when any record changes, all embeddings for that patient are deleted and recomputed from scratch. This is simpler than tracking which specific embedding corresponds to which record, and guarantees consistency — there is no risk of stale or orphaned embeddings. The cost is re-embedding unchanged records, but this is acceptable because: (a) embedding computation is fast (~50–200ms per patient), (b) most patients have hundreds, not thousands, of records, and (c) the AOP hooks fire on clinical data saves, which happen infrequently relative to reads. A future incremental approach (see Planned future work) would avoid this redundancy for patients with very large charts.

  The one exception is `EncounterService`, which uses an **incremental** strategy: it upserts only the encounter's obs and diagnoses rather than re-indexing the entire patient. This is a pragmatic optimization because encounters are the most frequent write path (every clinical visit creates one), and an encounter's obs/diagnoses are self-contained enough to update in isolation.

  The advised services are:
  - `EncounterService` — incremental encounter indexing (upserts only the encounter's obs and diagnoses)
  - `ObsService` — full patient re-index on save/void/unvoid/purge of standalone observations
  - `ConditionService` — full patient re-index on save/void/unvoid/purge
  - `DiagnosisService` — full patient re-index on save/void/unvoid/purge
  - `PatientService` — full patient re-index on allergy changes; on `mergePatients`, re-indexes the preferred patient and deletes the non-preferred patient's embeddings
  - `OrderService` — full patient re-index on save/void/unvoid/purge/discontinue
  - `ProgramWorkflowService` — full patient re-index on save/void/unvoid/purge of program enrollments
  - `MedicationDispenseService` — full patient re-index on save/void/unvoid/delete of medication dispenses
- **Backfill**: A one-time scheduled task (`EmbeddingIndexTask`) indexes all patients that don't yet have embeddings. Handles initial population when the module is installed on a system with existing data. Skips already-indexed patients, so it is safe to re-run and picks up where it left off if stopped. Admins trigger it from the scheduler UI; it does not run automatically.

## Decision 9: Text serialization — ClinicalTextSerializer pattern

### Decision

A generic `ClinicalTextSerializer<T>` interface with one implementation per OpenMRS resource type:

| Serializer | Output example |
|---|---|
| `ObsTextSerializer` | `"Systolic Blood Pressure: 120 mmHg (ABNORMAL). Note: Taken after exercise"` |
| `ConditionTextSerializer` | `"Condition: Type 2 Diabetes Mellitus. Status: ACTIVE. Verification: CONFIRMED"` |
| `AllergyTextSerializer` | `"Allergy: Penicillin (drug allergen). Severity: Severe. Reactions: Anaphylaxis, Rash"` |
| `DiagnosisTextSerializer` | `"Diagnosis: Malaria. Certainty: CONFIRMED. Rank: Primary"` |
| `OrderTextSerializer` | `"Drug order: Metformin 500mg. Dose: 1.0 Tablet(s) Oral twice daily. Duration: 30 Day(s). Action: NEW. Urgency: ROUTINE"` |
| `PatientProgramTextSerializer` | `"Program: HIV Treatment. Enrolled: 2024-01-15. Status: Active. Current state: On ART"` |
| `MedicationDispenseTextSerializer` | `"Dispensed: Metformin 500mg. Status: Completed. Quantity: 30 Tablet(s). Dose: 1 Tablet(s) Oral twice daily"` |

Key design choices:
- The **record date** (when it was observed/created) is not produced by the serializer itself. Instead, `PatientChartSerializer` prepends it as a citation label when constructing the LLM prompt (e.g., `[1] (2025-10-30) Systolic Blood Pressure: 120 mmHg`). This means the **LLM does see every record's date** — it is just added at the prompt assembly level rather than the serializer level. The date is also included in the API response's `references` array for the UI to display. Records are sorted most-recent-first, giving the LLM a positional recency signal in addition to the explicit date.

  The date is excluded from the serializer because **the same serialized text is used for both embedding and LLM input**. `EmbeddingIndexer` passes `record.getText()` directly to the embedding model. If the serializer included the date, then `"(2024-06-15) Systolic Blood Pressure: 120 mmHg"` and `"(2025-10-30) Systolic Blood Pressure: 120 mmHg"` would produce different embedding vectors despite being clinically identical observations. The date text would pollute the semantic similarity — a query like "blood pressure" would get slightly different similarity scores for the same reading depending on when it was recorded. By keeping dates out of the serialized text, embeddings reflect pure clinical content, and the date is added only at prompt assembly time when it is needed for the LLM.

  This distinction matters because the embedding model and the LLM use text differently. The embedding model computes a fixed vector where every token influences the math — date tokens shift the vector away from the pure clinical meaning. The LLM, by contrast, *reads and reasons* over the text — it can use the date to answer "when was the last blood pressure taken?" or simply ignore it when the date is irrelevant. Dates pollute embedding vectors but enrich LLM input.

  Note that small LLMs (1.5B–14B) are unreliable at date arithmetic — even 7B models struggle with "How many days between March 15 and June 2?" or "Was this before or after that?" This improves around 13B+ parameters but remains unreliable until much larger models. The dates are still included because they are clinically important (e.g., "when was the last blood pressure taken?"), but users should be aware that date-based reasoning may be inaccurate.

  In addition to the citation label date, serializers include **clinically significant dates within a record** when they represent facts distinct from the record date:
  - `ConditionTextSerializer`: end date (`Resolved: 2023-02-01`) — when a condition resolved is a clinical fact distinct from when it was recorded.
  - `OrderTextSerializer`: date stopped (`Stopped: 2024-06-20`) — when an order was discontinued.
  - `MedicationDispenseTextSerializer`: date handed over (`Handed over: 2025-01-10`) — when the patient actually received the medication.
  - `PatientProgramTextSerializer`: enrollment and completion dates (`Enrolled: 2024-01-15`, `Completed: 2023-01-15`) — both are intrinsic clinical facts about the enrollment.
- Voided and retired records are excluded by relying on OpenMRS service methods, not by checking voided flags in serializers. For example, `getObservationsByPerson()` returns only non-voided obs, and `getAllergies()` returns only non-voided allergies. This keeps the serializers focused on text formatting and avoids duplicating filtering logic that the platform already handles correctly.
- Both active and inactive conditions are loaded via `getAllConditions()`, not just active ones. This ensures resolved conditions (e.g., past malaria, resolved pneumonia) are visible to the LLM. Without inactive conditions, a clinician asking "Has this patient ever had malaria?" would get "No relevant information found" even if a resolved malaria condition exists — a wrong answer. The `ConditionTextSerializer` distinguishes between them via the `clinicalStatus` field (e.g., `Status: ACTIVE` vs `Status: INACTIVE`), so the LLM can differentiate when answering questions about current vs historical conditions. The token cost is modest — most patients have 5-20 conditions, and each serializes to ~10-15 tokens.
- `ObsTextSerializer` flattens obs groups into a single text record rather than serializing each group member as a separate record. This preserves clinical context — a blood pressure group with systolic and diastolic members stays together as one record (e.g., `"Blood Pressure: Systolic 120 mmHg; Diastolic 80 mmHg"`) rather than being split into two unrelated records that lose their association. Group members are delimited by semicolons. Nested groups (groups within groups) are flattened recursively.
- `ObsTextSerializer` includes concept name, value (coded/numeric/text/datetime/drug), value modifier (e.g., `>`, `<` for lab results like ">200 copies/mL"), units, interpretation, comments, and flattened group members. Omitted: reference ranges (`Obs.referenceRange` added in OpenMRS 2.7.0, and `ConceptNumeric` hi/low normal/critical/absolute thresholds) — including ranges adds ~8 tokens per numeric obs, and a patient with 50+ lab values would add 400+ tokens just for ranges, working against the goal of concise text for small LLMs with limited context windows; the LLM's role is to find and cite relevant records, not interpret lab abnormality, and clinicians can look up reference ranges outside the LLM. Also omitted: location (administrative), accession number (lab logistics), order linkage (structural reference, not clinical text), and status (PRELIMINARY/FINAL/AMENDED — adds tokens for a distinction small LLMs are unlikely to reason about meaningfully).
- Units are extracted from `ConceptNumeric`, not `Concept` (which has no `getUnits()` in OpenMRS 2.8.x).
- `ConditionTextSerializer` includes condition name, clinical status, verification status, additional detail, end date, and end reason. Omitted: onset date (handled by record sort position, same reasoning as other dates). Note: onset date is the most debatable omission across all serializers — unlike transactional obs dates, onset date is an intrinsic clinical fact ("When did diabetes start?"), similar to why PatientProgram enrollment dates ARE included. The difference is that conditions are sorted by onset date so position already conveys recency, and the actual date is available in the API response's references array. If future use cases require explicit onset dates (e.g., larger models that can reason about durations), this is the first field to reconsider.
- `AllergyTextSerializer` includes allergen name, allergen type, severity, reactions, and comments. No clinically meaningful fields are omitted — allergy serialization is comprehensive.
- `DiagnosisTextSerializer` includes diagnosis name, certainty, and rank. Omitted: linked condition (structural reference — the condition itself is serialized separately), and custom attributes (deployment-specific, may be empty).
- `OrderTextSerializer` handles the full Order hierarchy: base `Order`, `DrugOrder`, and `ServiceOrder` (which includes `TestOrder` and `ReferralOrder`). Base orders include concept name, action, urgency, instructions, reason, and date stopped. Drug orders additionally include drug name (coded or non-coded), dose/units, route, frequency, duration/units, as-needed flag with condition, and dosing instructions. Service/test/referral orders additionally include laterality (LEFT/RIGHT/BILATERAL — critical for imaging and procedures, e.g., "X-Ray Left Knee" vs "X-Ray Knee"), specimen source (e.g., "Venous blood"), and clinical history (free-text context for the order). Omitted across all order types: orderer (who placed the order — administrative), care setting (inpatient/outpatient — contextual metadata), fulfiller status/comments (pharmacy workflow, not prescription content), commentToFulfiller (could carry clinical context for the fulfiller, but overlaps with `instructions` which is already serialized and is primarily fulfillment-workflow-oriented), refills/brand name/dispense-as-written (pharmacy logistics). Omitted from ServiceOrder: frequency/numberOfRepeats (scheduling logistics), location (administrative).
- `PatientProgramTextSerializer` includes program name, enrollment/completion dates, active status, outcome, and current workflow state (via `getCurrentState(null)`). Only the current state is serialized; historical state transitions (e.g., "First Line → Second Line") are omitted. Including them would add ~15-20 tokens per transition (e.g., `States: First Line (2023-01-15 to 2024-06-01), Second Line (2024-06-01 to present)`), and most programs have 1-3 transitions. This is a potential future enhancement for questions about treatment changes (e.g., "Why was the patient's ARV regimen changed?"), but the token cost compounds across multiple program enrollments. Location is omitted — it is administrative metadata rarely part of a clinical question about a program enrollment.
- `MedicationDispenseTextSerializer` includes drug name, status (completed/declined/cancelled), quantity/units, dose/units/route/frequency, dosing instructions, status reason, substitution flag/type/reason, and date handed over. Status is clinically critical — a declined or cancelled dispense means the patient did NOT receive the medication, which changes the clinical picture entirely. Substitution details (type and reason) are included because a generic substitution for cost reasons is clinically different from a therapeutic substitution for a drug interaction. Omitted: date prepared (pharmacy workflow detail, not clinical content).

### Resource type coverage analysis

The seven resource types above (Obs, Condition, Allergy, Diagnosis, Order, PatientProgram, MedicationDispense) were chosen after a systematic review of every patient-facing domain class in OpenMRS core 2.8.0. The following data types were considered and intentionally excluded:

| Data type | Reason for exclusion |
|-----------|---------------------|
| Visit | Pure metadata grouper for encounters. Clinical content lives in the encounters' obs and diagnoses, which are already embedded. |
| Encounter | Container for obs and diagnoses, which are already captured individually. Encounter type (e.g., "Admission") is implicit in the obs recorded during that encounter. |
| PersonAttribute | Deployment-specific custom fields (phone, occupation, next of kin). Too variable to serialize generically — may be empty or administrative. |
| Patient demographics | Not embedded as individual records, but age and gender are included as a header line in the LLM prompt (e.g., `Patient: 45-year-old Male`). This gives the LLM context for age- and sex-dependent clinical reasoning without inflating the record count or embedding index. For example: blood pressure 130/85 is concerning for a 25-year-old but unremarkable for a 75-year-old; cancer screening recommendations depend on sex and age (cervical vs prostate); and medication dosing may vary by age. |
| Relationship | Administrative (e.g., "Mother of Patient X", "Emergency contact"), not clinical. |
| OrderGroup | Groups related orders (e.g., chemotherapy regimen). The individual orders within the group are already serialized; the group itself is structural. |
| VisitAttribute | Deployment-specific custom fields on visits. Too variable to serialize generically. |
| PatientIdentifier | Administrative identifiers, not clinical content. |

Cross-checked against FHIR clinical resources to verify completeness:

| FHIR resource | OpenMRS mapping | Status |
|---------------|----------------|--------|
| Observation | Obs | Embedded |
| Condition | Condition | Embedded |
| AllergyIntolerance | Allergy | Embedded |
| MedicationRequest | DrugOrder | Embedded |
| MedicationDispense | MedicationDispense | Embedded |
| ServiceRequest | ServiceOrder / TestOrder / ReferralOrder | Embedded |
| DiagnosticReport | Obs (text reports from imaging/labs) | Embedded |
| EpisodeOfCare | PatientProgram | Embedded |
| Immunization | Not in OpenMRS core (separate module) | N/A |
| CarePlan | Not in OpenMRS core | N/A |
| Appointment | Not in OpenMRS core (separate module) | N/A |

### Resource types as string constants

Resource types (e.g., `"obs"`, `"condition"`, `"order"`) are defined as `public static final String` constants in `ChartSearchAiConstants`, not as a Java enum. This is because resource type values are stored as strings in the `chartsearchai_embedding` database table and returned as strings in the REST API's JSON response. Using an enum would require mapping between the enum and its string representation at every persistence and serialization boundary. String constants avoid this overhead while still providing compile-time references and a single source of truth for the values.

## Decision 10: Single LLM architecture with optional embedding pre-filter

### Context

The current architecture (Decisions 3–9) uses a two-model pipeline: an embedding model for semantic search retrieval, plus a generative LLM for query understanding and response synthesis. This requires vector storage, cosine similarity search, and an embedding indexing strategy.

However, if two conditions are met, this complexity can be eliminated entirely:

1. **The full patient chart fits within the LLM's context window.** A patient with 2000 records, each serialized to ~15 tokens by the `ClinicalTextSerializer`, produces ~30K tokens. Models like Mistral 7B (32K context) and Llama 3.2 3B (128K context) can accommodate this.
2. **A local LLM is available with acceptable latency.** Quantized models (3B–14B parameters) can run on CPU via [java-llama.cpp](https://github.com/kherud/java-llama.cpp), which provides Java JNI bindings to llama.cpp and is available on Maven (`de.kherud:llama`). The recommended 8B model requires ~10GB RAM and produces ~8–12 tokens/sec on CPU. This keeps the module self-contained with no external service dependency.

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

Both approaches carry hallucination risk, but the failure modes differ. In the current system, these correspond to the `chartsearchai.embedding.preFilter` toggle: `true` (default) uses embedding-based pre-filtering, while `false` sends the full chart to the LLM.

#### Embedding-based pre-filtering hallucinations (`preFilter=true`)

With pre-filtering enabled, the LLM only sees the top-K retrieved records (default 15, configurable via `chartsearchai.embedding.topK`). This limits how much it can hallucinate *about*, but it introduces a different risk: hallucinating from *missing context*. Examples:

- The retrieval step misses a relevant record (e.g., a resolved penicillin allergy) because the query and record text are semantically distant. The LLM confidently says "no known drug allergies" based on the records it received.
- The LLM sees a single elevated blood pressure reading without the surrounding context of the patient exercising beforehand (that context is in a different obs comment that was not retrieved). It may overstate the clinical significance.
- The LLM receives a medication order and a lab result but not the clinician's note explaining why the medication was started. It invents a plausible but incorrect reason.

#### Full-chart hallucinations (`preFilter=false`)

With pre-filtering disabled, the LLM sees the full patient chart. It will not miss relevant records, but more input means more opportunity to hallucinate from *over-interpreting context*. Examples:

- The LLM sees a headache obs and a new hypertension medication started the same week. It infers the medication caused the headache, when the timing was coincidental.
- The LLM notices elevated liver enzymes and a hepatitis B diagnosis. It concludes the hepatitis is active and causing the elevation, when the enzymes were actually elevated due to a statin started recently.
- The LLM sees multiple records mentioning fatigue across several visits and synthesizes a narrative about chronic fatigue syndrome, when each instance had a different, resolved cause.

#### Mitigation

The mitigation is the same for both modes: **never present LLM output as clinical fact**. The module should always show the source records alongside the LLM's answer so the clinician can verify. The full-chart mode actually makes this easier — since there is no retrieval step, every record the LLM saw is known and can be cited. With pre-filtering, the clinician must additionally trust that the retrieval step found the right records.

### Source citations

Source citations are straightforward because we control exactly what the LLM sees. Each serialized record is numbered sequentially before being included in the prompt (sorted most recent first):

```
Patient: 45-year-old Male

[1] (2025-10-30) Systolic Blood Pressure: 120 mmHg (ABNORMAL)
[2] (2018-03-10) Condition: Type 2 Diabetes Mellitus. Status: ACTIVE
[3] (2025-01-10) Order: Metformin. Action: NEW
[4] (2025-09-15) HbA1c: 8.2%
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
| Llama 3.3 8B | ~4.5GB | ~8GB | 128K tokens | ~8–12 tok/s |
| Qwen 2.5 7B | ~4GB | ~8GB | 128K tokens | ~8–12 tok/s |
| Gemma 2 9B Instruct | ~5GB | ~10GB | 8K tokens | ~5–10 tok/s |
| Mistral Nemo 12B | ~7GB | ~12GB | 128K tokens | ~4–8 tok/s |
| Phi-3-Medium 14B | ~8GB | ~14GB | 4K tokens (128K variant available) | ~3–6 tok/s |
| Qwen 2.5 14B | ~8GB | ~14GB | 128K tokens | ~3–6 tok/s |

### Recommended model: Llama 3.3 8B

Llama 3.3 8B is the default recommendation. Its 128K token context window can hold approximately 6,000 serialized patient records (~15 tokens each), which comfortably accommodates even the largest patient charts. At 8B parameters with Q4_K_M quantization, it is ~5GB on disk and requires ~10GB total RAM. It offers strong instruction following and medical text comprehension — significantly better than 3B models at reasoning, citation accuracy, and avoiding hallucinations.

### Alternative models

| Model | RAM Needed | Chat Template | Why |
|-------|-----------|---------------|-----|
| **Llama 3.2 3B** | ~6GB total | `llama3` | For low-resource deployments where 10GB RAM is not available. Faster inference but weaker instruction following and reasoning. |
| **Mistral Nemo 12B** | ~12GB total | `mistral` | Best sub-15B option for clinical Q&A. Strong medical text comprehension and 128K context window. Requires changing both model path and chat template. |

These models are from US/EU organizations (Meta and Mistral AI respectively), have strong performance on medical benchmarks, and use chat templates already supported by the module. Switching requires only two global property changes (`modelFilePath` and `chatTemplate`) — no code changes or module rebuild.

### Other alternatives

- **Qwen 2.5 1.5B** is faster and smaller but its 32K context window limits it to ~2,000 records, and its reasoning capability is weaker at 1.5B parameters.
- **Phi-3 Mini 3.8B** (Microsoft) has slightly better reasoning per parameter than Llama 3.2 3B, but its default 4K context window is far too small for full patient charts. The 128K variant exists but is slower on CPU due to the longer context handling. Phi models are trained primarily on synthetic/textbook data and tend to be weaker on messy, real-world clinical text compared to Llama and Mistral models at similar sizes.
- **Phi-3-Medium 14B** (Microsoft) is the largest model in the Phi-3 family at 14 billion parameters. It uses the same Transformer decoder architecture as Phi-3 Mini but scaled up, trained on a mix of synthetic data generated by larger models and heavily filtered web data (Microsoft's "textbook quality" data curation pipeline). It scores competitively on reasoning benchmarks — outperforming Llama 3.1 8B and Mistral Nemo 12B on MMLU (~78%), GSM8K (~89%), and HumanEval (~62%), and approaching GPT-3.5-Turbo on several benchmarks. However, the same caveats that apply to Phi-3 Mini apply here: the training data skews toward clean, synthetic, and textbook-style text, which means it can underperform on messy, real-world clinical notes with abbreviations, typos, and inconsistent formatting compared to Llama and Mistral models trained on broader web corpora. The default context window is only 4K tokens, which limits it to ~250 serialized patient records without embedding pre-filtering — far too small for large patient charts. A 128K variant (`Phi-3-medium-128k-instruct`) exists but is significantly slower on CPU due to the RoPE scaling required for long contexts. At 14B parameters with Q4_K_M quantization, it is ~8GB on disk and requires ~14GB total RAM — the same resource footprint as Qwen 2.5 14B but with a much smaller default context window. It uses the `phi3` chat template already supported by the module. Licensed under MIT, which is a genuinely permissive open-source license with no usage restrictions — more permissive than Llama's community license. For deployments where licensing simplicity matters (e.g., government or NGO procurement), this is an advantage. However, given the 4K default context limitation and the weaker performance on unstructured clinical text, Qwen 2.5 14B or Mistral Nemo 12B are generally better choices at this parameter class for clinical Q&A.
- **Mistral 7B** has strong reasoning but at 7B parameters it is noticeably slower on CPU (~10–15 tok/s) and requires ~8GB RAM. Superseded by Llama 3.3 8B which offers better quality at a similar resource cost.
- **Qwen 2.5 7B/14B** (Alibaba) offers strong instruction following and large context windows. However, Qwen is developed by a Chinese company subject to China's data laws — while GGUF models run locally with no data leaving the machine, US healthcare organizations may face compliance or perception concerns. Consider Llama or Mistral alternatives first.
- **Gemma 2 9B Instruct** (Google) has excellent reasoning and instruction following at 9B parameters, but its 8K context window limits it to ~500 records without embedding pre-filtering. Requires ~10GB RAM.

All models run via java-llama.cpp with Q4_K_M quantization in GGUF format.

### Licensing

Llama 3.3 8B is free for both research and commercial use under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). No fees or royalties apply. The license permits use, modification, and distribution.

The model is not technically "open source" by the [Open Source Initiative's definition](https://opensource.org/blog/metas-llama-license-is-still-not-open-source) — Meta's license includes restrictions that do not meet OSI criteria. However, for practical purposes the only meaningful restriction is that products with over 700 million monthly active users require a separate license from Meta, which is not a concern for OpenMRS.

The license requires the following attribution to be included in all distributed copies: *"Llama 3.3 is licensed under the Llama 3.2 Community License, Copyright (C) Meta Platforms, Inc. All Rights Reserved."*

### Deployment and memory requirements

The model runs **in-process** inside the OpenMRS JVM via [java-llama.cpp](https://github.com/kherud/java-llama.cpp) JNI bindings. No separate web server, process, or HTTP service is needed. The deployment consists of two files:

1. The `.omod` module file (includes the java-llama.cpp dependency)
2. The `.gguf` model file (placed in the OpenMRS application data directory)

The model path is configured via the `chartsearchai.llm.modelFilePath` global property. The model loads into memory on first query and stays resident for subsequent requests. If the model path global property is changed, the model is automatically unloaded and the new model is loaded on the next query — no restart required.

Inference uses temperature 0.0, a fixed seed, and prompt caching disabled (`setCachePrompt(false)`) for deterministic output. Prompt caching had to be disabled because llama.cpp's KV cache reuse caused non-deterministic responses — the first query for a given prompt produced a different answer than subsequent identical queries. With caching off, identical inputs always produce identical answers, which is important for clinical trust and for the answer cache to be meaningful.

Model file paths are resolved relative to the OpenMRS application data directory. Path traversal (`..`) is rejected and the resolved path is verified to stay within the data directory, preventing an admin from accidentally (or maliciously) pointing the module at arbitrary files on the filesystem.

### Chat template configurability

Different GGUF models require different prompt formats. The `chartsearchai.llm.chatTemplate` global property accepts either a **preset name** or a **custom template string**:

| Preset | Format |
|--------|--------|
| `llama3` (default) | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>...` |
| `mistral` | `[INST] {system}\n\n{user} [/INST]` |
| `phi3` | `<\|system\|>\n{system}<\|end\|>\n<\|user\|>\n{user}<\|end\|>\n<\|assistant\|>\n` |
| `chatml` | `<\|im_start\|>system\n{system}<\|im_end\|>\n<\|im_start\|>user\n...` |
| `gemma` | `<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n` |

For models not covered by a preset, set the property to a custom template string with `{system}` and `{user}` placeholders. Stop strings are resolved automatically from the preset; custom templates use no stop strings (the model's own EOS token terminates generation).

This means switching LLM models requires only two global property changes (`modelFilePath` and `chatTemplate`) — no code changes or module rebuild.

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

**8B models** (e.g. Llama 3.3 8B) are the recommended default — significantly better instruction following and clinical reasoning than 3B, while still feasible on a server with 10GB RAM.

**9B models** (e.g. Gemma 2 9B Instruct) offer excellent reasoning and instruction following. Note that Gemma 2's 8K context window is smaller than Llama or Qwen models, so embedding pre-filtering is strongly recommended.

**14B models** (e.g. Qwen 2.5 14B, Phi-3-Medium 14B) provide the best response quality among CPU-viable options, with strong reasoning. They require 14–16GB total RAM and produce slower inference (~2–4 tok/s). Suitable for well-resourced deployments where response quality is prioritized over speed. Note that context window size varies significantly at this tier — Qwen 2.5 14B offers 128K tokens natively, while Phi-3-Medium defaults to 4K (128K variant available but slower on CPU).

A server running OpenMRS typically uses 1–2GB for the JVM heap. A 4GB machine is insufficient to run this module — the LLM alone requires at least 3–4GB for the smallest viable model.

### Hardware requirements

The module requires sufficient RAM for both the OpenMRS JVM and the LLM model:
- **Minimum**: ~6GB total (1–2GB JVM + 3–4GB for a 3B model). Usable but with weaker instruction following and reasoning.
- **Recommended**: ~10GB total for the default 8B model, which provides significantly better response quality.
- The embedding pre-filter (default: enabled) reduces the number of tokens sent to the LLM, which improves both response quality and latency for large patient charts.

### Decision

A single architecture is used: all queries go through the LLM for reasoning and synthesis. An embedding pre-filter (`chartsearchai.embedding.preFilter`, default `true`) narrows the patient chart to the most relevant records before sending them to the LLM (default top 15, configurable via `chartsearchai.embedding.topK`). This solves the "lost in the middle" problem where small LLMs struggle to find relevant information in large contexts. Set to `false` to send the full chart instead.

Embeddings are indexed on first patient chart access and kept up to date automatically via AOP hooks on data changes. A bulk backfill task is also available for pre-indexing all patients.

Embeddings use all-MiniLM-L6-v2 via ONNX Runtime (~90MB model file, configured via `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`).

### Embedding model selection

The default embedding model (all-MiniLM-L6-v2) is general-purpose. For clinical text, it produces narrow similarity score ranges (e.g., 0.20–0.31) with modest separation between relevant and irrelevant records. A clinical-domain model like [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) (Apache 2.0, 768 dimensions, ~440MB), fine-tuned on PubMed text, was tested as an alternative. While pubmedbert produced higher absolute similarity scores (~0.57 vs ~0.31), it performed worse at retrieval ranking — for a query like "any past history of tumors?", pubmedbert returned pulse readings while all-MiniLM-L6-v2 correctly returned Kaposi sarcoma records. Higher absolute scores do not guarantee better retrieval; what matters is the relative ranking of relevant vs irrelevant records. all-MiniLM-L6-v2 remains the default.

Any BERT-based ONNX embedding model can be used as a drop-in replacement by updating `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`. Embedding dimensions are auto-detected from the model output — both 384-dim and 768-dim models (and any other size) work without code changes. After switching models, all existing embeddings must be recomputed by running the backfill task, since embeddings from different models are not compatible.

A `chartsearchai.embedding.similarityRatio` setting (default 0.80) filters out low-relevance records by requiring each record to score at least 80% of the top result's similarity score. This works alongside `topK` as a quality floor — `topK` sets the maximum number of records, while `similarityRatio` drops noise within that cap.

### Retrieval precision improvements

The general-purpose embedding model (all-MiniLM-L6-v2) ranks records by lexical overlap rather than clinical semantics. For example, a query like "any medications?" would rank an allergy record containing "DRUG" higher than actual drug orders, because the word "DRUG" has higher surface-level similarity to "medications" than dosing details like "500mg Oral twice daily." Three techniques address this:

**Semantic embedding prefixes.** Each record's text is prepended with a type-specific prefix before computing embeddings (but not in the LLM prompt). For example, a drug order is embedded as `"Medication prescription: Drug order: Azithromycin..."` while an allergy is embedded as `"Patient allergy: Allergy: Penicillin..."`. This shifts the embedding vectors toward the right semantic space, so medication queries rank drug orders higher. Prefixes are further specialized by order sub-type: `"Medication prescription:"` for drug orders, `"Lab or diagnostic test:"` for test orders, and `"Clinical referral:"` for referral orders. In testing, this moved drug orders from 8th/9th place to 1st/3rd place for medication queries.

**Query stopword normalization.** Common filler words ("does", "the", "patient", "have", "any") are stripped from the query before embedding, so that "any medications?" and "does the patient have any medications?" produce the same embedding vector and identical retrieval results. Without this, different phrasings of the same question return different filtered record sets, leading to inconsistent LLM answers. Stopwords are loaded from `<application-data-directory>/chartsearchai/query-stopwords.txt` if present, otherwise from a bundled default. Only true filler words are stripped — clinical qualifiers like "no", "not", "current", "recent", "last", and "active" are preserved because they change the query's meaning.

**Type-cohort expansion.** After applying the similarity ratio threshold, remaining top-K records whose resource type already has at least one record above the threshold are included. This prevents related records (e.g., two drug orders for the same patient) from being split by marginal score differences. For example, if one drug order scores 0.41 (above threshold) and another scores 0.39 (below threshold), both are included because the "order" type already has a passing record.

### Chunking strategy

No chunking is used. Each patient record (obs, condition, diagnosis, allergy, order, program enrollment, medication dispense) is serialized as a single text string and embedded as one unit. This is possible because individual clinical records are naturally short — typically a sentence or two — so they fit well within the embedding model's 256-token window without splitting. This avoids the complexity of chunk boundary management, overlap strategies, and reassembly that document-oriented RAG systems require.

### Serialization format

Records are serialized as labeled plain text (e.g., `Condition: Diabetes. Status: ACTIVE`). This was chosen over structured formats for token efficiency:

| Format | Example | Tokens (approx.) |
|--------|---------|-------------------|
| **Plain text (chosen)** | `Condition: Diabetes. Status: ACTIVE` | ~8 |
| JSON | `{"type":"condition","name":"Diabetes","status":"ACTIVE"}` | ~18 |
| FHIR JSON | `{"resourceType":"Condition","code":{"text":"Diabetes"},"clinicalStatus":{"coding":[{"code":"active"}]}}` | ~30+ |
| XML | `<condition><name>Diabetes</name><status>ACTIVE</status></condition>` | ~16 |

With 15 records per query and potentially hundreds of patients per day, the token savings compound. Plain text also reads naturally, which helps smaller LLMs that perform better with human-readable input than structured formats. Field labels (e.g., "Status:", "Severity:") provide enough structure for the LLM to extract information without the overhead of delimiters, braces, or tags.

### Serialized fields per record type

Each record type is serialized into a concise text string. The fields below are chosen for clinical value while minimizing token count.

#### Obs (observations, vitals, lab results)

**Included fields:** concept name, value (coded/numeric/text/datetime/drug/boolean), units, value modifier (e.g., "<", ">"), interpretation (NORMAL, ABNORMAL), comment, group members (flattened)

**Excluded fields:** reference range (not exposed in OpenMRS 2.8.x API), linked order (serialized separately as its own record), obs datetime (already in the citation date label)

Examples:
```
[1] (2025-10-30) Systolic Blood Pressure: 120 mmHg (ABNORMAL). Note: Taken after exercise
[2] (2025-10-30) Blood Panel: Hemoglobin: 12.5 g/dL; White Blood Cells: 8000 cells/uL (NORMAL)
```

#### Condition

**Included fields:** condition name, clinical status (ACTIVE/INACTIVE), verification status (CONFIRMED, etc.), additional detail, end date, end reason

**Excluded fields:** onset date (already in the citation date label)

Examples:
```
[3] (2018-03-10) Condition: Type 2 Diabetes Mellitus. Status: ACTIVE. Verification: CONFIRMED.
    Detail: Stage 3, GFR 45 mL/min
[4] (2023-01-15) Condition: Malaria. Status: INACTIVE. Resolved: 2023-02-01 (Treatment completed)
```

#### Diagnosis

**Included fields:** diagnosis name, certainty (CONFIRMED/PROVISIONAL), rank (Primary/Secondary)

**Excluded fields:** linked condition (serialized separately)

Example:
```
[5] (2025-06-29) Diagnosis: Tuberculosis. Certainty: CONFIRMED. Rank: Secondary
```

#### Allergy

**Included fields:** allergen name, allergen type (drug allergen/food allergen/environmental allergen), severity, reactions, comments

Example:
```
[6] (2024-12-29) Allergy: Penicillin (drug allergen). Severity: Severe. Reactions: Anaphylaxis, Rash.
    Comments: Confirmed by allergist
```

#### Order (drug orders, test orders, referral orders)

**Included fields:** concept name, action (NEW/REVISE/DISCONTINUE/RENEW), urgency (ROUTINE/STAT), instructions, order reason, date stopped. Drug orders additionally: drug name, dose/units/route/frequency, duration/units, quantity/units, as-needed flag and condition, dosing instructions. Service/test/referral orders additionally: laterality, specimen source, clinical history.

**Excluded fields:** number of refills (pharmacy detail), brand name (drug name already captured), dispense-as-written flag (pharmacy detail), care setting — inpatient/outpatient (adds tokens to every order for marginal value), scheduled date (only relevant for rare ON_SCHEDULED_DATE urgency), number of repeats (rarely relevant)

Examples:
```
[7]  (2025-01-10) Drug order: Metformin 500mg. Dose: 1.0 Tablet(s) Oral twice daily.
     Duration: 30 Day(s). Quantity: 60.0 Tablet(s). Action: NEW. Urgency: ROUTINE
[8]  (2025-03-15) Drug order: Ibuprofen 400mg. Dose: 1.0 Tablet(s) Oral.
     As needed (for pain). Action: NEW. Urgency: ROUTINE
[9]  (2025-06-29) Test order: X-Ray Chest. Laterality: LEFT.
     Clinical history: Persistent cough for 3 weeks. Action: NEW. Urgency: STAT
[10] (2025-04-01) Drug order: Lisinopril 10mg. Action: DISCONTINUE. Urgency: ROUTINE.
     Reason: Persistent dry cough. Stopped: 2025-04-01
```

#### Medication dispense

**Included fields:** drug name, status (completed/declined/cancelled), quantity/units, dose/units/route/frequency, dosing instructions, status reason, substitution flag/type/reason, date handed over

**Excluded fields:** date prepared (pharmacy workflow detail)

Examples:
```
[11] (2025-01-10) Dispensed: Metformin 500mg. Status: Completed.
     Quantity: 60.0 Tablet(s). Dose: 1.0 Tablet(s) Oral twice daily.
     Handed over: 2025-01-10
[12] (2025-01-10) Dispensed: Metformin 500mg. Status: Completed.
     Status reason: Out of stock. Substituted: Generic equivalent.
     Substitution reason: Cost. Handed over: 2025-01-10
[13] (2025-02-15) Dispensed: Amoxicillin 250mg. Status: Declined.
     Status reason: Patient refused
```

#### Patient program

**Included fields:** program name, enrollment date, completion date, active status, outcome, current state

**Excluded fields:** location (where enrolled — marginal for clinical queries), program attributes (implementation-specific, unknown content)

Examples:
```
[14] (2024-01-15) Program: HIV Treatment. Enrolled: 2024-01-15. Status: Active.
     Current state: On ART
[15] (2022-06-01) Program: TB Treatment. Enrolled: 2022-06-01.
     Completed: 2023-01-15. Outcome: Treatment completed
```

### Medical imaging data (X-rays, scans, etc.)

The recommended Llama 3.3 8B model is text-only. Multimodal variants that can interpret images directly (Llama 3.2 11B and 90B) are too large for CPU inference in low-resource settings.

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
- **Patient access control**: A `PatientAccessCheck` interface controls whether a user can query a specific patient's chart. The default implementation (`DefaultPatientAccessCheck`) permits all access — any user with the `AI Query Patient Data` privilege can query any patient. Deployments requiring patient-level restrictions (e.g., location-based or care-team-based) can override this by registering a custom Spring bean with id `chartSearchAi.patientAccessCheck`. This separates privilege-based access (handled by OpenMRS) from patient-level access (handled by the module).

## Decision 12: Concurrency model

### Constraint

Both the LLM (llama.cpp via jllama) and the embedding model (ONNX Runtime) use native memory that is **not thread-safe**. To prevent memory corruption:

- `LlmProvider.search()` and `searchStreaming()` are `synchronized` — only one LLM inference runs at a time.
- `OnnxEmbeddingProvider.embed()` is `synchronized` — only one embedding computation runs at a time.

### Why `synchronized` instead of other concurrency primitives?

Java's `synchronized` keyword is the simplest correct choice here. A `ReentrantLock` with a bounded queue, a thread pool, or `CompletableFuture` would add complexity without benefit — the fundamental constraint is that the native library allows only one inference at a time, so there is nothing to parallelize. A queue with position feedback (see Future options below) would improve the user experience for waiting requests, but the serialization itself is unavoidable without loading multiple model instances. For v1 targeting small clinics with 1–3 concurrent users, `synchronized` is sufficient and easy to reason about.

### Impact on concurrent users

When multiple users submit queries simultaneously, requests are serialized:

1. The first request acquires the LLM lock and begins inference.
2. Subsequent requests queue on the `synchronized` block and wait.
3. Each request times out after `chartsearchai.llm.timeoutSeconds` (default 120s).

With an 8B model on CPU, a single query typically takes 15–45 seconds. This means roughly **2–3 concurrent users** can be served before requests start timing out. Smaller models (3B) are faster but produce lower quality responses; larger models (12B) have slower inference and reduce concurrency further.

Embedding computation is faster (~50–200ms per patient) so the embedding lock is rarely a bottleneck.

### Existing mitigations

- **Answer cache** (`chartsearchai.cacheTtlMinutes`): Identical (patient, question) pairs return cached results without acquiring the LLM lock.
- **Rate limiter** (`chartsearchai.rateLimitPerMinute`): Limits per-user query frequency, reducing queue depth.
- **Configurable timeout**: Prevents requests from waiting indefinitely.

### Future options (not yet implemented)

- **Multiple model instances**: Load the LLM into separate native contexts and round-robin across them. Trades RAM for throughput (each 8B instance adds ~6–8GB).
- **Request queuing with position feedback**: Return queue position to the client via SSE so the UI can show "you are #3 in queue" instead of hanging silently.
- **External inference server**: Offload to a dedicated inference server (e.g., llama.cpp server mode, vLLM, Ollama) that manages its own concurrency. This decouples the module from native memory constraints but adds an external dependency.

For the initial release targeting small clinics with low concurrent usage, the serialized approach is acceptable and avoids the complexity of managing multiple native contexts.

## Known limitations

- **Counting questions**: LLMs are unreliable at precise counting tasks (e.g., "how many weight records in the last 10 years?"). The model may undercount or overcount even when all relevant records are provided. Larger, more capable models perform better at counting but are still not perfectly reliable. This is a fundamental limitation of LLM inference, not a retrieval issue. Questions that require exact counts are better suited to structured queries.

## Planned future work

- **Incremental embedding indexing**: Currently, `indexPatient()` deletes all embeddings for a patient and recomputes them from scratch on every data change. This is simple and guarantees consistency, but recomputes embeddings for records that haven't changed. An incremental approach would track which record maps to which embedding row and only add, update, or delete the specific embeddings affected. This matters for patients with large charts where AOP hooks fire frequently.
- **Reranking**: A cross-encoder reranking step between embedding retrieval and LLM inference could improve relevance. Currently deferred because: individual records are short (embedding similarity works well), the LLM itself reasons over all 15 retrieved records (acting as an implicit reranker), and adding another model increases RAM usage, latency, and concurrency bottlenecks. Worth revisiting if real-world usage reveals relevance gaps, if top-K is increased significantly (50+), or if patient charts grow very large with thousands of records.
- **Concept graph traversal**: Complement embedding search with OpenMRS concept relationship traversal to improve retrieval for queries involving related concepts (e.g., finding NSAID allergies when asking about ibuprofen).
- **Pre-computed summaries**: Cache LLM-generated summaries for common query patterns (e.g., "current medications", "active problems") to reduce inference latency for frequently asked questions.
- **Agent/tool-use pattern**: Enable multi-step reasoning where the LLM can request additional data or perform follow-up queries. Deferred until local models with reliable tool-use capabilities are available.
- **Unstructured data / image OCR**: Extract text from photos of paper forms at write time so the content flows through the existing serializer and embedding pipeline.
