# Chart Search AI Module

An OpenMRS module that lets clinicians ask natural language questions about a patient's chart and get answers with source citations.

For project background, community discussion, and roadmap, see the [wiki project page](https://openmrs.atlassian.net/wiki/spaces/projects/pages/373325839/Chart+Search+aka+ChartSearchAI).

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
  - [1. Build](#1-build)
  - [2. Download the LLM model](#2-download-the-llm-model-local-mode-only)
  - [3. Download the embedding model](#3-download-the-embedding-model)
  - [4. Install](#4-install)
  - [5. Configure](#5-configure)
  - [6. Grant privileges](#6-grant-privileges)
  - [7. Indexing](#7-indexing)
  - [Testing the Elasticsearch pipeline locally](#testing-the-elasticsearch-pipeline-locally)
- [Query behavior](#query-behavior)
- [API](#api)
  - [Search](#search)
  - [Streaming search (SSE)](#streaming-search-sse)
  - [Feedback](#feedback)
  - [Audit log](#audit-log)
- [Patient access control](#patient-access-control)
- [Evals](#evals)
- [Evaluated models](#evaluated-models)
- [Architecture](#architecture)
- [License](#license)

## Requirements

- Java 11+
- OpenMRS Platform 2.8.0+
- Webservices REST module 2.44.0+
- 10GB+ RAM recommended (for local LLM inference with the default 8B model; not required when using a remote LLM)
- Elasticsearch 8.14+ *(optional, for the hybrid retrieval pipeline; the default embedding and Lucene pipelines require no external services)*

## Setup

### 1. Build

```
mvn package
```

The `.omod` file is in `omod/target/`.

### 2. Download the LLM model *(local mode only)*

> **Skip this step** if you plan to use a remote LLM (see [LLM engine](#llm-engine) below).

Download Llama 3.3 8B (Q4_K_M quantization) in GGUF format (~5GB) from [Hugging Face](https://huggingface.co/bartowski/Llama-3.3-8B-Instruct-GGUF).

Place the `.gguf` file inside the OpenMRS application data directory (e.g., `<openmrs-application-data-directory>/chartsearchai/`). Model paths are resolved relative to this directory for security.

**Recommended models for local inference:**

| Model | RAM Needed | Chat Template | Download |
|-------|-----------|---------------|----------|
| Llama 3.2 3B | ~6GB total | `llama3` | [GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| MedGemma 4B | ~5GB total | `gemma` | [GGUF](https://huggingface.co/bartowski/google_medgemma-4b-it-GGUF) |
| Llama 3.3 8B *(default)* | ~10GB total | `llama3` | [GGUF](https://huggingface.co/bartowski/Llama-3.3-8B-Instruct-GGUF) |
| Mistral Nemo 12B | ~12GB total | `mistral` | [GGUF](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF) |

To switch models, update `chartsearchai.llm.modelFilePath` and `chartsearchai.llm.chatTemplate` — no rebuild needed. See [Evaluated models](#evaluated-models) for a full comparison of all models tested, including size trade-offs and licensing.

### 3. Download the embedding model

If embedding pre-filtering is enabled (default), download the all-MiniLM-L6-v2 ONNX model (~90MB) from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). You need both `model.onnx` and `vocab.txt` from the repository.

Place them alongside the LLM model (e.g., `<openmrs-application-data-directory>/chartsearchai/`).

### 4. Install

Copy the `.omod` file into the `modules` folder of the OpenMRS application data directory (e.g., `<openmrs-application-data-directory>/modules/`). The module will be loaded on the next OpenMRS startup.

### 5. Configure

Set these global properties in **Admin > Settings**:

#### LLM engine

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.llm.backend` | `local` | LLM inference engine: `local` runs a GGUF model in-process via llama.cpp; `remote` calls an OpenAI-compatible API |

**Local engine** (default) — requires a downloaded GGUF model file (see step 2):

| Property | Description |
|----------|-------------|
| `chartsearchai.llm.modelFilePath` | Relative path (within the OpenMRS application data directory) to the `.gguf` model file, e.g. `chartsearchai/Llama-3.3-8B-Instruct-Q4_K_M.gguf` |

**Remote engine** — set `chartsearchai.llm.backend` to `remote` and configure:

| Property | Where | Description |
|----------|-------|-------------|
| `chartsearchai.llm.remote.endpointUrl` | Global property | Chat completions endpoint URL (e.g. `https://api.openai.com/v1/chat/completions`) |
| `chartsearchai.llm.remote.apiKey` | `openmrs-runtime.properties` | API key for authentication (sent as `Bearer` token). Stored in runtime properties instead of the database for security |
| `chartsearchai.llm.remote.modelName` | Global property | Model identifier (e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`) |

The API key is read from `openmrs-runtime.properties` (not from the database) so it is never exposed in the Admin UI or database backups. Add it to your runtime properties file:

```
chartsearchai.llm.remote.apiKey=sk-your-api-key-here
```

The remote engine works with any provider that implements the OpenAI chat completions API format, including OpenAI, Azure OpenAI, Google AI, Anthropic (via proxy), vLLM, Ollama, and other self-hosted inference servers. No GGUF model download is needed when using the remote engine.

#### Retrieval pipeline

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.embedding.preFilter` | `true` | When `true`, uses the selected retrieval pipeline to narrow patient records to the most relevant ones before sending to the LLM. Set to `false` to send the full chart instead |
| `chartsearchai.retrieval.pipeline` | `embedding` | Selects the retrieval pipeline: `embedding` (default) uses vector similarity via an ONNX model with custom scoring; `lucene` uses Apache Lucene BM25 text search; `elasticsearch` uses Elasticsearch hybrid search combining BM25 text and kNN vector search via Reciprocal Rank Fusion (requires Elasticsearch 8.14+ configured in OpenMRS). All require `preFilter` to be `true`. Records are indexed automatically on first access. Changing this setting takes effect on the next query |

#### Embedding pipeline tuning

These settings only apply when `chartsearchai.retrieval.pipeline` is `embedding` (the default). They have no effect on the Lucene or Elasticsearch pipelines.

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.embedding.topK` | `10` | Maximum number of records sent to the LLM per query. When the query mentions a specific clinical type (e.g., "medications", "allergies", "lab results"), all records of that type are included regardless of topK, and remaining slots are filled with contextual records from other types. For queries without a detected type, topK is the hard cap. Type detection uses keyword matching — for example, "medications" and "drugs" both match drug orders, while "blood pressure" and "bp" both match observations |
| `chartsearchai.embedding.similarityRatio` | `0.80` | Minimum similarity score as a fraction of the top result's score. Records scoring below this ratio are excluded even if within the topK limit. Must be between 0 and 1 |
| `chartsearchai.embedding.scoreGapMultiplier` | `2.5` | Controls adaptive topK by detecting natural cluster boundaries in similarity scores. Higher values include more records; lower values cut more aggressively. Set to a very large value (e.g. 999) to disable gap detection |
| `chartsearchai.embedding.minScoreGap` | `0.10` | Minimum absolute gap between consecutive similarity scores required for the adaptive cutoff detector to trigger. Prevents premature cutting when a relatively large gap (compared to a tight cluster's running average) is still small in absolute terms. Only applies when gap detection is active |
| `chartsearchai.embedding.keywordWeight` | `0.3` | Additive keyword bonus weight in the hybrid retrieval formula: `finalScore = semanticScore + weight × keywordScore`. Keyword overlap can only increase the score, never decrease it. Set to `0` to disable keyword matching |
| `chartsearchai.embedding.typeBoostFactor` | `1.0` | Score multiplier applied to records whose resource type matches the query intent (e.g., drug orders when the query is about medications). Set to `1.0` to disable type boosting (default). Values like `1.2`–`1.5` provide moderate boosting. Must be between 1.0 and 3.0 |
| `chartsearchai.embedding.queryPrefix` | *(empty)* | Prefix prepended to the user query before embedding. Leave empty for models like all-MiniLM-L6-v2 that were not trained with instruction prefixes. Set to `search_query: ` or `Represent this sentence for searching relevant passages: ` for models that support instruction-aware queries (e.g., BGE) |
| `chartsearchai.embedding.maxSequenceLength` | `256` | Maximum WordPiece token sequence length for embedding input. Increase when using models that support longer contexts (e.g., 512 for BGE models). Must be between 32 and 8192 |
| `chartsearchai.embedding.modelFilePath` | — | Required when using the embedding or elasticsearch pipeline. Relative path to the ONNX model file (all-MiniLM-L6-v2), e.g. `chartsearchai/all-MiniLM-L6-v2.onnx`. Not needed for the Lucene pipeline |
| `chartsearchai.embedding.vocabFilePath` | — | Required when using the embedding or elasticsearch pipeline. Relative path to the WordPiece `vocab.txt` file, e.g. `chartsearchai/vocab.txt`. Not needed for the Lucene pipeline |

#### LLM tuning

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.llm.systemPrompt` | *(built-in clinical prompt)* | System prompt that guides how the LLM responds — e.g. answering only the question asked, using only the provided patient records, citing records by number, naming what is missing when records lack relevant information (e.g. "There are no records about diabetes in this patient's chart"), keeping answers concise, and returning structured JSON |
| `chartsearchai.llm.timeoutSeconds` | `120` | Maximum seconds to wait for LLM inference before timing out |
| `chartsearchai.llm.chatTemplate` | `llama3` | *(Local engine only)* Chat template for formatting prompts. Presets: `llama3`, `mistral`, `phi3`, `chatml`, `gemma`. Set to `auto` to use the model's built-in GGUF chat template. Or a custom template string with `{system}` and `{user}` placeholders |
| `chartsearchai.llm.idleTimeoutMinutes` | `30` | *(Local engine only)* Minutes of inactivity after which the LLM model is unloaded from memory to free RAM. The model is automatically reloaded on the next query. Set to `0` to keep the model loaded indefinitely |

#### Rate limiting and caching

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.rateLimitPerMinute` | `10` | Maximum queries per user per minute. Set to `0` to disable |
| `chartsearchai.cacheTtlMinutes` | `0` | Minutes to cache identical (patient, question) answers. Set to `0` to disable (default) |

#### Audit

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.auditLogRetentionDays` | `90` | Audit log entries older than this are purged daily. Set to `0` to retain all |

### 6. Grant privileges

| Privilege | Purpose |
|-----------|---------|
| **AI Query Patient Data** | Execute chart search queries |
| **View AI Audit Logs** | Access the audit log endpoint |

### 7. Indexing

When `chartsearchai.embedding.preFilter` is `true` (default), patient records are automatically indexed on first chart access for whichever retrieval pipeline is active. Subsequent data changes trigger automatic re-indexing via AOP hooks on encounter, obs, condition, diagnosis, allergy, order, program enrollment, medication dispense, and patient merge operations.

**Embedding pipeline** (default): Uses an ONNX embedding model for vector similarity search. A bulk backfill task (**"Chart Search AI - Embedding Backfill"**) is available in **Admin > Scheduler > Manage Scheduler** to pre-index all patients. The default model is all-MiniLM-L6-v2 (general-purpose, 384 dimensions). Any BERT-based ONNX embedding model can be used as a drop-in replacement by updating `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`. Embedding dimensions are auto-detected from the model output, so models with any dimension size work without code changes. After switching models, existing embeddings are incompatible — run the backfill task to re-index all patients with the new model.

**Lucene pipeline** (`chartsearchai.retrieval.pipeline=lucene`): Uses Apache Lucene BM25 text search with English stemming. No ONNX model files are required. The Lucene index is stored at `<openmrs-application-data-directory>/chartsearchai/lucene-index/` and is built automatically on first patient access. This pipeline is simpler to set up (no model download needed) and may be preferred for environments where the ONNX model is unavailable.

**Elasticsearch pipeline** (`chartsearchai.retrieval.pipeline=elasticsearch`): Uses Elasticsearch hybrid search combining BM25 text search with kNN dense vector search via Reciprocal Rank Fusion (RRF). Requires Elasticsearch 8.14+ configured in OpenMRS (set `hibernate.search.backend.uris` in runtime properties). Also requires the ONNX embedding model (same as the embedding pipeline) to compute vectors for the kNN side of the hybrid search. Patient records are indexed into a shared `chartsearchai-patient-records` Elasticsearch index with both text and embedding vector fields. The RRF algorithm fuses rankings from both signals — this means queries like "any cancer?" can find semantic matches (e.g. Kaposi sarcoma) via kNN even when the literal term is absent from the records, while also benefiting from BM25's lexical matching. If Elasticsearch is not available at query time, the pipeline automatically falls back to the embedding pipeline. After switching embedding models, delete the `chartsearchai-patient-records` index from Elasticsearch — it will be recreated with the new model's dimensions on the next patient access.

**Choosing a pipeline:**

| Consideration | Embedding *(default)* | Lucene | Elasticsearch |
|--------------|----------------------|--------|---------------|
| External dependencies | ONNX model files only | None | Elasticsearch 8.14+ cluster + ONNX model files |
| Semantic matching (e.g., "cancer" finds "Kaposi sarcoma") | Yes | No | Yes (via kNN) |
| Absent-data detection (returns "no records about X" instead of false positives) | Yes (z-score gate) | No | No |
| Type-aware auto-expand (e.g., "any conditions?" returns all conditions) | Yes | No | No |
| Adaptive result filtering (gap detection, similarity ratio) | Yes | No | No |
| Keyword matching | Yes (hybrid scoring) | Yes (BM25 with stemming) | Yes (BM25 + kNN via RRF) |
| Tunable parameters | Many (topK, similarityRatio, scoreGapMultiplier, keywordWeight, etc.) | Few (topK only) | Few (topK only; scoring delegated to Elasticsearch) |
| Compute location | In-process (JVM) | In-process (JVM) | Elasticsearch cluster |
| Graceful fallback | N/A (default) | Falls back to full chart on error | Falls back to embedding pipeline |

The **embedding pipeline** is recommended for most deployments — it runs entirely in-process, has the most sophisticated filtering (z-score gate for absent-data detection, gap detection for adaptive result cutoff, type-aware expansion), and requires no external services. The **Lucene pipeline** is the simplest option when the ONNX model is unavailable, but lacks semantic understanding. The **Elasticsearch pipeline** is best when you already have an ES cluster in your infrastructure and want to offload retrieval compute, but it lacks the embedding pipeline's absent-data detection and adaptive filtering — RRF always returns results from at least the kNN side, even when the patient has no relevant records.

### Testing the Elasticsearch pipeline locally

To test the Elasticsearch pipeline with the OpenMRS SDK:

**1. Start Elasticsearch 8.14+ with Docker:**

```
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.17.2
```

Verify it's running: `curl http://localhost:9200/_cluster/health`

**2. Configure OpenMRS to use Elasticsearch:**

Add to your OpenMRS runtime properties file (e.g., `~/openmrs/openmrs-runtime.properties`):

```
hibernate.search.backend.type=elasticsearch
hibernate.search.backend.uris=http://localhost:9200
```

Or if using the SDK with Docker, pass the environment variable when running the server:

```
OMRS_SEARCH=elasticsearch mvn openmrs-sdk:run
```

**3. Set the retrieval pipeline:**

In **Admin > Settings**, set:

| Property | Value |
|----------|-------|
| `chartsearchai.retrieval.pipeline` | `elasticsearch` |

Also ensure the ONNX embedding model and vocab files are configured (same as the default embedding pipeline).

**4. Query a patient** — records are indexed automatically on first access. To verify indexing, check the ES index:

```
curl http://localhost:9200/chartsearchai-patient-records/_count
```

**5. To reset and re-index**, delete the ES index:

```
curl -X DELETE http://localhost:9200/chartsearchai-patient-records
```

Records will be re-indexed on the next patient access.

## Query behavior

### Absent-data detection

When the embedding pipeline is active and a query has no keyword matches in the patient's records (e.g., asking "any cancer?" for a patient with no cancer-related records), the system uses a z-score gate to detect whether the top semantic match is a genuine result or just noise. If the patient has 30+ records and the best semantic score is not a statistical outlier (z-score < 1.5), the query returns "There are no records about [topic] in this patient's chart" instead of false positives. This prevents the system from returning unrelated records that happen to have slightly elevated similarity scores.

### Recency cap

Questions with numeric recency constraints are automatically detected and honored. For example, "last 3 blood pressure readings" or "most recent 5 lab results" will cap the results per concept group to the specified number, keeping only the most recent measurements. This applies across all retrieval pipelines.

### Input validation

Questions are checked against common prompt injection patterns (e.g., "ignore previous instructions", "you are now", "system prompt:") and rejected with HTTP 400 if matched. This is a defense-in-depth measure — the primary protection is the GBNF grammar that constrains LLM output to a fixed JSON structure regardless of prompt content. Normal clinical questions containing words like "ignore" or "instructions" in non-adversarial contexts (e.g., "What instructions were given at discharge?") are not affected.

## API

### Search

```
POST /ws/rest/v1/chartsearchai/search
Content-Type: application/json

{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?"
}
```

Response:

```json
{
  "answer": "The patient is currently on Metformin [1] and Lisinopril [3]...",
  "disclaimer": "This response is AI-generated and may not be accurate...",
  "questionId": "42",
  "references": [
    { "index": 3, "resourceType": "order", "resourceId": 789, "date": "2025-03-15" },
    { "index": 1, "resourceType": "order", "resourceId": 456, "date": "2025-01-10" }
  ]
}
```

`questionId` is a string identifier for this query, used to submit feedback (see below). It is omitted if audit logging fails.

### Streaming search (SSE)

For real-time token-by-token streaming:

```
POST /ws/rest/v1/chartsearchai/search/stream
Content-Type: application/json
Accept: text/event-stream

{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?"
}
```

SSE events:

| Event | Description |
|-------|-------------|
| `token` | A chunk of the answer text as it is generated |
| `done` | Final JSON with the complete answer, references (sorted most recent first, with `index`, `resourceType`, `resourceId`, `date`), `questionId`, and disclaimer |
| `error` | Error message if something goes wrong |

### Feedback

Submit user feedback (thumbs up/down) for an AI response. Requires the **"AI Query Patient Data"** privilege.

```
POST /ws/rest/v1/chartsearchai/feedback
Content-Type: application/json

{
  "questionId": "42",
  "rating": "positive",
  "comment": "Accurate and helpful"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `questionId` | Yes | The `questionId` from the search response |
| `rating` | Yes | `"positive"` or `"negative"` |
| `comment` | No | Optional text (max 500 characters, truncated if longer) |

Users can only submit feedback on their own queries. Submitting again overwrites the previous feedback.

### Audit log

Requires the **"View AI Audit Logs"** privilege.

```
GET /ws/rest/v1/chartsearchai/auditlog?patient=...&user=...&fromDate=...&toDate=...&startIndex=0&limit=50
```

All query parameters are optional. `fromDate` and `toDate` are epoch milliseconds. Returns paginated results ordered by most recent first, with a `totalCount` for pagination. Each entry includes `rating` and `feedbackComment` fields (null if no feedback was submitted).

## Patient access control

By default, any user with the **"AI Query Patient Data"** privilege can query any patient. To add patient-level restrictions (e.g., location-based or care-team-based), provide a custom Spring bean that implements the `PatientAccessCheck` interface:

```xml
<bean id="chartSearchAi.patientAccessCheck"
      class="com.example.LocationBasedPatientAccessCheck"/>
```

This overrides the default permissive implementation.

## Evals

The project includes an eval framework that tests retrieval quality, citation accuracy, absent-data detection, and prompt injection resistance without requiring a running LLM or external services.

### Running evals

```
mvn test -pl api -Dtest="*EvalTest"
```

Or run a specific suite:

```
mvn test -pl api -Dtest="RetrievalQualityEvalTest"
mvn test -pl api -Dtest="CitationEvalTest"
mvn test -pl api -Dtest="AbsentDataEvalTest"
mvn test -pl api -Dtest="PromptInjectionEvalTest"
```

### Adding cases

Each suite is driven by a JSON dataset in `api/src/test/resources/eval/`. To add a case, append an entry to the relevant file:

| File | What it tests |
|------|---------------|
| `retrieval-eval-dataset.json` | Query → expected record indices (recall\@30) |
| `citation-eval-dataset.json` | Simulated LLM JSON → expected citation indices (F1) |
| `absent-data-eval-dataset.json` | Query → expected keywords in "no records" answer |
| `prompt-injection-eval-dataset.json` | Adversarial payload → special tokens stripped |

### Metrics report

Each run appends per-case and summary metrics to `api/target/eval-results.csv` for tracking regressions over time.

## Evaluated models

The following models were evaluated for local inference via java-llama.cpp (Q4_K_M quantization, GGUF format). All figures are approximate and depend on hardware.

| Model | Params | File Size | Total RAM | Context Window | CPU Speed | Chat Template |
|-------|--------|-----------|-----------|----------------|-----------|---------------|
| Qwen 2.5 1.5B | 1.5B | ~1GB | ~2GB | 32K tokens | ~40–50 tok/s | chatml |
| Llama 3.2 3B | 3B | ~2GB | ~6GB | 128K tokens | ~20–30 tok/s | llama3 |
| Phi-3 Mini 3.8B | 3.8B | ~2GB | ~4GB | 4K tokens | ~15–25 tok/s | phi3 |
| MedGemma 4B | 4B | ~2.5GB | ~6–8GB | 128K tokens | ~10–20 tok/s | gemma |
| Mistral 7B | 7B | ~4GB | ~8GB | 32K tokens | ~10–15 tok/s | mistral |
| Qwen 2.5 7B | 7B | ~4GB | ~8GB | 128K tokens | ~8–12 tok/s | chatml |
| **Llama 3.3 8B** *(default)* | 8B | ~4.5GB | ~10GB | 128K tokens | ~8–12 tok/s | llama3 |
| Gemma 2 9B Instruct | 9B | ~5GB | ~10GB | 8K tokens | ~5–10 tok/s | gemma |
| Mistral Nemo 12B | 12B | ~7GB | ~12GB | 128K tokens | ~4–8 tok/s | mistral |
| Phi-3-Medium 14B | 14B | ~8GB | ~14GB | 4K tokens | ~3–6 tok/s | phi3 |
| Qwen 2.5 14B | 14B | ~8GB | ~14GB | 128K tokens | ~3–6 tok/s | chatml |
| MedGemma 27B Text | 27B | ~16.5GB | ~20–24GB | 128K tokens | ~1–2 tok/s | gemma |

### Model size guidance

- **3B models** (Llama 3.2 3B): Most deployable in low-resource settings but weaker instruction following — may produce verbose or hedging responses.
- **4B models** (MedGemma 4B): Medical-domain fine-tuning at ~3B resource cost. Good for low-resource deployments where clinical accuracy matters. Licensed under [Health AI Developer Foundations Terms](https://developers.google.com/health-ai-developer-foundations/terms) — requires validation before clinical deployment.
- **8B models** (Llama 3.3 8B): Recommended default. Significantly better reasoning and instruction following than 3B, feasible on 10GB RAM.
- **12B models** (Mistral Nemo 12B): Best sub-15B option for clinical Q&A. Strong medical text comprehension.
- **14B models** (Qwen 2.5 14B, Phi-3-Medium 14B): Best CPU-viable response quality, but slower (~2–4 tok/s) and need 14–16GB RAM.
- **27B models** (MedGemma 27B Text): Highest potential clinical accuracy, but CPU inference (~1–2 tok/s) is too slow for interactive use — practical only with GPU acceleration.

A server running OpenMRS typically uses 1–2GB for the JVM heap. A 4GB machine is insufficient — the smallest viable model requires at least 3–4GB on its own.

### Licensing notes

- **Llama 3.x** (Meta): Free for research and commercial use under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). Not technically "open source" by OSI definition — the only meaningful restriction is that products with over 700M monthly active users require a separate license.
- **Mistral** (Mistral AI): Apache 2.0 license.
- **MedGemma** (Google): [Health AI Developer Foundations Terms](https://developers.google.com/health-ai-developer-foundations/terms) — more restrictive than Llama. Requires validation before clinical deployment.
- **Phi-3** (Microsoft): MIT license — fully permissive with no usage restrictions.
- **Qwen 2.5** (Alibaba): Apache 2.0 license. Developed by a Chinese company subject to China's data laws — while GGUF models run locally with no data leaving the machine, some organizations may have compliance concerns.
- **Gemma 2** (Google): [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

See [docs/adr.md](docs/adr.md) (Decision 10) for detailed per-model analysis, trade-off discussion, and architectural rationale.

## Architecture

See [docs/adr.md](docs/adr.md) for architectural decisions and design rationale.

## License

This project is licensed under the [MPL 2.0](http://openmrs.org/license/).

Llama 3.3 is licensed under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/), Copyright (C) Meta Platforms, Inc. All Rights Reserved.

MedGemma is licensed under the [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms), Copyright (C) Google LLC. All Rights Reserved.
