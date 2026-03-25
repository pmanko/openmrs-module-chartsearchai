# Chart Search AI Module

An OpenMRS module that lets clinicians ask natural language questions about a patient's chart and get answers with source citations.

For project background, community discussion, and roadmap, see the [wiki project page](https://openmrs.atlassian.net/wiki/spaces/projects/pages/373325839/Chart+Search+aka+ChartSearchAI).

## Requirements

- Java 11+
- OpenMRS Platform 2.8.0+
- Webservices REST module 2.44.0+
- 10GB+ RAM recommended (for LLM inference with the default 8B model)

## Setup

### 1. Build

```
mvn package
```

The `.omod` file is in `omod/target/`.

### 2. Download the LLM model

Download Llama 3.3 8B (Q4_K_M quantization) in GGUF format (~5GB) from [Hugging Face](https://huggingface.co/bartowski/Llama-3.3-8B-Instruct-GGUF).

Place the `.gguf` file inside the OpenMRS application data directory (e.g., `<openmrs-application-data-directory>/chartsearchai/`). Model paths are resolved relative to this directory for security.

**Available models:**

| Model | RAM Needed | Chat Template | Download |
|-------|-----------|---------------|----------|
| Llama 3.2 3B | ~6GB total | `llama3` | [GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Llama 3.3 8B *(default)* | ~10GB total | `llama3` | [GGUF](https://huggingface.co/bartowski/Llama-3.3-8B-Instruct-GGUF) |
| Mistral Nemo 12B | ~12GB total | `mistral` | [GGUF](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF) |

Larger models produce more accurate answers with better instruction following. Smaller models use less RAM but may produce lower quality responses. To switch models, update `chartsearchai.llm.modelFilePath` and `chartsearchai.llm.chatTemplate` — no rebuild needed.

### 3. Download the embedding model

If embedding pre-filtering is enabled (default), download the all-MiniLM-L6-v2 ONNX model (~90MB) from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). You need both `model.onnx` and `vocab.txt` from the repository.

Place them alongside the LLM model (e.g., `<openmrs-application-data-directory>/chartsearchai/`).

### 4. Install

Copy the `.omod` file into the `modules` folder of the OpenMRS application data directory (e.g., `<openmrs-application-data-directory>/modules/`). The module will be loaded on the next OpenMRS startup.

### 5. Configure

Set these global properties in **Admin > Settings**:

#### Required

| Property | Description |
|----------|-------------|
| `chartsearchai.llm.modelFilePath` | Relative path (within the OpenMRS application data directory) to the `.gguf` model file, e.g. `chartsearchai/Llama-3.3-8B-Instruct-Q4_K_M.gguf` |

#### Embedding pre-filter

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.embedding.preFilter` | `true` | When `true`, uses embedding similarity to narrow patient records to the most relevant ones before sending to the LLM. Set to `false` to send the full chart instead |
| `chartsearchai.embedding.topK` | `10` | Maximum number of records to retrieve via embedding similarity when pre-filtering is enabled |
| `chartsearchai.embedding.similarityRatio` | `0.80` | Minimum similarity score as a fraction of the top result's score. Records scoring below this ratio are excluded even if within the topK limit. Must be between 0 and 1 |
| `chartsearchai.embedding.scoreGapMultiplier` | `2.5` | Controls adaptive topK by detecting natural cluster boundaries in similarity scores. Higher values include more records; lower values cut more aggressively. Set to a very large value (e.g. 999) to disable gap detection |
| `chartsearchai.embedding.keywordWeight` | `0.3` | Additive keyword bonus weight in the hybrid retrieval formula: `finalScore = semanticScore + weight × keywordScore`. Keyword overlap can only increase the score, never decrease it. Set to `0` to disable keyword matching |
| `chartsearchai.embedding.typeBoostFactor` | `1.0` | Score multiplier applied to records whose resource type matches the query intent (e.g., drug orders when the query is about medications). Set to `1.0` to disable type boosting (default). Values like `1.2`–`1.5` provide moderate boosting. Must be between 1.0 and 3.0 |
| `chartsearchai.embedding.queryPrefix` | *(empty)* | Prefix prepended to the user query before embedding. Leave empty for models like all-MiniLM-L6-v2 that were not trained with instruction prefixes. Set to `search_query: ` or `Represent this sentence for searching relevant passages: ` for models that support instruction-aware queries (e.g., BGE) |
| `chartsearchai.embedding.maxSequenceLength` | `256` | Maximum WordPiece token sequence length for embedding input. Increase when using models that support longer contexts (e.g., 512 for BGE models). Must be between 32 and 8192 |
| `chartsearchai.embedding.modelFilePath` | — | Required when pre-filtering is enabled. Relative path to the ONNX model file (all-MiniLM-L6-v2), e.g. `chartsearchai/all-MiniLM-L6-v2.onnx` |
| `chartsearchai.embedding.vocabFilePath` | — | Required when pre-filtering is enabled. Relative path to the WordPiece `vocab.txt` file, e.g. `chartsearchai/vocab.txt` |

#### LLM tuning

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.llm.chatTemplate` | `llama3` | Chat template for formatting prompts. Presets: `llama3`, `mistral`, `phi3`, `chatml`, `gemma`. Set to `auto` to use the model's built-in GGUF chat template. Or a custom template string with `{system}` and `{user}` placeholders |
| `chartsearchai.llm.systemPrompt` | *(built-in clinical prompt)* | System prompt that guides how the LLM responds — e.g. answering only the question asked, using only the provided patient records, citing records by number, declining to answer when records lack relevant information, keeping answers concise, and returning structured JSON |
| `chartsearchai.llm.timeoutSeconds` | `120` | Maximum seconds to wait for LLM inference before timing out |

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

### 7. Embeddings

When `chartsearchai.embedding.preFilter` is `true` (default), patient records are automatically indexed on first chart access. Subsequent data changes trigger automatic re-indexing via AOP hooks on encounter, obs, condition, diagnosis, allergy, order, program enrollment, medication dispense, and patient merge operations. A bulk backfill task (**"Chart Search AI - Embedding Backfill"**) is also available in **Admin > Scheduler > Manage Scheduler** if you prefer to pre-index all patients at once.

**Switching embedding models:** The default model is all-MiniLM-L6-v2 (general-purpose, 384 dimensions). Any BERT-based ONNX embedding model can be used as a drop-in replacement by updating `chartsearchai.embedding.modelFilePath` and `chartsearchai.embedding.vocabFilePath`. Embedding dimensions are auto-detected from the model output, so models with any dimension size work without code changes. After switching models, existing embeddings are incompatible — run the **"Chart Search AI - Embedding Backfill"** task to re-index all patients with the new model.

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
  "references": [
    { "index": 3, "resourceType": "order", "resourceId": 789, "date": "2025-03-15" },
    { "index": 1, "resourceType": "order", "resourceId": 456, "date": "2025-01-10" }
  ]
}
```

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
| `done` | Final JSON with the complete answer, references (sorted most recent first, with `index`, `resourceType`, `resourceId`, `date`), and disclaimer |
| `error` | Error message if something goes wrong |

### Audit log

Requires the **"View AI Audit Logs"** privilege.

```
GET /ws/rest/v1/chartsearchai/auditlog?patient=...&user=...&fromDate=...&toDate=...&startIndex=0&limit=50
```

All query parameters are optional. `fromDate` and `toDate` are epoch milliseconds. Returns paginated results ordered by most recent first, with a `totalCount` for pagination.

## Patient access control

By default, any user with the **"AI Query Patient Data"** privilege can query any patient. To add patient-level restrictions (e.g., location-based or care-team-based), provide a custom Spring bean that implements the `PatientAccessCheck` interface:

```xml
<bean id="chartSearchAi.patientAccessCheck"
      class="com.example.LocationBasedPatientAccessCheck"/>
```

This overrides the default permissive implementation.

## Architecture

See [docs/adr.md](docs/adr.md) for architectural decisions and design rationale.

## License

This project is licensed under the [MPL 2.0](http://openmrs.org/license/).

Llama 3.3 is licensed under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/), Copyright (C) Meta Platforms, Inc. All Rights Reserved.
