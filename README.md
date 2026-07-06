# Chart Search AI Module

[![Download Standalone](https://img.shields.io/badge/Download-O3_Standalone_with_Chart_Search_AI-blue?style=for-the-badge)](https://nightly.link/openmrs/openmrs-module-chartsearchai/workflows/build-standalone/main/openmrs-standalone-chartsearchai.zip)

An OpenMRS module that lets clinicians ask natural language questions about a patient's chart and get answers with source citations.

For project background, community discussion, and roadmap, see the [wiki project page](https://openmrs.atlassian.net/wiki/spaces/projects/pages/373325839/Chart+Search+aka+ChartSearchAI).

The standalone download above includes the backend module and frontend ESM. Chat requires an OpenAI-compatible LLM endpoint configured through global properties; for local deployments, run a local serving process such as med-agent-hub, vLLM, Ollama, or LM Studio and point chartsearchai at it.

- **LLM**: served outside the OpenMRS module behind an OpenAI-compatible chat-completions API. A local med-agent-hub sidecar is the preferred local path for staged answer/validation/in-depth behavior.
- **Retrieval + embedding**: the [querystore module](https://github.com/openmrs/openmrs-module-querystore) with [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) (~440 MB ONNX) — querystore is a required module and owns the retrieval path. chartsearchai no longer ships its own embedder.

> **Before running the download, see [Standalone platform notes](#standalone-platform-notes)** — in particular the **Windows JDK requirement** (the local embedder won't load on an old or Oracle JDK).

## Table of Contents

- [Try it on the demo server](#try-it-on-the-demo-server)
- [Standalone platform notes](#standalone-platform-notes)
- [Requirements](#requirements)
- [Docker](#docker)
- [Setup](#setup)
  - [1. Build](#1-build)
  - [2. Configure an LLM endpoint](#2-configure-an-llm-endpoint)
  - [3. Download the embedding model](#3-download-the-embedding-model-optional)
  - [4. Install](#4-install)
  - [5. Configure](#5-configure)
  - [6. Grant privileges](#6-grant-privileges)
  - [7. Indexing](#7-indexing)
- [Query behavior](#query-behavior)
- [API](#api)
  - [Chat](#chat)
  - [Streaming chat (SSE)](#streaming-chat-sse)
  - [Feedback](#feedback)
  - [Audit log](#audit-log)
- [Patient access control](#patient-access-control)
- [Evals](#evals)
- [Evaluated models](#evaluated-models)
- [Architecture](#architecture)
- [License](#license)

## Try it on the demo server

A live demo runs at **https://chartsearchai.openmrs.org** with the standard O3 reference patient set, so you can try Chart Search AI without installing anything.

1. Open https://chartsearchai.openmrs.org and log in (default credentials: `admin` / `Admin123`).
2. Click the magnifying-glass icon in the top header and search for **Betty Williams** — she is the reference patient with the most data on the demo (medications, vitals, conditions), so the AI has something to ground its answers in. Open her chart from the dropdown.

   ![Patient search overlay with "Betty" typed and Betty Williams in the result list](docs/images/ai-chart-search-patient-search.png)

3. Click the floating blue AI sparkle icon in the bottom-right corner of the chart (tooltip: *Ask AI about this patient*). A chat panel slides in.
4. Type a clinical question — e.g. *What medications is this patient on?*, *Any allergies?*, *Last 3 blood pressure readings* — and press **Send**, or click the microphone for voice input.
5. The answer appears when the backend emits the answer phase. Staged hub-backed models may then update the same message with answer-check status and a later in-depth section. The records the answer cites appear under **References**, numbered to match the inline citations (`[1]`, `[2]`, ...). Both the inline citations and the chips under **References** are clickable — they navigate to the relevant chart tab (Orders, Results, Allergies, Conditions, Programs, etc.) and highlight the source record. Every response carries the AI-generated disclaimer.

   ![AI Chart Search panel showing an answer with numbered citations on Betty Williams' chart](docs/images/ai-chart-search-demo.png)

6. Optionally rate the answer under **Was this helpful?** with **Helpful** / **Not helpful** and an optional comment. Feedback is recorded in the audit log alongside the question.

Notes:

- The AI button is only rendered for users with the **AI Query Patient Data** privilege.
- The launch surface is configurable via the frontend `chatLaunchMode` setting: `floating` (the bottom-right circular button used above), `workspace` (an icon in the top-right workspace strip that opens the chat as a docked workspace), or `both` (default).
- First-query latency on the demo reflects the configured LLM endpoint and any serving-side warmup. chartsearchai itself no longer owns model warmup or local model process lifecycle.
- The demo currently calls a remote/hub-backed LLM endpoint; latency reflects that endpoint, not an embedded OpenMRS JVM inference process.

## Standalone platform notes

Per-platform setup for the [downloaded standalone](#chart-search-ai-module) (Java 21+ required; see the Windows note for *which* JDK):

- **Windows:** run it with a JDK whose bundled MSVC runtime is **≥ 14.40** (Visual Studio 2022 17.10+) — [**Microsoft Build of OpenJDK 21.0.8+**](https://learn.microsoft.com/en-us/java/openjdk/download) (recommended), a current [**Eclipse Temurin 21**](https://adoptium.net/temurin/releases/?version=21), or **Azul Zulu**. The local ONNX embedder (querystore retrieval) is compiled against that runtime; an older JDK — **Oracle JDK in particular ships an outdated `msvcp140.dll` and fails even at 21/24** — makes ONNX fail to initialize (`onnxruntime.dll: A dynamic link library (DLL) initialization routine failed`) and chart queries error out. Check yours with `java -version` (vendor matters more than the number). See [onnxruntime#24287](https://github.com/microsoft/onnxruntime/issues/24287).
- **macOS — requires macOS 14 (Sonoma) or newer.** The bundled database and embedder dylibs are built for the macOS 14 SDK, so older releases — e.g. High Sierra 10.13 — can't launch them and the standalone exits at startup. On an unsupported macOS, use [Docker](#docker) or the [demo server](#try-it-on-the-demo-server) instead. If a (supported) build fails to start with a `libpcre2` dyld error, run `xattr -dr com.apple.quarantine <extracted-directory>` once and retry — current builds self-heal this at launcher startup, rebuild the patient search index on first run, and land on the login page.
- **Apple Silicon vs Intel Mac:** Apple Silicon is the supported Mac target and runs the bundled database out of the box. Intel (x86_64) Macs ship without a bundled MariaDB — there's no prebuilt x86_64 macOS binary to bundle (Homebrew publishes no x86_64 bottle, MariaDB has no macOS `.pkg`, and the bundleable `mariaDB4j-db-mac64` stops at 10.2.11) — so install one with `brew install mariadb` (on Intel, Homebrew builds a current MariaDB from source) and the standalone uses it automatically. This still requires macOS 14+; otherwise prefer Apple Silicon, Windows, or Docker.

## Requirements

- Java 11+
- OpenMRS Platform 2.8.0+
- Webservices REST module 2.44.0+
- Access to an OpenAI-compatible chat-completions endpoint. This can be a local sidecar/service such as med-agent-hub, vLLM, Ollama, or LM Studio, or a cloud/self-hosted remote endpoint.
- The [openmrs-module-querystore](https://github.com/openmrs/openmrs-module-querystore) module — required; it owns all retrieval, indexing, and embedding.

## Docker

```bash
git clone https://github.com/openmrs/openmrs-module-chartsearchai.git
cd openmrs-module-chartsearchai
docker compose up --build
```

No JDK or embedded LLM download is needed for the module container. The Docker deployment must still have a configured OpenAI-compatible LLM endpoint; the local development stack can provide this through med-agent-hub. The e5-base-v2 sentence embedder (~440MB) is provisioned for the [querystore deployment](#querystore-deployment) — set the matching querystore GPs after first start (see that section for the exact wiring).

First startup takes several minutes for database initialization and retrieval model provisioning. Once the logs show that OpenMRS has started, open http://localhost/openmrs/spa (default credentials: `admin` / `Admin123`). Subsequent starts are fast since the data volume persists.

Alternatively, download the [O3 Standalone with Chart Search AI](https://nightly.link/openmrs/openmrs-module-chartsearchai/workflows/build-standalone/main/openmrs-standalone-chartsearchai.zip) — a single zip with everything included, no Docker required (Java 21+ needed). See the [OpenMRS Standalone guide](https://openmrs.atlassian.net/wiki/spaces/docs/pages/25472583/OpenMRS+Standalone) for instructions.

## Setup

### 1. Build

```
mvn package
```

The `.omod` file is in `omod/target/`.

### 2. Configure an LLM endpoint

chartsearchai no longer bundles or starts a local GGUF model. It relays chat requests to an OpenAI-compatible chat-completions endpoint configured with global properties. For local/offline deployments, run a local serving service such as med-agent-hub, vLLM, Ollama, or LM Studio and point chartsearchai at that service.

Minimum required configuration:

| Property | Where | Description |
|----------|-------|-------------|
| `chartsearchai.llm.engine` | Global property | Must be `remote`. Any other value causes chat to fail clearly rather than falling back to an embedded engine |
| `chartsearchai.llm.remote.endpointUrl` | Global property | Chat completions endpoint URL, e.g. `http://localhost:8000/v1/chat/completions`, `http://localhost:11434/v1/chat/completions`, or a med-agent-hub `/v1/chat/completions` endpoint |
| `chartsearchai.llm.remote.modelName` | Global property | Model/profile identifier to send to the endpoint, e.g. a hub profile such as `single-12b-checked` |
| `chartsearchai.llm.remote.apikey` | `openmrs-runtime.properties` | Optional Bearer token for endpoints that require authentication |

The API key is read from `openmrs-runtime.properties` (not from the database) so it is never exposed in the Admin UI or database backups:

```
chartsearchai.llm.remote.apikey=sk-your-api-key-here
```

The same API path works for local sidecars, self-hosted GPU inference servers, and cloud providers. The OpenMRS module does not manage model downloads, process lifecycle, KV caches, or prompt-cache warmup; those belong to the serving endpoint.

### 3. Download the embedding model *(optional)*

The embedding model belongs to querystore — chartsearchai no longer ships its own. It is used both for querystore's retrieval index and for chartsearchai's citation grounding (the verifier embeds with the same model that built the index).

**Querystore-backed retrieval** — the querystore module handles retrieval; the LLM filters the top-K it returns. See [Querystore deployment](#querystore-deployment) below for the global properties this path expects and [ADR Decision 22](docs/adr.md#decision-22-e5-base-v2-for-the-querystore-backed-retrieval-path) for the model rationale. The LLM endpoint configured in [step 2](#2-configure-an-llm-endpoint) is still required. Download `intfloat/e5-base-v2` (~440MB):

- ONNX model: https://huggingface.co/Xenova/e5-base-v2/resolve/main/onnx/model.onnx *(self-contained — see [ADR Decision 22](docs/adr.md#decision-22-e5-base-v2-for-the-querystore-backed-retrieval-path) for why this source over the canonical `intfloat/e5-base-v2`)*
- Vocab: https://huggingface.co/Xenova/e5-base-v2/resolve/main/vocab.txt

Place both at `<openmrs-application-data-directory>/querystore/` and wire the global properties documented in [Querystore deployment](#querystore-deployment) below.

> chartsearchai's own embedding/Lucene/Elasticsearch pre-filter pipelines were removed in the querystore migration (#51); querystore is now the only retrieval and grounding embedder.

### 4. Install

Copy the `.omod` file into the `modules` folder of the OpenMRS application data directory (e.g., `<openmrs-application-data-directory>/modules/`). The module will be loaded on the next OpenMRS startup.

### 5. Configure

Set these global properties in **Admin > Settings**:

#### LLM engine

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.llm.engine` | `remote` | LLM inference engine. Chat requires `remote`; there is no bundled local engine mode |

Configure the remote endpoint:

| Property | Where | Description |
|----------|-------|-------------|
| `chartsearchai.llm.remote.endpointUrl` | Global property | Chat completions endpoint URL (e.g. `http://localhost:11434/v1/chat/completions` for Ollama, `http://gpu-server:8000/v1/chat/completions` for vLLM, `https://api.openai.com/v1/chat/completions` for OpenAI, `https://api.anthropic.com/v1/chat/completions` for Anthropic) |
| `chartsearchai.llm.remote.apikey` | `openmrs-runtime.properties` | API key for authentication (sent as `Bearer` token). Stored in runtime properties instead of the database for security. Optional — omit for self-hosted servers that don't require auth |
| `chartsearchai.llm.remote.modelName` | Global property | Model identifier (e.g. `llama3.3` for Ollama, `meta-llama/Llama-3.3-8B-Instruct` for vLLM, `gpt-4o` for OpenAI, `claude-opus-4-7` for Anthropic) |

The API key is read from `openmrs-runtime.properties` (not from the database) so it is never exposed in the Admin UI or database backups. Add it to your runtime properties file:

```
chartsearchai.llm.remote.apikey=sk-your-api-key-here
```

The remote engine works with any server that implements the OpenAI chat completions API format, including med-agent-hub, self-hosted inference servers (vLLM, Ollama, text-generation-inference), and cloud providers (OpenAI, Azure OpenAI, Google AI, Anthropic). Self-hosted/local sidecars keep patient data on-premise while still keeping inference runtime management outside the OpenMRS module.

For Anthropic's OpenAI-compat endpoint, point `chartsearchai.llm.remote.endpointUrl` at it and set `chartsearchai.llm.remote.modelName` to a Claude model identifier (e.g. `claude-opus-4-7`). The module emits Anthropic-compatible request bodies automatically: `response_format: json_schema` (Anthropic's compat endpoint rejects `json_object`) and, on Claude Opus 4.7, `top_k: 1` instead of `temperature` (Anthropic deprecated `temperature`/`top_p` on that model). Other Claude models (Opus 4.5/4.6, Haiku 4.5) keep using `temperature: 0`.

#### Querystore deployment

chartsearchai delegates all retrieval to the [openmrs-module-querystore](https://github.com/openmrs/openmrs-module-querystore) module (a required dependency) — querystore handles indexing and top-K retrieval, and the configured LLM endpoint reasons over the result set. chartsearchai's own embedding/Lucene/Elasticsearch pipelines were removed in the querystore migration (#51), so this is the only retrieval path. It is what the Docker image (`Dockerfile.backend` + `backend-init.sh`) provisions by default. See [ADR Decision 22](docs/adr.md#decision-22-e5-base-v2-for-the-querystore-backed-retrieval-path) for the full architectural narration.

**Deployment checklist:**

1. LLM endpoint available and configured ([step 2](#2-configure-an-llm-endpoint)).
2. e5-base-v2 ONNX + vocab placed at `<openmrs-application-data-directory>/querystore/` ([step 3](#3-download-the-embedding-model-optional)).
3. Global properties set per the table below.
4. Indexing is lazy on first chart access — no backfill task needed.

| Property | Value | Description |
|----------|-------|-------------|
| `querystore.embedding.modelFilePath` | `querystore/model.onnx` | Path to the ONNX embedder, relative to `<openmrs-application-data-directory>`. Querystore ships this with an empty default (the module is model-agnostic), so a fresh install must set it |
| `querystore.embedding.vocabFilePath` | `querystore/vocab.txt` | Path to the WordPiece vocab, same convention |
| `querystore.embedding.queryModelFilePath` | *(empty)* | Leave empty for `e5-base-v2`; set only for dual-encoder models like MedCPT |

A follow-up will populate these defaults in the querystore module's `config.xml` so fresh deploys work without manual GP wiring. The GPs are already declared there with empty values, which is why they appear in **Admin > Settings** today; until the defaults land, set them yourself after first start. See [ADR Decision 22](docs/adr.md#decision-22-e5-base-v2-for-the-querystore-backed-retrieval-path) for why this path uses `e5-base-v2`.

#### Hub-owned answer checks

Answer synthesis, temporal validation, citation grounding, drug-safety checks, and any staged answer review are owned by the configured hub/model endpoint. chartsearchai relays the endpoint's answer envelope, persists the phased updates, and renders metadata such as `answerValidation`, `references[].groundingStatus`, `inDepth`, and `safetyWarnings` when they are present.

#### Rate limiting and caching

| Property | Default | Description |
|----------|---------|-------------|
| `chartsearchai.rateLimitPerMinute` | `10` | Maximum queries per user per minute. Set to `0` to disable |

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

Retrieval indexing is owned entirely by the [openmrs-module-querystore](https://github.com/openmrs/openmrs-module-querystore) module — it performs its own lazy per-patient projection on first chart access and keeps the retrieval index current via core events. There is no chartsearchai-side index to build or maintain; see the querystore repo for indexing details.

## Query behavior

chartsearchai is now a relay/persistence/UI layer for chat. It sends the patient uuid, prior conversation turns, and the current user question to the configured OpenAI-compatible endpoint. The hub/model endpoint owns chart retrieval, prompt assembly, temporal validation, citation grounding, answer review, in-depth generation, and optional safety checks. chartsearchai preserves and renders the returned metadata, including `confidence`, `answerValidation`, `references[].groundingStatus`, `inDepth`, and `safetyWarnings` when present.

This split keeps OpenMRS responsible for session state, authorization, audit, feedback, and source navigation, while the hub owns model orchestration and answer-quality gates.

## API

### Chat

Multi-turn: pass a prior response's `session` uuid (or the one returned by `GET /chat`) to continue that
conversation; omit it to use or open the caller's active session for the patient. Every chat turn relays
through the configured remote LLM endpoint — chartsearchai has no local inference — see [Configure](#5-configure).

```
POST /ws/rest/v1/chartsearchai/chat
Content-Type: application/json

{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?",
  "session": "existing-session-uuid (optional)",
  "endpointUrl": "per-request backend override (optional)",
  "modelName": "per-request backend override (optional)"
}
```

Response:

```json
{
  "answer": "The patient is currently on Metformin [1] and Lisinopril [3]...",
  "disclaimer": "This response is AI-generated and may not be accurate...",
  "references": [
    { "index": 3, "resourceType": "order", "resourceUuid": "a8f5f167-4ee2-4d2a-94f9-3f3f86d2e9b6", "date": "2025-03-15" },
    { "index": 1, "resourceType": "order", "resourceUuid": "5946f880-b197-400b-9caa-a3c661d71165", "date": "2025-01-10" }
  ],
  "blocks": [],
  "confidence": { "answer": { "level": "green", "note": "" } },
  "answerValidation": { "status": "checked", "label": "Checked" },
  "safetyWarnings": [],
  "session": "session-uuid",
  "messageId": "assistant-message-uuid",
  "model": "resolved-model-id"
}
```

`confidence`, `answerValidation`, and `safetyWarnings` are only present when the resolved model/level
produces them. `safetyWarnings` entries are non-blocking advisories from the configured hub/model
endpoint.

### Streaming chat (SSE)

```
POST /ws/rest/v1/chartsearchai/chat/stream
Content-Type: application/json
Accept: text/event-stream

{
  "patient": "patient-uuid-here",
  "question": "What medications is this patient on?",
  "session": "existing-session-uuid (optional)",
  "staged": "true (optional — see below)"
}
```

The `X-ChartSearchAi-Session` response header carries the session uuid before the stream opens.

For non-staged endpoints, the stream emits one final `done` event with the same JSON shape as the sync
`POST /chat` response above.

Setting `staged: "true"` opts into the staged answer/validation/in-depth event sequence — only meaningful
for a model the configured remote endpoint advertises as staged-capable (the `staged` field on each entry
returned by `GET /endpoints`):

| Event | Description |
|-------|-------------|
| `answer_done` | The direct answer is complete; the envelope's `answerValidation.status` is `validating` and `inDepth.status` is `pending` |
| `answer_validation` | *(only when the level has a validator)* the same message updated after its self-check |
| `indepth_pending` | The in-depth analysis is about to start |
| `indepth_done` / `indepth_error` | The in-depth analysis completed or failed |
| `done` | Final envelope with the settled answer and the completed in-depth |
| `error` | Error message if something goes wrong |

Every JSON event payload also carries `session`, `messageId`, `model`, and `disclaimer`.

### Serving-side warmup

chartsearchai no longer exposes an in-module warmup endpoint or manages an embedded LLM prompt cache. If the configured LLM endpoint supports model loading or prompt-cache warmup, run that through the serving layer itself. For local demos, warm the med-agent-hub/router sidecar before recording latency-sensitive sessions.

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

The project includes an eval framework that tests citation accuracy, absent-data answering, and prompt injection resistance without requiring a running LLM or external services.

### Running evals

```
mvn test -pl api -Dtest="*EvalTest"
```

Or run a specific suite:

```
mvn test -pl api -Dtest="CitationEvalTest"
mvn test -pl api -Dtest="AbsentDataEvalTest"
mvn test -pl api -Dtest="PromptInjectionEvalTest" -Dchartsearchai.prompt.injection.test=true
```

### Adding cases

Each suite is driven by a JSON dataset in `api/src/test/resources/eval/`. To add a case, append an entry to the relevant file:

| File | What it tests |
|------|---------------|
| `citation-eval-dataset.json` | Simulated LLM JSON → expected citation indices (F1) |
| `absent-data-eval-dataset.json` | Query → expected keywords in "no records" answer |
| `prompt-injection-eval-dataset.json` | Adversarial payload → LLM produces safe JSON, no system prompt leakage |

### Metrics report

Each run appends per-case and summary metrics to `api/target/eval-results.csv` for tracking regressions over time.

## Evaluated models

The following open-weight models have been evaluated as candidate serving models for local or self-hosted OpenAI-compatible endpoints. The module does not serve these itself; RAM and speed figures apply to the serving sidecar/runtime.

| Model | Params | File Size | Total RAM | Context Window | CPU Speed | Chat Template |
|-------|--------|-----------|-----------|----------------|-----------|---------------|
| Qwen 2.5 1.5B | 1.5B | ~1GB | ~2GB | 32K tokens | ~40–50 tok/s | chatml |
| Gemma 3 1B | 1B | ~0.7GB | ~2GB | 32K tokens | ~40–50 tok/s | gemma |
| Gemma 3n E2B | E2B (5B total) | ~1.5GB | ~3GB | 32K tokens | ~25–35 tok/s | gemma |
| Gemma 4 E2B | E2B (2.3B eff) | ~1.5GB | ~3–5GB | 128K tokens | ~25–35 tok/s | gemma |
| Llama 3.2 3B | 3B | ~2GB | ~6GB | 128K tokens | ~20–30 tok/s | llama3 |
| Phi-3 Mini 3.8B | 3.8B | ~2GB | ~4GB | 4K tokens | ~15–25 tok/s | phi3 |
| Gemma 3 4B | 4B | ~2.5GB | ~6–8GB | 128K tokens | ~10–20 tok/s | gemma |
| Gemma 3n E4B | E4B (8B total) | ~2.5GB | ~3–5GB | 32K tokens | ~15–25 tok/s | gemma |
| **Gemma 4 E4B** | E4B (4.5B eff) | ~2.5GB | ~6–8GB | 128K tokens | ~10–20 tok/s | gemma |
| MedGemma 1.5 4B | 4B | ~2.5GB | ~6–8GB | 128K tokens | ~10–20 tok/s | gemma |
| MedGemma 4B | 4B | ~2.5GB | ~6–8GB | 128K tokens | ~10–20 tok/s | gemma |
| Mistral 7B | 7B | ~4GB | ~8GB | 32K tokens | ~10–15 tok/s | mistral |
| Qwen 2.5 7B | 7B | ~4GB | ~8GB | 128K tokens | ~8–12 tok/s | chatml |
| Llama 3.3 8B | 8B | ~4.5GB | ~10GB | 128K tokens | ~8–12 tok/s | llama3 |
| Gemma 2 9B Instruct | 9B | ~5GB | ~10GB | 8K tokens | ~5–10 tok/s | gemma |
| Gemma 3 12B | 12B | ~7GB | ~12GB | 128K tokens | ~4–8 tok/s | gemma |
| Mistral Nemo 12B | 12B | ~7GB | ~12GB | 128K tokens | ~4–8 tok/s | mistral |
| Phi-3-Medium 14B | 14B | ~8GB | ~14GB | 4K tokens | ~3–6 tok/s | phi3 |
| Qwen 2.5 14B | 14B | ~8GB | ~14GB | 128K tokens | ~3–6 tok/s | chatml |
| **Gemma 4 26B MoE** | 26B (3.8B active) | ~15GB | ~18–22GB | 256K tokens | ~3–6 tok/s | gemma |
| Gemma 3 27B | 27B | ~16.5GB | ~20–24GB | 128K tokens | ~1–2 tok/s | gemma |
| MedGemma 27B Text | 27B | ~16.5GB | ~20–24GB | 128K tokens | ~1–2 tok/s | gemma |
| Gemma 4 31B | 31B | ~18GB | ~22–26GB | 256K tokens | ~1–2 tok/s | gemma |

### Model size guidance

- **1–2B models** (Gemma 3 1B, Gemma 3n E2B, Gemma 4 E2B): Ultra-low-resource or on-device deployments. Gemma 3n and Gemma 4 "E" models use Per-Layer Embeddings (PLE) for memory efficiency — E2B runs in as little as ~3GB RAM. Weaker reasoning but fast inference. Gemma 4 E2B offers 128K context; Gemma 3 1B and 3n E2B are limited to 32K.
- **3B models** (Llama 3.2 3B): Most deployable in low-resource settings but weaker instruction following — may produce verbose or hedging responses.
- **4B models** (MedGemma 1.5 4B, Gemma 4 E4B): Recommended default tier. MedGemma 1.5 4B provides medical-domain fine-tuning with improved medical imaging support. Gemma 4 E4B is a strong general-purpose alternative under the permissive Apache 2.0 license. Both offer 128K context and ~10–20 tok/s CPU inference at ~6–8GB total RAM.
- **8B models** (Llama 3.3 8B): Significantly better general reasoning and instruction following than 4B, feasible on 10GB RAM.
- **12B models** (Gemma 3 12B, Mistral Nemo 12B): Best sub-15B options for clinical Q&A. Gemma 3 12B offers 128K context with strong reasoning. Mistral Nemo 12B has strong medical text comprehension.
- **14B models** (Qwen 2.5 14B, Phi-3-Medium 14B): Best CPU-viable response quality, but slower (~2–4 tok/s) and need 14–16GB RAM.
- **26–31B models** (Gemma 4 26B MoE, Gemma 4 31B, MedGemma 27B Text): Highest quality tier. Gemma 4 26B MoE activates only 3.8B parameters per token, offering faster inference than dense models at this size. Gemma 4 31B Dense offers the best general reasoning under Apache 2.0. MedGemma 27B Text is the medical-domain specialist. All require ~20GB+ RAM and are practical mainly with GPU acceleration.

A server running OpenMRS typically uses 1–2GB for the JVM heap. Size the LLM serving endpoint separately from the OpenMRS JVM; colocating both on one machine requires enough memory for both.

### Licensing notes

- **Gemma 4** (Google): Apache 2.0 license — fully permissive, no usage restrictions. The first Gemma family release under a standard open-source license.
- **Gemma 3, Gemma 3n** (Google): [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — custom license that permits commercial use but reserves Google's right to terminate access for policy violations. More restrictive than Apache 2.0.
- **Gemma 2** (Google): [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **MedGemma** (Google): [Health AI Developer Foundations Terms](https://developers.google.com/health-ai-developer-foundations/terms) — more restrictive than Gemma. Requires validation before clinical deployment. Applies to both MedGemma 1.5 4B and MedGemma 27B Text.
- **Llama 3.x** (Meta): Free for research and commercial use under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). Not technically "open source" by OSI definition — the only meaningful restriction is that products with over 700M monthly active users require a separate license.
- **Mistral** (Mistral AI): Apache 2.0 license.
- **Phi-3** (Microsoft): MIT license — fully permissive with no usage restrictions.
- **Qwen 2.5** (Alibaba): Apache 2.0 license. Developed by a Chinese company subject to China's data laws — while GGUF models run locally with no data leaving the machine, some organizations may have compliance concerns.

See [docs/adr.md](docs/adr.md) (Decision 10) for detailed per-model analysis, trade-off discussion, and architectural rationale.

## Architecture

See [docs/adr.md](docs/adr.md) for architectural decisions and design rationale.

## License

This project is licensed under the [MPL 2.0](http://openmrs.org/license/).

MedGemma is licensed under the [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms), Copyright (C) Google LLC. All Rights Reserved.

Gemma 4 is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

Gemma 3 and Gemma 3n are licensed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), Copyright (C) Google LLC. All Rights Reserved.

Llama 3.3 is licensed under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/), Copyright (C) Meta Platforms, Inc. All Rights Reserved.
