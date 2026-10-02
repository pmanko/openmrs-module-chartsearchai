# OpenMRS provider interface reference

The [provider contract](../README.md#provider-integration-contract) owns shared
behavior; this reference describes the module boundary and adapter vocabulary.
The [Hub](https://github.com/pmanko/med-agent-hub/blob/main/README.md) owns its native
API and the [frontend](https://github.com/pmanko/openmrs-esm-chartsearchai/blob/main/README.md) owns rendering.

## 1. The three hops

```
ESM (openmrs-esm-chartsearchai)
   |  REST + SSE, provider-neutral TurnEventType vocabulary            (section 4)
OpenMRS module (openmrs-module-chartsearchai)
   |-- bundled provider --> llama-server            OpenAI + llama.cpp extensions  (section 2)
   '-- hub provider     --> med-agent-hub           OpenAI + hub extensions        (section 3)
                              '--> llama-router     OpenAI + llama.cpp extensions  (section 3.4)
```

Provider selection is configuration (`chartsearchai.providers.enabled`, default `bundled`;
`chartsearchai.providers.default`). An enabled Hub provider is ready only when `chartsearchai.hub.endpointUrl` is set.
There is no fallback between providers: an unknown, disabled or unready provider fails the turn with an
explicit `ProviderUnavailableException` (`api/.../provider/ClinicalAnswerProviderRegistry.java`,
`require()`), and the picker is hidden when only one provider is enabled (`isPickerVisible()`).

## 2. Bundled provider to llama-server

Engine selection: `chartsearchai.llm.engine` = `local` (the module manages a llama-server subprocess) or
`remote` (any OpenAI-compatible chat-completions endpoint). Source: `api/.../api/impl/LocalLlmEngine.java`.

| Call | Standard | Used for | Source |
|---|---|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions; body carries `messages`, `stream`, `temperature`, `max_tokens` | the answer, streamed | `LocalLlmEngine.java` |
| `cache_prompt` in that body | llama.cpp extension | KV-cache reuse across turns on one patient | `LocalLlmEngine.java` (request body builder) |
| `POST /tokenize` | llama.cpp | exact token count of a text | `LocalLlmEngine.java` |
| `POST /v1/chat/completions/input_tokens` | llama.cpp extension of the OpenAI path | exact input-token count of the complete prompt, before generation | `LocalLlmEngine.java`, used by `LlmInferenceService.ensurePromptFits` |
| `GET /health` | llama.cpp | readiness | `LocalLlmEngine.java` |
| `GET /slots` | llama.cpp | slot occupancy for KV-cache scoping | `LocalLlmEngine.java` |

### 2.1 The three text channels

The bundled pipeline streams three distinct kinds of text, and the provider stream keeps them
distinct because two facts about the preview cannot be recovered from its text:

| Channel | Event | What it is |
|---|---|---|
| preview | `preliminary_delta` | optional progressive reasoning over an independently-numbered top-K chart (`chartsearchai.progressiveReasoning.topK`). Its `[N]` markers do NOT index the records the committed answer cites, so a client must strip them; and it is provisional, so committed reasoning REPLACES it. |
| committed reasoning | `reasoning_delta` | chain-of-thought over the full chart view. Render distinctly, never as the answer. |
| answer | `answer_delta` | the answer text itself. |

The legacy `/search/stream` names the same three (`preliminary`, `thinking`, `token`). Folding the
preview into `reasoning_delta` loses the independent citation scope and provisional status.

Budget enforcement on this hop is the bundled path's own: a prompt whose exact count exceeds the configured
input budget fails the turn with `ChartTooLargeException` (`LlmInferenceService.ensurePromptFits`), and
mandatory evidence that cannot fit makes the turn abstain with `InsufficientContextException`
(`QueryStoreChartBuilder`). Cancellation reaches the running inference through `CancellationSignal`
(`LlmProvider.searchStreaming(..., CancellationSignal)`).

## 3. Hub provider to med-agent-hub

### 3.1 Request

`chartsearchai.hub.endpointUrl` must end with `/v1/chat/completions` (`HubProfileService.java`);
`/v1/models` is derived from it for profile discovery (`HubProfileService.java`). The call is
`POST` with `Content-Type: application/json`, `Accept: text/event-stream`, and `Authorization: Bearer
<chartsearchai.hub.apikey>` when the key is set (`HttpHubStreamTransport.java`).

The body is an OpenAI `ChatCompletionRequest` (`med-agent-hub/server/openai_compat.py`, class
`ChatCompletionRequest`): `model`, `messages`, `stream`, `temperature`, `max_tokens`,
`response_format`, plus these extensions:

| Extension field | Meaning | Module source |
|---|---|---|
| `model` | the hub product profile id (from `/v1/models`) | `HubCallRequest.profileId` |
| `messages` | the question and prior clinical turns as chat messages | `HubCallRequest.question`, `priorTurns` |
| `patient` | the OpenMRS patient uuid the hub may retrieve context for | `HubCallRequest.patientUuid` |
| `session` | the conversation id, for multi-turn context | `HubCallRequest.conversationId` |
| `request_id` | correlation id for one turn | `HubCallRequest.requestId` |
| `require_product_profile` | refuse a bare model; a staged product profile is required | `HubClinicalAnswerProvider` |
| `context` | module-supplied context object (chart slice, scopes) | `HubClinicalAnswerProvider` |

### 3.2 Response stream

The hub answers with server-sent events. Each `data:` frame is an OpenAI `chat.completion.chunk`
(`choices[0].delta`, `finish_reason`; `openai_compat.py`) whose payload also carries the hub's
vendor keys: `answer`, `answerValidation` (`status`, `label`, `summary`), `inDepth` (`status`,
`answer`, `error`), `safetyStatus`, `safetyCheck` (`schema_version`, `status`, `warnings`, `package`,
`coverage`, `identity_confidence`, `issues`), `references`, `blocks` (`server/engine.py`,
`server/drug_safety.py`). The module parses each frame into a `HubWireEvent(event, payload)` and maps
the SSE `event:` name onto the canonical lifecycle (`HubClinicalAnswerProvider.mapEvent`, `handleWire`):

| Hub SSE event | Module `TurnEventType` |
|---|---|
| `answer_done` | `answer_done` |
| `answer_validation` | `answer_validation` |
| `evidence_updated`, `grounded` | `evidence_updated` |
| `indepth_pending` | `indepth_pending` |
| `indepth_done` | `indepth_done` |
| `indepth_error` | `indepth_error` |
| `heartbeat` | `heartbeat` |
| `error` | `turn_error`, problem code taken from the body |
| `done` | `turn_done` |
| `status`, `label`, `summary` | payload keys inside `answerValidation` and `inDepth`, not events |
| `stream_cancelled` (hub-emitted on cancel) | not mapped; `mapEvent` returns null and the frame is ignored |

Anything after the first terminal event is dropped (`handleWire` returns once `doneSeen` is set). The
hub provider declares `answer`, `answer_check`, `answer_review`, `indepth`, `grounding`, `drug_safety`,
`structured_blocks`, `multi_turn_context`; it does not declare `token_streaming`, and it emits no
`answer_delta`: the answer arrives whole on `answer_done`, then is checked and deepened.

### 3.3 Safety status on this hop

`safetyStatus` and `safetyCheck.status` are the outcome of the check that ran: `unavailable` when
source or patient context was missing or the check threw (`issues` carries `execution_failed`,
`coverage.execution_complete` is false), `limited` when issues were recorded, `checked` only when the
run completed clean (`server/drug_safety.py`, `check_answer_safety`). The bundled provider reports the
same three values with the same meaning from `DrugSafetyValidator.validateWithStatus`
(`api/.../reference/DrugSafetyValidator.java`), carried on `ChartAnswer.safetyStatus`.

### 3.4 Hub to llama-router

The hub itself speaks OpenAI to the router: `/v1/chat/completions` for generation, `/v1/models` for
discovery, and llama.cpp's `/tokenize` and `/v1/chat/completions/input_tokens` for exact budgets
(`server/engine.py`). Router profiles are product configuration in `server/levels.yaml`.
The caller prepares the model service; OpenClinAI's
[router lifecycle script](https://github.com/pmanko/openclinai.org/blob/main/scripts/llama-router-up.sh)
runs from the umbrella root, never as part of harness execution.

## 4. Module to ESM: the provider-neutral surface

Base path `/openmrs/ws/rest/v1/chartsearchai` (`ChartSearchAiRestController.java`).

| Method and path | Purpose |
|---|---|
| `GET /providers` | enabled providers: `id`, `ready`, `default`, `modes`, `capabilities`, `unavailableReason`; picker visibility |
| `GET /models` | hub product profiles, when the hub provider is enabled |
| `POST /chat/stream` | one provider turn as SSE (below) |
| `GET /chat` | the patient's active conversation and prior turns |
| `POST /chat/new` | close the active conversation and open a fresh one |
| `POST /search`, `POST /search/stream` | the pre-provider endpoints, unchanged, always bundled |
| `POST /feedback`, `GET /auditlog`, `GET /chartalerts`, `GET /drugreferencestatus`, `POST /prewarm`, `GET /prewarmstatus`, `POST /warmup` | module services outside the answer lifecycle |

### 4.1 The turn stream

`POST /chat/stream` emits the canonical lifecycle as SSE `event:` names taken from `TurnEventType`
(`api/.../provider/TurnEventType.java`), validated by `TurnLifecycleValidator`:

| Event | Payload | Rule |
|---|---|---|
| `turn_started` | `session`, `messageId`, `provider` | exactly one, first |
| `heartbeat` | none | keep-alive, any time |
| `preliminary_delta` | text | the optional progressive PREVIEW (`chartsearchai.progressiveReasoning.enabled`, default off); only with `token_streaming`; repeats; may not resume once committed reasoning begins |
| `reasoning_delta` | text | committed reasoning; only with `token_streaming`; ends once answer deltas begin |
| `answer_delta` | text | only with `token_streaming` |
| `answer_done` | the answer envelope | required before `turn_done` |
| `answer_validation` | envelope with `answerValidation` | only with `answer_check` or `answer_review` |
| `evidence_updated` | envelope with final `references` and safety findings | only with `grounding` |
| `indepth_pending`, `indepth_done`, `indepth_error` | envelope with `inDepth` | only with `indepth`; at most one of done/error, after pending |
| `turn_done` | the final envelope | exactly one terminal event |
| `turn_error` | `problemCode` | the other terminal event; may occur before any answer |

Every envelope-bearing event's payload is the provider's `AnswerEnvelope.payload` plus `session`,
`messageId`, `provider`, `disclaimer` and, once persisted, `auditLogId`
(`ChartSearchAiRestController.writeTurnEvent`). The bundled envelope carries `answer`, `blocks`,
`references`, `safetyWarnings`, `safetyStatus`, `safetyCheck`, `inputTokens`, `outputTokens`,
`cachedTokens` (`BundledClinicalAnswerProvider.toAnswerEnvelope`), and the module's answer-limit statements (`misattributedOrderCitations`,
`unstatedFindingSeverities`, `conditionRuleCoverage`, `interactionPairs`, `activeOrderClaims`) plus the
wire shape of the safety chips: the controller publishes them from the `ChartAnswer` the envelope keeps
(`AnswerEnvelope.getSource()`), through the same `putModuleStatements` that `/search` and
`/search/stream` use, on every envelope-bearing event and on the persisted turn. A relayed (hub)
envelope has no source and passes through unchanged.

Capabilities on the wire (`ProviderCapability.java`): `answer`, `token_streaming`, `answer_check`,
`answer_review`, `indepth`, `grounding`, `drug_safety`, `structured_blocks`, `multi_turn_context`.

### 4.2 What the ESM consumes

`src/api/chartsearchai.ts` (`chatPatientChartStream`) handles `turn_started`, `preliminary_delta`,
`answer_delta` and `reasoning_delta` (text frames; one leading space stripped per SSE line, so a
token's own leading space survives), `answer_done`, `answer_validation`, `evidence_updated`, `indepth_pending`,
`indepth_done`, `indepth_error`, `turn_done`, `turn_error`, and folds each envelope into one message
through the turn-phase model in `src/hooks/useChartSearchAi.ts` (`answering`, `checking`, `settled`,
`in-depth`, `complete`, `error`). Deltas accumulate only while the phase is `answering`; `answer_done`
restates the whole answer, so a provider that streams no tokens (the hub) renders the same as before
and a stopped turn ignores late frames. The preview accumulates into `ChatMessage.preliminaryReasoning`
with the shared `citationStripPattern` applied to the whole accumulation (so a marker split across
frames is still removed), and the first committed reasoning delta or answer token clears it.
The safety badge reads `safetyCheck.status ?? safetyStatus`; the same three values render the same way
for both providers (`src/components/ai-response-panel.component.tsx`).

## 5. Behavior with the hub off

- `chartsearchai.providers.enabled` defaults to `bundled`; `getDefaultProviderId()` uses
  `bundled` when the configured default is disabled and bundled is enabled; otherwise it uses
  the first enabled provider. If none are enabled, discovery is empty and execution is unavailable.
- `/providers` lists enabled providers only. An enabled Hub without its endpoint has
  `ready=false` and an `unavailableReason` naming `chartsearchai.hub.endpointUrl`; a disabled
  Hub is absent from discovery.
- `/search` and `/search/stream` do not consult the provider registry at all.
- The bundled path's own behaviors (context slice from querystore, exact token budgets, abstention on
  truncated or overflowing mandatory context, cancellation, `safetyStatus`) apply regardless of the hub.
  The module's tests and runtime evidence establish these behaviors independently of Hub configuration.

## 6. Where each fact lives

| Fact | File |
|---|---|
| bundled engine calls | `api/src/main/java/org/openmrs/module/chartsearchai/api/impl/LocalLlmEngine.java` |
| hub endpoint rule, profile discovery | `api/src/main/java/org/openmrs/module/chartsearchai/api/impl/HubProfileService.java` |
| hub HTTP transport | `api/src/main/java/org/openmrs/module/chartsearchai/api/provider/HttpHubStreamTransport.java` |
| hub event mapping, terminal handling | `api/src/main/java/org/openmrs/module/chartsearchai/api/provider/HubClinicalAnswerProvider.java` |
| lifecycle vocabulary and rules | `.../provider/TurnEventType.java`, `.../provider/TurnLifecycleValidator.java`, `.../provider/ProviderCapability.java` |
| provider selection | `.../provider/ClinicalAnswerProviderRegistry.java` |
| REST surface and turn serialization | `omod/src/main/java/org/openmrs/module/chartsearchai/web/rest/ChartSearchAiRestController.java` |
| hub request model and chunk shape | `med-agent-hub/server/openai_compat.py` |
| hub safety status rule | `med-agent-hub/server/drug_safety.py` |
| ESM stream client and phases | `chartsearchai-esm/src/api/chartsearchai.ts`, `chartsearchai-esm/src/hooks/useChartSearchAi.ts` |
