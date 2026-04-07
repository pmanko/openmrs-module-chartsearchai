# Rules

- When a test fails, fix the pipeline/production code — never weaken assertions, revert test data, or change expected values to match wrong behavior.
- Prefer root-cause fixes over incremental symptom patches. Diagnose *why* something is broken before proposing a fix. Start with the best solution, not the quickest.
- Do not go in circles analyzing a problem without fixing it.
- Tests must call the actual production pipeline — no simulations, mocks, or reimplementations of pipeline logic in test code. This includes unit tests for internal methods: do not call internal methods directly with hand-crafted inputs. Every test must exercise the full production code path using real data (e.g. the ONNX embedding model and patient test datasets).
- When a multi-step pipeline exists (e.g. normalize → transform → embed), tests must call the composed production method, not manually chain the individual steps. Manually chaining steps in tests can mask bugs where the production code assembles the pipeline differently. If no composed method exists, extract one and use it from both production and test code.
- Tests are the specification. Modifying tests to make them pass is changing the spec — only fix the production code to satisfy the existing tests.
- Before presenting changes, review your own diff multiple times. On each pass, check every item below. Keep reviewing until a pass finds nothing to fix.
  1. Does the change match the plan?
  2. Is the logic correct — edge cases, off-by-one errors, null/empty handling?
  3. Are there any regressions — does existing behavior stay intact?
  4. Is anything missing — incomplete implementation, forgotten call sites?
  5. Does it use the real production pipeline, not a simulation or mock?
  6. Were any tests modified? If so, that's a rule violation — revert and fix the production code instead.
  7. Were all rules in this file followed?
- Follow test-driven development: for every bug fix or new feature, first write a failing test that defines the expected behavior, then write production code to make it pass.
- Always create a plan before writing code. Read the relevant code, outline the approach, then implement.

# API surface rules — do not bypass these methods

These methods are the ONLY correct entry points for their respective operations. Do not reimplement their logic inline, call their internal helpers directly, or hardcode their output values.

- **Prefixed text**: Use `ChartSearchAiConstants.buildPrefixedText(resourceType, text)`. Never call `getEmbeddingPrefix()` directly (it is private) or hardcode prefix strings like `"Clinical observation: "`.
- **Query embedding input**: Use `LlmInferenceService.prepareEmbeddingInput(question, queryPrefix)`. Never manually chain `stripQueryStopwords` → `buildEmbeddingQuery` → embed.
- **Retrieval pipeline**: Use `LlmInferenceService.findSimilar()`. Never reimplement scoring/ranking/filtering by calling `cosineSimilarity`, `computeKeywordScore`, or `extractQueryTerms` in a loop.
- **Building embeddings**: Use `EmbeddingIndexer.buildEmbeddings()`. Never construct `ChartEmbedding` objects manually with `provider.embed()`.
- **Cosine similarity**: Use `ChartSearchAiConstants.cosineSimilarity()`. Never reimplement the formula.
- **Test datasets**: Use `TestDatasetHelper.FULL_PATIENT_DATASET`, `TestDatasetHelper.SECOND_PATIENT_DATASET`, `TestDatasetHelper.toSerializedRecords()`, `TestDatasetHelper.inferResourceType()`, `TestDatasetHelper.stripDatasetPrefixAndDate()`. Never duplicate these helpers in individual test files.
