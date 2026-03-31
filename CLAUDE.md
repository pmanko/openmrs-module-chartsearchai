# Rules

- When a test fails, fix the pipeline/production code — never weaken assertions, revert test data, or change expected values to match wrong behavior.
- Prefer root-cause fixes over incremental symptom patches. Diagnose *why* something is broken before proposing a fix. Start with the best solution, not the quickest.
- Do not go in circles analyzing a problem without fixing it.
- Tests must call the actual production pipeline — no simulations, mocks, or reimplementations of pipeline logic in test code.
- Tests are the specification. Modifying tests to make them pass is changing the spec — only fix the production code to satisfy the existing tests.
- Before presenting changes, review your own diff multiple times. On each pass check: correctness, completeness, that no rules in this file were violated, and that no issues from previous passes were missed. Keep reviewing until a pass finds nothing to fix.
- Follow test-driven development: for every bug fix or new feature, first write a failing test that defines the expected behavior, then write production code to make it pass.
- Always create a plan before writing code. Read the relevant code, outline the approach, then implement.
