# Delta Spec: testing/coverage-gate

## ADDED Requirements

### Requirement: Coverage is measured and gated in CI

CI SHALL measure line coverage with `cargo llvm-cov` and fail when
`kitty-harness` line coverage falls below 70%.

#### Scenario: coverage gate enforces the threshold

- **GIVEN** the coverage job in CI
- **WHEN** harness coverage is below 70%
- **THEN** the job fails with the coverage summary in the log

### Requirement: Property-based tests cover the high-risk parsers

The workspace SHALL include `proptest` suites that guarantee:

- `ToolCallParser` never panics on arbitrary malformed tool-call text and
  always returns a parse result or error (never hangs, never UB).
- `HashlineEditor` edit operations never panic on arbitrary
  (file-content, hash-prefix, replacement) combinations and preserve the
  file-changed invariant.
- Guard block invariants hold over generated shell command strings: any
  generated command containing a blocked payload inside an interpreter
  wrapper is blocked.

#### Scenario: parser fuzzer never panics

- **GIVEN** proptest with arbitrary Unicode strings
- **WHEN** `ToolCallParser::feed`/parse variants are exercised
- **THEN** no panic occurs in 1,000+ generated cases per run

#### Scenario: hashline fuzzer never panics

- **GIVEN** proptest with arbitrary file contents and edit arguments
- **WHEN** hashline edit paths are exercised
- **THEN** no panic occurs and `FileChanged` is returned whenever the
  content hash changed

#### Scenario: guard fuzzer holds the invariant

- **GIVEN** proptest generating interpreter-wrapped payloads
  (`sh -c`, `bash -c`, `python3 -c`, `awk`) around known-blocked bodies
- **WHEN** `Guard::check_command` evaluates them
- **THEN** each is blocked (deny verdict)

### Requirement: OllamaClient has HTTP contract tests

`OllamaClient` SHALL have wiremock-based contract tests covering: successful
chat with `truncate:false` payload, HTTP 400 context-overflow mapping to the
overflow error, request timeout mapping, and token-count endpoint fallback.

#### Scenario: overflow maps to typed error

- **GIVEN** a wiremock server returning HTTP 400 with the overflow body
- **WHEN** `OllamaClient::chat` is called
- **THEN** the returned error is the context-overflow error variant

#### Scenario: timeout maps to typed error

- **GIVEN** a wiremock server that delays beyond the client timeout
- **WHEN** `chat` is called
- **THEN** the returned error is the timeout/transport error variant
