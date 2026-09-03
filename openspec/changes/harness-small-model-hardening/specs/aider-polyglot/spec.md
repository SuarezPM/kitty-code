# Aider Polyglot Benchmark Loader

## Purpose
Load and run the real Aider Polyglot 225 benchmark with Exercism exercises and unit test grading, producing results comparable to the public leaderboard.

## Requirements

### R1: Exercise Loading
- GIVEN the `bench/official/polyglot-benchmark/` directory
- WHEN the polyglot bench command is invoked
- THEN exercises are loaded from `<lang>/exercises/practice/<exercise>/`
- AND prompts are extracted from `.meta/.docs/problem.md` or `.meta/description.md`
- AND test files are identified per language (pytest, go test, cargo test, etc.)

### R2: Unit Test Execution
- GIVEN a loaded exercise with prompt and test files
- WHEN the model generates a solution
- THEN the solution is written to the exercise directory
- AND the appropriate test runner is executed
- AND pass/fail is determined by test exit code

### R3: Language Support
- GIVEN exercises in Python, JavaScript, Rust, Go, C++, Java
- WHEN grading
- THEN the correct test runner is used:
  - Python: `pytest <test_file>`
  - JavaScript: `node <test_file>` or `npm test`
  - Rust: `cargo test` in exercise directory
  - Go: `go test` in exercise directory
  - C++: compile + run
  - Java: compile + run

### R4: Result Format
- GIVEN completed exercises
- WHEN results are generated
- THEN output matches the Aider Polyglot format:
  - `pass_rate_1`: percentage of exercises passing on first try
  - `pass_rate_2`: percentage passing after retry
  - Per-exercise details with latency, verdict, language

### R5: Comparable Metrics
- GIVEN a completed benchmark run
- WHEN comparing with little-coder (45.56%) or other harness results
- THEN the same 225 exercises and grading methodology are used

### R6: Smoke Test
- GIVEN `--limit N` flag
- WHEN running with N < 225
- THEN only N exercises are run (subset)

### R7: Unit Tests
- GIVEN the polyglot loader
- WHEN tests are run
- THEN exercise loading, prompt extraction, and test runner selection work correctly

## Non-Goals
- No Aider CLI integration (standalone harness)
- No Docker sandbox (runs locally)
- No parallel exercise execution (sequential for determinism)
