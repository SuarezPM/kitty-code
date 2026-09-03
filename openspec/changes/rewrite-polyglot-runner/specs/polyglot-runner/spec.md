# Polyglot Runner Rewrite

## Purpose
Rewrite the Aider Polyglot benchmark runner to follow the official pattern: config.json parsing, test file hiding, retry with error feedback, and proper timeouts.

## Requirements

### R1: Config.json Parsing
- GIVEN an exercise directory at `bench/official/polyglot-benchmark/<lang>/exercises/practice/<slug>/`
- WHEN the runner loads the exercise
- THEN it reads `.meta/config.json` to obtain `files.solution`, `files.test`, `files.example`

### R2: Test File Hiding
- GIVEN solution_files and test_files from config.json
- WHEN the exercise is copied to tempdir
- THEN ONLY solution_files are copied (plus `.docs/` for instructions)
- AND test_files are NOT present during the agent's turn

### R3: Prompt Construction
- GIVEN the exercise's `.docs/` directory
- WHEN the prompt is built
- THEN it concatenates introduction.md + instructions.md + instructions.append.md + suffix listing solution files

### R4: Agent Execution
- GIVEN the constructed prompt and tempdir with solution stubs
- WHEN the agent runs
- THEN timeout is 900s and workspace is set to tempdir

### R5: Test Materialization
- GIVEN the agent has finished
- WHEN tests are about to run
- THEN test_files are copied from original into tempdir

### R6: Retry Loop
- GIVEN test execution failed
- WHEN retries remain (2 tries)
- THEN test output is appended and retry prompt sent

### R7: Test Commands
- Python: `python3 -m pytest -x -q`
- Rust: `cargo test -- --include-ignored`
- Go: `go test ./...`
- JavaScript: `npm test --silent`
- C++: `make test` or cmake+ctest
- Java: `./gradlew test`

### R8: Timeouts
- Test execution: 180s subprocess timeout
- Agent turn: 900s

### R9: Result Format
- Aider-compatible: pass_rate_1, pass_rate_2, by_language
