# Terminal-Bench 2.1 Runner

## Purpose
Add an integrated TB2 runner that delegates to the official Harbor CLI, producing results comparable to the leaderboard.

## Requirements

### R1: Harbor CLI Integration
- WHEN `kitty bench tb2` is run
- THEN it invokes `harbor run -d terminal-bench/terminal-bench-2 -a terminus-2 -m "ollama_chat/{model}" --ak api_base=http://127.0.0.1:11434`
- AND passes through model, timeout, and concurrency flags

### R2: Preflight Checks
- WHEN the TB2 runner starts
- THEN it checks: harbor CLI installed, Docker running, Ollama accessible
- AND reports clear error if any prerequisite is missing

### R3: Result Parsing
- WHEN harbor run completes
- THEN parse the output for resolution rate, per-task pass/fail
- AND format results consistently with other benchmarks

### R4: Homemade TB2 Deprecation
- WHEN `bench tb2` is documented
- THEN it clearly states "8 internal tasks, NOT comparable to official TB2.1 leaderboard"
- AND the new runner is available as `bench tb2-official` or similar

### R5: Unit Tests
- GIVEN the TB2 runner
- WHEN tests are run
- THEN preflight checks work, command construction is correct, result parsing works
