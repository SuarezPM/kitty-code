# ClawEval Fix

## Purpose
Fix the critical bug where the model flag is ignored and clean up dead code.

## Requirements

### R1: Pass --model flag correctly
- WHEN `kitty bench claweval --model ornith:9b` is run
- THEN `claw-eval batch --model ornith:9b` is invoked (NOT `openrouter/default`)
- AND the model from the CLI flag is passed through to the subprocess

### R2: Remove dead code
- WHEN reviewing `claweval.rs`
- THEN `run_claweval_batch` function is either integrated into `run_claweval_suite` or removed
- AND no dead code paths remain

### R3: Integrate run_claweval_batch
- WHEN the ClawEval suite runs
- THEN it uses the functions from `claweval.rs` (not duplicated logic in main.rs)

### R4: Unit Tests
- GIVEN the ClawEval fixes
- WHEN tests are run
- THEN model flag is correctly passed, no dead code, integration works
