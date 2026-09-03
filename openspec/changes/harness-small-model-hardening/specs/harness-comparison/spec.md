# Harness Comparison Table

## Purpose
Compare baseline vs candidate harness configurations with correctness lift, wins/ties/losses metrics.

## Requirements

### R1: Paired Comparison
- GIVEN results from two harness runs (baseline and candidate) on the same exercises
- WHEN `compare_harnesses()` is called
- THEN a comparison table is generated with:
  - Correctness lift (candidate pass rate - baseline pass rate in percentage points)
  - Baseline wins, candidate wins, ties count
  - Per-exercise breakdown

### R2: Token/Latency/Cost Metrics
- GIVEN paired results
- WHEN comparison is generated
- THEN token usage delta, latency delta, and estimated cost delta are included

### R3: Output Format
- GIVEN a completed comparison
- WHEN output is generated
- THEN it matches the format:
  ```
  Baseline  <name>
  Candidate  <name> (N/M pairs)
  Pass rate  +X.X pp (candidate Y%, baseline Z%)
     Tokens  +X.X (candidate N, baseline M)
    Latency  -X.Xms (candidate Nms, baseline Mms)
  Est. cost  -$X.XXXX (candidate $Y, baseline $Z)
  ```

### R4: Artifact Persistence
- GIVEN a comparison result
- WHEN saved
- THEN a JSON file with full per-exercise data is written to the results directory

### R5: Unit Tests
- GIVEN the comparison module
- WHEN tests are run
- THEN comparison calculations are correct, format is valid, artifacts are persisted

## Non-Goals
- No statistical significance testing (simple delta comparison)
- No multi-run aggregation (single-run comparison)
