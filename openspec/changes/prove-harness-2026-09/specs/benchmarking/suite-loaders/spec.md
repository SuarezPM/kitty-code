# Delta Spec: benchmarking/suite-loaders

## ADDED Requirements

### Requirement: TB2 runs on the pinned 2.1 dataset with an oracle smoke

TB2 invocations SHALL target the `terminal-bench-2-1` dataset (registry
id or verified local pin — never the 2.0 dataset by default), support
task-count selection instead of ignoring the limit flag, and pass an
oracle `-k5` smoke before any model run counts as evidence.

#### Scenario: oracle smoke gates the model run

- **GIVEN** a TB2.1 campaign about to start
- **WHEN** the oracle smoke has not passed `-k5`
- **THEN** no model result from that campaign is publishable

### Requirement: GPQA and ClawEval unblock on artifacts, not on code changes

Both suites keep their offline-first gated preflight; unblocking them
SHALL mean materializing the missing artifacts (GPQA `diamond.jsonl` +
manifest sidecar validated to the kitty cache path; ClawEval CLI +
`OPENROUTER_API_KEY` + `data/fixtures` + verified sandbox image) so
`is_ready`/preflight turns green with zero harness changes.

#### Scenario: missing artifact stays blocked, loudly

- **GIVEN** any required artifact absent
- **WHEN** the suite is invoked
- **THEN** it writes `blocked` (or the equivalent not-ready verdict)
  without invoking the model, naming the missing piece

### Requirement: Blocked-harness labels survive until official grading

Results from the GPQA/ClawEval approximation paths SHALL keep the
`local_approximation_not_official` label until they are produced
through the official harness and grader.

#### Scenario: approximation never upgrades itself

- **GIVEN** a GPQA or ClawEval result from the local path
- **WHEN** it is published
- **THEN** it carries the approximation label and is not compared to
  official GPQA-Diamond or ClawEval numbers
