# Delta Spec: benchmarking/official-methodology

## ADDED Requirements

### Requirement: Every suite has a pinned methodology card

Each benchmarked suite SHALL record its exact replication protocol:
benchmark version/commit, harness version, model id + quant + serving
stack version, sampling params (temperature, top_p, num_ctx, thinking
budget, profile), submission format, grading procedure, environment,
attempts, and reporting statistics.

#### Scenario: polyglot-225 card matches little-coder parity

- **GIVEN** a polyglot-225 run with ornith 1.5 9B
- **WHEN** its methodology is inspected
- **THEN** it states full 225 tasks, clean workspace copy per exercise,
  native runner exit-code grading, ≤20 turns, up to 2 attempts with
  retry on failing output, `pass@2` primary plus first-attempt rate,
  2 runs in different order with mean±SD, temp 0.3, ctx 32768,
  thinking 2048, and a test-edit audit with per-exercise verdicts

#### Scenario: TB2.1 card matches ornith parity

- **GIVEN** a TB2.1 run
- **WHEN** its methodology is inspected
- **THEN** it states dataset `terminal-bench-2-1`, pinned Harbor
  version, agent `terminus-2`, temperature 1.0 + top_p 1.0, `-k 5`
  trials with mean±SD, a prior oracle `-k5` sanity, kept trajectories,
  and disclosed env/timeout

#### Scenario: SWE-bench card uses the official eval

- **GIVEN** a SWE-bench-verified run
- **WHEN** its methodology is inspected
- **THEN** it states pinned OpenHands version, official `swebench`
  eval over `FAIL_TO_PASS` + `PASS_TO_PASS` in official Docker images,
  all 500 instances, disabled network, stripped git history, published
  `preds.json` + `results.json` + per-instance logs, a fresh `run_id`,
  and disclosed temp/iterations

### Requirement: Heavy runs are pre-registered before launch

No heavy bench run SHALL start without a manifest that predates its
results and records timestamp, HEAD sha of a clean tree, suite,
N/trials, dataset fingerprint, full params, and the valid-run gate.

#### Scenario: unregistered run is void

- **GIVEN** results without a prior manifest (or a dirty-tree HEAD)
- **WHEN** publication is proposed
- **THEN** the run is discarded as evidence, never published as a rate

#### Scenario: mid-campaign swap voids the run

- **GIVEN** a running campaign whose daemon or model fingerprint at
  end differs from the manifest start fingerprint
- **WHEN** results are evaluated
- **THEN** the run is void, never patched — manifests SHALL record
  daemon/model fingerprints at start and end

### Requirement: Grading methods never share a column

Published tables SHALL separate `string-match-homemade`,
`unit-test-local-harness-reproduction`, and official rates, with the
grading method printed on every metric.

#### Scenario: homemade rate stays in its own table

- **GIVEN** the B1 98.2% string-match result and a polyglot-225
  unit-test result
- **WHEN** they are published
- **THEN** they appear in different tables and neither is compared to
  an official leaderboard number or to little-coder 45.56% except the
  unit-test one as same-class native

### Requirement: Replication baselines are pinned (2026-09-09)

Every pre-registration SHALL record the exact pins below. Any delta
SHALL be disclosed and voids all comparability claims unless
re-baselined in a new pre-registration. "Newer" is a delta, not an
exemption.

#### Scenario: polyglot-225 pins

- **GIVEN** a polyglot-225 pre-registration
- **WHEN** its pins are inspected
- **THEN** it records dataset `Aider-AI/polyglot-benchmark@7e0611e`
  (frozen 2024-12-22, no drift vs little-coder era), reference
  `little-coder v0.0.2 (1d62bde)`, the Ollama version served; any
  delta vs 0.20.5 limits the max claim to same-class
  (version-drifted), never parity — parity requires an isolated
  0.20.5 run, caps 12+8 turns, timeouts
  py/go 60, js 90, rust 180, cpp 240, java 300 (seconds), retry
  truncation 2000 chars, 2 attempts max

#### Scenario: TB2.1 pins

- **GIVEN** a TB2.1 pre-registration
- **WHEN** its pins are inspected
- **THEN** it records dataset `terminal-bench-2-1` version 2.1.0
  (source commit `7131e43`), Harbor `v0.22.0`, agent `terminus-2`,
  temp 1.0 / top_p 1.0 passed explicitly (Harbor ≥0.6.5 sends no
  default), the served model id, and `-k 5`. A Fase 5.2
  pre-registration on Harbor 0.21.0 is void — the Fase 4 upgrade is a
  prerequisite gate (local 0.21.0 on 2026-09-09, non-normative).

#### Scenario: SWE-bench pins

- **GIVEN** a SWE-bench-verified pre-registration
- **WHEN** its pins are inspected
- **THEN** it records `swebench==5.0.2`, the OpenHands revision
  served (`d839f369` SHALL be labeled community-known-good, never
  ornith-parity — the ornith-parity label is forbidden unless Ornith
  publishes their revision), `max_iterations`, the served temp, a
  fresh `run_id` (UUIDv4, never reused), the network-off/history-
  stripped attestation, and install fingerprints (packages, images,
  cache sha). Note (non-normative): on 2026-09-09 all three local
  installs were absent.

#### Scenario: model pins

- **GIVEN** any ornith 1.5 9B run
- **WHEN** its pins are inspected
- **THEN** it records the model digest (`e5df7dcdd8a2`, Q4_K_M,
  9.0B), the served `num_ctx` (not the 262144 model max), the sha256
  of the actually-served `chat_template.jinja` bytes plus the
  byte-compare verdict vs HEAD `489cb97` (UNVERIFIED forbids equality
  claims, not runs), and `ollama show` + Modelfile sha proving empty
  params/system/license at serve time
