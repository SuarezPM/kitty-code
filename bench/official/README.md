# bench/official — Official benchmark datasets

This directory holds **official** benchmark datasets cloned from upstream sources,
used as ground truth for comparison against the homemade benchmarks in `benches/`.

## Contents

| Directory | Source | Tasks | Notes |
|---|---|---|---|
| `terminal-bench-2/` | https://github.com/harbor-framework/terminal-bench-2.git (shallow, depth 1) | 89 | Official Terminal-Bench 2.0 Harbor dataset. Each task has `environment/Dockerfile`, `instruction.md`, `solution/`, `tests/`, `task.toml`. |
| `aider-harness/` | https://github.com/Aider-AI/aider.git (shallow, depth 1) | — | Official aider repo; harness entrypoint at `aider-harness/benchmark/benchmark.py`. |
| `polyglot-benchmark/` | https://github.com/Aider-AI/polyglot-benchmark.git (shallow, depth 1) | 225 | Exercism exercises: cpp(26) go(39) java(47) javascript(49) python(34) rust(30) under `<lang>/exercises/practice/`. |
| `aider-harbor/` | `harbor download aider-polyglot@1.0` (legacy registry) | 225 | Harbor tasks `polyglot_<lang>_<slug>/` with `instruction.md`, `solution/`, `tests/`, `task.toml`, `environment/`. |

## TB2 official vs homemade comparison

- **Official** (`bench/official/terminal-bench-2/`): 89 real-world tasks from the
  Harbor Terminal-Bench 2.0 dataset. Each task ships a full Docker environment
  (`environment/Dockerfile`), a natural-language `instruction.md`, a reference
  `solution/`, and `tests/` for grading. Run via the `harbor` CLI
  (`harbor run -p <dir> -a <agent> --k <n>`).
- **Homemade** (`benches/exercises_tb2.json`): lightweight synthetic TB2-style
  exercises (file ops, simple shell tasks) with inline `prompt`/`setup`/
  `check_type`/`check_path` grading — no Docker, run directly by
  `benches/run_tb2_89.py`. Used for fast iteration and CI; NOT a substitute for
  the official dataset.

## ADR: TB2 oracle smoke test deferred (2026-08-19)

- **Status:** Deferred — not executed.
- **Reason:** `harbor` CLI is not installed on this machine and Docker is not
  available (`docker info` fails). The 5-task oracle smoke
  (`harbor run -p bench/official/terminal-bench-2 -a oracle --k 5`) cannot run
  without a Docker daemon.
- **When to revisit:** after installing `harbor` and starting the Docker daemon.
  Expected result if env is healthy: 5/5 pass with the oracle agent.
- **Not done (by design):** full 89-task run was never attempted (smoke only).

## ADR: Aider polyglot infra cloned, run deferred (2026-08-19)

- **Status:** Infra cloned ✅ — official run deferred.
- **What was done:** cloned `aider-harness` (benchmark/benchmark.py present),
  cloned `polyglot-benchmark` (225 Exercism exercises verified), and downloaded
  the Harbor dataset `aider-polyglot@1.0` (225 tasks) into `aider-harbor/`.
- **Reason for deferral:** Docker is NOT installed on this machine (no `docker`,
  no `podman`). The official aider polyglot harness runs each exercise inside a
  Docker container, so the F6.2 run (Plan T15 SC-DOCKER-POLYGLOT) requires
  Docker first (e.g. `sudo pacman -S docker` on CachyOS + start the daemon).
- **Harbor gotcha:** `aider/aider-polyglot@1.0` (with `org/` prefix) is NOT in
  the new git registry — it lives in the **legacy registry**. Use
  `harbor download aider-polyglot@1.0` (no prefix). `--legacy` is only a flag on
  `harbor dataset list`, not on `download`. Harbor installed via
  `uv tool install harbor` (binaries: `harbor`, `hb`, `hr`).
- **Not done (by design):** no Docker build/run attempted; no `benches/*.json`
  modified; nothing committed (Wave4 parallel lane).