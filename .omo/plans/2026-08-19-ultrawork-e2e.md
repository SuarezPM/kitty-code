# Ultrawork E2E — Kitty Code P1.1 + Docs Commit + Floor 225 + OpenSpec Archive + F6
**Fecha:** 2026-08-19 22:50 -03 | **Orquestador:** Sisyphus (muse-spark-1.2) | **Plan Agent:** Prometheus (ses_fe32a6165ffeoN8QeEBIlHNSFV) | **Notepad:** /tmp/ulw-20260819-223528.hHegF6.md
**Estado base validado:** 239 tests, clippy clean, build 34s, 5 commits 2026-08-19, Gates 1-4 aprobados, 6 OpenSpec validate 6/6, F0-F5+P0/P1 done, P1.1 debt 30min pendiente
**Modo:** ultrawork end-to-end, TDD RED→GREEN+SURFACE, 12 commits atómicos, sin remote GitHub para CI/push pero sí todo local + benchmarks Docker locales

---

## 1. Context — 5 Frentes SACRED (no scope reduction)

| # | Frente | Scope exacto | Archivos clave | Blockers |
|---|--------|--------------|----------------|----------|
| 1 | **P1.1 deuda 30min** | mcp persist Mutex<ToolRegistry> + contract integración mock LLM + baseline CARGO_MANIFEST_DIR + is_small_model→profile + enable_grammar dedup doc + grammar spike defer doc | `crates/kitty-cli/src/mcp.rs` (289L), `crates/kitty-harness/src/contract.rs` (377L), `crates/kitty-harness/src/agent.rs` (1098L), `crates/kitty-cli/src/main.rs` (729L), `crates/kitty-runtime/src/grammar.rs` (387L), `crates/kitty-harness/src/model_profile.rs` (153L) | Ninguno — todo local |
| 2 | **Commit docs untracked** | `git add docs/COMPETITIVE-INTEL.md docs/STRATEGY.md openspec/` + commit | `docs/COMPETITIVE-INTEL.md` (15K), `docs/STRATEGY.md` (7.9K), `openspec/changes/*` (6 changes) | Ninguno |
| 3 | **Floor 225 full smollm2 10min** | `kitty --model smollm2:1.7b bench polyglot --limit 225 --exercises benches/exercises_polyglot.json --out benches/bench_225_smollm2.json` | `benches/exercises_polyglot.json` (1577L, 225 items), `benches/baseline.json` | Ollama smollm2:1.7b 1.8GB must be loaded |
| 4 | **Archivar OpenSpec 6 cambios** | `openspec archive <name> -y` ×6 → `openspec/changes/archive/2026-08-19-*` + `openspec/specs/*` 6 capabilities | `openspec/changes/*/proposal.md`, `openspec/changes/*/specs/*/spec.md`, `openspec/config.yaml` | `isPlanningComplete:false` (design/tasks missing) → Option B backfill |
| 5 | **F6 semanas** | F6.1 A/B thinking 1024/2048/4096 + ejercicios + knowledge cards; F6.2 polyglot/TB2 Docker oficial; F6.3 OpenSSF ≥7; F6.4 llama.cpp GBNF real | `crates/kitty-harness/src/thinking_budget.rs`, `benches/*.json`, `knowledge/`, `Cargo.toml`, `crates/kitty-runtime/src/client.rs` | F6.2 Docker install, F6.3 CI requires push, F6.4 decision bind vs spike |

**Exclusiones explícitas (no scope creep):** qwen3 90min defer, TUI interactivo, cloud fallback, sandbox Landlock, MemScenes — quedan en backlog F6.5

---

## 2. Sizing — 20 Surfaces / 32 Files / 23 Tasks

**Surfaces:** 1) mcp persist 2) contract guard 3) baseline path 4) is_small_model 5) enable_grammar 6) grammar defer 7) docs commit 8) openspec archive 9) floor225 10) bench ab 11) exercises 12) knowledge cards 13) knowledge default 14) polyglot Docker 15) TB2 Docker 16) SECURITY.md 17) deny.toml 18) CODEOWNERS 19) rust-toolchain 20) llama.cpp adapter

**Files touched:** 25 .rs + 7 md/toml/yaml/json (mcp.rs, contract.rs, agent.rs, main.rs, grammar.rs, model_profile.rs, tools.rs, thinking_budget.rs, context.rs, knowledge.rs, client.rs, lib.rs×3, codemap.md, docs/*, benches/*, openspec/*, SECURITY.md, CONTRIBUTING.md, CODEOWNERS, deny.toml, rust-toolchain.toml, .github/workflows/*)

| Task | File:Line verified | Effort | Type |
|------|-------------------|--------|------|
| T1 mcp persist | mcp.rs:41 `ToolRegistry::new()` | 20min | code |
| T2 contract integration | contract.rs:199 + agent.rs:592 | 15min | code+test |
| T3 baseline path | main.rs:301 hardcoded | 10min | code+test |
| T4 is_small_model | main.rs:193 heuristic | 15min | code+test |
| T5 enable_grammar dedup | agent.rs:88/189 | 10min | code+doc |
| T6 grammar defer doc | grammar.rs:1 spike | 10min | doc |
| T7 docs commit | git status untracked | 5min | git |
| T8 openspec backfill design/tasks | openspec/changes/*/proposal.md | 20min | doc |
| T9 openspec archive ×6 | openspec archive -y | 15min | cli |
| T10 floor 225 smollm2 | benches/exercises_polyglot.json | 12min* | bench |
| T11 bench ab subcommand | main.rs:81 BenchCommand | 45min | code+test |
| T12 exercises add | benches/exercises_polyglot.json 0 matches | 20min | data |
| T13 knowledge cards Python | knowledge/ 2 cards | 15min | data |
| T14 knowledge default | main.rs bench no --knowledge | 10min | code |
| T15 polyglot Docker harness | bench/official/ | 60min | infra |
| T16 TB2 Docker harness | harbor | 60min | infra |
| T17 SECURITY.md | missing | 15min | doc |
| T18 deny.toml + audit | missing | 15min | config |
| T19 CODEOWNERS + workflows | missing | 20min | config |
| T20 rust-toolchain + CONTRIBUTING | missing | 15min | config |
| T21 llama.cpp adapter decision | grammar.rs spike | 30min | decision+code |
| T22 CNAS stale fix | CNAS.md ~40 | 5min | doc |
| T23 harness tests + codemap | codemap.md 239 tests | 10min | doc |

*10min wall + 2min analysis

---

## 3. Scenario Contract — 9 Escenarios BINDING

Cada escenario: **binary pass + real surface + test file+id**. No DONE sin ambos artefactos.

| ID | Clase | Escenario | Pass binary | Real surface | Test file+id |
|----|-------|-----------|-------------|--------------|--------------|
| **SC-MCP-PERSIST** | Happy | `handle_request` secuencial Read→Edit mismo path funciona | `cargo test mcp_persist_read_then_edit` PASS | `echo '{"method":"tools/call","params":{"name":"Read","arguments":{"path":"/tmp/kitty-test-xxx"}}}' \| kitty mcp` luego Edit mismo path → success:true | `crates/kitty-cli/src/mcp.rs` `test_mcp_persist_read_then_edit_sequential` |
| **SC-CONTRACT-NUDGE** | Happy | Agent con `set_contract(pending)` + LLM devuelve "All done" bloquea y nudge | `cargo test contract_integration_nudge` PASS + `messages.last()` contains `[CONTRACT-GUARD]` | `agent.run("All done")` con mock LLM → stdout contiene `[CONTRACT-GUARD] Blockers:` | `crates/kitty-harness/src/agent.rs` `test_contract_integration_nudge_injected` |
| **SC-BASELINE-CWD** | Edge | Baseline resuelve desde subdir vía CARGO_MANIFEST_DIR | `cargo test -p kitty-cli --manifest-path crates/kitty-cli/Cargo.toml` desde `crates/` PASS | `cd crates && cargo test baseline_path_resolves` + `benches/baseline.json` existe | `crates/kitty-cli/src/main.rs` `test_baseline_path_cargo_manifest` |
| **SC-PROFILE-HEURISTIC** | Edge | `is_small_model` reemplazado por `ModelProfile::from_cli("tiny")` | `cargo test profile_is_small` PASS para `:4b/:2b/1.7b` | `kitty bench --model smollm2:1.7b --profile tiny` → auto --no-memory | `crates/kitty-cli/src/main.rs` `test_is_small_uses_profile` |
| **SC-GRAMMAR-DEDUP** | Regression | `enable_grammar` doble-set documentado, no regresión | `cargo test grammar_double_set` PASS | `AgentConfig {profile: tiny, enable_grammar:false}` → after new_inner enable_grammar true | `crates/kitty-harness/src/agent.rs` `test_tiny_forces_grammar_dedup` |
| **SC-BENCH-AB** | Happy | `kitty bench ab --thinking 1024,2048,4096 --limit 5` corre 3 budgets interleaved | `cargo test bench_ab_parses` PASS + `bench_ab.json` exists con per_budget + pairwise | `kitty bench ab --thinking 1024,2048 --limit 2 --out /tmp/ab.json` → JSON con budgets array | `crates/kitty-cli/src/main.rs` `test_bench_ab_interleave` |
| **SC-DOCKER-POLYGLOT** | Happy | Polyglot oficial harness clonado, Docker build, oracle smoke 5 tests | `harbor run -d polyglot-benchmark -a oracle --k 5` → 5/5 PASS | `ls bench/official/aider-harness/Dockerfile` + `bench/official/polyglot-benchmark/` exists | `bench/official/README.md` existence check (infra, no unit test) |
| **SC-OPENSSF-7** | Happy | SECURITY.md + deny.toml + CODEOWNERS + rust-toolchain + workflows → score ≥7 local | `cargo deny check` PASS + `ls SECURITY.md CODEOWNERS` → exists | `cat SECURITY.md \| grep -i "report.*vuln"` + `cargo deny check advisories` 0 vulns | `deny.toml` + `.github/workflows/ci.yml` `pinned SHA` |
| **SC-GBNF-VALIDITY** | Happy | Grammar spike documentado como defer + decision ADR, o adapter real >95% validity | `cargo test gbnf_spike_documented` PASS | `grep -r "SPIKE defer" crates/kitty-runtime/src/grammar.rs` → found | `docs/STRATEGY.md` ADR entry |

**Regresión adyacente obligatoria para todos:** `cargo test --workspace` 239→249+ PASS, `cargo clippy --all-targets -- -D warnings` clean, `cargo build --release` 34s, `openspec validate --all` 6/6

---

## 4. Task Dependency Graph

```
T1(mcp) ─┐
T2(contract) ─┤
T3(baseline) ─┤ → T7(docs commit) ─→ T8(backfill) ─→ T9(archive) ─→ T23(codemap)
T4(profile) ─┤              ↑
T5(grammar dedup) ─┘        │
T6(grammar doc) ────────────┘

T10(floor225) independent (needs ollama)
T11(bench ab) → T12(exercises) no dep, T13(cards) → T14(knowledge default) depends T13
T15(polyglot) independent infra
T16(TB2) independent infra
T17-20(OpenSSF local) independent, parallel
T21(llama adapter) depends T5+T6
T22(CNAS fix) depends T7
```

---

## 5. Parallel Execution Graph — 6 Waves

**Wave 1 (7 parallel, 0 deps):** T1 + T2 + T3 + T4 + T13 + T17 + T19
**Wave 2 (2 parallel, after W1):** T5 + T6 (need T1-T4 context) + T20 (parallel)
**Wave 3 (4 parallel):** T7 (needs T1-T6) → then T8 + T11 + T22 parallel after T7
**Wave 4 (5 parallel):** T9 (needs T8) + T14 (needs T13) + T15 + T16 + T21 (needs T5/T6)
**Wave 5 (3 parallel):** T10 (floor, needs bench stable) + T12 (needs T11) + T18
**Wave 6 (3):** T23 (needs all) + final verification

**Estimado wall:** 6.5h paralelo vs 14h secuencial. P1.1 (Wave1-2) = 45min.

---

## 6. Tasks — Detalle por T con Delegation (category+skills evaluation)

### T1: mcp.rs persist — `crates/kitty-cli/src/mcp.rs:41`
- **Description:** Reemplazar `ToolRegistry::new()` por `Arc<Mutex<ToolRegistry>>` persistente en McpServer
- **Delegation:** `category="deep"` (goal-oriented, harness critical), `load_skills=["rust-router","m06-error-handling","m07-concurrency"]` — rust-router para ownership/borrow de Arc<Mutex>, m07 para concurrency, m06 para error handling. Omit frontend/cli etc.
- **Depends:** Ninguno
- **Acceptance:** `test_mcp_persist_read_then_edit_sequential` RED→GREEN + `echo tools/call Read | kitty mcp` → `success:true` then Edit same path success:true

### T2: contract integration — `crates/kitty-harness/src/agent.rs:929` + `contract.rs:199`
- **Description:** Test integración Agent::run mock LLM "All done" con contract pending → nudge injected
- **Delegation:** `category="deep"`, `load_skills=["rust-router","m06-error-handling","test-driven-development"]`
- **Depends:** Ninguno (puede en paralelo con T1)
- **Acceptance:** MockClient queue ["All done"] + set_contract(pending) → messages.last() contiene `[CONTRACT-GUARD]` + `a01` blocker

### T3: baseline path — `crates/kitty-cli/src/main.rs:301`
- **Description:** Reemplazar `"benches/baseline.json"` por `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benches/baseline.json")`
- **Delegation:** `category="quick"`, `load_skills=["rust-router","m12-lifecycle"]`
- **Depends:** Ninguno
- **Acceptance:** `cd crates && cargo test -p kitty-cli` PASS + `cat benches/baseline.json` existe

### T4: is_small_model → profile — `crates/kitty-cli/src/main.rs:193`
- **Description:** Reemplazar heuristic contains chain por `ModelProfile::from_cli(&cli.profile).name=="tiny"` o helper `is_tiny_model(&model,&profile)`
- **Delegation:** `category="quick"`, `load_skills=["rust-router","m05-type-driven"]`
- **Depends:** Ninguno
- **Acceptance:** `test_is_small_uses_profile` para `qwen3:4b`+`tiny` → true, `ornith:9b`+`medium` → false

### T5: enable_grammar dedup doc — `crates/kitty-harness/src/agent.rs:88/189`
- **Description:** Documentar doble-set `with_profile` + `new_inner` ambos `tiny→enable_grammar=true` con comentario + test
- **Delegation:** `category="quick"`, `load_skills=["rust-router"]`
- **Depends:** T1-T4
- **Acceptance:** `grep "dedup.*with_profile.*new_inner.*tiny→grammar" agent.rs` → found + test pass

### T6: grammar spike defer doc — `crates/kitty-runtime/src/grammar.rs:1`
- **Description:** Documentar spike como defer explícito con ADR + validar sin llama.cpp bind
- **Delegation:** `category="quick"`, `load_skills=["rust-learner"]`
- **Depends:** T5
- **Acceptance:** `grep "SPIKE defer.*no llama.cpp bind" grammar.rs` → found + docs/STRATEGY.md ADR

### T7: docs commit — `git status untracked`
- **Description:** `git add docs/COMPETITIVE-INTEL.md docs/STRATEGY.md openspec/ && git commit -m "docs: intel+strategy+specs 6 aprobados"`
- **Delegation:** `category="quick"`, `load_skills=["git-master"]`
- **Depends:** T1-T6 (para incluir fixes en commit si se desea atomic)
- **Acceptance:** `git status --porcelain | grep openspec` → vacío, `git log -1 --stat` shows 3 files

### T8: openspec backfill design/tasks — `openspec/changes/*/proposal.md`
- **Description:** Generar `design.md` + `tasks.md` por change vía `openspec instructions design --change <name> --json` luego write minimal design/tasks para isPlanningComplete=true
- **Delegation:** `category="unspecified-high"`, `load_skills=["openspec-propose"]`
- **Depends:** T7
- **Acceptance:** `openspec status --change bench-reproducibility --json | jq .isPlanningComplete` → true ×6

### T9: openspec archive ×6 — `openspec archive <name> -y`
- **Description:** `for c in bench-reproducibility grammar-enforcement mcp-server model-profiles safety-hardening small-model-tool-loop; do openspec archive $c -y --json; done`
- **Delegation:** `category="quick"`, `load_skills=["openspec-archive-change"]`
- **Depends:** T8
- **Acceptance:** `ls openspec/changes/archive/2026-08-19-*` 6 dirs + `ls openspec/specs/*/spec.md` 6 files + `openspec validate --all` 6 passed

### T10: floor 225 smollm2 — `benches/exercises_polyglot.json`
- **Description:** `kitty --model smollm2:1.7b bench polyglot --limit 225 --exercises benches/exercises_polyglot.json --out benches/bench_225_smollm2.json` (10min, timeout 30s, max_iter 5, gate 50)
- **Delegation:** `category="deep"`, `load_skills=["rust-learner"]`
- **Depends:** Ninguno (pero bench harness debe estar stable)
- **Acceptance:** `cat benches/bench_225_smollm2.json | jq .rate_pct` <50% + `.total==225` + JSON exists

### T11: bench ab subcommand — `crates/kitty-cli/src/main.rs:70`
- **Description:** Añadir `Commands::BenchAb { thinking: String, limit: usize, reps: usize, suite, out, interleave: bool }` + loop interleaved StdRng(42) + BenchAbResult JSON con per_budget Wilson CI + pairwise Fisher
- **Delegation:** `category="deep"`, `load_skills=["rust-router","domain-cli","golang-testing"]` (para pattern, no golang code)
- **Depends:** Ninguno
- **Acceptance:** `kitty bench ab --thinking 1024,2048 --limit 2 --out /tmp/ab.json` → JSON con budgets [1024,2048] + per_budget 2 entries

### T12: exercises add — `benches/exercises_polyglot.json`
- **Description:** Añadir 3 ejercicios Python: list_comprehension, decorator_with_args, async_await (con check_str + check_file)
- **Delegation:** `category="quick"`, `load_skills=["rust-learner"]`
- **Depends:** T11
- **Acceptance:** `jq '.[] | select(.id=="py-list-comp")' benches/exercises_polyglot.json` → found ×3, total 228

### T13: knowledge cards Python — `knowledge/`
- **Description:** Crear `knowledge/python-list-comp.md`, `python-decorators.md`, `python-async.md` (scoring word1.0+bigram2.0)
- **Delegation:** `category="quick"`, `load_skills=["rust-learner"]`
- **Depends:** Ninguno
- **Acceptance:** `ls knowledge/*.md` 5 files + `cargo test knowledge_loads_5_cards` PASS

### T14: knowledge default in bench — `crates/kitty-cli/src/main.rs:47`
- **Description:** Activar `--knowledge` por defecto en bench tras A/B, o flag `--knowledge` default true para bench ab
- **Delegation:** `category="quick"`, `load_skills=["rust-router"]`
- **Depends:** T13
- **Acceptance:** `kitty bench --help | grep knowledge` shows default true

### T15: polyglot Docker harness — `bench/official/`
- **Description:** `mkdir -p bench/official && git clone Aider-AI/aider bench/official/aider-harness && git clone polyglot-benchmark bench/official/polyglot-benchmark && harbor download aider/aider-polyglot@1.0`
- **Delegation:** `category="deep"`, `load_skills=["domain-cloud-native","clonedeps"]`
- **Depends:** Ninguno
- **Acceptance:** `ls bench/official/aider-harness/benchmark/benchmark.py` + `ls bench/official/polyglot-benchmark/` → exists

### T16: TB2 Docker harness — `harbor`
- **Description:** `harbor download terminal-bench/terminal-bench-2@latest --output-dir bench/official/tb2-local && harbor run -p bench/official/tb2-local -a oracle --k 5` smoke
- **Delegation:** `category="deep"`, `load_skills=["domain-cloud-native"]`
- **Depends:** Ninguno
- **Acceptance:** `harbor run -a oracle --k 5` → 5/5 PASS (or Docker not available → defer with ADR)

### T17: SECURITY.md — missing
- **Description:** Crear `SECURITY.md` con email/private advisory, vuln+disclos+days, 14d response, credit
- **Delegation:** `category="quick"`, `load_skills=["golang-security"]` (pattern cross-lang)
- **Depends:** Ninguno
- **Acceptance:** `cat SECURITY.md | grep -i "vulnerab.*report"` → found + `grep -E "disclos|days"` → found

### T18: deny.toml + cargo audit — missing
- **Description:** Crear `deny.toml` (cargo-deny) + `cargo deny check advisories` 0 vulns + `cargo audit`
- **Delegation:** `category="quick"`, `load_skills=["golang-security","rust-learner"]`
- **Depends:** Ninguno
- **Acceptance:** `cargo deny check` → 0 errors + `deny.toml` exists

### T19: CODEOWNERS + workflows — missing
- **Description:** Crear `.github/CODEOWNERS`, `.github/workflows/ci.yml` (test/clippy/fmt/deny pinned SHA), `codeql.yml`, `dependabot.yml`, `scorecard.yml`
- **Delegation:** `category="deep"`, `load_skills=["domain-cloud-native","golang-security"]`
- **Depends:** Ninguno
- **Acceptance:** `ls .github/workflows/ci.yml` + `grep "actions/checkout@" .github/workflows/ci.yml | grep -E "[0-9a-f]{40}"` → pinned

### T20: rust-toolchain + CONTRIBUTING — missing
- **Description:** Crear `rust-toolchain.toml` (channel stable, MSRV), `CONTRIBUTING.md` (DCO, coding standard, test policy, governance), `CHANGELOG.md`
- **Delegation:** `category="quick"`, `load_skills=["rust-learner"]`
- **Depends:** T19
- **Acceptance:** `cat rust-toolchain.toml | grep channel` + `cat CONTRIBUTING.md | grep DCO` → found

### T21: llama.cpp adapter decision — `crates/kitty-runtime/src/grammar.rs`
- **Description:** ADR: documentar spike defer vs real `llama-cpp-2` crate 0.1.154 (`LlamaSampler::grammar`, `json_schema_to_grammar`, `grammar_lazy` pitfall custom+tools exclusive). Si GO, implementar `LlamaCppClient` implementing `ChatClient` con `json_schema_to_grammar` + `LlamaSampler::grammar_lazy` con triggers.
- **Delegation:** `category="ultrabrain"`, `load_skills=["rust-learner","rust-router"]`
- **Depends:** T5+T6
- **Acceptance:** `docs/STRATEGY.md` ADR entry + `grep llama-cpp-2 Cargo.toml` OR `grep "SPIKE defer.*decision.*ADR" grammar.rs`

### T22: CNAS stale fix — `CNAS.md ~40`
- **Description:** Fix "Episodic memory exists but is not yet wired" → "Wired at agent.rs:650/679"
- **Delegation:** `category="quick"`, `load_skills=["writing"]`
- **Depends:** T7
- **Acceptance:** `grep "wired at agent.rs" CNAS.md` → found

### T23: harness tests + codemap — `codemap.md`
- **Description:** `cargo test --workspace` (239→249+), `cargo clippy --all-targets -- -D warnings` clean, actualizar `codemap.md` 239→249+ tests + nuevos módulos
- **Delegation:** `category="quick"`, `load_skills=["rust-learner"]`
- **Depends:** T1-T22 (final gate)
- **Acceptance:** `cargo test --workspace 2>&1 | grep "test result: ok"` ×4 crates + `codemap.md` updated

---

## 7. Commit Strategy — 12 Atomic Commits (C1-C12)

| Commit | Wave | Tasks | Message | Files |
|--------|------|-------|---------|-------|
| C1 | 1 | T1+T3 | `fix(mcp): persist ToolRegistry Arc<Mutex> for Read→Edit + baseline CARGO_MANIFEST_DIR` | mcp.rs, main.rs |
| C2 | 1 | T2+T4 | `feat(contract): integration nudge test + is_small_model→profile` | contract.rs, agent.rs, main.rs, model_profile.rs |
| C3 | 1 | T13+T17 | `docs(knowledge): add Python cards + SECURITY.md` | knowledge/*.md, SECURITY.md |
| C4 | 1 | T19(part) | `ci: add CODEOWNERS + workflows pinned SHA + dependabot` | .github/* |
| C5 | 2 | T5+T6+T20 | `docs(grammar): dedup enable_grammar doc + spike defer ADR + toolchain` | agent.rs, grammar.rs, rust-toolchain.toml, CONTRIBUTING.md |
| C6 | 3 | T7 | `docs: intel+strategy+specs 6 aprobados` | docs/*, openspec/* |
| C7 | 3 | T8 | `docs(openspec): backfill design.md+tasks.md for 6 changes` | openspec/changes/*/design.md, tasks.md |
| C8 | 3 | T11 | `feat(cli): bench ab --thinking 1024,2048,4096 --limit 50 interleaved` | main.rs, bench_ab.rs |
| C9 | 4 | T9 | `chore(openspec): archive 6 changes → 2026-08-19-* + specs` | openspec/changes/archive/*, openspec/specs/* |
| C10 | 4 | T14+T22 | `fix(bench): knowledge default true + CNAS wired fix` | main.rs, CNAS.md |
| C11 | 5 | T12+T18+T21 | `feat(bench): add 3 Python exercises + deny.toml + llama ADR` | benches/*.json, deny.toml, docs/STRATEGY.md |
| C12 | 6 | T23+T15+T16 | `docs: codemap 239→249+ + bench official harness clone` | codemap.md, bench/official/* |

Cada commit: TDD (RED→GREEN capturado) + surface artifact + `git log --oneline -20` mimic style + `cargo clippy` clean antes de commit.

---

## 8. Success Criteria — 9 Gates

1. SC-MCP-PERSIST PASS (Read→Edit secuencial via mcp)
2. SC-CONTRACT-NUDGE PASS (nudge injected)
3. SC-BASELINE-CWD PASS (subdirs resuelven)
4. SC-PROFILE-HEURISTIC PASS (profile dispatch)
5. SC-GRAMMAR-DEDUP PASS (dedup doc)
6. SC-BENCH-AB PASS (JSON per_budget)
7. SC-DOCKER-POLYGLOT PASS (harness clonado, Docker smoke o ADR defer si no Docker)
8. SC-OPENSSF-7 PASS (cargo deny 0 + files exist, sin remote CI run pero files pinned)
9. SC-GBNF-VALIDITY PASS (spike defer ADR o adapter real)

+ `cargo test --workspace` GREEN + `openspec validate --all` 6 passed + `cargo clippy -D warnings` clean

---

## 9. Execution Instructions — 8 Wave Commands

```bash
# Wave 1: 7 parallel deep/quick
task(category="deep", load_skills=["rust-router","m07-concurrency"], prompt="T1 mcp persist...") &
task(category="deep", load_skills=["rust-router","m06-error-handling"], prompt="T2 contract...") &
task(category="quick", load_skills=["rust-router","m12-lifecycle"], prompt="T3 baseline...") &
task(category="quick", load_skills=["rust-router","m05-type-driven"], prompt="T4 profile...") &
task(category="quick", load_skills=["rust-learner"], prompt="T13 cards...") &
task(category="quick", load_skills=["golang-security"], prompt="T17 SECURITY...") &
task(category="deep", load_skills=["domain-cloud-native"], prompt="T19 CODEOWNERS...") &

# Wave 2: after Wave1
task(category="quick", load_skills=["rust-router"], prompt="T5 dedup...") &
task(category="quick", load_skills=["rust-learner"], prompt="T6 spike...") &
task(category="quick", load_skills=["rust-learner"], prompt="T20 toolchain...") &

# Wave 3: after T7
task(category="quick", load_skills=["git-master"], prompt="T7 docs commit...") ->
task(category="unspecified-high", load_skills=["openspec-propose"], prompt="T8 backfill...") &
task(category="deep", load_skills=["rust-router","domain-cli"], prompt="T11 bench ab...") &
task(category="quick", load_skills=["writing"], prompt="T22 CNAS...") &

# Wave 4: after T8
task(category="quick", load_skills=["openspec-archive-change"], prompt="T9 archive...") &
task(category="quick", load_skills=["rust-router"], prompt="T14 knowledge default...") &
task(category="deep", load_skills=["domain-cloud-native","clonedeps"], prompt="T15 polyglot...") &
task(category="deep", load_skills=["domain-cloud-native"], prompt="T16 TB2...") &
task(category="ultrabrain", load_skills=["rust-learner","rust-router"], prompt="T21 llama...") &

# Wave 5: after T11
task(category="deep", load_skills=["rust-learner"], prompt="T10 floor225...") &
task(category="quick", load_skills=["rust-learner"], prompt="T12 exercises...") &
task(category="quick", load_skills=["golang-security"], prompt="T18 deny...") &

# Wave 6: final
task(category="quick", load_skills=["rust-learner"], prompt="T23 codemap...")

# Each task: RED test → GREEN → SURFACE artifact + commit
```

---

## 10. TODO List — 23 Items `path: action for scenario — verify by check`

- [ ] `crates/kitty-cli/src/mcp.rs: persist Arc<Mutex<ToolRegistry>> for SC-MCP-PERSIST — verify by cargo test mcp_persist RED→GREEN + curl mcp Read→Edit success:true`
- [ ] `crates/kitty-harness/src/agent.rs: integration nudge for SC-CONTRACT-NUDGE — verify by mock LLM All done → [CONTRACT-GUARD] in messages`
- [ ] `crates/kitty-cli/src/main.rs: baseline CARGO_MANIFEST_DIR for SC-BASELINE-CWD — verify by cd crates && cargo test baseline_path`
- [ ] `crates/kitty-cli/src/main.rs: is_small_model→profile for SC-PROFILE-HEURISTIC — verify by test_is_small_uses_profile qwen3:4b tiny=true`
- [ ] `crates/kitty-harness/src/agent.rs: document dedup for SC-GRAMMAR-DEDUP — verify by grep dedup comment + test_tiny_forces_grammar`
- [ ] `crates/kitty-runtime/src/grammar.rs: spike defer ADR for SC-GBNF-VALIDITY — verify by grep SPIKE defer + docs/STRATEGY ADR`
- [ ] `git: commit docs untracked for SC-OPENSSF-7 — verify by git status --porcelain open→clean + git log -1`
- [ ] `openspec/changes/*/design.md: backfill design/tasks for archive — verify by openspec status isPlanningComplete=true ×6`
- [ ] `openspec: archive 6 changes for SC-GBNF-VALIDITY — verify by ls archive/2026-08-19-* 6 + specs/*/spec.md 6`
- [ ] `benches/bench_225_smollm2.json: floor 225 for SC-MCP-PERSIST — verify by jq .total==225 + rate_pct<50`
- [ ] `crates/kitty-cli/src/main.rs: bench ab subcommand for SC-BENCH-AB — verify by kitty bench ab --thinking 1024,2048 --limit 2 → JSON budgets`
- [ ] `benches/exercises_polyglot.json: add 3 Python exercises — verify by jq select .id==py-list-comp 3 found`
- [ ] `knowledge/*.md: add Python cards for SC-BENCH-AB — verify by ls 5 cards + cargo test knowledge_loads`
- [ ] `crates/kitty-cli/src/main.rs: knowledge default true — verify by kitty bench --help grep knowledge`
- [ ] `bench/official/aider-harness: clone polyglot for SC-DOCKER-POLYGLOT — verify by ls benchmark.py + polyglot-benchmark/`
- [ ] `bench/official/tb2-local: clone TB2 for SC-DOCKER-POLYGLOT — verify by harbor download + oracle 5 smoke`
- [ ] `SECURITY.md: create for SC-OPENSSF-7 — verify by grep vuln+disclos+days + cargo deny`
- [ ] `deny.toml: create for SC-OPENSSF-7 — verify by cargo deny check 0 errors`
- [ ] `.github/CODEOWNERS: create for SC-OPENSSF-7 — verify by ls + grep pinned SHA in ci.yml`
- [ ] `rust-toolchain.toml: create for SC-OPENSSF-7 — verify by cat channel + CONTRIBUTING DCO`
- [ ] `docs/STRATEGY.md: llama ADR for SC-GBNF-VALIDITY — verify by ADR entry + Cargo.toml llama-cpp-2 OR spike defer`
- [ ] `CNAS.md: fix wired for SC-GBNF-VALIDITY — verify by grep wired at agent.rs:650`
- [ ] `codemap.md: update 239→249+ for SC-GBNF-VALIDITY — verify by cargo test --workspace GREEN + clippy clean`

---

## 11. Findings (append as work progresses)

- `mcp.rs:41` verbatim `let mut registry = ToolRegistry::new()` — bug confirmed cross-turn Read→Edit
- `main.rs:301` hardcoded `benches/baseline.json` — CWD fragile
- `main.rs:193` heuristic 5 contains — ad-hoc vs ModelProfile
- `agent.rs:88/189` doble-set tiny→grammar — intentional but undocumented
- `grammar.rs:1` spike post-hoc — GBNF nunca enviado a llama.cpp, OllamaClient.chat sin grammar field, no deps bindgen
- `openspec/` untracked + isPlanningComplete false ×6 — gap proceso, no implementación
- `knowledge/` 2 cards → 5, `benches/exercises_polyglot.json` 0 comprehensions → +3
- Docker/podman not installed — F6.2 smoke defer si no Docker, ADR required
- `CNAS.md` stale wired false

## 12. Learnings

- P1.1 es único código acotado — debe ir primero Wave1 para desbloquear archive
- OpenSpec archive requiere backfill design/tasks antes de archivar — no archivar con isPlanningComplete false directo
- Bench ab necesita ThinkingBudget observability (session_tokens en BenchResult) — agregar
- F6.2 Docker es infra (1 día) no code — parallel con OpenSSF local

---

## 13. Verification Anti-Patterns (BLOCKING)

- No "should work" — cada task requiere RED→GREEN proof + real surface artifact path en notepad
- No code sin failing test primero — TDD mandatory, exemption solo formatting con justificación
- No scope reduction — 23 tasks completas, no demo/skeleton
- No excusas — si blocker, consultar Oracle/Artistry o pedir al usuario

**Prometheus READY — awaiting approval to start Wave 1 (7 parallel deep/quick)**
