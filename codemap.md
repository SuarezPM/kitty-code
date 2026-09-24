# Repository Atlas: Kitty Code

## Project Responsibility
`Kitty Code` is a **local coding agent** written in Rust (mono-repo: harness +
embedded inference engine), optimized to run small GGUF models (1–35B) on a
consumer GPU (RTX 2060 Super 8 GB) via Ollama HTTP API **or** the embedded
`kitty-infer` backend (`KITTY_BACKEND=kitty-infer`, default Ollama byte-identical).
It is a research harness, not a production system. The differentiators are the
**harness** (tool loop, safety guards, hashline edits, quality monitor,
context budget, knowledge injection, episodic memory) around small local LLMs,
plus an **honest-bench culture** (manifests per run, non-claims, k≥3) and a
**private engine fork** (K2-Horizon support nobody else ships).

## System Entry Points
- `crates/kitty-cli/src/main.rs` (thin dispatch: `Cli`/`Commands` + `resolve_project_root` + `resolve_session_num_ctx`, bench delegated to `bench/`) — CLI binary (`run`, `bench`, `bench-ab`, `info`, `mcp`, `tui`); backend switch `backend_selected(ENV_BACKEND)` → `Box<dyn ChatClient>` (Ollama vs `KittyInferClient`); bench suite in `crates/kitty-cli/src/bench/{common,run,ab,polyglot,tb2,gpqa,claweval}.rs` + suite support libs.
- `crates/kitty-runtime/src/kitty_infer.rs` — embedded backend: `KittyInferConfig` (env `KITTY_INFER_*`: model, tokenizer, ctx, dtype policy, GPU layers, sampler, tool format), `KittyInferClient` (single-resident registry + evict, VRAM precheck, tool_call_id threading) behind `ChatClient`, feature-gated (`kitty-infer[-cuda]`, zero cost default).
- `crates/kitty-cli/src/bench/timeouts.json` — per-(model, backend) timeout manifest (minicpm5 60/150s, qwen3 120/180s); thresholds/seeds/interleave are contract, never tuned to pass.
- `engine/mistralrs-core/src/serving/` — engine serving layer: `ModelSpec::auto`, single-resident registry + evict, OpenAI-shape API, SSE, errors with hints, GPU snapshot health, `infer_policy` (dtype F16 + F32 overrides, tokenizer sidecar-or-error), warm preload.
- `Cargo.toml` — single virtual workspace: `crates/*` + 7 `engine/mistralrs*` (mono M1–M3; `mistralrs-cli` dormant), one lock, release profile (LTO, strip, `panic="abort"`).
- `AUDIT.md` / `CNAS.md` — honesty-first benchmark documentation (homemade quizzes,
  explicitly **not** Aider Polyglot / Terminal-Bench 2.0; Harbor oracle verifier 15/15 + 5/5 100%).
- `docs/STRATEGY.md` — F6.4 ADR-005 GBNF defer spike final + roadmap P0/P1.
- `docs/FLOOR-BENCHMARK.md` — W4 GBNF A/B + W5 Consumer vs Vendor evidence.
- `docs/SDD-MONOREPO-UPGRADE.md` — mono-repo + upgrade program (decisions: squash+mirror, archive, 40min).
- `docs/BACKEND-DEFAULT-CRITERIA.md` — 8 criteria + NO-SWITCH verdict with numbers.

## Directory Map (Aggregated)
| Directory | Responsibility Summary | Detailed Map |
|-----------|------------------------|--------------|
| `crates/kitty-runtime/` | `ChatClient` trait + Ollama/Cloud/llama.cpp clients + embedded `KittyInferClient` (config, single-resident, tool ids, sampler envs) + wire types. | [View Map](crates/kitty-runtime/codemap.md) |
| `crates/kitty-harness/` | Agent loop, tool registry, layered guard, sandbox, fuzzy edits, parser (channel/kwarg/_raw + tool_call_id), quality monitor, router, config, compaction, telemetry, skills. | [View Map](crates/kitty-harness/codemap.md) |
| `crates/kitty-memory/` | Episodic memory (blake3 + SQLite WAL) + BM25 + opt-in dense RRF hybrid. | [View Map](crates/kitty-memory/codemap.md) |
| `crates/kitty-cli/` | Thin-dispatch CLI + `bench/` suite (common/run/ab + runners, wilson CI, per-cell timeouts, `BenchClient` Ollama/Infer) + MCP + TUI. | [View Map](crates/kitty-cli/codemap.md) |
| `crates/kitty-cli/src/bench/` | Bench dispatch per suite + timeout manifest data + gates. | [View Map](crates/kitty-cli/src/bench/codemap.md) |
| `engine/mistralrs-core/src/serving/` | Model registry + preload, API shapes, SSE, friendly errors, GPU health, inference policy. | [View Map](engine/mistralrs-core/src/serving/codemap.md) |
| `engine/mistralrs-core/src/pipeline/loaders/` | Loader registry incl. K2Horizon, GGUF loader + VRAM precheck, auto device-map (GPU-first, audible). | [View Map](engine/mistralrs-core/src/pipeline/loaders/codemap.md) |
| `engine/mistralrs-core/src/kv_cache/` | SingleCache (dtype-unified) + paged-attention block scheduler. | [View Map](engine/mistralrs-core/src/kv_cache/codemap.md) |
| `engine/mistralrs-core/src/models/` | Text towers (upstream inherited) + `k2_horizon.rs` port + registry. | [View Map](engine/mistralrs-core/src/models/codemap.md) |
| `engine/mistralrs/` | Facade crate: builders, `best_device`, re-exports, cuda/metal/flash-attn features. | [View Map](engine/mistralrs/codemap.md) |
| `engine/` | Fork mistral.rs (upstream MIT: `engine/NOTICE` + `engine/LICENSE`) como backend embebido GGUF; naming `mistralrs-*` conservado; `mistralrs-cli` dormant; docs `engine/F0-RECON.md` + `engine/docs/patches/` (guard candle#3909). | — |
| `bench/official/` | Vendored official benchmarks (aider-harbor 225, terminal-bench-2 89, polyglot-benchmark 225) — Harbor oracle verifier. | — |
| `benches/` | Homemade benches + exercises JSON (228) + baseline + A/B benches (recall, GBNF) + bench scripts + `ab/manifests/` (per-run manifests). | — |
| `docs/` | STRATEGY, FLOOR-BENCHMARK, PLUGIN, TUI, ADR-GBNF-GO, AUDIT mirror, SDD-MONOREPO-UPGRADE, BACKEND-DEFAULT-CRITERIA, EMBEDDED-BENCH-SPEC. | — |

## Data & Control Flow (one `run`)
1. CLI builds `AgentConfig` + `ChatClient` via backend switch (`KITTY_BACKEND`; OllamaClient default — session-fixed `num_ctx`, `"truncate": false`; `KittyInferClient::from_env()` for embedded: warm preload, VRAM precheck, dtype policy).
2. `Agent::run(prompt)`: bootstrap detect → preflight system hint (FIRST) → episodic recall → knowledge inject → push user msg → loop:
   - `ChatClient::chat` → `ToolCallParser` (native + text) → tool-call normalization → `GrammarClient` post-hoc filter → `QualityMonitor`
     → `ToolRegistry::execute` (HashlineEditor + fuzzy ladder + Guard / Bash via Sandbox + Guard / Glob / Grep / Plugin) → push result (tool_call_id threaded end-to-end).
   - context pressure → `trigger_compaction`; empty streak 2 / iteration cap → CloudClient fallback (1 retry) else fail-fast.
3. Final text returned; episode saved (best-effort).
4. Bench path: same loop per exercise with per-cell timeouts (manifest), warm resident, manifests persisted per run (never `/tmp`).

## Structure Remediations (2026-09-03, audit Fase 1-6)
- `kitty-cli/src/main.rs` is a thin dispatch layer (<1K LOC); benchmark
  suites + stats live in `crates/kitty-cli/src/bench/{common,run,gpqa,claweval,polyglot,tb2,ab}.rs`.
- `kitty-harness/src/agent.rs` → `agent/{mod,run,response}.rs`; `guard.rs` →
  `guard/{mod,patterns,shell_parse,interpreters}.rs`.
- All harness `KITTY_*` env names + typed accessors: `kitty-harness/src/config.rs`
  (runtime-owned vars stay in `kitty-runtime`; sandbox child vars stay in `sandbox.rs`).
- Per-run overrides via `Agent::run_with(prompt, RunOverrides)`;
  `config_mut()`/`budget_mut()` removed from the public API.
- `EpisodeStore` ops run via `spawn_blocking` from the agent; BM25 index
  cached with write-side invalidation. Parser micro-benches:
  `cargo bench -p kitty-harness`.
- CI gates: strict clippy (`--all-targets --all-features -D warnings`),
  `--all-features` tests, CycloneDX SBOM job, weekly `cargo audit`,
  llvm-cov gate (kitty-harness ≥70%, measured 89.7% at remediation time).
- Zero `#[allow(dead_code)]` in production code; root orphan scripts removed.
- Mono-repo M1–M3 (2026-09-23): `engine/` + `crates/` one workspace, one lock,
  dual-dep epochs documented, `pre-mono` tags as rollback.

## Honesty Notes (verified 2026-08-30 — f8 + F2 B1-l225 autonomous)
- **560 tests pass** (`cargo test --workspace --all-features`: 46 cli +425 harness +25 memory +64 runtime) 0 failed, `cargo clippy --all-targets --all-features -- -D warnings` **FAIL 2× await_holding_lock in tools.rs:3189 pre-existente (no introducido por benches, cargo test OK)**, `cargo fmt --all --check` clean. Release build ~30s LTO strip. `cargo deny check` clean.
- **F6 W5 Consumer vs Vendor published**: Harbor 0.21.0 + Docker 29.7.2 — Oracle 15/15 100% (polyglot) 5m06s + 5/5 100% (TB2 adaptive-rejection-sampler x5) 3m19s prove infra healthy (extrapolated 225/225 + 89/89 valid as gate, not vendor claim). Consumer ornith:9b proxies: 13/15 86.7% polyglot + 8/8 100% TB2 builtin + 2/15 13.3% file_ops (gate 9.2% PASSED barely). Grading `harbor-verifier-unit-tests` vs `string-match-homemade NOT comparable` — do NOT compare. See `AUDIT.md` §2, `docs/FLOOR-BENCHMARK.md` W5, `CNAS.md`.
- **Homemade** `KittyQuiz-225` 96.44% 217/225 (string-match) + `KittyShell-89` 12.36% remain NOT comparable to official Aider Polyglot / TB2. Documented honestly in `AUDIT.md`/`CNAS.md`.
- **F6 P0s LANDED (merged, counted in the 331):** `sandbox.rs` real kernel containment ported from Catalyst `apohara-sandbox` — Landlock ruleset (read `/`, rw allow-roots, ABI-probed V3 gating TRUNCATE), seccompiler EPERM fail-closed allowlist + clone3 ENOSYS overlay, userns/mountns/pidns + double-fork, RLIMIT_AS, 16 MiB output cap; wired into `ToolRegistry::execute_bash` (fail-closed) with loud per-run `KITTY_SANDBOX=off` opt-out and INFO first-arm log of resolved fs_allow roots; sandbox allow-root follows `AgentConfig.workspace` explicitly (`kitty-cli` resolves project root by marker walk-up — never silent cwd/$HOME). `cloud.rs` SecretString zeroize + Debug redact (Catalyst `apohara-secrets` pattern) + secret-env strip. Behavioral tests: write /etc/passwd denied with unsandboxed negative control, rlimit abort, net deny; Miri scoped to pure-Rust sandbox units.
- Benchmarks are genuinely homemade (string-match `content.contains(check_str)`) — documented honestly in `docs/FLOOR-BENCHMARK.md` (smollm2 53% 2.76s, qwen3 53% 24s, ornith 96.44% 4.6s + W4 GBNF spike 60% tiny validity (simulated 98% SUPERSEDED by empirical 94%/94% — GO REVERSED, `docs/ADR-GBNF-GO.md`) + W5 oracle vs proxy). Tier sub-2B re-encuadrado a demo (<50% en 225 full), 2-4B GO condicional.
- Memory recall baseline was lexical Jaccard → BM25 (k1 1.2/b 0.75) as IR baseline; dense embeddings via fastembed opt-in + RRF K=60 hybrid is the upgrade (blocked by rten/Gemma int8 → ort chosen). `save_episode` records real completion status (fail on exhausted iterations).
- `tree-sitter` dependency and `trait Tool` removed; `ShellStart`/`Test` advert removed (never registered). Security guard is interpreter-aware (`sh -c` bypass closed). Token counting is REAL via Ollama `/api/tokenize` (model's own tokenizer), heuristic fallback only on server error.
- Architecture of 4 crates is the correct sweet spot — not refactored for its own sake, now with `model_profile.rs` (tiny 4K/5iter/4tools, medium 16K/20iter, large 32K/20iter + per-model `ModelTuning` resolution: exact → family "ornith"/"qwen" → default; sampling/thinking profiles, unified turn caps; nano preset + `PromptStyle::{Full,Minimal}` with conservative auto-detection for sub-3B models), `edit_replacers.rs` (fuzzy edit ladder behind hashline/search-replace failure, bounded candidate search), `router.rs`, `contract.rs`, `grammar.rs` (spike), `mcp.rs`, `sandbox.rs`, `plugin.rs`, `cloud.rs`/`llama.rs`, `embedding.rs`.
- **f8 batch levers (landed code, unmeasured):** autorun continuation hops in the agent loop (`AgentConfig.autorun_hops` clamp 0..=12, consumed only on iteration exhaustion, MAX_STEPS suppressed while hops remain; bench flag-only via `--hops`, interactive `KITTY_AUTORUN_HOPS`); native context probe (`/api/show` suffix-match on `.context_length` keys, once-per-process startup resolution, KV warning rerouted); honest Read windows (header + offset/limit clamped, profile read_cap plumbed); compaction suite (summarizer overrides temp 0.1/predict 400/≤280 words, derived threshold min(profile, max(0.62,(ctx−2048)/ctx)), user-boundary prune walk-back K=12 with exact landing); `Box<T>` chat_with_options forwarding fix (RequestOptions were silently dropped on boxed clients); async lint gate (`spawn_blocking`) + token-count LRU cache (~256 entries).
- **F2 B1-l225 autonomous (2026-08-30, G1 APPROVE):** `b1-l225-rerun-20260830T042258Z.json` **221/225 98.2%** (0 errored ≤22 PASS, `grep400=0`, wall 182m21s) mix real `50py→50js→30ts→45rust→30go→10cpp→10java` (225 base, 3py bonus excluded, fingerprint `0d38e3c`), manifest pre-registro. GPQA Diamond / ClawEval 300t **blocked-harness** (`kitty bench` solo `polyglot/tb2/smoke`, sin loader) — gaps explícitos en `AUDIT §7.1b` / `STRATEGY`.
- **F0-F5 hardened**: guard (command substitution, eval block, timeout 30s bench), hashline prefix matching, tool error handling, budget single-point accounting (`push_message` sole), cross-turn tool-loop detection (args-aware blake3), message ordering, thinking retry (cap 2), Read truncation (300 lines), Grep via ignore crate, dead code removal, CLI flags (`--profile tiny|medium|large`, `--no-memory` auto, `--semantic-memory`, `--cloud-fallback`), JSON exercise loading, `bench-ab` interleaved + wilson CI.
- **Empty response fix**: assistant msg pushed to history, no orphaned LLM round-trips, `MAX_EMPTY_STREAK=2` + fail-fast → latency 87s→4s smollm2, 39s→24s qwen3.
- **Bench/Knowledge (F6 W5):** `bench/official/` exists (aider-harbor, polyglot-benchmark 225, terminal-bench-2 89, aider-harbor 225), `benches/exercises_polyglot.json` 228 (225 +3 Python: list-comp/decorators/async), floor 225 smollm2 45.3% (<50% → demo re-frame, `docs/FLOOR-BENCHMARK.md`), knowledge 5 cards (`knowledge/*.md` rust-traits/go-generics/python-list-comp/python-decorators/python-async) scoring word1.0/bigram2.0, `benches/baseline.json` 96.44, `kitty mcp` stdio 6 tools via Guard, `cargo deny` + `deny.toml` allowlist, `CODEOWNERS`, workflows pinned SHA.
- **Septiembre 2026 (mono-repo + infer embebido):** workspace único `crates/*` + `engine/mistralrs*` (M1–M3, un lock); backend `kitty-infer` tras `ChatClient` (`KITTY_BACKEND`, feature-gated, default Ollama intacto); paridad motor bit-igual CPU 3/3 y mismo-GPU funcional; bench k≥3 con manifiestos en repo + timeouts por (modelo,backend) + warm-resident; veredicto NO-SWITCH con números; K2-Horizon porteado con envelope documentado; guard candle#3909 persistido; auto-map GPU-first audible.

## Codemap Freshness (2026-09-23)
- Full run post-fusión: init con 150 ficheros trackeados (`.slim/codemap.json`), backup de 10 maps previos (intactos), 114 stubs vacíos eliminados.
- Maps refrescados/creados: 10 crates (incl. `bench/` nuevo) + 5 engine (serving, loaders, kv_cache, models-K2, facade) + este atlas; 0 marcadores `Fixer:` restantes.
- `AGENTS.md` ya referencia este atlas — Step 5 exento como idempotente.
