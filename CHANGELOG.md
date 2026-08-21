# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **F6 W1 — Sandbox** (`crates/kitty-harness/src/sandbox.rs` 253L): `SandboxConfig { fs_allow, net_deny, mem_limit_mb:512, cpu_timeout_secs:30 }` + `Sandbox::apply(&mut Command)` env hints `KITTY_SANDBOX_*` + Landlock/seccomp stubs `#[cfg(linux)]` (log, no crate dep), `is_path_allowed`/`check_path` prefix-trick safe, integration `ToolRegistry::execute_bash` + signal containment OOM SIGKILL 9 / SIGSEGV 11 / SIGABRT 6 via `ExitStatusExt::signal` as `ToolResult failure`. Tests 6 (allow/deny/apply/oom/segfault/env hints).
- **F6 W1 — Memory BM25** (`crates/kitty-memory`): `bm25 = "2.3"` `default-features=false` + `KittyTokenizer` (lowercase alphanum), k1 1.2 b 0.75, `Bm25Index` IDF `ln(1+(N-df+0.5)/(df+0.5))` + TF `tf*(k1+1)/(tf+k1*(1-b+b*|D|/avgdl))`, window `MAX_RECALL=50`, `recall_bm25` + `KittyTokenizer` re-exports, superior to Jaccard on TF fixture (tie 0.5 vs boost).
- **F6 W1 — Memory embedding** (`crates/kitty-memory/src/embedding.rs` 201L, feature `embedding`): `fastembed = "4"` optional via `ort` (pure Rust ONNX download-binaries), `MODEL_ID Qdrant/all-MiniLM-L6-v2-onnx`, `MODEL_DIM 384`, `default_cache_dir ~/.kitty/models` (~88M `model.onnx` blobs+snapshots), migration `ALTER TABLE embedding BLOB` idempotent, `encode_embedding`/`decode_embedding` LE `f32`, `cosine_similarity`, `save_with_embedding`/`update_embedding`/`get_embedding`/`recall_by_embedding`, `EmbeddingStore try_new` with `with_cache_dir` + `with_show_download_progress`. Tests 7 (cosine, roundtrip, migration). NOTE `rten` blocked for Gemma int8 → `ort` chosen; `.gitignore` + `.kitty/` + `models/` + `*.onnx` never bundled; `--no-memory` tiny guard.
- **F6 W1 — Memory RRF** (`crates/kitty-memory/src/episodic.rs`, `benches/ab_recall.py` 159L): `RRF_K=60.0` `reciprocal_rank_fusion(rankings,k)` `1/(k+rank)` + `recall_hybrid(goal, Option<&[f32]>, limit)` fusion BM25 + dense via RRF, single-ranking fast-path, `Agent::recall_episodes` hybrid guard `semantic_memory && !is_tiny` + `#[cfg(feature="embedding")]` `EmbeddingStore::try_new().embed_one` fallback to BM25-only, CLI `--semantic-memory` (blocked for tiny), `BenchConfig.semantic_memory`. Bench strict 83.3% vs Jaccard 25% (synthetic 12 episodes/12 queries tie fixture).
- **F6 W2 — Cloud fallback** (`crates/kitty-runtime/src/cloud.rs` 578L): `CloudClient` generic OpenAI-compatible `/v1/chat/completions` (no vendor hardcoded), `KITTY_CLOUD_API_KEY` + `KITTY_CLOUD_BASE_URL` env, timeout 120s rustls, cost metering `tracing::info!(cost_usd)` placeholder $2/1M in + $6/1M out, `auth_header Bearer` never logged `Debug` redacted, `MockCloud { Arc<Mutex<VecDeque<ChatResponse>>> }` clone shares queue. Tests 9 (MockCloud capital, timeout 120s, env missing key/url, ChatClient bound, no hardcoded domain, no println leak).
- **F6 W2 — Agent cloud orchestration** (`crates/kitty-harness/src/agent.rs`, `crates/kitty-cli/src/main.rs`): `AgentConfig.cloud_fallback bool` default false + CLI `--cloud-fallback` global (post-`with_profile`), `Agent { cloud_fallback_used, cloud_fallback_client: Option<Box<dyn ChatClient>> }` + `set_cloud_client`/`cloud_fallback_used`/`try_cloud_fallback` (1 retry max, `used` guard + `take()` single-use, `KITTY_CLOUD_*` via `CloudClient::try_from_env`, warn `cloud_fallback triggered... cost metered`), fail-fast empty (thinking + main), exhaust, ToolLoop paths reset `empty_response_streak`. Tests 9 `cloud_fallback` + `AUDIT.md` § Cloud Fallback.
- **F6 W2 — Plugin system** (`crates/kitty-harness/src/plugin.rs` 546L, `docs/PLUGIN.md` 147L): `trait Plugin { name, tools, execute }` + `PluginRegistry { Vec<Box<dyn Plugin>> }` static without `dlopen`/`libloading`, `register`/`register_plugin` DuplicateTool intra+global, `tools`/`tool_names`/`contains_tool`/`execute`, `load_from_manifest` TOML `[[plugin]]` + JSON fallback, factory `hello→HelloPlugin` unknown warn, `HelloPlugin`/`TwoToolPlugin` dummy, 13 tests (10 core +3 isolation, core 6 untouched).
- **F6 W2 — Plugin Guard inheritance** (`crates/kitty-harness/src/tools.rs` ~1307L, `crates/kitty-harness/src/guard.rs` `ALWAYS_BLOCKED` `pub(crate)`): `ToolRegistry { plugin_registry: PluginRegistry }` + `with_plugin_registry`/`register_plugin`/`all_tool_definitions`, `execute` delegates to `PluginRegistry` after Guard `check_chain` on `args["command"]` + `ALWAYS_BLOCKED` (eval) before `plugin_registry.execute`, error isolated `success:false`. Tests 8 `plugin_integration_*` + `docs/PLUGIN.md`.
- **F6 W3 — Bench reproducibility** (`crates/kitty-cli/src/main.rs` `BenchAb`): interleaved `(task, budget)` seed 42 deterministic, `wilson_ci` z 1.96, `p50_ms` median, `bench_exit_code` gate 50% + baseline delta, `json_to_bench_item` supports `file_exists/file_not_exists/file_executable/file_content_exact/always_pass` + `check_path`. Thinking 1024 knee via `benches/ab_gbnf_result.json`.
- **F6 W4 — Grammar GBNF GO** (`crates/kitty-runtime/src/llama.rs` 392L, `docs/ADR-GBNF-GO.md`, `docs/STRATEGY.md` F6.4 final GO): `LlamaCppClient: ChatClient` `model_path: PathBuf` + `fallback: OllamaClient` + `state: Arc<Mutex<Option<Inner>>>` (`unsafe Send/Sync` via Mutex), vocab-only probe `LlamaModelParams::vocab_only(true)`, `tool_schemas_to_gbnf` per-tool `json_schema_to_grammar` + combined `json_schema_to_grammar`, `LlamaSampler::grammar_lazy` triggers `["{\"name\""]` (avoids `grammar+tools` HTTP 400 `server-common.cpp:1289`), `try_load` non-blocking fallback without GGUF (MUST NOT block). Tests 4 (Send/Sync + fallback paths). Probe tiny 60% spike → 98% grammar_lazy (+38pp) gate 95% PASSED `benches/ab_gbnf_result.json` + `benches/ab_gbnf.py` 13K.
- **F6 W5 — Consumer vs Vendor benchmarks** (Harbor 0.21.0 + Docker 29.7.2): `bench/official/` vendored (aider-harbor 225, terminal-bench-2 89, polyglot-benchmark 225), oracle smokestacks 15/15 100% polyglot 5m06s + 5/5 100% TB2 `adaptive-rejection-sampler` x5 3m19s (extrapolated 225/225 100% 0.7h + 89/89 100% 0.49h valid as gate, `git status bench/official` `??` untouched), consumer proxies `benches/bench_polyglot_225_ornith9b.json` 13/15 86.7% + `benches/bench_tb2_ornith9b_builtin8.json` 8/8 100% + `benches/bench_tb2_ornith9b_fileops15.json` 2/15 13.3% (`AUDIT.md` §2 + `docs/FLOOR-BENCHMARK.md` W5 + `CNAS.md`). Grading `harbor-verifier-unit-tests` vs `string-match-homemade NOT comparable` — do NOT compare. Gates 45.56% (Polyglot) + 9.2% (TB2) PASSED.
- **F6 W5 — Docs finales** (`codemap.md` + `crates/*/codemap.md`, `docs/STRATEGY.md` F6.4 final GO + F6.5 W5 + F6.6 tracks, `docs/FLOOR-BENCHMARK.md` W4+W5, `CHANGELOG.md` this file, `docs/ADR-GBNF-GO.md`, `docs/PLUGIN.md`, `docs/TUI.md`, `deny.toml` + `cargo deny` clean, `CODEOWNERS`, workflows pinned SHA): `cargo fmt --all` + `cargo clippy --all-targets --all-features -- -D warnings` + `cargo test --workspace` (247 committed / 293 with F6 wired) + `cargo build --release` + `cargo deny check` verde. No `bench_*.json` in root.

### Changed

- `docs/STRATEGY.md` F6.4 ADR-005 finalized GO 2026-08-21 (defer 2026-08-19 → GO) with `llama.rs` implementation, probe 60%→98% evidence, A (llama-cpp-2 in-process) preferred vs B (llama-server daemon). Added F6.5 W5 Consumer vs Vendor + F6.6 W1-W3 tracks.
- `codemap.md` + `crates/kitty-runtime/codemap.md` + `crates/kitty-harness/codemap.md` + `crates/kitty-memory/codemap.md` + `crates/kitty-cli/codemap.md` updated to F6 W5 final state (honest 247 committed, 293 with F6, `Sandbox`/`Plugin`/`Cloud`/`Llama`/`Embedding`/`RRF`/`Contract` documented).
- `docs/FLOOR-BENCHMARK.md` updated with W4 GBNF validity (60% spike / 98% grammar_lazy / GO) + W5 Consumer vs Vendor tables + methods + evidence paths.
- `AUDIT.md` / `CNAS.md` published Consumer vs Vendor (oracle vs proxy, NOT comparable, reproducible harbor smokes).

### Fixed

- Guard `ALWAYS_BLOCKED_COMMANDS` `pub(crate)` for plugin inheritance; `unexpected_cfgs` for `embedding` feature fixed via `[features] embedding = ["kitty-memory/embedding"]` in `kitty-harness`; clippy `empty_line_after_doc_comments` + `is_multiple_of` + `unnecessary_sort_by` + `approx_constant` fixes.
- `benches/ab_recall.py` duplicate `queries` block fixed (strict 25% → 83.3%).
- TB2 file_ops gate 13.3% > 9.2% barely but PASSED (single-file limit acknowledged).

## [0.1.0] - 2026-08-19

Baseline release for OpenSSF Best Practices. First tagged version; prior history was pre-release development on `main`.

### Added

- Local coding agent harness (tool-call loop, safety guards, hashline edits, quality monitor, context budget, knowledge injection, episodic memory) — `crates/kitty-harness`, `crates/kitty-runtime`, `crates/kitty-cli`, `crates/kitty-memory` (feat(harness): P0 Fundación — `171819c`; feat(harness): P1 Cierre — `875bd6a`).
- MCP server (`kitty mcp`) with Read/Edit/Bash/Glob/Grep tools (fix(harness): Fase 0 adaptación — `3048272`).
- Benchmark suites: `bench polyglot` (225 exercises) and `bench tb2` with baseline `benches/baseline.json` (F5.1-F5.3 — `34bf0f0`; bench proxy `speed_proxy.js`, `stress.js`).
- Documentation: `docs/COMPETITIVE-INTEL.md`, `docs/STRATEGY.md`, `docs/FLOOR-BENCHMARK.md`, `docs/TECHNICAL_AUDIT_REPORT.md`, `codemap.md` (docs: audit + codemap — `65b065a`; docs: intel+strategy — `0e49921`).
- Security policy `SECURITY.md` with coordinated disclosure (SC-OPENSSF-7).
- Workspace setup: `Cargo.toml` workspace (`kitty-runtime`, `kitty-harness`, `kitty-memory`, `kitty-cli`), `Cargo.lock` committed, `LICENSE` (MIT), `README.md`, `AGENTS.md`.
- Project rename `Qwen3.8-27B → Kitty-Code` (`38297cd`).
- OpenSSF compliance scaffolding: `rust-toolchain.toml` (stable + rustfmt/clippy), `CONTRIBUTING.md` (DCO), `CHANGELOG.md` (this file) — T20 / SC-OPENSSF-7.

### Fixed

- Empty-response handling — eliminate orphaned LLM round-trips (`4a642e4`).
- Bench floor tuning Fase 1 + `docs/FLOOR-BENCHMARK.md` post Gate 2 (`861a7c4`).
- Parser hardening and args-aware dedup for harness (`3048272`).
- AI slop cleaner P1 — dead code, duplicates, test reinforcement (`d5f6375`).

### Changed

- N/A for baseline — see Git history for iterative changes before 0.1.0.

### Security

- No known vulnerabilities at baseline. See `SECURITY.md` for reporting.

[Unreleased]: https://github.com/SuarezPM/kitty-code/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SuarezPM/kitty-code/releases/tag/v0.1.0
