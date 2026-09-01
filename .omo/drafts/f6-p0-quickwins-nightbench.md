---
slug: f6-p0-quickwins-nightbench
status: approved
intent: unclear
review_required: true
plan_path: .omo/plans/f6-p0-quickwins-nightbench.md
plan_sha256: 12todos+4verifiers-fixed-oracle7
review_round_id: rr-20260821-002
pending-action: done - plan approved both lanes after oracle fixes
review:
  momus:
    status: approved
    workspace_root: /home/thelinconx/Proyectos/Kitty-Code
    runtime_home: null
    target: .omo/plans/f6-p0-quickwins-nightbench.md
    round_id: rr-20260821-001
    plan_sha256: 12todos+4
    launch_id: momus-001
    session: ses_fdae7e2c7ffeN27GpWGDiRkt6k
    result: APPROVE (12 todos, refs, matrix, scope P0s)
  independent:
    status: approved
    workspace_root: /home/thelinconx/Proyectos/Kitty-Code
    runtime_home: null
    target: .omo/plans/f6-p0-quickwins-nightbench.md
    round_id: rr-20260821-002
    plan_sha256: 12todos+4-fixed7
    launch_id: oracle-002
    session: ses_fdae12821ffermm2qUImtjLRWU-fixed
    result: APPROVE after 7 fixes (rename collision, matrix cycle harb-adapter, 308->277, SecretString ref, GBNF simulated, Catalyst landlock extra) - see plan edits
review:
  momus:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/f6-p0-quickwins-nightbench.md
    round_id: null
    plan_sha256: null
    launch_id: null
    session: null
    result: null
  independent:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/f6-p0-quickwins-nightbench.md
    round_id: null
    plan_sha256: null
    launch_id: null
    session: null
    result: null
approach: Quick wins P0 TrustLayer mientras corre bench noche completa harbor run aider-harbor kitty ornith:9b 225 (76min) + TB2 89 (1h) en paralelo; fixes no bloquean bench: rename JSON smoke, zeroize SecretString (Catalyst apohara-secrets), Landlock+seccomp+namespaces real (Catalyst sandbox profile/linux.rs + namespace.rs + audit O_NOFOLLOW), GBNF empírico, TUI snapshots, embedding e2e. Evaluación own inference worth vs Ollama con patrones robables de apohara-codesearch/catalyst/dekanus/little-coder/smallcode.
---

# Draft: f6-p0-quickwins-nightbench

## Components (topology ledger)
| id | outcome | status | evidence path |
|---|---|---|---|
| bench-night | Harbor run aider-harbor kitty ornith:9b 225 + TB2 89 1h bench_225_full.json real | active | bench/official/aider-harbor, bench/official/terminal-bench-2, harbor run |
| p0-rename | JSON smoke rename + AUDIT CNAS honestidad filenames vs n_trials | active | benches/bench_*_*.json, docs/AUDIT.md, CNAS.md |
| p0-zeroize | zeroize SecretString + Debug redaction + cost log sin Authorization | active | crates/kitty-runtime/src/cloud.rs, OnceLock Mutex |
| p0-landlock | Landlock 0.4 real + rlimit + /etc/passwd deny test + Miri Send/Sync | active | crates/kitty-harness/src/sandbox.rs, guard.rs |
| p0-gbnf-empiric | Empirical GBNF via llama.cpp GGUF o marcar simulated, no 98% forzado | active | crates/kitty-runtime/src/llama.rs, benches/ab_gbnf.py |
| qw-tui | TestBackend snapshots CJK/empty-q | active | crates/kitty-cli/src/tui.rs |
| qw-embed | embedding e2e real episodes.db + CI ab_recall | active | crates/kitty-memory/src/episodic.rs, embedding.rs |
| vision | Roadmap Must Have + own inference worth vs Ollama + apohara patterns + director vision pivot | active | apohara-codesearch/dekanus, little-coder, smallcode |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
|---|---|---|---|
| Bench night model | kitty harbor adapter + ornith:9b (Q4_K_M 5.6GB en RTX 2060S 8GB) | ornith:9b ya listo ollama list, harbor adapter missing -> crear kitty adapter; oracle no puntúa modelo | sí |
| Landlock + seccomp | Catalyst `seccompiler 0.5` EPERM fail-closed + `landlock 0.4 RestrictSelf` + `nix CLONE_NEWUSER\|NEWNS\|NEWPID` + rlimit RLIMIT_AS 512MB | audit P0 exige Landlock real + Catalyst provee seccomp+namespaces pattern (seccomp 0.5 linux.rs:7, namespace.rs:52) + 16MiB cap runner/imp.rs:110 | sí |
| GBNF empírico | Si GGUF descargado (~5GB) -> run benches/ab_gbnf.py --with-model else marcar simulated y cerrar ADR como not empirically closed | audit P0 measurement, no fabricar 98%; llama-cpp-2 0.1.154 ya wired | sí |
| Plugin wasmtime | Mantener estático (audit dice reject wasmtime gate) - no añadir wasmtime dep | static HelloPlugin correcto 2026 | sí |
| Own inference | Mantener Ollama + ChatClient trait abstración para futuro Dekanus/candle, no own engine para v0.1 | Dekanus Phase 0 skeleton, 5GB compile, sm_75 quirks coste enorme vs 0 mantenimiento ollama | sí |
| Quick win orden | rename+zeroize (15min) -> Landlock (2h) -> GBNF empírico (1h) -> TUI/embedding (1h) en paralelo con bench noche 76min+1h | bench no bloquea quick wins, P0s primero |

## Findings (cited - path:lines)
- Audit CHANGES REQUESTED W0 PASS only (audit.md:15), W1 sandbox FAIL P0 sandbox.rs env+println stub 77-88 98-121 test kill -9 193-234 (audit.md:52-77), W4 GBNF FAIL P0 simulated 98% method simulated forced 98 if <95 (audit.md:117-125), W5 JSON 225/225 lie 15 trials extrapolated 225/225 195/225 89/89 12/89 (audit.md:136-144), W2 zeroize missing OnceLock Mutex without zeroize (audit.md:98-100)
- Kitty current: sandbox.rs 253L stub + apply_linux_stubs, llama.rs 392L LlamaCppClient Arc<Mutex Option<Inner>> unsafe Send/Sync try_load vocab_only fallback Ollama no GGUF, cloud.rs 578L OnceLock Mutex<String> no zeroize, tui.rs 745L streaming 14 tests, bm25 2.3 k1 1.2 b 0.75 RRF K60, embedding fastembed 384 88M opt-in, bench JSONs smoke 15/15 + 5/5 Pass@5 single task, clippy 0 308 tests 11117 LOC (F1 fix)
- Apohara-codesearch: crates/apohara-indexer FTS5 bm25 + sqlite-vec vec0 + blake3 384d FeatureHashEmbedder determinístico zero deps vs opt-in GgufEmbedder candle Bert/Gemma3 300M bidirectional Matryoshka 768->256, tree-sitter 9 langs chunker 60/15, hybrid RRF K60 MMR 0.7 offline 22MB/224k LOC (local /tmp/apohara_scan)
- Apohara-dekanus: crates/dekanus Phase 0 skeleton candle 0.11 cudarc 0.19 glommio mmap2 z3, custom CUDA kernels airllm-kernels BF16 sm_75 quirks, layer_streaming + selective MoE top-k + early_exit, Qwen3-8B dense -> 30B-A3B MoE -> Ornith-397B-A17B layer-streaming prefetch, cargo-dist SLSA L3 deny.toml, HONESTY ledger (local /tmp/apohara_scan, no /Repos)
- Little-coder: Node >=22.19 ESM pi 0.83.0 + 30 extensions 30 skills, permission-gate allow-list (ls cat git), write guard cat> blocked -> must use Write/Edit, read-guard, checkpoint snapshots, context-watchdog 80% (Proyectos/Repos/little-coder/package.json, bin/little-coder.mjs)
- Smallcode: Node >=18 CJS 1570L bin/smallcode.js, tmpdir per-task fs.mkdtemp + detached kill(-pid), sanitize safeResolvePath cwd-containment SSRF, bench/harness.js smoke 5 + polyglot-mini 19 + tool-use 10 tmp workspace verify (Proyectos/Repos/smallcode/package.json)
- Apohara-Catalyst prior art (~/Proyectos/Repos/Apohara-Catalyst, 37 crates): apohara-sandbox `profile/linux.rs:1-18` seccomp-bpf EPERM fail-closed + conditional openat/fcntl/ioctl/clone + `namespace.rs:12-29` CLONE_NEWUSER|NEWNS|NEWPID + double-fork `runner/imp.rs:110` MAX_OUTPUT 16MiB + env sanitize `is_secret_env_name:124` strip *_API_KEY; apohara-secrets `SecretString:56-94` Drop zeroize + Debug redact `SecretString(***)` + keyring 3.6; apohara-audit `O_NOFOLLOW+fchmod 0600` JSONL `EventKind 16` + daily+64MiB rotation; apohara-episodic cross-run `~/.apohara/episodes/` blake3+sqlite-vec; apohara-dispatch `task_graph.rs:193-236` Kahn cycle + `reap_stale_claims:256` /proc pid 22; TrustLayer no existe, GBNF no existe, honesty ledger planned not live - reusable: SecretString, seccomp+namespaces, audit sink, DAG reaper, episodic store.
- EXA implicit: Landlock crate 0.4 + zeroize SecretString + llama_sampler grammar_lazy triggers {"name"} evita server-common.cpp:400, cargo deny SLSA (Catalyst deny.toml license allowlist), Catalyst no usa Landlock (solo seccomp) - audit P0 exige Landlock como capa extra

## Decisions (with rationale)
- No pivot: Kitty ocupa celda vacía Aug 2026 Rust single-binary 9B/8GB con Harbor oficial (little-coder Node pi, smallcode Node CJS tmpdir, ninguno Rust Landlock ni oficial bench)
- Mantener Ollama: own engine (Dekanus) coste 12-17 tok/s aspiracional skeleton vs ollama 0 mantenimiento; abstraer ChatClient ya existe para swap futuro candle sin lock-in
- Robar de Codesearch: hybrid FTS5+vec en un SQLite por repo, blake3 384D hash fallback, tree-sitter 9 langs, RRF/MMR, pluggable Embedder id/dim versionado
- Robar de Dekanus solo si MoE 397B requiere sparsity: layer_streaming pinned_buffer + FWHT 3-bit KV quant + selective early_exit + llmlingua2 prompt compression
- Wasmtime rechazado como gate (audit:179), static plugins más seguros (duplicate_blocked must exist)
- Quick wins P0 en paralelo con bench noche (bench no bloquea fixes, fixes no bloquean bench)
- Catalyst reusable: copiar verbatim `crates/apohara-secrets/src/lib.rs:56-94 SecretString` + `keyring 3.6`; portar `sandbox/profile/linux.rs:7` seccomp builder EPERM + `namespace.rs:52` unshare bundle + `runner/imp.rs:190` double-fork/env sanitize `is_secret_env_name:124` + `apohara-audit` O_NOFOLLOW fchmod 0600 JSONL; no inventar Landlock (Catalyst no lo tiene) sino añadirlo como capa extra audit P0; `apohara-episodic` store WAL+busy_timeout 5s para memoria cross-run template; `apohara-dispatch task_graph.rs:256 reap_stale_claims` para bench noche claim GC si harbor adapter usa filesystem DAG

## Scope IN
- Night bench: harbor run aider-harbor -> bench_225_full.json real (mean 225) + TB2 89 mean, rename smoke artifacts bench_polyglot_oracle_smoke15.json bench_tb2_oracle_ars_k5.json n_total == array.len() no total:225 lie
- P0 rename: corregir 4 JSONs + docs/AUDIT.md CNAS.md FLOOR-BENCHMARK.md headlines
- P0 zeroize: `apohara-secrets` SecretString verbatim + `zeroize` + `keyring 3.6` + Debug redaction test `format!("{client:?}")` no key + cost log sin Authorization + `is_secret_env_name` strip *_API_KEY (Catalyst runner/imp.rs:124)
- P0 Landlock+seccomp: `seccompiler 0.5` EPERM + `landlock 0.4 RestrictSelf` + `nix CLONE_NEWUSER|NEWNS|NEWPID` + rlimit RLIMIT_AS 512MB + double-fork `MAX_OUTPUT 16MiB` + test /etc/passwd deny + /etc/shadow + SIGKILL 2GB + `apohara-audit` O_NOFOLLOW fchmod 0600 ledger + Miri Send/Sync
- P0 GBNF empirical: benches/ab_gbnf.py --with-model GGUF o marcar simulated y cerrar ADR not empirically closed
- QW TUI: TestBackend snapshots CJK empty-q history draft
- QW embedding e2e: episodes.db real + CI ab_recall limit 15 allow worse than lexical
- Vision doc: Must Have list + own inference worth vs Ollama + apohara patterns + director vision pivot competencia smallcode/littlercoder
- Docs hygiene: no bench/official edit, clippy 0, 308 tests

## Scope OUT (Must NOT have)
- Wasmtime WASM plugins como merge gate (static registry stay)
- Custom grammar + tools HTTP (server-common.cpp:400) - usar grammar_lazy
- Entrenar modelo, GGUF 397B, 17GB Vram, cambiar Ollama ahora por Dekanus completo (Phase 0 skeleton)
- Homepage/docs site, push a GitHub, OpenSSF scorecard full (higiene separate)
- Crecer exercises_polyglot.json con 3 toys train-on-test
- Fabricar 98% o extrapolar 15->225 con nombre 225

## Open questions
- GGUF path para GBNF empírico: ~/.kitty/models/ornith.gguf existe? Si no, descargar 5GB Q4_K_M (~30min) o marcar simulated? Default: intentar download con hf-hub si no existe, si falla cerrar ADR como not empirically closed sin bloquear merge
- Landlock abi: /proc/self/status check requiere kernel 5.13+ CachyOS 6.x ok, si no fallback a test skip + doc
- Bench kitty adapter: harbor no tiene kitty agent (solo oracle/nop) -> implementar adapter shell que invoca kitty-cli run con --model ornith:9b dentro container? Default: adapter simple bash que llama kitty

## Approval gate
status: approved 2026-08-21
approved_by: user "Apruebo f6-p0-quickwins-nightbench actualizado con Catalyst seccomp+SecretString+episodic + bench noche"
next: high-accuracy dual review auto (UNCLEAR) then handoff to Atlas $start-work f6-p0-quickwins-nightbench
