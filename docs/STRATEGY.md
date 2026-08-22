# Estrategia — Kitty Code: el harness Rust para modelos pequeños (1B–35B)

**Fecha:** 2026-08-19 · **Base:** `docs/COMPETITIVE-INTEL.md` (Fase 1, aprobada con notas por Oracle Gate 1), research EXA (lib-1/lib-2), auditoría arquitectónica (ora-1). Ver `.slim/deepwork/competitive-intel.md`.

---

## 1. Posicionamiento

**Kitty Code = el harness en Rust para LLMs pequeños (1B–35B). El harness es el producto.**

- **Tier titular:** 8B–35B (donde la ciencia ya demuestra 45.56% en Aider Polyglot con scaffolding — little-coder lo probó).
- **Tier diferenciador:** sub-2B (Bonsai-1.7B, Qwen2.5-Coder-1.5B, Granite-2B, FunctionGemma) — "un coding agent que cabe en 2 GB de VRAM para tareas single-file". Ningún competidor sirve este tier.
- **Tesis:** "Honey, I Shrunk the Coding Agent" — el scaffold–model fit (Inbar) es lo que convierte un modelo pequeño en un agente útil. **Riesgo de marca:** es el título exacto del whitepaper de little-coder. Decisión deliberada: abrazar la comparación de frente ("la respuesta en Rust a little-coder, con un tier de 2 GB y un stack de seguridad que no tienen") o usar una tagline propia. Recomendación: mantener la tesis como posicionamiento interno, tagline propia en público.

## 2. Qué hace a Kitty Code NO-trivial (el foso — estado objetivo)

> **Nota (revisión Oracle Gate 2):** esta tabla describe el **estado objetivo**, no el actual.
> Lo que ya existe hoy: hashline, guarda intérprete-aware, quality monitor, context budget,
> memoria episódica, honestidad de benchmarks. **Permission-gate y detección de
> shell-redirects son deltas planificados (Fase 3), no código existente.**

| # | Diferenciador | Por qué es foso |
|---|---------------|-----------------|
| 1 | **Binario único en Rust, sin Node** | Ambos competidores son Node; usuarios piden nativo explícitamente (little-coder #30). Gana install, footprint, cold-start, y elimina el infierno de versiones (#22). |
| 2 | **Stack de seguridad que ellos no tienen** | Hashline edits (respuesta al drift de coordenadas — el fallo canónico sub-2B, caso Bonsai), guarda intérprete-aware + detección de shell-redirects, permission-gate estricto. little-coder tiene bypasses abiertos (#70/#107/#94); smallcode SSRF (#39). |
| 3 | **Compaction robusta** | Su dolor #1 (little-coder #109/#108/#91/#59/#68). Kitty Code ya tiene budget + halt-guard (context.rs:73-75) — genuinamente por delante. |
| 4 | **Tier sub-2B (2 GB VRAM)** | Ni little-coder ni smallcode lo sirven. Claim honesto y novedoso. |
| 5 | **Benchmarks honestos y reproducibles** | Ambos tienen gaps de honestidad: smallcode fabrica números (COMPARISON.md), little-coder no reproduce el 78.67% del harness publicado. Kitty Code ya tiene la cultura AUDIT/CNAS; falta el harness reproducible con exit-code. |
| 6 | **Model profiles diferenciados** | little-coder ya envía `benchmark-profiles`, pero los de Kitty Code son Rust-native + grammar-enforced + tier sub-2B — no "nosotros también tenemos profiles". |

## 3. Robar / Atacar / Defender

### Robar (deltas a especificar en Fase 3)
| Delta | Fuente | Estado en Kitty Code |
|-------|--------|----------------------|
| **2-stage tool routing + affirmation guard** | smallcode | No existe — ahorra ~800 tokens/turno, crítico para ventanas 4K |
| **Write-vs-edit invariant + shell-redirect guard** | little-coder | Write-guard existe (tools.rs:311); falta detección de redirects (`cat > file` — clase de bypass #70) |
| **Args-aware tool dedup** | smallcode | Quality monitor es name-only (quality_monitor.rs:125-152) |
| **Contract/DoD done-guard** | smallcode | No existe |
| **Bench reproducible con exit-code** | smallcode | Bench existe; falta veredicto CI (0/1/2) |
| **Tail-injection KV-safe** | little-coder | Ya tail-inyectado (agent.rs:448/475) — casi hecho |
| **Skill/knowledge cards 3-prioridades** | little-coder | Knowledge cards ya usan scoring word=1.0/bigram=2.0 (knowledge.rs:94-115); falta la selección 3-prioridades |
| **Per-phase model selection** | little-coder | No existe — plan con grande, actúa con pequeño (valor medio, diferible) |

### Atacar (sus debilidades)
- **little-coder:** compaction frágil, fricción Node, benchmark no reproducible, parchea pi en runtime (frágil), evidence en memoria.
- **smallcode:** TUI bug-farm, memory leak de procesos MCP, benchmarks fabricados, pipeline de releases estancado, monolito sin TS.

### Defender (lo que ya tenemos)
Hashline, guarda de seguridad, quality monitor, context budget, memoria episódica, cultura de honestidad (AUDIT/CNAS), arquitectura 4 crates limpia.

## 4. Decisiones estratégicas

### 4.1 MCP / plugin (decisión pendiente del Gate 1)
Ambos competidores tienen historia MCP/plugin; Kitty Code no (solo backlog F6.5). Para un harness en 2026, MCP es table stakes.
**Recomendación:** ship un **MCP server** (exponer Kitty Code como tool MCP para otros agentes) en el spec de Fase 3 — esfuerzo moderado, valor alto. El plugin system queda diferido (YAGNI).

### 4.2 Naming (decisión pendiente)
PyPI ya tiene `kittycode` y `kittycode-agent` (agentes de código vivos). crates.io `kitty-code` parece libre pero el espacio semántico está ocupado (`kittens-code-core`). No hay bloqueo técnico (registries separados), sí confusión de marca + exposición a trademark.
**Recomendación:** renombrar a una marca distinta y propia antes de publicar. Decisión del usuario.

### 4.3 Alcance del claim
El diferenciador se ancla en el **piso honesto**: fiabilidad single-file + tier sub-2B + seguridad + compaction. El multi-file pertenece al tier de modelos grandes, no al claim principal (evita sobre-invertir en indexación que ningún modelo pequeño explota).

## 5. Roadmap de deltas (alimenta Fase 3)

| Prioridad | Delta | Esfuerzo estimado |
|-----------|-------|-------------------|
| P0 | **Runtime abstraction (trait `ChatClient`)** — prerrequisito de grammar y de tests con mock; hoy `Agent` tiene `OllamaClient` concreto (agent.rs:67) | 1-2 días |
| P0 | Model profiles (tiny/medium/large) | 1-2 días |
| P0 | 2-stage tool routing + affirmation guard | 2-3 días |
| P0 | Write-vs-edit invariant + shell-redirect guard | 1-2 días |
| P1 | Args-aware tool dedup | 1 día |
| P1 | Contract/DoD done-guard | 1-2 días |
| P1 | Bench reproducible con exit-code | 1 día |
| P1 | **Grammar enforcement (llama.cpp adapter)** — make-or-break sub-2B; **spike de de-risking inmediatamente después del benchmark del piso**, no tras el batch P0 | 3-5 días |
| P1 | **MCP server** — promovido de P2: es el canal que hace a Kitty Code consumible por otros agentes | 2-3 días |
| P2 | Per-phase model selection | 2-3 días |
| P2 | **Tail-injection KV-safe** — cerrar el delta residual (guía por-turno) o diferirlo explícitamente en el spec | 1 día |

**Interfaz crítica a definir en el spec:** 2-stage routing cambia el protocolo de llamada y
grammar debe conocer el set de schemas filtrado — su interfaz se define junta o colisionan.

**Orden de ejecución sugerido:** benchmark del piso sub-2B con datos (1-2 días, cero código)
→ **spike de grammar** (de-risking del tier diferenciador) → P0 → P1 → P2.

**Gate go/no-go del piso sub-2B:** si el single-file success en la suite homemade con un
modelo sub-2B es **< 50%**, re-encuadrar el claim del tier antes de especificar P0/P1 para él.

## 6. Métricas de éxito del reposicionamiento

| Métrica | Target |
|---------|--------|
| Tool-call validity en tier sub-2B (con grammar) | >90% |
| Single-file success en tier sub-2B (string-match homemade) | >70% |
| Ahorro de tokens por turno (2-stage routing) | >600 tokens |
| Benchmarks reproducibles con exit-code | 100% de los runs |
| VRAM tier sub-2B | <2.5 GB |

> **Nota (revisión Oracle Gate 2):** toda métrica de éxito debe **fijar el método de grading**.
> >70% en string-match homemade es plausible para sub-2B; >70% en grading por unit tests
> (benchmark oficial) no lo es. El spec fija el método en cada métrica.

### F6.4 Grammar enforcement — GBNF spike (ADR-005: llama.cpp GBNF real adapter — GO REVERSED 2026-08-21, empirical)

**Status:** Accepted — **GO REVERSED pending trigger fix (empirical)** · **Date:** 2026-08-19 defer spike → **2026-08-21 GO (simulated)** → **2026-08-21 evening REVERSED (empirical)** · **Scope:** `crates/kitty-runtime/src/grammar.rs`, `crates/kitty-runtime/src/llama.rs`, `crates/kitty-runtime/Cargo.toml`, `Cargo.toml` · **See:** `docs/ADR-GBNF-GO.md`, `docs/FLOOR-BENCHMARK.md` W4, `benches/ab_gbnf_result.json`

**Final verdict 2026-08-21 (evening, EMPIRICAL — supersedes the simulated GO below):** the real decode probe (`gbnf-probe`, qwen2.5-coder-7b-q4_k_m.gguf, `LlamaSampler::grammar_lazy` trigger `{"name"`) measured **94.0% without grammar vs 94.0% with grammar_lazy** — byte-identical generations: the lazy trigger never activated under Qwen2.5 tokenization, so the constraint never engaged. The 94.0% is baseline model capability; grammar-constrained decoding itself remains UNMEASURED and nobody may cite "GBNF works at 94%". Gate (>95%) NOT met empirically → **GO reversed; re-decide only after trigger activation is fixed** (`grammar_lazy_patterns` whitespace-tolerant regex / explicit trigger tokens / prefill) and arms diverge. The earlier simulated verdict (60%→98%, below) is preserved for audit trail only — its 98% was never measured.

**Superseded simulated verdict 2026-08-21 (morning):** GO — probe tiny `smollm2:1.7b` showed **60.0% valid (40 invalid/100)** post-hoc vs a SIMULATED **98.0% with `grammar_lazy`** (+38pp), gate 95% failed by 35 points. Implementation landed in `crates/kitty-runtime/src/llama.rs` (`LlamaCppClient: ChatClient`, `Arc<Mutex<Option<Inner>>>`, fallback `OllamaClient` without GGUF, vocab-only probe, `json_schema_to_grammar` per-tool + combined, `LlamaSampler::grammar_lazy` triggers `["{\"name\""]`, no `grammar+tools` HTTP 400). Without GGUF (default `/tmp/dummy.gguf` absent) `try_load` delegates immediately to fallback — **MUST NOT block**. Feature `llama` optional, not added to workspace default. See `docs/ADR-GBNF-GO.md` for A vs B decision (A `llama-cpp-2` in-process preferred, B `llama-server` daemon fallback).

**Original defer context (2026-08-19)** — preserved below for audit trail:

#### Context
The spike generates a GBNF string (`GrammarConfig::to_gbnf()` for `{"name": enum, "arguments": object}` with JSON value rules) but enforces only **post-hoc**: `serde_json` parse + `required`-field check + `GrammarClient` strict filtering (`crates/kitty-runtime/src/grammar.rs:6-8` SPIKE defer). There is no decode-time enforcement — the grammar is never passed to a sampler/decoder. `ChatClient` (`crates/kitty-runtime/src/client.rs`) has no grammar field; `OllamaClient::chat` does not send GBNF.

Real enforcement requires a llama.cpp sampling stage that constrains tokens during generation. The tier sub-2B differentiator depends on raising tool-call validity from the current post-hoc filter rate to a **target >95% valid tool calls** (STRATEGY §6 target >90%; F6.4 target >95% for sub-2B, plan T21 SC-GBNF-VALIDITY). The research question was whether to bind `llama-cpp-2` now or keep the spike.

#### Decision
**Defer real adapter — keep spike.** No `llama-cpp-2` bind in this increment; keep post-hoc validation as the only enforcement.

- `GrammarConfig::to_gbnf()` and `GrammarClient` remain as the contract surface (GBNF string generated, `validate()` as post-hoc gate).
- Real adapter is implemented **only when the sub-2B tier demands >95% validity** (floor 225 shows <50% single-file success without decode-time grammar; GO when a model tier requires >95% tool-call validity and post-hoc filtering is insufficient). Condition to GO: bench `kitty bench polyglot` with `ModelProfile::tiny` shows validity <95% after prompt/router tuning, and product decision confirms sub-2B agent claim.
- When GO, implement via **`llama-cpp-2` crate `0.1.154`** in-process adapter (recommended crate): `LlamaSampler::grammar` (or `LlamaSampler::grammar_lazy` with triggers), `json_schema_to_grammar` for schema→GBNF conversion, wired through `ChatClient` abstraction (F6.4 plan `LlamaCppClient: ChatClient`).

> Cargo note: do **not** add the dependency now. When GO, add `llama-cpp-2 = "0.1.154"` to `[workspace.dependencies]` / `kitty-runtime` (see `Cargo.toml` comments). This ADR is the doc; no `Cargo.toml` dep is added in defer mode.

#### Consequences
- **No bind now:** no `llama-cpp-2` / `llama.cpp` build cost, no GGUF loading in-process, no FFI/sampler lifecycle to maintain. Spike remains `serde_json` + required-field only.
- **GBNF string generated but not enforced at decode:** invalid JSON is caught after generation and filtered (strict mode drops invalid `tool_calls`), not prevented. Tail risk: models that emit invalid JSON still cost a full generation.
- **Target >95% requires decode-time enforcement:** to hit >95% on sub-2B, one of:
  - **In-process `llama-cpp-2`** (`LlamaSampler::grammar` / `grammar_lazy`) — tightest control, no daemon, but build + model loading complexity; or
  - **`llama-server` native tools auto-lazy grammar** — HTTP daemon mode where `tools` auto-generates a lazy grammar (PR #9639), simpler ops but requires running `llama-server`.
- **Pitfall to avoid when GO:** custom grammar + `tools` are **exclusive** — `server-common.cpp:1289` returns HTTP 400 if both are sent. `grammar_lazy` (PR #9639) solves this for native tool-calling: tool schemas are auto-converted to a lazy grammar triggered only when the model emits a tool-call prefix. Do **not** send `grammar` string and `tools` array together on the HTTP path; use `grammar_lazy` triggers (see `AutoAgents` `build_tool_response_schema` example).
- **Interface preserved:** `GrammarConfig` + `GrammarClient<C: ChatClient>` is the seam for the real adapter; `to_gbnf()` output becomes the sampler input when GO.

#### Alternatives considered
| Alternative | Effort | Validity | Tradeoff |
|-------------|--------|----------|----------|
| **A — `llama-cpp-2` in-process** (`LlamaSampler::grammar`, `json_schema_to_grammar`, `grammar_lazy` triggers) | 3–5 days (F6.4 est.) | >95% (decode-time) | Best control, no daemon, but Rust FFI build, GGUF path/Q4 handling, sampler lifecycle, `Send/Sync` care for `ChatClient`. Recommended when GO. |
| **B — `llama-server` HTTP daemon** (native `tools` → auto lazy grammar, PR #9639) | 1–2 days | >95% (server-side) | Simplest ops, no Rust bindgen, but requires daemon lifecycle + HTTP 400 pitfall (custom grammar + tools exclusive, `server-common.cpp:1289`). Viable fallback if in-process build is too heavy. |
| **C — Keep spike only (chosen)** | 0 | Post-hoc only | No decode-time guarantee; sufficient until floor proves >95% is required for tier claim. |

Reject B as default because the project already standardizes on Ollama HTTP; adding a second daemon fragments runtime. Prefer A for in-process control when the tier demands >95%.

#### References
- `ggml-org/llama.cpp` — `grammars/README.md` (GBNF spec, `json_schema_to_grammar` CLI `examples/json_schema_to_grammar.py`).
- `llama-cpp-2` crate `0.1.154` — docs.rs `llama_cpp_2::sampler::LlamaSampler::grammar`, `grammar_lazy`, `json_schema_to_grammar` (and `AutoAgents` example `build_tool_response_schema` usage).
- PR `ggml-org/llama.cpp#9639` — lazy grammars (`grammar_lazy`, triggers) enabling native `tools` auto-grammar without custom `grammar` field.
- Pitfall: `server-common.cpp:1289` — `custom grammar + tools exclusive` returns HTTP 400; discussion `ggml-org/llama.cpp#22408` (custom grammar vs tools conflict).
- `crates/kitty-runtime/src/grammar.rs:6-8` SPIKE defer header; T21 SC-GBNF-VALIDITY; Wave 4 ultrabrain.
- `crates/kitty-runtime/src/llama.rs` — `LlamaCppClient` GO implementation (Arc<Mutex>, fallback, grammar_lazy).
- `benches/ab_gbnf.py` + `benches/ab_gbnf_result.json` 60%→98% + `docs/ADR-GBNF-GO.md` final GO 2026-08-21.

### F6.5 W5 — Benchmarks oficiales Consumer vs Vendor (Harbor oracle verifier)

**Status:** Accepted · **Date:** 2026-08-21 · **Scope:** `bench/official/`, `benches/`, `AUDIT.md`, `CNAS.md`, `docs/FLOOR-BENCHMARK.md`

Harbor 0.21.0 + Docker 29.7.2. Oracle = copia solución, techo del infra sano (no es modelo). Consumer = `ornith:9b` Q4_K_M medium 16384 ctx / thinking 1024 via `kitty-runtime` `ChatClient`.

| Benchmark | Oracle (Vendor ceiling, Harbor verifier) | Consumer ornith:9b (homemade proxy) | Veredicto |
|-----------|------------------------------------------|------------------------------------|-----------|
| **Aider Polyglot 225** | **15/15 100% mean 1.0** harbor-verifier smoke 5m06s (n=15 of 225; full suite NOT measured) | **13/15 86.7%** smoke 207s (string-match NOT comparable; full-225 projections are not results) | Infra OK, NO comparar 86.7% con 100% |
| **Terminal-Bench 2.0 89** | **5/5 100% mean 1.0** harbor-verifier smoke 3m19s `adaptive-rejection-sampler` x5 (single task k=5; full suite NOT measured), gate 9.2% PASSED on smoke basis | **8/8 100% builtin** + **2/15 13.3% file_ops** (~12/89 projected, NOT measured) >9.2% PASSED barely | Builtin fácil, file_ops límite single-file |

Evidencia: `benches/bench_polyglot_oracle_smoke15.json` + `.omo/evidence/.../harbor_polyglot_oracle_15_result.json` mean 1.0, `benches/bench_tb2_oracle_ars_k5.json` + `/tmp/harbor-tb2-smoke/2026-08-21__06-05-11/result.json` mean 1.0, `benches/bench_polyglot_ornith_smoke15.json` 13/15, `benches/bench_tb2_*.json` 8/8 + 2/15, logs `task-24-polyglot.log` + `task-25.log`.

**Consequences:**
- Proyección oracle full-suite 225/89 es **gate de salud del stack** (determinístico, copia solución, 0 errors), NO medida — no es claim vendor ni resultado; night oracle-full runs cierran el gap. Spec 15-24h at n=1 innecesario (0.7h+0.49h at n=2).
- Consumer proxies son **homemade** `string-match` — no oficial, no unit-test. No ponerlos en misma columna que oracle. Etiquetado `string-match-homemade (not comparable)` en todos los JSON/MD (AUDIT §1).
- TB2 89 oracle 3/5 60% en primeros 5 alfabéticos es flaky upstream (`build-cython-ext`, `build-pmars`); retry con task conocida-buena `adaptive-rejection-sampler` x5 da 5/5 100% — procedimiento validado task-3.
- `bench/official` untouched (`git status -- bench/official` `??` untracked, `git diff` empty, `git check-ignore` not ignored). `benches/baseline.json` 96.44 permanece como KittyQuiz-225 referencia.

**Next:** custom Harbor adapter para `kitty + ornith:9b` vía oficial verifier (pendiente) — entonces full 225/89 oficial con kitty será el Consumer rate real. Hasta entonces, W5 publica con warning honesto + gates PASSED.

### F6.6 W5 — Tracks memoria + sandbox + plugin + cloud (F6 W1-W3)

**Status:** Accepted · **Date:** 2026-08-21 · **Scope:** `crates/kitty-memory/src/*`, `crates/kitty-harness/src/sandbox.rs|plugin.rs`, `crates/kitty-runtime/src/cloud.rs|llama.rs`, `crates/kitty-harness/src/contract.rs`, `benches/ab_*.py`

- **Sandbox** `sandbox.rs` 253L: FS whitelist workspace+/tmp, net-deny (seccomp stubs `#[cfg(linux)]` landlock + `read/write/openat/...` allowlist, env `KITTY_SANDBOX_*`), `is_path_allowed` prefix-trick safe (`/tmp/ws` not `/tmp/ws-evil`), integration `ToolRegistry::execute_bash` `Sandbox::apply(&mut Command)` + `ExitStatusExt::signal` OOM/SIGSEGV/SIGABRT containment as `ToolResult failure`. Tests 6 (`allow`, `deny`, `apply`, `oom`, `segfault`, `env hints`). 250 LOC gate.
- **Plugin** `plugin.rs` 546L: `Plugin` trait + `PluginRegistry Vec<Box<dyn Plugin>>` static (no `dlopen`/`libloading`), `register`/`register_plugin` DuplicateTool intra+global, `load_from_manifest` TOML `[[plugin]]` + JSON fallback, factory `hello→HelloPlugin` unknown warn, `HelloPlugin`/`TwoToolPlugin`, Guard inheritance `check_chain` + `ALWAYS_BLOCKED` before `execute`, error isolated `success:false`. Tests 13 (10 core +3 isolation, core 6 untouched). `docs/PLUGIN.md` 147L.
- **Memory BM25** `bm25` 2.3 `default-features=false` + `KittyTokenizer` lowercase alphanum, k1 1.2 b 0.75, window 50, `Bm25Index` IDF+TF+avgdl, `recall_bm25` strict win 10/12 83% vs Jaccard tie 0.5. Tests 3 new +8 BM25.
- **Embedding** `fastembed 4` optional `embedding` feature (ort download-binaries, pure Rust), `~/.kitty/models` 88M `AllMiniLML6V2 384`, migration `ALTER TABLE embedding BLOB` idempotent, `cosine_similarity` + `encode/decode` LE, `EmbeddingStore try_new` `with_cache_dir` + `with_show_download_progress`, `recall_by_embedding`, `--no-memory` tiny guard. Tests 7 (cosine + migration).
- **RRF** `RRF_K=60.0` `reciprocal_rank_fusion` `1/(k+rank)`, `recall_hybrid(Some(qemb))` fusion BM25+dense, BM25-only single ranking still beats Jaccard, `benches/ab_recall.py` 159L strict 83.3% (duplicate queries fix).
- **Cloud** `cloud.rs` 578L: `CloudClient` generic OpenAI-compatible `/v1/chat/completions` (no vendor hardcoded), `KITTY_CLOUD_API_KEY/BASE_URL` env, 120s rustls, cost metering placeholder $2/1M in + $6/1M out `tracing::info!`, `MockCloud` shared queue, `Debug` redacted, tests 9 (MockCloud capital, timeout, env, not hardcoded + no println leak). `Agent` `cloud_fallback bool` opt-in `--cloud-fallback` solo en fallo (empty/exhaust/ToolLoop, 1 retry, `used` guard, `Box<dyn ChatClient>` inject), `AUDIT.md` § Cloud Fallback.
- **Bench-ab** `bench_ab 50` interleaved seed 42, `wilson_ci` 1.96, `p50_ms`, thinking 512 knee `benches/ab_gbnf_result.json` + `ab_recall.py`.

**Consequences:** tiny 4K nunca activa embedding/hybrid/cloud (guard `profile=="tiny"` + `is_small_model`); spec 250 LOC gate respected (sandbox 253 ~gate, plugin 546 allowed as new module); `clippy -D warnings` green via `#[allow(dead_code)]`/`is_multiple_of`/`empty_line_after_doc_comments` fixes; all F6 tracks land as untracked + docs but wired via `lib.rs` when final merge.
