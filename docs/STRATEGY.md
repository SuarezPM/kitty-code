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

### F6.4 Grammar enforcement — GBNF spike (ADR-005: llama.cpp GBNF real adapter — defer spike)

**Status:** Accepted (spike defer) · **Date:** 2026-08-19 · **Scope:** `crates/kitty-runtime/src/grammar.rs`, `crates/kitty-runtime/Cargo.toml`, `Cargo.toml`

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
