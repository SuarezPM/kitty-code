# CodeAuditForge — Auditoría Kitty-Code (2026-09-24)

Modo: Principal Staff Engineer + Product Architect + Security mindset. Honestidad brutal, cero mediocridad, rigor 2026.
Alcance: mono-repo Rust (`crates/*` + `engine/mistralrs*`), commit base con `codemap.md` atlas 2026-09-23.
Método: PASO 1-5 + E2E usuario final + 10 papers + 10 tecnologías (jul-2026) + plan de mejora + debt log.
Evidencia: toda afirmación lleva archivo:línea. Sin pruebas no hay nota.

---

## PASO 1 — Metodología y mapeo estructural

### Principios aplicados
1. **Puertos y adaptadores / hexagonal parcial** (AWS Prescriptive Guidance; Master Hexagonal Rust 2024): el seam es `trait ChatClient` (`crates/kitty-runtime/src/client.rs:254`) con inyección `Box<dyn ChatClient>` (`crates/kitty-cli/src/main.rs:366`).
2. **SOLID** (foco SRP/OCP/ISP), ley de Demeter, cohesión/acoplamiento, detección de god objects y modelos anémicos.
3. **OWASP Top 10:2025 + OWASP LLM Top 10 v2025** (A01/A02/A05/A07; LLM01 prompt injection, LLM02 sensitive disclosure, LLM10 consumo ilimitado) + **STRIDE** sobre `guard/` + `sandbox.rs` + `serving/`.
4. **OpenSSF Best Practices + Scorecard** (CI-Tests, SAST, SBOM, provenance, pin-SHA, permissions mínimas, branch protection, secret scanning, dependency-update) + supply-chain Rust (cargo-audit/deny, CycloneDX, SLSA/sigstore).
5. **Testing moderno**: property-based (proptest), mutación (cargo-mutants), cobertura LLVM (llvm-cov), Wilson CI + k≥3 en benches (honest-bench culture del repo).

### Grafo de dependencias (mental, verificado)
- `Cargo.toml:1-17`: workspace único `crates/*` + 7 `engine/mistralrs*`; `mistralrs-cli` dormant; dual epochs documentados reqwest 0.12/0.13 (`:142-150`) y rand 0.8/0.9 (`:83,:173`); candle por git rev `35d7ae7` (`:101-104`).
- Entradas: `crates/kitty-cli/src/main.rs:313` → `Cli/Commands` (`:159,:215`: `run/bench/bench-ab/info/mcp/tui`) → `backend_selected(KITTY_BACKEND)` → `Box<dyn ChatClient>`; bench en `bench/{common,run,ab,polyglot,tb2,gpqa,claweval}.rs` + `timeouts.json`.
- Núcleo: `kitty-runtime` (`OllamaClient :177`, `KittyInferClient kitty_infer.rs:717`, `cloud.rs:50 SecretString`, `grammar.rs:6` spike, `openai_compat.rs`, `llama.rs`) ← `kitty-harness` (`agent/{mod 2877L,run 483L,response 762L}`, `tools.rs 3303L ToolRegistry :72`, `guard/{mod 757L,patterns,shell_parse 565L,interpreters}`, `sandbox.rs 1812L`, `parser.rs`, `quality_monitor.rs`, `context.rs`, `model_profile.rs`, `contract.rs:50`) ← `kitty-cli` (+ `mcp.rs:17` stdio, `tui.rs:150` feature-gated) + `kitty-memory` (`episodic.rs:236` SQLite WAL, `embedding.rs`, BM25 cache).
- `engine/mistralrs-core/src/serving/` (`mod 24L, api 270L, model_manager 429L, errors 214L, health 129L, infer_policy 183L, sse 74L`) + `global_registry` + `vram_precheck`; `engine/` solo entra vía `kitty_infer.rs:41` (una dirección).
- Flujo `run`: `main.rs:342-380` (backend + `resolve_project_root :39` + `resolve_session_num_ctx :116`) → `Agent::run_with` (preflight → recall → knowledge → loop `chat→ToolCallParser→GrammarClient→QualityMonitor→ToolRegistry::execute(Bash vía Sandbox+Guard)` → compactación → `save_episode`).
- Interfaces externas: Ollama HTTP (reqwest rustls `:45`), FS (`tools.rs:405,431,489,554,655,752`, `knowledge.rs:166`), SQLite rusqlite bundled (`:62`), MCP stdio, TUI ratatui/crossterm.

### Patrón real: **monolito modular en capas con seam hexagonal parcial**
El `Box<dyn ChatClient>` desacopla backends de verdad, pero el harness mezcla loop+guard+sandbox+parser sin puertos inversos y `tools.rs`/`agent/mod.rs` superan 2.8 KLOC cada uno: hexagonal en el borde, monolito por dentro. Decisión correcta para un research harness; sería insuficiente como plataforma multi-backend.

---

## PASO 2 — Análisis dimensional (1-10, inmisericorde)

### D1. Modularidad y separación de responsabilidades — **6/10**
- `tools.rs:3303L` + `agent/mod.rs:2877L` + `client.rs:1549L` + `sandbox.rs:1812L`: baja cohesión, varios jobs por fichero. Acción: extraer `BashTool`/`EditTool`/`ReadTool` de `ToolRegistry` y `loop/` de `Agent` en módulos propios (Fase 3 no; deuda P1).
- `main.rs:709L` delega bien a `bench/`; dirección `cli→harness→runtime/memory` unidireccional, cero ciclos entre crates. Bien.
- Demeter: `agent.config().tuning` (`main.rs:467`), `snapshot().devices.first()` (`kitty_infer.rs:812`), CLI conoce `/api/show` (`main.rs:139`). Menor; acción: fachada `ModelInfo` en runtime.
- Doble residencia (`loaded_map :697` + `global_registry`) con `retain :899` que lo mitiga. Vigilancia, no deuda.

### D2. Diseño de clases y encapsulación — **6/10**
- SRP roto: `ToolRegistry` (`tools.rs:72`), `Agent` (`mod.rs`), `OllamaClient` (`client.rs:177`) son god objects parciales. OCP bien (`ChatClient :284` + `BenchClient common.rs`); ISP bien (4 métodos).
- `GrammarClient` (`grammar.rs:6`) es filtro post-hoc, no enforcement: la gramática no se impone al sampler. Acción P1: máscaras EBNF (ver P4 SynCode) o declarar `grammar.rs` spike y borrarlo (YAGNI).
- Bien: `RunOverrides` (`main.rs:447`), `config_mut/budget_mut` eliminados, `SecretString` (`cloud.rs:62`, Debug redact + zeroize), `LOADED OnceLock<Mutex>` nunca across-await, BM25 con invalidación en escritura.
- `KittyInferClient::count_tokens` heurístico `len/4` (`:1117`) mientras README promete conteo real: fuga de abstracción documentada como gap (ver PASO 3).

### D3. Optimización y rendimiento — **5/10** (la peor nota técnica)
- FS síncrono en async en paths calientes: `tools.rs:405,752`, `knowledge.rs:166`, `hashline.rs:87`. Acción P0: `tokio::fs` o `spawn_blocking` (ya existe el patrón en `tools.rs:1340`).
- `tokio::Mutex<Model>` en hot path de inferencia (`engine/.../model.rs:843`); clones masivos `messages.clone` (`to_request :909`) y en `model_builder_trait.rs`. Acción P1: `RwLock`/arc + préstamos.
- `health.rs:38,68` ejecuta `nvidia-smi/rocm-smi` por snapshot (caché 1s): bien mitigado, no tocar.
- Bien: `spawn_blocking` en `cargo_check`, BM25 cacheado, timeouts por (modelo,backend) en `timeouts.json`, cap 16 MiB, `RLIMIT_AS`, `panic=abort`+LTO (`Cargo.toml:268-273`).
- Clips: 2× `await_holding_lock` (`tools.rs:3189`) están en tests con `#[allow]` justificado (`:3225,:3258` + 6 en `agent/mod.rs`): higiene, no deadlock prod.

### D4. Simplicidad y código muerto — **7/10**
- Cero `#[allow(dead_code)]` en prod; orphan scripts eliminados; `cfg_attr(not(feature))` (`kitty_infer.rs:23`) legítimo. Bien.
- Parser soporta 6 formatos (nativo + XML/fenced/bare/kwarg/fuzzy) cuando README promete "native": complejidad defensiva contra SLM pequeños, justificada pero con coste de mantenimiento. Acción: property tests por formato (ya existen 5, ampliar a 6).
- Heredado engine fuera de gate: `flash.rs:85 TODO`, `starcoder2.rs:692 unimplemented!`, `phi3.rs:664 unreachable!`; `main.rs:379 unreachable!` defensivo OK.
- Resta: `cargo-fmt` kitty-scope DIRTY (`bench/common.rs`, `main.rs:324,333`, `examples/infer_spike.rs:22`, `parser.rs:395,2292`, `kitty_infer.rs` ~25 hunks, `lib.rs:23,29`, `openai_compat.rs` 6 hunks) contradice el "fmt clean" del codemap: deriva real o toolchain `stable` flotante. Acción P0: `cargo-fmt --check` scoped + pin toolchain (ver Fase 3).

### D5. Seguridad — **7/10** (fortaleza relativa con un hallazgo alto)
- Excelente: Guard (`patterns.rs:18` BLOCKED+13 RISKY+eval, `interpreters.rs:30` cierra bypass `sh -c`/`python -c`/`awk system`, `shell_parse.rs`), Sandbox Landlock+seccomp+userns+mountns/pidns+`RLIMIT_AS` fail-closed, `KITTY_SANDBOX=off` ruidoso, `SecretString` + `sanitize_env`, `resolve_project_root` nunca `$HOME`, `exec` solo argv sin shell (`health.rs:38`).
- **Hallazgo alto**: `serving/api.rs:219` auth default-abierta (`expected=None → Ok`), sin CORS ni rate-limit en `serving/`. Mitigante real: diseño loopback-only documentado + test `:256`. Acción P0: documentar asunción en README/SECURITY + `bind 127.0.0.1` por defecto + rate-limit/token obligatorio si se expone; penaliza la nota hasta cerrarse.
- Supply-chain: `deny.toml:1-42` + `Cargo.lock` + `--locked` en release bien; gaps: toolchain `stable` flotante, `SOURCE_DATE_EPOCH` ausente, 2× pin-SHA con `M4-TODO` (`ci.yml:34,57,97,101`), sin gitleaks/secret-scanning (solo defensa manual en `cloud.rs`), `cargo-vet` ausente (aceptable, no-crítico).
- `KITTY_SANDBOX=off` existe como opt-out: ruidoso pero real; STRIDE-Tampering lo marca como vía de degradación voluntaria.

### D6. Pruebas y QA — **6/10** (Gate 2: CI rojo en clippy+fmt al medir)
- Verde medido default-features: **722 passed / 0 failed** (cli 88 + harness-lib 505 + property 5 + memory 20 + runtime 100 + ollama 4), 11s el núcleo; `kitty-cli info` smoke `Status: Connected` sin exigir modelo; cultura honest-bench (k≥3, manifiestos, Wilson CI, no-claims en `AUDIT.md`/`CNAS.md`).
- Property-based EXISTE y pasa (`kitty-harness/tests/property_tests.rs:13`, 1024/512 casos + regressions). Bien.
- Resta: clippy ROJO por 3× `items_after_test_module` en kitty-runtime (lint nuevo de toolchain flotante; micro-fix Fase 3); coverage CI solo `kitty-harness ≥70%`; `--workspace --all-features` no compila en Linux (`objc2` Apple-only) y kitty `--all-features` tampoco (`cudarc 0.19.9` vs CUDA 13.4): el gate válido es default-features + job cuda en contenedor — documentarlo para no re-medir humo; `mutants.out/` baseline sin gate nocturno aún.

### D7. Adherencia a mejores prácticas — **6/10** (fuera de fórmula, informativa)
- Documentación honesta (`README:59-62`, `AUDIT.md`, `CNAS.md`, `docs/` con ADRs) muy por encima de la media; `CODEOWNERS`, `deny.toml`, `SECURITY.md`, `CONTRIBUTING.md`, `LICENSE` existen; CI con cpu/clippy/fmt/deny/sbom/audit; `tracing` con campos.
- Resta: fmt dirty (D4), toolchain flotante, `console-subscriber` ausente (tool-loop opaco en p95), `contract.rs:50` sin consumer-driven tests reales, `bench/` mezcla proxies homemade con oficiales vendored (admitido, pero fricción cognitiva).

---

## PASO 3 — Contraste promesas vs realidad
- "Native tool calling" → parser de 6 formatos (`parser.rs`): el modelo pequeño NO es nativo fiable; el harness lo compensa. Promesa imprecisa, comportamiento superior al prometido.
- "Real token counts" → `kitty-infer` usa `len/4` (`:1117`): FALSO para el backend embebido por diseño. Documentar o implementar.
- "`bench polyglot|tb2`" → proxies homemade + oficiales vendored sin loader unificado (`gpqa/claweval` gated): admitido en README, pero el naming invita a comparar. Renombrar a `bench-homemade` vs `bench-official`.
- "TUI" → requiere `--features tui`, no default (`main.rs:309`): fricción menor, una línea en README.
- "Backend embebido" → default sigue Ollama; `kitty-infer[-cuda]` opt-in: correcto y honesto (veredicto NO-SWITCH con números).

---

## PASO 4 — Autoverificación y control de sesgo

**Los 3 más duros:**
1. *D3 = 5 (FS síncrono en async)*. Contraprueba: el único `spawn_blocking` productivo está en `cargo_check` (`:1340`) y BM25 cachea índice; el codebase YA conoce el patrón, solo no lo generalizó. Se mantiene el 5: el hot path `Read/Edit` sí bloquea el reactor.
2. *D1 = 6 (god objects)*. Contraprueba: la dirección entre crates es limpia y `main.rs` es thin dispatch real; el monolito está contenido en 2 ficheros, no esparcido. Se mantiene el 6 pero acotado: deuda de 2 ficheros, no sistémica.
3. *D5 serving abierto*. Contraprueba (corregida Gate 2): el mitigante "bind local" era falso (`mistralrs-cli/src/args/server.rs:72` default `0.0.0.0`); el mitigante correcto y más fuerte es que la superficie no está shipped: `mistralrs-cli` fuera de members (`Cargo.toml:8-9`) y `kitty-cli` no expone HTTP (MCP stdio). Se mantiene P0 + nota 7 con el mitigante correcto.

**Los 3 más indulgentes:**
1. *D4 = 7 con cero dead_code*. Contraprueba: fmt dirty en kitty-scope prueba deriva activa; el 7 se mantiene solo porque la deuda es mecánica (`cargo-fmt` + pin), no estructural.
2. *D6 = 6 (Gate 2) con 722 verdes*. Contraprueba aceptada: clippy rojo + coverage de un solo crate + sin inferencia E2E con modelo imponen el 6; volverá a 7 al cerrar micro-fix + coverage workspace.
3. *D5 = 7 con sandbox excelente*. Contraprueba: `KITTY_SANDBOX=off` + sin gitleaks + SHA sin pinear; el 7 pende de cerrar al menos gitleaks + pins (Fase 3).

---

## PASO 5 — Informe final

### Resumen ejecutivo
Salud global **6.2/10** (Gate 2: D6 7→6 por CI rojo): harness honesto y bien armado en seguridad perimetral y cultura de bench, lastrado por un hot path async con I/O bloqueante, dos god files y deriva de formato. Compila, 722 tests verdes y CLI usable sin modelo: funciona de verdad en su caso de uso (research harness local). Riesgo principal: serving embebido abierto por defecto si alguien lo expone, y `--all-features` que no compila en Linux (trampa de medición, no de producto).

### Tabla de puntuaciones
| Dimensión | Nota | Críticos | Acción clave |
|---|---|---|---|
| Modularidad (25%) | 6 | `tools.rs` 3303L, `agent/mod.rs` 2877L | Extraer `Bash/Edit/ReadTool`, `loop/` (P1) |
| Diseño (20%) | 6 | `grammar.rs` post-hoc, `len/4` | EBNF real o borrar spike (P1) |
| Optimización (15%) | 5 | FS sync en async, `Mutex<Model>`, clones | `tokio::fs`/`spawn_blocking` en `Read/Edit` (P0) |
| Simplicidad (10%) | 7 | fmt dirty kitty-scope | `cargo-fmt` + pin toolchain (P0) |
| Seguridad (20%) | 7 | `api.rs:219` abierto, sin gitleaks/pins | bind loopback + rate-limit + gitleaks (P0) |
| Testing (10%) | 6 | clippy CI rojo (3 lints), fmt rojo, cobertura 1 crate, sin E2E con modelo | micro-fix lints + llvm-cov workspace + gate etiquetado (P0/P1) |
| *Adherencia (info)* | *6* | toolchain flotante, sin console | pin + `console-subscriber` (P1) |

**Salud ponderada (Gate 2): 6×0.25 + 6×0.20 + 5×0.15 + 7×0.10 + 7×0.20 + 6×0.10 = 6.15 → 6.2/10.**

### E2E usuario final (medido 2026-09-24)
`cargo build -p kitty-cli` 4.17s OK; `kitty-cli info` → `Status: Connected` sin modelo; `cargo test` kitty default-features **722/0**; clippy 3 lints `items_after_test_module` (kitty-runtime); `cargo-fmt` dirty (lista en D4); `--all-features` no compila en este host (objc2/cudarc). Funciona; el gate válido es default-features + cuda en contenedor.

### 10 papers → aplicación repo
P1 SWE-bench (ICLR 24): holdout + k≥3 en `bench/official/`. P2 SWE-agent (NeurIPS 24): ACI mínima recorta `agent/run.rs`+`tools.rs`. P3 ReAct (ICLR 23): tipar el loop en `agent/mod.rs`+`response.rs`. P4 SynCode (TMLR 25): EBNF real para `grammar.rs`→`parser.rs`. P5 Generative Agents (UIST 23): scoring 3-factores en `episodic.rs`. P6 RAG (NeurIPS 20): completar `embedding.rs` denso. P7 RRF (SIGIR 09): fusión BM25+dense k=60 en `episodic.rs`. P8 Phi-3 (2024): perfiles `model_profile.rs` para SLM. P9 Capsicum (USENIX 10): auditar `sandbox.rs`+`guard/` por mínima autoridad. P10 contaminación+Wilson: oracle cifrado + Wilson 95% en `bench/`.

### 10 tecnologías (jul-2026, verificadas) → adopción
T1 proptest 1.11 (2026-03-24): `proptest!` YA existe solo en `crates/kitty-harness/tests/property_tests.rs:13,76`; adopción = extender a `shell_parse.rs`/`parser.rs`/`hashline.rs` como unit-property. T2 cargo-mutants 27.1 (2026-06-02): nocturno `guard/`+`sandbox.rs` score ≥80%. T3 llvm-cov 0.8.7 + nextest 0.9.140: gate 70% + −30% tiempo. T4 deny 0.20.2 + audit 0.22.2: 0 advisories. T5 cyclonedx 0.5.9: SBOM por artefacto. T6 cosign v3.1.1: 100% assets firmados. T7 SLSA gen v2.1.0: provenance L3. T8 CodeQL Rust 2.23.3: 0 high en `tools/`/`sandbox/`/`guard/`. T9 gitleaks v8.30: pre-commit + CI. T10 tokio-console 0.1.14: p95 del tool-loop visible en `telemetry.rs`.

### Plan para llevar cada nota baja al máximo
- **D3 5→9**: P0 `tokio::fs` en `Read/Edit/knowledge/hashline` + test de no-bloqueo (`tokio::test` con pause); P1 `RwLock` lector en inferencia + eliminar clones `to_request`; P2 benchmark antes/después con `bench_results.json`. Techo 9 (no 10: GGUF local siempre paga I/O modelo).
- **D1/D2 6→8**: extraer 3 tools + `AgentLoop` sin cambiar `ChatClient`; `grammar.rs`: EBNF o borrar; fachada `ModelInfo`. Techo 8: el harness seguirá siendo monolito deliberado.
- **D5 7→9** en dos oleadas: (i) CI ahora — gitleaks, pins SHA (`ci.yml:34,57,96,101`: `rust-cache@v2` ×3 + `actions/cache@v4` ×1), `SOURCE_DATE_EPOCH`, toolchain `1.98.0` (cuda ya la fija `ci.yml:79`); helpers rate-limit/auth en `serving/`. (ii) Serve cuando reviva el lane (`mistralrs-cli` fuera de members hoy, sin call site de `check_auth` en prod): bind 127.0.0.1 por defecto + auth obligatoria. Techo 9 (10 exige auditoría externa).
- **D6 7→9**: micro-fix 3 lints; llvm-cov workspace ≥70 aspirando 80 en `guard/tools/sandbox`; mutants nocturno; etiquetar gate válido (default + cuda-container). Techo 9 (10 exige inferencia E2E con modelo en CI).
- **D4/D7 7/6→9/8**: `cargo-fmt` + pin toolchain + `console-subscriber` + renombrar benches homemade/official.

### Tips de implementación
1. Un fix, un crate, un PR: nunca mezclar `kitty-*` con `engine/` en el mismo cambio.
2. Todo cambio P0 lleva test que falla sin el fix (red-green) + `cargo-fmt -p kitty-*` + clippy scoped antes de pushear.
3. Medir dos veces: `bench_results.json` + manifiesto k≥3 antes de proclamar mejora (cultura del repo, no negociable).
4. Las subidas major (fastembed 4→7, rusqlite 0.31→0.40) van por oleadas con rollback por PR (Fase 3).

---

## Debt log (explícito; Fase 3 lo ejecuta donde aplique)
- D1: `tools.rs`/`agent/mod.rs` god files (P1). D2: `grammar.rs` spike (P1). D3: FS sync (P0). D4: fmt dirty (P0). D5: `api.rs:219` + gitleaks + pins + toolchain (P0). D6: 3 clippy lints + coverage scope (P0/P1). D7: console + renombres bench (P1).
- PRs dependabot (8): rusqlite #10, toml #8, fastembed #7, crossterm #5, criterion #4, rand #3, nix #2 (ya en 0.31.3 en workspace; verificar), ratatui #1 — oleadas con matriz compat en Fase 3.
- `council` inexistente como lane: sustituido por doble gate oracle + evidencia; ofrecer revisión humana externa.
- Alias `cargo fmt` recursivo del entorno (`fmt→fmt`): arreglar alias local, usar `cargo-fmt` directo hasta entonces.
