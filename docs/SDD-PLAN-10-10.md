# SDD-PLAN-10-10 — Kitty-Code 6.2/10 → 10/10

Plan de especificación (sin código). Base: auditoría 6.2/10 (`docs/AUDIT-CODEAUDITFORGE-2026-09-24.md`, Gate 3 PASS, main=71c89ae, 723 tests verdes). Gate válido: kitty default-features + cuda-container (`--all-features` no compila en Linux). Boundaries de extracción: por símbolo, nunca por línea. Versiones no confirmadas van como `verify-at-execution`, nunca como hechos.

---

## 1. REQUISITOS Y ALCANCE

**Problema**: el harness funciona pero frena el reactor (I/O bloqueante), vive en 2 god files, la gramática no impone nada, el serving nace abierto, la cobertura es de 1 crate y los papers del roadmap no son código.

**Historias + AC (Given/When/Then)**:
- HU-1 async: Given `cargo test -p kitty-*` + workload Read/Edit/Grep, When se ejecuta con tokio-console, Then 0 tareas `tools::execute` con poll>100ms y 723→N tests verdes.
- HU-2 god files: Given `tools.rs`/`agent/mod.rs`, When se extraen módulos por símbolo, Then cada fichero <800L, API pública idéntica (`cargo check --workspace` + tests verdes, 0 cambios de firma).
- HU-3 grammar: Given `GrammarClient` post-hoc, When se impone máscara llguidance en el sampler, Then tool-calls inválidos = 0 en bench k≥3 sin subir latencia p99 >10%.
- HU-4 serving: Given `api.rs:219` default-abierto, When se expone fuera de loopback, Then exige token + bind 127.0.0.1 + rate-limit (tests de cada rama).
- HU-5 cobertura: Given llvm-cov 1 crate ≥70%, When se extiende a scope `kitty-*` (engine excluido), Then ≥70% global y ≥80% en `guard/tools/sandbox` + mutants nocturno score ≥80% (advisory).
- HU-6 memoria: Given BM25 solo, When RRF BM25+dense k=60 + 3-factores, Then recall@10≥0.78 y nDCG@10≥BM25+Δ en SciFact/qrels fijos.
- HU-7 loop: Given loop implícito, When se tipa como ReAct (Thought→Action→Observation), Then 0 regresiones + doom-loop detection con test.
- HU-8 supply/deploy: Given release actual, When SBOM firmada + provenance SLSA + `SOURCE_DATE_EPOCH`, Then artefactos 100% verificables (`slsa-verifier` + `cosign verify` verdes).

**Fuera de alcance**: `engine/` upstream heredado (solo integration points), `bench/official/*` vendored, `.slim/*`, clone anidado polyglot-benchmark, reescribir benches oficiales.
**DoD global por tarea**: fmt 0 + clippy `-D warnings` 0 + tests del crate + bench k≥3 si toca loop/sampler + manifiesto si toca bench.
**Techo honesto**: ~8.2 interno con estas dianas; el 10/10 exige auditoría externa (W11).

## 2. MODELO DE DATOS Y ENTIDADES

- `Episode{goal, files_touched, outcome, timestamp, importance}` (`episodic.rs:10-18` + campo `importance` nuevo) — scoring `recencia+relevancia+importancia`.
- `RrfParams{k=60, alpha, N=100}` parametrizable (tests con k alternativo 10-20).
- `ToolCall{normalized, source_format}` — `normalize_tool_call` ya existe (`tools.rs:1153`); ReAct añade `Thought{summary}` efímero (no persistido).
- `SandboxPolicy`/`GuardVerdict` — sin cambios de forma; Capsicum-audit añade `CapabilitySet` doc + tests de mínima autoridad.
- `BenchManifest` + `GoldenVector` (patrón `ab.rs:315` a extender a `polyglot/tb2` órdenes).
- `ServingConfig{bind, require_token, rate_limit, cors}` — nuevo, default `127.0.0.1` + token exigido fuera de loopback.
- Almacén: SQLite WAL (sin cambio de motor); qrels fijos para memoria en `kitty-memory/tests/`; SBOM CycloneDX por artefacto.

## 3. ENFOQUE TÉCNICO Y ARQUITECTURA

**Componentes**: `tools/{io,bash,search,normalize,lint}.rs` + `agent/{config,policy,setup,output}.rs` (extracciones puras, misma API); `grammar.rs` (llguidance mask en sampler, `GrammarClient` como fallback-compat); `serving/` (auth/bind/rate); `episodic.rs` (`rrf_fuse` + 3-factores); CI (llvm-cov `kitty-*`, nextest, mutants nocturno, checks features ya existentes + test-run embedding/tui).

**Librerías permitidas (ancladas)**: llguidance `1.7.x` (1.7.5 verificada; baseline local 1.2.0) — toktrie `verify-at-execution` (1.8.0 sin confirmar); llvm-cov `0.9.0` (0.9.1 verify-at-execution); nextest `0.9.146`; mutants `27.1.0` (verify-at-execution); proptest `1.11.0`; deny `0.20.2`; audit `0.22.2` (verify-at-execution); cyclonedx `0.5.9`; cosign `v3.1.3`; gitleaks-action `v3` + engine `8.30.1`; tokio-console `0.1.14` (verify-at-execution); toolchain `1.98.0`.
**Prohibido**: GBNF llama.cpp hasta gate Harbor (veto AGENTS.md vigente); Outlines en `mistralrs-core` (startup/memoria); `tokio::fs` en pipes (solo ficheros ordinarios); `cargo update` global (solo `-p` acotado); `--all-features` como gate en Linux.

**Flujos**: Read/Edit/Grep → `tokio::fs` (todo-fichero en 1 op) o `spawn_blocking` donde ya existe el patrón (`tools.rs:1338`, `response.rs:329,382`); sampler: `GrammarConfig→llguidance mask→sesgo logits` vía `build_llg_factory` + `llg_factory` (anclas por símbolo en T3.0); memoria: BM25+dense→RRF→rerank opcional.
**Normas**: un fix un crate un PR; red-green (test que falla sin el fix); `bench_results.json` + manifiesto k≥3 antes de proclamar mejora; boundaries por símbolo.
**Safeguards**: Sandbox fail-closed intacto + `KITTY_SANDBOX=off` ruidoso; ningún cambio relaja `Guard`; `deny.toml` 0 advisories nuevos (3 licencias + paste preexistentes documentados, no reabiertos); lock reproducible `--locked` verde siempre.

## 4. DESCOMPOSICIÓN DE TAREAS (oleadas multi-día; [P] = paralelizable)

**W0 sanitize verify-then-cut** (D4 7→8): T0.1 TwoToolPlugin/compare_harnesses/discover_skills/grammar-pub — paso 1 `grep -rn` por símbolo + `cargo tree -i`; cortar solo si test-only [P]; T0.2 anidado `crates/kitty-cli/crates/` — inspeccionar y borrar o documentar [P]; T0.3 `Contract`/`ToolRouter` = ADR retirada (no corte mecánico). Validación: `cargo check --workspace` + tests.
**W1 async reactor** (D3 5→8): T1.1 `tools.rs` puntos (405/406/431/432/489/554/655/749/752/1028 — verificar conteo en ejecución) → `tokio::fs` [P por fichero]; T1.2 `knowledge.rs` + `hashline.rs` [P]; T1.3 `agent/mod.rs:559` open + `:822` write + `response.rs:398` + `run.rs:413` + `skills/telemetry/contract/plugin` [P]; regla de firmas: async solo en hojas I/O, constructores vía `spawn_blocking` (patrón `tools.rs:1338`) — el "API idéntica" de W2 se refiere a firmas post-W1; T1.4/T1.5 cierran W1 (evidencia antes/después + bench k≥3). Validación HU-1.
**W2 god files** (D1/D2 6→8): T2.1 `tools/{io,bash,search,normalize,lint}` por símbolo, API idéntica; T2.2 `agent/{config,policy,setup,output}` idem [P vs T2.1]; T2.3 `Fachada ModelInfo` (Demeter `main.rs:139` + `kitty_infer.rs:812`); `mod tests` viaja con su símbolo. Validación HU-2 + `cargo public-api` diff vacío si disponible.
**W3 grammar real** (D2): T3.0 plumbing `GrammarConfig → KittyInferClient → pipeline llg` (hoy solo llega al wrapper post-hoc `agent/mod.rs:529-530`; anclas `llg.rs`/`pipeline/mod.rs` a verificar por símbolo) o decisión de pasar grammar por `RequestOptions`; T3.1 spike llguidance 1.7.x en `grammar.rs:39` (mask offline, SynCode como spec de caché) tras `sampler_opts_from` (`kitty_infer.rs:320`); T3.2 bench A/B k≥3 vs post-hoc (invalidez 0, p99 ≤+10%); T3.3 GO/NO-GO Harbor (si <95% se revierte a post-hoc). Validación HU-3.
**W4 serving cerrado** (D5 7→9, integration point declarado — `engine/` sigue fuera salvo este punto): T4.1 `ServingConfig` + bind + token + rate-limit + CORS (`tower-http`) [P↔T10.1]; T4.2 auth en Admin/Logs/Metrics + tests por rama; T4.3 STRIDE+Capsicum audit `sandbox.rs`+`guard/` con tests de mínima autoridad. Validación HU-4.
**W5 testing** (D6 6→8): T5.1 llvm-cov 0.9.0 scope `kitty-*` (engine excluido: no compila en default, `ci.yml:41-42`) ≥70% (≥80% guard/tools/sandbox) [P]; T5.2 nextest en CI; T5.3 mutants nocturno guard/sandbox ≥80% (advisory, no blocking); T5.4 test-run embedding/tui en CI (hoy solo compile); T5.5 proptest unitario `shell_parse/parser/hashline` (hoy solo tests/); T5.6 goldens `polyglot/tb2` (patrón ab.rs:315); T5.7 `llama-vulkan`: `cargo check --features` o declaración fuera de alcance. Validación HU-5.
**W6 memoria** (papers): T6.1 sobre RRF existente (`episodic.rs:401,459`, RRF_K=60): Bgem3 sparse + rerank + parametrizar `RrfParams` [P]; T6.2 3-factores en `Episode`; T6.3 qrels SciFact + recall@10≥0.78 (depende de T6.1+T6.2). Validación HU-6.
**W7 loop ReAct**: T7.1 tipar Thought→Action→Observation (efímero, sin migración de datos); T7.2 doom-loop con test; bench k≥3 sin regresión.
**W8 benches honestos**: T8.1 renombrar homemade vs official; T8.2 holdout (SWE-bench spec) + goldens `polyglot/tb2` (oracle cifrado y Wilson ya existen).
**W9 docs vivos**: T9.1 `count_tokens len/4` — implementar real o documentar límite; T9.2 TUI `--features` en README; T9.3 `contract.rs` consumer-tests (Contract vivo en `agent/mod.rs`; borrar exigiría unwire — fuera de este plan).
**W10 deploy** (HU-8): T10.1 `SOURCE_DATE_EPOCH` + SBOM por artefacto + cosign v3.1.3 + SLSA L3 [P]; T10.2 `cargo-vet` sí/no (ADR); T10.3 gap deny (licencias + paste) llevar a 0 o documentar excepción; T10.4 dashboard CI (scorecard + secret scanning ya verdes).
**W11 cierre**: T11.1 re-auditoría interna PASO 1-5 (esperado ~8.2 — techo de estas dianas); T11.2 **auditoría externa** (requisito del 10/10); T11.3 archivar plan + debt log residual.

**Dependencias declaradas**: T6.3→T6.1+T6.2; T4.2→T4.1; W7→W2 (loop post-split); T1.4/T1.5 cierran W1; T5.3 advisory hasta score sostenido; [P] T4.1↔T10.1 (CI/supply en paralelo).

**Mapa score** (pesos auditoría 25/20/15/10/20/10): D1 +0.5 (6→8), D2 +0.4 (6→8), D3 +0.45 (5→8), D4 +0.1 (7→8), D5 +0.4 (7→9), D6 +0.2 (6→8) = +2.05 → 6.2+2.05 ≈ **8.2** interno (D7 informativa, fuera de fórmula); W11 externa → 10.

---

## Cierre del programa (7.6/10)

Score final por dimensión (pesos auditoría 25/20/15/10/20/10):

| Dim | Score | Peso | Aporte |
|-----|-------|------|--------|
| D1 | 8 | 25 | 2.00 |
| D2 | 7 | 20 | 1.40 |
| D3 | 7 | 15 | 1.05 |
| D4 | 8 | 10 | 0.80 |
| D5 | 8 | 20 | 1.60 |
| D6 | 7 | 10 | 0.70 |
| **Total** | | 100 | **7.55 → 7.6** |

Razones de gap (corregidas):
- D3 (7, no 8): T1.4 tokio-console + T1.5 p50/p99 ausentes — NO constructores-main (el offload `spawn_blocking` en CLI ya cierra ese frente, ctores siguen sync por diseño).
- D5 (8, no 9): serving sin wire (policy + 6 tests verdes, listener axum pendiente) + HU-8 pending-on-release.
- D2 (7, no 8): ReAct en paralelo (tipado Thought→Action→Observation no cerrado en este ciclo).
- D6 (7, no 8): coverage % sin medir + mutants sin correr (infra lista: llvm-cov/nextest pinados, goldens y proptest W5 verdes).

HU-8 (`slsa-verifier` + `cosign verify`): pending-on-release explícito — el workflow firma y genera provenance/SBOM en cada release tag, pero la verificación verde contra un release publicado no se ha ejecutado en este ciclo.

Serving: 6 tests (`serving_config.rs` unitarios + espejo de integración en `engine/mistralrs-core/tests/`), fail-closed + rutas + rate + CORS.
