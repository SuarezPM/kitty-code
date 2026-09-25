# SDD-PLAN-82-10 — Kitty-Code 7.9 → 8.2 → 10

Plan de especificación (sin código). Base: re-auditoría Nº2 7.9 (`docs/AUDIT-CODEAUDITFORGE-2-2026-09-24.md`, Gate 3). Deber nuevo (orden usuario): corregir errores heredados del engine para validación directa. Techo interno con estas dianas: 8.2; 10 solo con T11.2 externa.

## 1. REQUISITOS Y ALCANCE
Problema: hueco efectivo 0.35 (D3+D5; el 0.65 fue el contexto histórico del programa gap con base 7.55) + engine-heredado + readiness maratón (K2/MiniCPM5 por kitty-infer, ornith por Ollama; ornith sin GGUF, K2 sin tag Ollama).
Historias + AC:
- HU-E1 engine-directo: Given errores LSP/heredados (gguf/sampling/sequence/lib/toml_selector), When se corrigen por fichero con tests, Then `cargo test -p mistralrs-core` sin errores de compilación + clippy con `-A` declarado para los ~373 lints preexistentes (ci.yml:66-70); bajar esos lints = tarea separada con recuento medido.
- HU-E2 wire-compilable: Given manifest dormido roto, When se revive (crear crates ausentes o extraer binario), Then serve compila + sus tests corren en CI.
- HU-M1 maratón: Given GGUFs K2/MiniCPM5/Qwen3 + Ollama 10, When smoke por (modelo, backend) con sidecars (`KITTY_INFER_TOKENIZER_DIR`, HF cache IFM), Then matriz readiness verde + manifiesto pre-registrado.
- HU-Q1 mutants≥80: Given baseline 57.8% + nightly, When se escriben tests para missed (top: `||→&&` en `Guard::scan`), Then score ≥80% medido.
- HU-Q2 latencia: Given T1.4/T1.5 pendientes, When tokio-console + bench k≥3 con modelo, Then p50/p99 persistidos + 0 tasks poll>100ms.
- HU-C1 CI: Given 5 rojos preexistentes, When re-pin 3 SHAs + pyo3 (bump documentado o ignore) + hosted-fixes, Then CI verde.
Fuera: reescribir upstream engine, BGE-M3/rerank offline (HELD), SGLang (frozen), auditoría externa (T11.2, humano).
DoD por tarea: fmt 0 + clippy `-D warnings` 0 + tests + rollback si rompe + evidencia persistida en repo (no `.slim`). Excepción E1: clippy con `-A` para lints preexistentes declarados (ver HU-E1).

## 2. MODELO DE DATOS Y ENTIDADES
- `ServeGuard` ya existe (config + limiter + middleware); E2 añade `McpGuard` si el router MCP lo exige.
- Manifiestos JSON `benches/ab/manifests/` + `TimeoutManifest` (`common.rs:421`) + celda `ornith:9b/ollama=60s` (ya); E2 añade celdas infer por (modelo,backend) medidas, nunca inventadas.
- `RrfParams`/`Episode.importance` intactos; Q1 añade qrels para mutants-killers si aplica.
- `ReadinessMatrix{(modelo,backend): estado+evidencia}` (doc efímero de maratón, no entidad código).
- Sin migraciones de datos en todo el plan.

## 3. ENFOQUE TÉCNICO Y ARQUITECTURA
Componentes: `engine/mistralrs-core` (fix por fichero, diff mínimo vs upstream; cada fix con test que falle sin él) → `mistralrs-cli` revive (opción A: crear `mistralrs-sandbox`+`server-core` mínimos; opción B: extraer `serve` a bin propio; decidir por coste tras spike 2h) → maratón (matriz + manifiesto) → mutants (tests dirigidos por missed) → latencia (console + bench) → CI (pins + pyo3-ADR + hosted).
Librerías: llguidance `1.7.x` (1.7.5 verificada; pin preventivo para W3-follow-up fuera de este mapa); llvm-cov `0.9.0`; nextest `0.9.146`; cosign `v3.1.3`; pyo3 **NO bump** sin spike (candle-rev); axum/tower solo tras revive.
Prohibido: tocar `mistralrs-cli` sin revivir manifest antes; `--all-features` como gate en Linux; `cargo update` global; inventar timeouts/celdas sin fuente repo-interna.
Normas: un fix un crate un PR; boundaries por símbolo; verify-then-cut; evidencia en repo.
Safeguards: sandbox fail-closed, guard intacto, deny 0 advisories nuevos, `--locked` verde, sin secretos/binarios.

## 4. DESCOMPOSICIÓN DE TAREAS ([P] = paralelo)
- E1 engine-directo: T-E1.1 gguf.rs:706 (confirmar test-only o fix) [P]; T-E1.2 sampling aridad+import (por feature) [P]; T-E1.3 sequence/image gates [P]; T-E1.4 lib.rs try_find* + toml_selector variants (¿código vivo? si muerto → CUT con verificación) [P]; T-E1.5 `cargo test -p mistralrs-core` verde directo + recuento baseline de lints (crecimiento = regresión); T-E1.6 follow-up: bajar lints preexistentes con recuento medido (fuera del 8.2). Validación: sin espejos.
- E2 wire: T-E2.1 spike revive (2h, opción A/B) en serie tras E1 (misma área engine); T-E2.2 compilar serve + tests CI; T-E2.3 crear tag de release + HU-8 verify (cadena tag→HU-8→cierre: sin tag el techo es 8.0).
- M1 maratón: T-M1.1 sidecars (K2 tokenizer en cache HF verificado; MiniCPM5 `hf download` si falta) en serie antes de T-M1.2-infer; T-M1.2 smoke (modelo,backend) matriz incl. infer-Qwen3 (base hecha) + K2 + MiniCPM5 + ornith-ollama [P por backend]; T-M1.3 manifiesto pre-registrado + celdas infer medidas.
- Q1 mutants: T-Q1.1 tests para `||→&&` Guard::scan + top-missed [P]; T-Q1.2 full nocturno ≥80% (si no cierra → D6→7 honesto, plan lo declara).
- Q2 latencia: T-Q2.1 console-subscriber cableado (receta: dep `console-subscriber 0.5` + `RUSTFLAGS="--cfg tokio_unstable"` + coexistencia con el `fmt().init()` existente de `main.rs:315-320` — un solo dispatcher global: componer `Registry+fmt+console` o sustituir por `try_init` guardado; `init()` a secas panica — + `tokio-console` puerto 6669) + sesión poll>100ms [P]; T-Q2.2 bench k≥3 con modelo p50/p99 → `bench_results.json` + manifiesto.
- C1 CI: T-C1.1 re-pin 3 SHAs (deref tag→commit) [P]; T-C1.2 pyo3: bump-spike o ignore documentado [P]; T-C1.3 hosted (user-ns off, serializar revisión/env, cuda compile-only) [P].
- Cierre: re-auditoría PASO 1-5 (esperado 8.2) + T11.2 externa (humano) para el 10.

**Mapa**: E1+E2 (D5 8→9 +.2) + Q2 (D3 7→8 +.15) + Q1 (D6 sostén) + M1/C1 (credibilidad E2E/CI) = 7.85+0.35 ≈ **8.2**; ramas: sin HU-8 → 8.0, sin Q1 → 8.1 o 7.9, todo-verde → 8.2. 10 = T11.2 externa revocando techos D1/D2/D3/D5/D6 de Nº1 + unlock E2E-con-modelo-en-CI (definición literal, no "mejor estado").
