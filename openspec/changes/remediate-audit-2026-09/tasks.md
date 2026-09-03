# Tasks: remediate-audit-2026-09

## 1. Fase 1 — Modularidad

- [x] 1.1 Extraer `bench/gpqa.rs`, `bench/claweval.rs`, `bench/polyglot.rs`, `bench/tb2.rs`, `bench/common.rs` desde `main.rs` (suites + stats: `wilson_ci`, `p50_ms`, `build_ab_plan`, `budget_summary`, tipos compartidos). `main.rs` queda < 1K prod LOC, comportamiento idéntico.
- [x] 1.2 Split `guard.rs` → `guard/mod.rs` (re-exports), `guard/patterns.rs` (BLOCKED/RISKY/SAFE_OUTPUT tables), `guard/shell_parse.rs` (tokenizer + `detect_write_targets`), `guard/interpreters.rs` (INTERPRETERS raw-scan). Tests intactos.
- [x] 1.3 Split `agent.rs` → `agent/mod.rs`, `agent/run.rs` (run/verify/hops), `agent/response.rs` (process_response, llm_call, telemetry fields), tipos compartidos en mod. Tests intactos (477/477).
- [x] 1.4 Módulo `config.rs` central en kitty-harness: consts únicos + accesores tipados `_raw`/env para SANDBOX/TELEMETRY/TRUNCATION_DIR/VERIFY_LOOP/BENCH_MODE/EDIT_CHECK/LINT_GATE/AUTORUN_HOPS/BENCH_TIMEOUT_SECS/GPQA/WORKSPACE_ROOT + re-export NO_THINK/NUM_CTX (runtime); migrados lectores duplicados; excepciones documentadas (sandbox child vars, runtime).
- [x] 1.5 Gate Fase 1: `cargo-fmt --all --check` + `cargo clippy --workspace --all-targets --all-features -- -D warnings` + `cargo test --workspace --all-features` verdes (único fallo: permute_four pre-existente, verificado en HEAD).

## 2. Fase 2 — Diseño/Encapsulación

- [x] 2.1 `RunOverrides { tuning, compact_threshold, autorun_hops }` + `run_with()`; eliminar `config_mut()`/`budget_mut()` migrando todos los callers (CLI flags, bench A/B, tests); eliminar el hack de re-seed G2-FIX3.
- [x] 2.2 Privatizar `ToolRegistry.hashline`; exponer `edit_by_hash`/accesos vía métodos del registry; ajustar usos internos.
- [x] 2.3 Gate Fase 2: suite completa + clippy estricto.

## 3. Fase 3 — Optimización

- [x] 3.1 `spawn_blocking` en `recall_episodes` y `save_episode` (agent) para todas las llamadas a `EpisodeStore`.
- [x] 3.2 Caché BM25 en `EpisodeStore` con invalidación on-write (`save`, `save_with_embedding`, `update_embedding`); tests de hit/invalidación.
- [x] 3.3 Criterion bench mínimo: `ToolCallParser` y dispatch de `ToolRegistry::execute` (Read/Edit) para medir antes de optimizar más.
- [x] 3.4 Gate Fase 3: suite completa + clippy estricto + criterion corre sin panic.

## 4. Fase 4 — Simplicidad

- [x] 4.1 Eliminar los 25+ `#[allow(dead_code)]` de producción borrando el código muerto (gpqa ×13, tui ×5, cloud ×3, guard ×2, episodic ×1, llama ×1, polyglot ×1, claweval ×1, etc.).
- [x] 4.2 Borrar huérfanos de raíz: `extract_speed.py`, `extract_speed.sh`, `oc_bench.js`, `speed_proxy.js`, `stress.js`, `vendor/` (vacío), `adapter-review-report.md`; comentar zombi `tui.rs:552` eliminado.
- [x] 4.3 `.gitignore`: regla raíz para artefactos `*.py` de bench (`/[a-z]*.py` estilo) + tracking de artefactos del agente.
- [x] 4.4 Gate Fase 4: build+tests verdes sin ninguno de los allows.

## 5. Fase 5 — Seguridad / Supply-chain

- [x] 5.1 Reemplazar `actions-rs/toolchain@v1` por pasos `rustup` mantenidos en los 3 workflows (test/clippy/fmt/deny), SHA-pinning intacto para actions externas.
- [x] 5.2 Endurecer CI: `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-features` en ci.yml.
- [x] 5.3 Añadir job SBOM (cargo cyclonedx, artifact en main) + job `cargo audit` (weekly schedule + pull_request con cambios en Cargo.lock).
- [x] 5.4 ADR `docs/ADR-PANIC-STRATEGY.md`: decisión (mantener abort), consecuencia, y criterios de reversión a unwind + catch_unwind.
- [x] 5.5 Gate Fase 5: CI local-equivalente verde (los cambios de workflow se validan con actionlint/lectura si no hay remoto).

## 6. Fase 6 — Testing

- [x] 6.1 `cargo llvm-cov` integrado en CI con gate ≥70% kitty-harness; medir línea base primero.
- [x] 6.2 proptest: parser (malformed tool-calls → no panic), hashline (edits arbitrarios → no panic + invariant FileChanged), guard (interpreter-wrapped payloads → blocked). Fixes de bugs reales que emerjan.
- [x] 6.3 wiremock contract tests: OllamaClient (chat con truncate:false, 400 overflow → typed error, timeout → typed error, fallback de token-count).
- [x] 6.4 Documentar comando `cargo mutants --file crates/kitty-harness/src/guard.rs,crates/kitty-harness/src/hashline.rs` y score objetivo >70% (ejecución local documentada, no en CI).
- [x] 6.5 Gate Fase 6: suite completa + proptest + contract tests verdes; coverage ≥70%.

## 7. Fase 7 — Verificación y cierre

- [x] 7.1 Gate final completo: fmt + clippy estricto + test --all-features + deny check + coverage + proptest.
- [x] 7.2 Sync codemap.md/README con la nueva estructura (bench/, guard/, agent/, config.rs) sin citar fuentes externas.
- [x] 7.3 Guardar hallazgos en Engram (mem_save) y resumen de sesión (mem_session_summary).
