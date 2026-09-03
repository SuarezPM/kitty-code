# Proposal: remediate-audit-2026-09

## Why

The 2026-09 code audit scored Kitty Code at **6.8/10** global health
(modularidad 7, diseño 6, optimización 6, simplicidad 6, seguridad 8,
testing 7). The findings are concrete, located, and fixable without touching
published benchmark numbers:

1. **Monolito CLI**: `crates/kitty-cli/src/main.rs` (2.804 LOC) orquesta 4
   suites de benchmark inline (`run_gpqa_suite`, `run_claweval_suite`,
   `run_polyglot_bench`, `run_tb2_official`) más estadística
   (`wilson_ci`, `p50_ms`, `build_ab_plan`).
2. **Dios archivos en harness**: `agent.rs` (3.980 LOC), `guard.rs` (1.578),
   ambos listados como split candidates en `crates/kitty-harness/AGENTS.md`.
3. **Config ambiental acoplada**: 36 variables `KITTY_*` leídas en 5 crates
   con lectores duplicados (`KITTY_SANDBOX`, `KITTY_NO_THINK`,
   `KITTY_MODEL_TURN_CAPS`, `KITTY_NUM_CTX`, `KITTY_TELEMETRY`).
4. **Encapsulación rota**: `Agent::config_mut()`/`budget_mut()` ya produjeron
   un bug de dead-store (G2-FIX3); `ToolRegistry.hashline` es campo público.
5. **Rendimiento**: `EpisodeStore` (rusqlite bloqueante) llamado en contexto
   async sin `spawn_blocking`; índice BM25 reconstruido en cada recall.
6. **Código muerto suprimido**: 25+ `#[allow(dead_code)]` en producción más
   scripts huérfanos en la raíz del repo.
7. **Supply-chain CI**: `actions-rs/toolchain` (archivado desde 2023),
   `cargo clippy` sin `--all-targets --all-features`, sin SBOM ni firma de
   artefactos.
8. **Testing**: sin cobertura medida, sin property-based testing, sin
   mutation testing en las áreas de mayor riesgo (parser, hashline, guard).

## What Changes

- **modularity/cli-bench-split**: extraer suites de benchmark de `main.rs` a
  módulos `crates/kitty-cli/src/bench/*`; partir `guard.rs` y `agent.rs` en
  módulos cohesivos; centralizar las variables `KITTY_*` en un módulo
  `config` tipado.
- **design/run-overrides**: reemplazar `config_mut()`/`budget_mut()` por
  `RunOverrides` explícito pasado a `run()`; privatizar `ToolRegistry::hashline`.
- **performance/async-memory**: envolver las operaciones de `EpisodeStore` en
  `spawn_blocking`; cachear el índice BM25 (rebuild solo on-write).
- **simplicity/dead-code**: eliminar los `#[allow(dead_code)]` y el código
  que suprimen; borrar huérfanos de la raíz (`extract_speed.py/.sh`,
  `oc_bench.js`, `speed_proxy.js`, `stress.js`, `vendor/` vacío).
- **security/supply-chain-2026**: reemplazar el toolchain action archivado,
  endurecer CI (clippy `--all-targets --all-features`, tests
  `--all-features`), añadir job SBOM (cargo cyclonedx) y `cargo audit`
  programado.
- **testing/coverage-gate**: llvm-cov con gate ≥70% en kitty-harness;
  proptest en parser/hashline/guard; wiremock para `OllamaClient`.

## Capabilities

### New capabilities
- `modularity/cli-bench-split`
- `design/run-overrides`
- `performance/async-memory`
- `simplicity/dead-code`
- `security/supply-chain-2026`
- `testing/coverage-gate`

## Impact

- **Código afectado**: `crates/kitty-cli/src/main.rs`, nuevos
  `crates/kitty-cli/src/bench/*`, `crates/kitty-harness/src/{agent,guard}.rs`
  (split), `crates/kitty-harness/src/lib.rs`, `crates/kitty-memory/src/episodic.rs`,
  `.github/workflows/ci.yml`, `.gitignore`, `Cargo.toml` (workspace + harness).
- **Sin impacto**: números publicados en `AUDIT.md`/`CNAS.md` (los benches de
  modelo no cambian), API pública de los comandos CLI existentes.
- **Riesgos**: el split de archivos grandes puede revelar dependencias
  implícitas entre módulos; mitigado con tests completos por paso
  (`cargo clippy --all-targets --all-features -- -D warnings` +
  `cargo test --workspace --all-features`).
- **Out of scope**: signatures cosign de releases (se deja como job manual
  documentado — requiere secretos de CI que no existen hoy), migración FTS5
  (se decide midiendo con criterion; si el cache BM25 basta, no se migra).
