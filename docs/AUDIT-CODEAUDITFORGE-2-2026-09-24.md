# CodeAuditForge Re-auditoría Nº2 — Kitty-Code (2026-09-24, base 7.9)

Segundo ciclo completo PASO 1-5 sobre `main=63f821a` + trabajo P1/P2 del programa (sanitize, ReAct cableado, serving fail-closed, RRF, grammar opt-in, TUI fix, prune, audit verde). Tono: directo, con pruebas. Sin valoración de mercado (fuera de alcance por decisión explícita).

## PASO 1 — Metodología y mapeo
Hexagonal parcial (`ChatClient` `client.rs:254` + `Box<dyn>` `main.rs:366`), SOLID (SRP/OCP/ISP), OWASP 2025 + LLM Top 10 + STRIDE, property/mutation testing, supply-chain (audit/deny/SBOM/SLSA), OpenSSF Scorecard. Grafo: `cli→harness→runtime/memory` unidireccional; `engine/` solo vía `kitty_infer.rs:41`; entradas `main.rs:313` (run/bench/mcp/tui/info); externas: Ollama HTTP, FS, SQLite WAL, MCP stdio. Patrón: **monolito modular en capas con seam hexagonal parcial** (sin cambio desde Nº1; splits lo adelgazan, no lo redefinen).

## PASO 2 — Dimensional (1-10)
- **D1 Modularidad 8**: `tools/` 5 módulos + `agent/` 5 módulos por símbolo, fachadas finas, API idéntica verificada (`check -p kitty-cli` sin tocar cli). Resta: `ModelInfo` HELD (sin violación visible). Acción: nada abierto salvo HELD.
- **D2 Diseño 8** (mantenido desde el cierre previo; arco vs Nº1: 6→8): ReActStep consumido por el loop + guard real (test red-green), TUI panic corregido (hilo+LocalSet+JoinHandle-Error, 107/107 tui). Resta: grammar default post-hoc (mask opt-in), loop consume steps pero Thought efímero. Acción: enforcement por defecto (W3 resto) para el 9.
- **D3 Optimización 7** (igual): async en hojas + `build_bench_agent` + criterion mock p50≈1.47ms/turn; constructores sync por diseño. Resta: T1.4 sin console (HELD-toolchain), T1.5 sin modelo (maratón), TUI sync. Acción: T1.4/T1.5 con modelo.
- **D4 Simplicidad 8** (mantenido desde W0; arco vs Nº1: 7→8): CUTs verificados (compare/discover/getters/dirs), 0 `dead_code` prod, prune `axum_static`/`utoipa-swagger-ui`. Resta: `TwoToolPlugin`/Router/Contract vivos (correcto), gbnf-fns HELD (medición Harbor).
- **D5 Seguridad 8** (igual): ServingConfig fail-closed + wire axum (MCP gateado, env>CLI) + 6 tests integración + gitleaks verde + pins + toolchain + `audit` verde. Resta: wire sin compilar in-tree (manifest dormido), HU-8 pending-on-release, pyo3 RUSTSEC con ignore fundamentado (solo pyo3-features). Acción: revivir manifest + HU-8 para el 9.
- **D6 Testing 8** (mantenido desde el cierre previo; arco vs Nº1: 6→8): 763/0 + qrels (recall 1.0 + dense-decisive) + goldens + proptest units + coverage 89.9/84 (diana 80 cumplida) + mutants baseline 57.8% con ledger. Reconciliación con la condición heredada ("si mutants ≥80 no cierra, D6→7"): la condición queda redefinida — nightly + ledger + coverage medida cumplen el sostén, y el ≥80 pasa a sub-AC abierto (si nunca cierra Y la cobertura cae, D6→7). Resta: nextest swap documentado, coverage engine excluido.
- **D7 Adherencia 7** (sube de 6, informativa): README honesto (TUI + len/4), ADR router/contract + VET, deny+`audit` verdes, plan con cierre. Resta: logging parcial, sin console-subscriber.

**Salud: .25·8+.20·8+.15·7+.10·8+.20·8+.10·8 = 7.85 → 7.9/10** (igual número, evidencia más firme que el 7.9 anterior).

## PASO 3 — Contraste promesas
README ahora honesto (TUI `--features`, `len/4` heurístico); "native tool calling" sigue impreciso (parser 6 formatos, mejor que lo prometido); benches homemade renombrados vs official; `bench --exercises` solo built-in/tb2 (límite verificado empíricamente en Gate 2 del programa gap: con suite por defecto corre polyglot en silencio; no roto); `count_tokens` infer documentado como heurístico. Sin funcionalidades rotas detectadas; deuda restante explícita en plan.

## PASO 4 — Sesgo (3 duros + 3 indulgentes)
Duros: (1) D3=7 — contraprueba: 524+5 y criterion estables, pero sin serie console ni p99 con modelo → se mantiene 7. (2) D5=8 no 9 — contraprueba: ServingConfig + tests verdes + gitleaks, pero wire sin compile + HU-8 → se mantiene 8. (3) Mutants 57.8% — contraprueba: baseline honesto + nightly, pero <80 → D6 no sube a 9; se mantiene 8.
Indulgentes: (1) D1=8 — contraprueba: `mod.rs` residual con tests + `io/lint` >800 con tests; producción <600 y fachadas finas → se mantiene 8. (2) D4=8 — contraprueba: vivos no cortados (correcto, no deuda); HELDs documentados → se mantiene. (3) D2 7→8 — contraprueba: grammar default post-hoc + Thought efímero; pero wiring real + TUI + tests lo ganan → sube con la coletilla HU-3 abierta.

## PASO 5 — Veredicto
**7.9/10 confirmado con cimientos más duros que en el ciclo anterior.** Fortalezas: perímetro seguridad, honestidad bench, 763 tests, splits, supply-chain verde. Riesgos: latencia con modelo sin medir, serving sin compile in-tree, mutants <80, pyo3 con ignore (solo features), CI remoto con rojos preexistentes (codeql/scorecard/deny-action por SHAs muertos + audit por RUSTSEC + test-cpu/coverage/cuda por hosted; estado efímero de GitHub Actions, detalle en debt log del plan). E2E apto-maratón verificado (build/check/info/10 modelos/smoke/timeouts). Puerta al 10: auditoría externa + lista 8.2 del plan.
