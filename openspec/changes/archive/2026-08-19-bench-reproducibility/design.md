## Context

See proposal.md Why: bench carece de veredicto exit-code y gate sub-2B; cultura AUDIT/CNAS existente pero sin baseline reproducible. Constraints: harness ya existe en `crates/kitty-cli/src/main.rs` y `benches/`; debe seguir siendo determinista y fijar método de grading (homemade string-match vs unit-test).

## Goals / Non-Goals

**Goals:**
- Veredicto CI con exit-code (0 mejoró / 1 regresó / 2 ruido) comparando contra baseline guardado.
- Gate go/no-go del piso sub-2B (KittyQuiz-225 / KittyShell-89, umbral 50%).
- Toda métrica etiqueta su método de grading.

**Non-Goals:**
- Nuevos benchmarks oficiales (se reutiliza suite homemade existente).
- Auto-actualización del baseline sin revisión humana.
- Profiling de performance del modelo (solo pass rate).

## Decisions

- **Baseline en `benches/` como artefacto versionado** — simple diff, reproducible en CI; alternativa descartada: BD externa (acoplamiento).
- **Exit-codes discretos 0/1/2** — mapeo directo a semáforo CI; alternativa: JSON + wrapper shell (más frágil).
- **Gate sub-2B como bench dedicado** — aislamiento del run costoso; alternativa: flag en bench general (mezcla ruido).

## Risks / Trade-offs

- [Ruido estadístico marca regresión falsa] → Mitigación: umbral de ruido (código 2) + ventana de tolerancia.
- [Baseline desactualizado diverge] → Mitigación: job CI que alerta si baseline sin actualizar tras N merges.
- [Homemade grading sobrevalora vs unit-test] → Mitigación: toda métrica etiqueta método y nota de no-comparabilidad.
