## Why

Ambos competidores tienen gaps de honestidad en benchmarks: smallcode fabrica números (COMPARISON.md "87/100" no reproducible del harness) y little-coder no reproduce el 78.67% del harness publicado (descriptores stubbed). Kitty Code ya tiene la cultura AUDIT/CNAS, pero el bench carece de veredicto con exit-code y no hay un gate go/no-go explícito para el piso sub-2B (si el single-file success es <50%, el claim del tier debe re-encuadrarse antes de construir para él).

## What Changes

- **Bench con veredicto exit-code**: el harness de bench produce un código de salida CI (0 = mejoró / 1 = regresó / 2 = ruido) comparando contra un baseline guardado.
- **Gate go/no-go del piso sub-2B**: un benchmark dedicado corre la suite homemade (KittyQuiz-225 / KittyShell-89) con un modelo sub-2B y reporta si el single-file success supera el umbral (50%); si no, el reporte marca el tier como "re-encuadrar".
- Toda métrica fija el método de grading (string-match homemade vs. unit-test oficial).

## Capabilities

### New Capabilities
- `benchmarking`: harness de bench reproducible con veredicto exit-code y gate go/no-go del piso sub-2B.

## Impact

- `crates/kitty-cli/src/main.rs` (bench), `benches/` (baselines guardados, gate sub-2B).
- Complementa la cultura AUDIT/CNAS existente.
- Interactúa con model-profiles (bench matrix por perfil).
