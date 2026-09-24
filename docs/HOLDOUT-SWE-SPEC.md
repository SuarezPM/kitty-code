# Holdout SWE-bench-spec (T8.2) — sin tocar bench/official/* vendored

Scope: `bench/holdout/` + `benches/` (ownership P3). Vendored
`bench/official/*` NO se toca (solo lectura como oracle).

## Protocolo
1. Split fijo: 80% dev / 20% holdout por `task_id` (lista en
   `benches/holdout_manifest.json`, seed 42, orden alfabético; ESQUELETO:
   `task_ids` vacío, lista pendiente del runner).
2. El holdout NUNCA se usa para tuning (timeouts, thresholds, prompts).
3. Oracle cifrado: el grading del holdout reutiliza `bench_results.json` +
   manifiesto k≥3 existente (Wilson CI ya en `bench/ab.rs`, oracle en
   `benches/*oracle*.json`). Sin nuevo oracle vendored.
4. Report: `suite=holdout` con `rate`, `wilson_ci`, `p50_ms`; gate 50%
   heredado del bench normal.

## Estado
Spec + manifiesto esqueleto. Runner `bench holdout` pendiente (follow-up,
requiere dispatch en `crates/kitty-cli`, permitido pero fuera de las
≤2 iteraciones P3).
