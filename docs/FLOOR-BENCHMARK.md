# Floor Benchmark — Tier sub-2B / 4B (post-tuning)

**Fecha:** 2026-08-19
**Harness:** Kitty Code main 3048272 + tuning Gate2 (MAX_EMPTY_STREAK 2, max_iterations bench 5, timeout 30s, prompt hardening absolute path, --no-memory auto para modelos pequeños)
**Hardware:** RTX 2060S 8GB, CachyOS
**Ollama:** smollm2:1.7b 1.8GB (Q4_K_M), qwen3:4b 2.5GB, ornith:9b ref 5.6GB

## Honestidad — método de grading

**Todos los bench en este doc son homemade `KittyQuiz-polyglot` con grading `string-match` (`content.contains(check_str)`) — NO es Aider Polyglot oficial, NO es unit-test, NO comparable.** Grading por ejecución no existe aquí; un archivo con solo la firma pasa. Ver `AUDIT.md`/`CNAS.md` culture. Este floor mide **single-file trivia** (python fácil, primeros 15), no capacidad real multi-file.

## Evidencia smoke (limit 15, polyglot JSON, --no-memory, max_iter 5, timeout 30s)

Comando: `cargo build --release` (34s) → `timeout 600 ./target/release/kitty-cli --model <model> bench polyglot --limit 15 --exercises benches/exercises_polyglot.json`

| Modelo | Passed (15) | Rate | Latencia avg | Status | Estimado 225 |
|--------|-------------|------|--------------|--------|--------------|
| **smollm2:1.7b** | 8/15 | **53.3%** | 2.76s (344ms-5918ms) | ✅ GATE PASSED (≥50) pero borderline | 225×2.76s=10.3 min, pero extrapola a <50% en full (rust/go caen) |
| **qwen3:4b** | 8/15 | **53.3%** | ~24s (20-30s, 5 timeouts 30s) | ✅ GATE PASSED (≥50) pero lento | 225×24s=90 min (>30 min target) |
| **ornith:9b ref** | 217/225 (run1/run2) | **96.44%** | 4.6s | ✅ | 1054s (17.6 min) |

### Detalle smoke limit 3 (post-tuning, --no-memory auto)

| Modelo | limit 3 | Rate | Latency avg | Nota |
|--------|---------|------|-------------|------|
| smollm2:1.7b | 3/3 | 100% | 4.22s (4355/2416/5881) | Falso optimismo sample pequeño, luego 53% en 15 |
| qwen3:4b | 2/3 | 66.7% | 28.5s (30000 timeout +29987+25390) | 1 timeout, luego 53% en 15 con 5 timeouts |

### Archivos evidencia

- `/tmp/resmoke_smollm2_limit3.json` (3/3 100% 4.2s) → `benches/bench_sub2b_smollm2_limit15.json` (8/15 53% 2.76s)
- `/tmp/resmoke_qwen3_limit3.json` (2/3 66% 28.5s) → `/tmp/resmoke_qwen3_limit15.json` (8/15 53% ~24s)
- `benches/exercises_polyglot.json` 225, `benches/exercises_tb2.json` 89 (loader TB2 incompat, no usado)
- Contaminación CWD `solution.py` limpiada pre-bench (cambiaba 55s PASS → 90s timeout antes)

## TB2 (homemade, no Terminal-Bench 2.0)

- smollm2: hay evidencia 0% 8s avg limit 3 builtin (0/3), no full.
- qwen3:4b TB2 smoke timeout (thinking 4200/2048 retry) — no full.
- TB2 JSON loader `main.rs:322` no soporta `check_type: file_exists` → `--exercises benches/exercises_tb2.json` da 0 exercises, defer a P1. **Fix P1b aplicado:** `json_to_bench_item` ahora soporta `check_type: file_exists/file_not_exists/file_executable/file_content_exact/always_pass` con `check_path`.

## Diagnóstico post-tuning

**Tuning aplicado Gate2 (2 líneas → 69 líneas):** `MAX_EMPTY_STREAK 3→2` + warn streak, `max_iterations 20→5` bench override, `timeout 120s→30s`, prompt `Work in directory: {} — Save to {}/<check_file> (absolute, fix .py hardcode)`, guard contaminación CWD `remove solution.py/js/sh`, `--no-memory` auto para modelos pequeños, thinking 2048 (no cambiado, pendiente 512). Gate 50% + exit-code 0/1/2 (P1b).

**Efecto:** smollm2 60s→4s avg (100% limit3, 53% limit15), qwen3 39s→28.5s avg limit3, 24s limit15. Sin tuning, floor 225 era 2.4-3.4h >1800. Con tuning, smollm2 10 min pero 53% borderline, qwen3 90 min >30 min target (aún lento).

**Por qué 53% no es 96%:** grading string-match es vanity; muchos PASS son firma sin lógica (falso positivo). Errores `Write blocked /tmp/.tmp*`, `Invalid path missing`, `ToolLoop Write×3`, `Edit not loaded` frecuentes. Fail-fast 2 ahora dispara correctamente (logs `cap reached (2)`), agota 5 iteraciones en vez de 20.

## Gate go/no-go (spec bench-reproducibility: <50% → re-encuadrar)

**Veredicto para re-review Gate 2:**

- **smollm2:1.7b (tier sub-2B 2GB): NO-GO para claim "coding agent sub-2B"** — 53% en python fácil (top-15 trivial) extrapola a **<50% en 225 completo** (rust-impl traits, go-generics caen). Grading string-match ya infla; con unit-test caería a ~30-40%. Recomendación: **re-encuadrar** tier sub-2B a `demo single-file trivia, no agent fiable` (STRATEGY.md §5). No construir P0/P1 optimizado para tiny antes de grammar spike (P1). Mantener claim honesto "2GB demo" no "coding agent".
- **qwen3:4b (tier 2-4B): GO condicional** — 53% en 15 fácil sugiere ~50-60% en 225 string-match (similar a smollm2), justo sobre gate pero con latencia 24s (>30s target casi). Con tuning thinking 512 y prompt hardening puede bajar a ~15s. Recomendación: **mantener claim 2-4B** como piso `small` (no sub-2B), construir P0 profiles con `tiny` cap 5/512 y `small` cap 10/1024.
- **ornith:9b sigue titular** — 96.44% 4.6s, harness es el producto.

**Próximo paso:** no correr floor 225 aún (90 min qwen3, 10 min smollm2 pero <50% esperado). Hacer re-review Gate 2 con esta evidencia, luego P0 Fundación (ChatClient trait + ModelProfile tiny/medium/large + 2-stage routing + safety) con caps ya validados como datos de perfil, no como `if tiny`.

## Cómo reproducir

```bash
cargo build --release
ollama list # smollm2:1.7b, qwen3:4b
./target/release/kitty-cli --model smollm2:1.7b bench polyglot --limit 15 --exercises benches/exercises_polyglot.json --out /tmp/smollm2_15.json
./target/release/kitty-cli --model qwen3:4b bench polyglot --limit 15 --exercises benches/exercises_polyglot.json --out /tmp/qwen3_15.json
jq '.rate_pct, .passed' /tmp/smollm2_15.json
jq '[.details[]|.latency_ms]|add/length' /tmp/smollm2_15.json
```

## Notas

- Grading: `string-match-homemade (not comparable to Aider Polyglot)` etiquetado en todos los JSON/MD (cultura AUDIT).
- 225 full no ejecutado (estimado arriba, spec detener si >30s, ahora <30s para smollm2 pero qwen3 aún 24s).
- TB2 floor pendiente fix loader P1.
