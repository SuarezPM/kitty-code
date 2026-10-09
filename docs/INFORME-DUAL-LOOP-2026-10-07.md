# INFORME DEFINITIVO — dual-loop-controller → Kitty Code (2026-10-07, cristalizado ora-2)

## 1. Qué es (ficha verificada)
`dual-loop-controller` (Ch3nOff, Python/PyTorch, MIT, v4.5.0): agente cognitivo que congela un SLM y le añade deliberación latente System-2 (outer loop, sin emitir tokens) + generación System-1 (inner loop). Módulos: `controller.py` (outer), `decoder.py` (inner), `memory.py`, `halting.py`, `firewall.py`, `evidential.py`, `cognitive_os.py` + ~15 engines experimentales. **NO es** disaggregación prefill/decode: es control de lazos estilo PonderNet/ACT (gate por conflicto → deliberar K pasos o bypass).
Advertencia de honestidad: v2.0.0a1 reporta resultados negativos reales (K extra no mejora, 29.4% vs 6.25% azar); v2.2.3 marketiniza (100% en 15/20 benches, "PPL que mejora"). Creer solo código + repro `comprehensive_suite.py`, nunca el README.

## 2. Veredicto final por pieza (explorer + oracle adversarial)

### HACER AHORA (inputs ya existen, riesgo mínimo)
1. **Telemetría hops/iter → manifiestos** (D6 puro): los contadores ya se imprimen en logs; manifiestos son evidencia aceptada; cero cambio de comportamiento.
2. **Vacuity → router de esfuerzo** (`policy.rs`/`router.rs`) con señales existentes (empty-streak, invalid-streak, quality_monitor): ataca el hops-ciego sin logits; ahorra iteraciones = latencia (D3).
3. **CWM → compresor de contexto heurístico** (`context.rs`, flag OFF en bench): menos tokens de prompt → p50/p99 medibles. Reset de expectativas: la calidad *entrenada* no se hereda.

### SPIKE-FIRST (bloqueado hasta input disponible)
- Halting entropía + margin gate: exigen **logits por token**, y kitty no los expone (`logprob|logit` en `crates/` = 1 comentario). Spike: plumbing logprobs por backend (Ollama `logprobs/top_logprobs` + seam engine→wire). Si sale → suben a #1-2.
- BoundedNorm: sin concepto "delta" fuera del loop torch → sin contraparte en kitty.

### REIMPLEMENTAR ADAPTADO (con condiciones)
- Episodic settled-anchors → `episodic.rs` (ya tiene blake3/SQLite): añadir `settled` + umbral **con TTL/invalidación** (la fuente limpia a mano).
- Critique textual post-tool → `quality_monitor.rs`: SOLO opt-in con presupuesto (cada crítica = +1 roundtrip = impuesto D3 en 8GB).
- Sleep-consolidation → `knowledge/*.md`: SOLO con QR/holdout previo (sin gate = envenenar benches).

### VETOS (ni loco)
1. EBA/wrong-log/embeddings en bench (contrato timeouts/semillas/interleave + BGE-M3 HELD-offline + infla k≥3).
2. BoundedNorm → engine/candle (divergencia por nada).
3. Pesos `.pt` / verificación "fiel" con W aprendidos / `export_gguf` (procedencia desconocida + torch + offline).
4. Sleep sin QR hacia vía de inyección.
5. Hooks por índice HF / monkey-patch vllm (corrupción silenciosa al cambiar de modelo).
6. Importar NINGUNA cifra suya (100%, PPL, +33-40%) sin repro + re-medición protocolo kitty (k≥3, wilson, manifiestos).
7. NOTICE: atribución para patrones no triviales (precedente `engine/NOTICE`); licencias de *datos* de sus `.pt` desconocidas.

## 3. Riesgos (no minimizar)
- Docs no autoritativos: 3 RELEASE_DRAFT + README_PYPI ≠ README + `HARNESS_SPEC.md` de otro proyecto → solo el código manda.
- Benchmarks sin réplica (sin k/seeds/CI) + 15 `.bat`, `archive/`, notebooks muertos.
- Deps torch/transformers/vllm/fastapi incompatibles con offline-Rust 8GB (cargo deny/SBOM).

## 4. Trampa de integración (la importante)
**Halting adaptativo en paths de bench**: los manifiestos registran budgets fijos y el contrato gates/semillas hace que celdas con presupuesto adaptativo dejen de ser comparables con todo lo medido (rompe k≥3 retroactivamente, sin test que lo diga). Segunda mitad: importar sus umbrales (1.30 nats/p75 de *sus* logits) — la entropía varía por familia/cuantización/backend → el halt dispararía siempre o nunca sin que nadie lo vea. Regla: **adaptativo solo interactivo, OFF en bench, recalibrado con logits de kitty cuando existan**.

## 5. Plan de ejecución propuesto
- Oleada 1 (ya): telemetría → manifiestos + vacuity → router (inputs existentes, 0 comportamiento).
- Oleada 2 (paralelo): spike logprobs por backend (si OK → halting/margin suben) + CWM heurístico OFF-bench.
- Oleada 3 (condicional): critique opt-in + Episodic-settled con TTL + sleep-con-QR.
- Nunca: vetos 1-7. Cada oleada con tests red-green + manifiesto k≥3 donde aplique. Sin commit (a tu orden).
