# INFORME DEFINITIVO — System One (jev-to-opencode) → Kitty Code (2026-10-07, cristalizado ora-2)

## 1. Qué es (ficha verificada)
`jev-to-opencode` = investigación System-One→opencode/slim, no runtime. System-One = clasificación tipada 1-forward sin generar (routing/triaje/rerank/guard con probs+conf) para gates pre/post-LLM. Núcleo: Laya (Apache-2.0: buckets+ECE+temp-fit, bug 0.4.0 → pin 0.3.28) + Jev wire (`/v1/systemone`, 1.13.0 congelado) + alternativas medidas (decider-2B, any2jev, system1, foq, edgejev) + 18 repos ecosistema + 7 alternativas. Handoff + INFORME (11 secciones) + VRAM math 2060S ya existen ahí.

## 2. Veredicto final por pieza (explorer + oracle adversarial)

### HACER AHORA (orden nota÷riesgo÷VRAM)
1. **O3 system1 closers — pero PRIMERO baseline TF-IDF propio** (0 VRAM, 0.4ms CPU): su propio doc concede paridad TF-IDF 2/3 → baseline puro-Rust días sin riesgo; port `.s1m` v2 solo si no llega. Conformal = umbral ordinario hasta recalibrar en datos kitty.
2. **O1 approve-tool vía Laya REIMPLEMENTADA sin torch** (0 VRAM, D5): fit con tráfico propio obligatorio (zero-shot 0.362 inútil); fail-open hasta precisión medida. Destino `guard/` + `router.rs`.
3. **O4 fast-jev-compact con QR = bench-ab propio** (0 VRAM, D3): A/B interleaved k≥3 + wilson contra compaction existente; "-60%" vendor hasta pasar ese QR.

### SPIKE-FIRST / CONDICIONAL
- Decider-2B: triple bloqueo (qwen35 con bug abierto + bf16 sin Turing + single-resident impide "lado a lado") → máximo wire-client + sidecar `llama.cpp /v1/systemone`; binario al fondo.
- any2jev: proyecto tamaño-K2 (tiene precedente: port k2_horizon; hero 42ms en ESTA GPU); head exige hidden-states (tore+head en engine o seam nuevo).
- Vacuity→router: con señales existentes, sin logits. Halting/margin: bloqueados hasta plumbing logprobs.
- CWM heurístico, Episodic-settled con TTL, critique opt-in con presupuesto, sleep SOLO con QR/holdout.

### VETOS (ni loco)
1. EBA/wrong-log/embeddings en bench (infla k≥3; BGE-M3 HELD-offline).
2. BoundedNorm → engine (sin contraparte tensorial).
3. Pesos `.pt`, verificación con W aprendidos, `export_gguf` (procedencia + torch + offline).
4. Sleep-consolidation sin QR hacia vía de inyección (envenena benches).
5. Hooks HF por índice / monkey-patch vllm; 4 repos sin LICENSE (ni código ni estructura); Jev-cloud/G5 y toda key (jubilada).
6. O7/foq eliminado de oportunidades (contradictorio: listado y vetado a la vez); edgejev-Xeon-only; verdict/von.
7. Importar umbrales/ECE/conformal/T de origen sin recalibrar + NOTICE al reimplementar (precedente `engine/NOTICE`).
8. Gates/closers en paths de bench por defecto (OFF + estado en manifiesto si se activa).

## 3. Riesgos (no minimizar)
- Umbrales no transfieren (ni los "buenos": ECE/decider/system1 son de sus suites) → recalibrar todo en tráfico kitty.
- Docs drift triple-capa (handoff + INFORME + .slim se corrigen entre sí) → verdad = código + MODEL_CARD.
- Vendor benches ≠ 2060S salvo any2jev-42ms → re-medir k≥3 todo lo citado.
- Estado por-query permanente en origen → TTL/invalidación obligatorios aquí.
- Critique post-tool = +1 roundtrip (impuesto D3): opt-in con presupuesto.

## 4. Trampa de integración (la importante, misma familia que halting-adaptativo)
`ToolRegistry::execute` + loop son compartidos por interactivo Y bench: cualquier gate/closer aterriza en bench y las celdas k≥3 dejan de ser comparables (los manifiestos no registran estado de gates). Regla: bifurcación obligatoria en camino bench (precedente `--hops` flag-only, `--no-memory`) + campo de manifiesto. El punto de integración más natural es exactamente el que rompe el contrato.

## 5. Plan de ejecución propuesto
- Oleada 1 (ya): TF-IDF baseline + vacuity-router + compact-QR (0 VRAM, 0 comportamiento en bench).
- Oleada 2 (paralelo): spike logprobs (desbloquea halting/margin) + CWM OFF-bench + Episodic-settled con TTL.
- Oleada 3 (condicional): any2jev-head (tras consolidar slot-fix) + critique opt-in + sleep-con-QR.
- Nunca: vetos 1-8. Cada oleada red-green + manifiesto k≥3 donde aplique. Sin commit (a tu orden).
