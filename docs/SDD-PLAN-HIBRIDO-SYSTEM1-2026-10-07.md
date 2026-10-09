# SDD HÍBRIDO — System-One × Dual-Loop → Kitty Code (2026-10-07)

Fusión de `docs/INFORME-SYSTEM1-JEV-2026-10-07.md` + `docs/INFORME-DUAL-LOOP-2026-10-07.md`, con contradicciones resueltas y duplicados unificados. Tesis: **no quemar el SLM en lo que un clasificador de 0.4ms resuelve** — routing, triaje, guard y esfuerzo fuera del LLM; el SLM solo genera.

## 1. REQUISITOS Y ALCANCE
- Problema: SLMs 1-9B queman iteraciones/latencia/VRAM en decisiones que no exigen generar (triaje, guard, esfuerzo, rerank, compactación ciega).
- HU-1 router de esfuerzo unificado: Given señales baratas (vacuity + margen top1-top2 + streaks + quality_monitor), When un router decide hops/fallback/off, Then -iteraciones inútiles medidas en manifiesto (k≥3), sin tocar paths de bench por defecto.
- HU-2 clasificadores 0-VRAM: Given TF-IDF baseline + system1-closers + Laya-reimplementada-sin-torch, When gates pre/post-tool, Then decisiones en ms con precisión recalibrada en datos kitty (fail-open hasta evidencia).
- HU-3 compactación con QR: Given bench-ab propio, When A/B compact-vs-actual, Then -tokens con wilson significativo o se revierte.
- HU-4 observabilidad: Given telemetría hops/iter/margen, When manifiesto la incluye, Then D6 con evidencia aceptada.
- Fuera: pesos `.pt`, torch en runtime, EBA/wrong-log en bench, sleep sin QR, hooks HF, Jev-cloud, importar cifras/umbrales ajenos, tocar `engine/` salvo spike logprobs.

## 2. MODELO DE DATOS Y ENTIDADES
- `EffortSignal{vacuity, margin_top1_top2, empty_streak, invalid_streak, verify_fail}` — fusión vacuity×conflict-gate (duplicado unificado: una sola señal, no dos gates).
- `GateDecision{route, hops_budget, fallback, off_reason}` + `off_in_bench: bool` (campo de manifiesto obligatorio si se activa).
- `ClassifierVerdict{label, conf_recalibrated, ttl}` — TTL/invalidación obligatorios (estados permanentes prohibidos).
- `KnowledgeCard` solo vía QR/holdout (sleep destilado con gate, nunca directo).
- `TelemetrySample{hops, iters, margin?, entropy?}` (margen/entropía solo si existe el spike logprobs).

## 3. ENFOQUE TÉCNICO Y ARQUITECTURA
- Capas: señales baratas (existentes) → `EffortRouter` (`policy.rs`/`router.rs`, fail-open) → clasificadores 0-VRAM (TF-IDF → system1 → Laya-reimpl) → SLM solo-genera → closers post-tool (opt-in, con presupuesto) → telemetría → manifiestos.
- Spike logprobs por backend (Ollama `logprobs/top_logprobs` + seam engine→wire): desbloquea halting/margin; hasta entonces, vacuity con señales existentes (orden resuelto: interim primero, óptimo después).
- Episodic-settled con TTL; critique opt-in presupuestado; CWM/compact OFF-bench con QR bench-ab.
- Prohibido: adaptativo en bench (OFF + campo manifiesto), umbrales ajenos sin recalibrar, NOTICE al reimplementar patrones no triviales.
- Resoluciones de contradicciones: EBA-rerank vs RRF/shortlist → se queda RRF/shortlist + wrong-log vetado; sleep dual-loop vs sleep-jev → una sola puerta con QR; CWM vs compact-QR → mismo bucket, QR bench-ab decide.

## 4. DESCOMPOSICIÓN DE TAREAS ([P] = paralelo)
- W0 Telemetría → manifiestos (contadores ya existen; 0 comportamiento) [P].
- W1 `EffortRouter` vacuity+margen con señales existentes + tests (fail-open) [P].
- W2 TF-IDF baseline closers + benchmark vs SLM-entero (si no llega → system1-port) [P].
- W3 Spike logprobs por backend (disponible; halting/margin NO suben a W4 por contrato Laya — W4 sin ellos, cableado bloqueado lane harness) [P].
- W4 Laya-reimpl sin torch + fit tráfico propio + QR (despliegue fail-open hasta precisión) [tras W2].
- W5 Compact-QR bench-ab (si wilson OK → adopta; si no → revierte) [P].
- W6 Episodic-settled con TTL + critique opt-in presupuestado [P].
- W7 Sleep-con-QR/holdout hacia knowledge/*.md [tras W5].
- W8 Recalibración global (todos los umbrales en datos kitty) + NOTICE + deuda.
- Nunca: vetos 1-8 unificados (EBA-bench, BoundedNorm-engine, .pt, sleep-sin-QR, hooks-HF, sin-LICENSE, cloud, cifras ajenas, gates-en-bench).
- Cada oleada: red-green + manifiesto k≥3 donde aplique + DoD (fmt/clippy/tests). Sin commit (a tu orden).

**Mapa de nota**: W0+W1 D6/D3 sin mover el loop; W2-W4 D3 medible (p50/p99) + D5 (guard real); W6-W8 sostén + credibilidad. Techo honesto: acerca a 8.2, no lo promete solo.
