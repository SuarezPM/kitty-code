# SDD Definitivo — Kitty Code + Harness²-reescrito + System-One (ship v0.1)

Fecha: 2026-10-23 · Estado: propuesta para ship · Objetivo: dejar de desarrollar y shippear un producto funcional, disruptivo y mediático sin inventar cifras.

## 0. Contrato de producto (no negociable)

TODO esto es UN SOLO BINARIO Rust, UN SOLO PROGRAMA, UN SOLO COMANDO.

- Abrir el terminal, tipear `kittycode` y que se abra la TUI para vibecodear y hablar con el agente. Sin subcomandos para lo básico, sin Python, sin Node, sin sidecars, sin daemon que instalar aparte.
- `kittycode` = binario `kitty` renombrado/aliasado (hoy compila como `kitty` + compat `kitty-cli` desde el mismo `src/main.rs`; el release publica UN artefacto con TUI compilada). System-One y Harness² NO son servicios: son módulos dentro del mismo proceso (System-One = clasificador 0-VRAM + decision-model GGUF opcional vía `/v1/systemone` del propio motor; Harness² = `bench ppc` + `memory_items` dentro del mismo SQLite, solo offline).
- System-One NO es sinónimo de Laya. Laya es UNA candidata (encoder 421M sin torch). El contrato es el wire tipado `choice/score/noul` 1-forward 0-output-tokens: vale TF-IDF puro-Rust, ONNX-encoder (183MB), d1-omni-600M GGUF, decider-2B `.s1m`, o el que gane la calibración W8. Se enchufa y desenchufa sin recompilar el binario (feature/modelo, no dependencia dura).
- Harness² NO es un repo aparte. Es código Rust dentro de `kittycode` (`bench/ppc.rs` + `memory/lessons.rs` + runmark/ledger/tripwires). Nada de `harness2/` Python al lado.

## 1. Tesis en una frase

Kitty Code sigue siendo Kitty Code (single-binary Rust, 1 residente, Ollama default, Landlock real): tras este SDD es **el único harness local 1B-35B que aprende solo (Harness² en Rust, offline) y decide barato (System-One 0-VRAM), con benches que no pueden mentir**.

## 2. De dónde viene cada pieza (y qué no es)

- **Kitty Code (base):** loop internalizado `agent/run.rs` (verify/autorun/max_iterations 20, retries, doom-loop, verificación workspace), backends OllamaClient default no-streaming + KittyInferClient opt-in in-process (mistralrs), memoria episodic SQLite + BM25/RRF/hybrid (hoy recall real = Jaccard top-3 sin gate), bench demo-12 k≥3 + manifiestos + Wilson en `ab.rs`, tiers Triage/Execution + EffortRouter (degrada esfuerzo, nunca modelo). Techo real 8GB = 7-9B Q4 16K.
- **Harness² (Google, arXiv 2610.14224, 9-oct-2026, Apache-2.0):** NO es inferencia (delega a OpenCode/Codex CLI, timeouts 2400-3600s, presupuesto en chars). Su oro es disciplina: Propose-Probe-Compose sin ground-truth (referencia → 2 candidatos invariante+domain → probe misma tarea → compose selectivo), library cue/lesson/evidence con gate + retain/forget, blind IDs, noise-floor k=1, ungraded≠zero, runmark fingerprint + resume atómico, ledger identidad R=4k/P=3k, tripwires anti-leak, preflight agregado. En Kitty se **reescribe en Rust, secuencial k=1→k=2, mismo SLM, verifier determinista, solo offline**. Paralelo, juez LLM y R=4k/P=3k son humo en 8GB (un residente; Ollama serializa o OOM).
- **System-One (doble sentido, no confundir):**
  - (a) Clásico Kahneman: S1 rápido / S2 lento (Talker-Reasoner 2024, SOFAI, D-Mem: retrieval barato S1 + deliberación S2 + quality gating).
  - (b) Nuevo 15-sep-2026 TypeSafe: decision-model tipado no-generativo (choice/score/noul, 1 forward, 0 output tokens, wire `/v1/systemone` que ya exponen llama.cpp/Ollama; OpenAI `/v1/decisions` GA 22-oct-2026).
  - En Kitty System-One = **clasificación tipada 1-forward sin generación** (tesis: no quemar el SLM en lo que un clasificador de 0.4ms resuelve). Pirámide 0-VRAM: TF-IDF puro-Rust baseline → port system1 `.s1m` solo si no llega → Laya reimplementada sin torch (pin 0.3.28). Vetados: `.pt`, torch runtime, hooks HF, Jev-cloud. Estado: EffortRouter + EffortSignal fail-open OFF-en-bench, TF-IDF closer `enabled=false`, logprobs spike en runtime sin calibrar (W8 HELD, 0/200 etiquetas), critique opt-in OFF. OJO homónimo: `hybrid-plugplay-fix` / `hybrid_cache` = SSM-híbrido (Mamba+atención), no System-One.
- **Contraste:** Harness² optimiza *el harness* entre tareas (lento, offline); System-One abarata *la decisión* dentro del turno (rápido, online). Uno sin el otro cojea: PPC con memoria sin gate envenena; System-One sin ledger/runmark mide humo.

## 3. Arquitectura objetivo (3 capas, todo Rust)

- **Capa 0 — Integridad (sin esto nada es publicable):** deny `bench/**` en Guard + tripwire answer-key por nombre y contenido; runmark blake3 (model+backend+perfil+ctx+max_iter+hops+suite+timeouts+seed+git) + manifiesto atómico + `--resume` solo celdas; ledger identidad (run_id, arm, git_head, dirty, fingerprint, blind); presupuesto a tokens punto único (contar tool_args, system, reconciliar `prompt_eval_count`); `for_bench` + `recall` OFF resuelto y volcado; doctor valida celdas `timeouts.json` + aviso Ollama CPU-only + perfil por VRAM.
- **Capa 1 — System-One online (barato, fail-open):** TF-IDF closer → Laya-sin-torch calibrada con datos kitty (W8: 200 etiquetas) → decision-model GGUF `/v1/systemone` (d1-omni-600M ~407MB) solo si TF-IDF no llega; hybrid thinking por rol (thinking OFF en triage 1-4B, ON capado en ejecución); ngram-spec gratis + MTP si el GGUF lo trae (DFlash no cabe); AgentDiet (dieta trayectoria: inútil/redundante/caducada, -40~-60% input) + prompt corto Pi-style (<2k, 4 tools); KV persistente por sesión en kitty-infer (stateful O(Δt) 2.1x/turno). Todo fail-open, todo OFF-en-bench con campo manifiesto.
- **Capa 2 — Harness² offline (lento, con gate):** `bench ppc` secuencial k=1→k=2, proposer = mismo SLM perfil triage temp baja + parser estricto, candidatos = deltas config (prompt style, tool subset, temp, iter cap), compose determinista por verifier + merge diff (no LLM judge); tabla `memory_items` (cue/lesson/evidence/domain, hash, provenance ledger, evidence_ref obligatoria, confidence, TTL, supersede) sin migrar episodios; recall_hybrid tras eval T6.3 (hybrid ≥ Jaccard@3) sin `fill_recent`; inyección bloque system reemplazable (no `role=user` acumulativo que contamina compactación); TTL/forget; solo runs con ledger limpio admiten lecciones.

## 4. Corte de ship

**v0.1 ENTRA:** repo público + tag `v0.2.0` (release.yml ya sabe: bin tui, sha256, cosign, SBOM, provenance) + E2E `curl|sh` en limpia + HU-8; doctor productizado (CPU-only + perfil VRAM); Ollama default (KEEP, VISION-ROADMAP), kitty-infer HELD-experimental; TUI W2 + bench demo-12 + undo/skills/plan/update + README/TUI al día (T9.1/T9.2); pack evidencia (tb2-8 24/24 vs 18/24, demo-12 35/36 vs 34/36 paridad, 0 ceros en k=3, serie VRAM 1-celda) con HEAD congelado; `cargo update -p rustls/zip` (7 RUSTSEC).
**v0.1 FUERA:** claims de victoria latencia (HU-LATENCIA ABIERTA, gates_off); 9B embebido como pitch (HELD 1501s); `--cloud-fallback` (solo warn, sin cablear); System-One calibrado y Harness²-PPC (v0.2 roadmap); E2E-CI GPU, mac/win, desktop/IDE, i18n, auditoría externa, mutants ≥80, holdout-runner, BGE-M3.

## 5. Titulares honestos (k≥3, con letra pequeña)

1. "24/24 (100%) vs 18/24 (75%) en suite Terminal-Bench casera 8, mismo modelo y GPU, con guard+sandbox" — `manifest-tb2-justicia-gpu.json`. Letra: casera no oficial; little más rápido (10.9s vs 27.2s).
2. "Empate técnico 35/36 (97.2%) vs 34/36 (94.4%), Δp50 190ms, brecha -96% sin tocar código" — `manifest-demo12-fasef-2026-10-10.json`. Letra: paridad, no victoria; HU-LATENCIA abierta.
3. "Cero ceros: ningún ejercicio 0/3 en dos series k=3 con sandbox on" — comparativo + fasef.
Reserva (serie 1-celda, no k≥3): "ornith:9b Q4 cabe en 8GB: pico 5391MiB 33/33 GPU sin OOM".

## 6. AC falsables (GIVEN/WHEN/THEN)

- F1: GIVEN 20 prompts adversarios (Read/Grep/cat bench) WHEN bench WHEN tripwire THEN 0 lecturas + verdict `leak`; GIVEN misma celda WHEN rerun WHEN runmark THEN rechazo con error; GIVEN 1 rep WHEN report THEN exit 2 no-evidencia.
- F2: GIVEN lift sin CI o sin run_id limpio WHEN report THEN "NO SIGNAL"; GIVEN timeouts WHEN p99 THEN `censored`.
- F3: GIVEN 50 goals no-relacionados WHEN recall THEN 0 inyecciones; GIVEN lesson sin evidence_ref o run con tripwire WHEN save THEN rechazo; GIVEN compaction WHEN memory THEN sobrevive y tarea no se poda.
- System-One: GIVEN 200 episodios etiquetados WHEN calibrar Laya THEN ECE medido; GIVEN TF-IDF ≥ paridad SLM THEN no port `.s1m`.
- PPC: GIVEN k=1 vs baseline 50 tareas k=3 WHEN McNemar/Wilson THEN exige +2 tareas CI>0 o se archiva k=2; GIVEN k=2 WHEN wall/tokens >2.5x o VRAM >7.5GB THEN fail.

## 7. Ficheros a tocar (orden)

F1: `bench/run.rs, common.rs, mod.rs, demo12.rs, ab.rs, runmark.rs(nuevo)`, `kitty-cli/main.rs`, `harness/agent/{config,setup,response}`, `tools/{io,search}`, `guard/*`, `sandbox.rs`.
F2: `harness/comparison.rs`, `bench/ab.rs`, `cli/doctor.rs`, `tui/*`.
F3: `memory/lessons.rs(nuevo), episodic.rs, embedding.rs`, `harness/agent/{response,run,config}`.
System-One: `harness/{router,tfidf_closer,laya_classifier,post_tool_critique}.rs`, `runtime/client.rs` (logprobs/margin), `kitty-infer` (grammar opt-in, seed, prefix-cache 1-2).
PPC: `bench/ppc.rs(nuevo)` offline; no tocar `router.rs` para modelo.

## 8. Vetos y riesgos

Vetos: sin cloud sin orden, sin dlopen/torch/`.pt`/hooks-HF, sin timeouts inventados (solo `timeouts.json` medido), sin `all-features` como gate, sin inventar celdas, serie exclusiva 1 residente, k≥3 + manifiesto.
Riesgos: (1) leak answer-key invalida todo lift futuro; (2) PPC triplica llamadas (20×3×13 ≈300/prompt) → OOM/timeouts sin ledger+cancel; (3) memoria sin gate envenena (Jaccard+fill_recent+proposer débil); (4) p50 27.2 vs 10.9 confundida por 32K/16K + loop sin early-exit —re-medir antes de atribuir; (5) anunciar sin release/repo-privado/CI-rojo = credibilidad destruida (ANUNCIO DO NOT PUBLISH).

## 9. Secuencia ship (dejar de desarrollar)

Público → tag v0.2.0 verificada (cosign/SLSA) → CI verde o waivers (sandbox hosted skip) → `cargo update rustls/zip` → reconciliar SUPPORTED-MODELS-8GB/README/TUI con manifiestos → doctor CPU-only+perfil VRAM → anuncio parity-only HEAD congelado → community-run doctor/run. Después: v0.2 = System-One calibrado + Harness²-PPC offline + lecciones con gate.
