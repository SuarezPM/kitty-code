# MEGA-PLAN SDD — Kitty Code estado del arte (1B–35B, hardware de consumo)

Plan mode: especificación, cero código. Base: 7.9/10, 763+ tests, serie VRAM, vetos vigentes. Tesis: el único harness diseñado para SLMs locales + motor embebido + sandbox real. Evidencia externa: stateful inference (2.1×/turno), PASTE (−43.5%), AgentDiet (−40–60% tokens), spec-decoding n-gram sin draft-model, Q4_K_M default, KV-quant q8_0, Qwen3.5-9B 6.6GB.

## 1. REQUISITOS Y ALCANCE
Problema: ganar en velocidad (no empatar), VRAM (no desperdiciar) y poder agéntico (no colapsar) con SLMs 1-35B en 8GB, hasta ser alternativa real a PI/opencode/little-coder + lanzamiento a comunidad.
Historias + AC:
- HU-INSTALAR: Given máquina limpia con Rust/curl, When `curl ... | bash` + `kitty doctor`, Then TUI abierto en <10min con modelo funcionando (fail si pide compilar engine).
- HU-PRIMER-RUN: Given Ollama sin modelos, When `kitty run "..."`, Then propone `pull` exacto + perfil por VRAM medida (nunca 404 silencioso, nunca regex).
- HU-TUI: Given terminal 80×24, When se abre TUI, Then streaming tokens + permisos + tokens/coste + selector modelo + resume (paridad PI/opencode).
- HU-LATENCIA: Given tb2-8 ambos, When router/triage/compact activos donde deciden, Then p50/p99 ≤ little-coder con pass-rate ≥ baseline−5pp (baseline = propio sin optimizaciones, mismo modelo/backend/turnos/timeouts pineados, k≥3, manifiesto).
- HU-VRAM: Given 8GB, When perfil auto, Then Q4 + ctx justo + KV-quant + tiers sin OOM (matriz publicada por modelo).
- HU-PODER: Given tareas 20+ turnos, When recovery + constrained + determinismo, Then tasa colapso (abandono/error-no-recuperado/timeout tras 20 turnos) < mitad del baseline tb2-8 medido y trazas de decisión reproducibles con semilla (no bit-a-bit en GPU).
Fuera: reescribir engine upstream, BGE-M3 offline, E2E-CI con GPU, desktop/IDE, auditoría externa (T11.2).
DoD por tarea: fmt/clippy/tests + bench k≥3 si toca loop + manifiesto + rollback; vetos 1-8 vigentes. Excepción engine (W5/W6 si lo tocan): diff mínimo vs upstream + test que falle sin el fix + `-A` declarado (protocolo `ci.yml`, alcance "engine tocable" explícito).

## 2. MODELO DE DATOS Y ENTIDADES
- `DoctorReport{ollama, models[], vram, backend_recomendado, acciones[]}` (nuevo `kitty doctor`).
- `ModelProfile` extendido: VRAM-medida + Q4-default + KV-quant flag + tier (triaje/ejecución) + temp/top_p por familia.
- `EffortSignal`/`GateDecision` (existen; activar donde deciden) + `TurnEvent` con tokens/turno (dieta medible).
- `BenchManifest` + campo `gates_off` (trampa conocida) + `origin` perfil build.
- `SupportedModel{arch, gguf, q4_size, ctx_rec, backend}` (tabla pública 1B-35B curada: Qwen3.5-9B, Granite-8B, Gemma-E4B, Phi-4-mini, Qwen3-4B…).
- Sin migraciones: todo aditivo con defaults compatibles.

## 3. ENFOQUE TÉCNICO Y ARQUITECTURA
- W-INSTALAR: binario `kitty` (`[[bin]]`, no `kitty-cli`) + script `install.sh` (`set -euo pipefail`, verifica exits, `curl -f`) + quickstart en README (3 comandos a TUI funcionando) + release matrix (ubuntu x86_64 en esta fase; mac/win declarados fuera hasta CI por objetivo) con `--features tui` + `cargo install` + feature-split (engine opcional); `doctor/init` con autodetección (Ollama/VRAM/disco) + bootstrap Ollama (detectar → comando exacto → re-check) — nunca regex de nombre; SBOM/cosign/provenance ya existen en `release.yml` (extender matriz, no construir).
- W-TUI (`@designer` + verificación visual): streaming + permisos + diffs + tokens/coste + selector + resume + hints por foco + consola overlay; UI = función pura del estado.
- W-MOTOR-SLM: tiers triaje/ejecución + compact-QR + closers + router activo + constrained enforcement + speculative tool-calls (PASTE: solo read-only, con co-scheduling) + PLD n-gram para tool-calls + prompt-deterministic cache para retries idénticos.
- W-VRAM: ctx justo + KV-quant q8_0 (+flash-attn, si no se ignora silencioso) + Q4 default + GPU-layers por medición + serie 1 modelo + toolchain CUDA alineado (contenedor o cudarc).
- W-PODER: recovery tras fallo + anti-loops + determinismo semillas + goldens + E2E con modelo en CI cuando haya GPU.
- Prohibido: adaptativo en bench, umbrales ajenos sin recalibrar, EBA-bench, `.pt`, hooks-HF, cloud sin orden, `--all-features` como gate, inventar timeouts/celdas.
- Normas: un fix un crate un PR; boundaries por símbolo; verify-then-cut; evidencia en repo. Evidencia GPU en ventana serial (HEAD congelado, árbol limpio, serie VRAM exclusiva, 1 celda a la vez); deps: W1←W6, W4←W6, W9←W8+W2; [P] solo código (W1/W2/W3 + parte código de W4/W5).

## 4. DESCOMPOSICIÓN DE TAREAS ([P] = paralelo)
- W0 Medir desglose real (TurnEvents + console, wall = modelo/guard/sandbox/tools). Serie VRAM exclusiva (ventana de evidencia).
- W1 Install + doctor + script [P] + defaults por hardware medido [tras W6; si la detección es runtime, W1 entero queda [P]]. AC: `kitty doctor` propone pull exacto + backend sin regex.
- W2 TUI streaming + permisos + sesiones (@designer, verificación visual) [tras W1].
- W3 MCP client + plan-mode + skills-producto + undo [P]. AC: MCP lista tools contra servidor de prueba + plan-mode bloquea edits en test + skill invocable por CLI + undo revierte un edit en test.
- W4 Tiers + compact-QR + closers + router-donde-decide (marcar latencia tb2-8) [P] (solo código; la medición GPU va en ventana serial).
- W5 Constrained + speculative-readonly + PLD-n-gram + response-cache idempotente [P] (solo código; validación con suite FC-15 propia a crear: 15 tool-calls sintéticos con veredicto conocido). AC: tool-calls malformados → 0 en suite FC-15 con gramática activa (vs baseline sin ella).
- W6 VRAM: ctx/KV-quant/tiers/GPU-lane + matriz publicada. Serie VRAM exclusiva (medición GPU).
- W7 Recovery + determinismo + goldens + E2E-CI. Serie VRAM exclusiva (goldens con modelo).
- W8 Modelos curados + enforcement + routing tiers + demo 12-ejercicios vs PI/opencode/little [tras W4-W6]. Demo = suite fija versionada `demo-12` (JSON propio; `--exercises` no filtra polyglot) con protocolo de justicia (versiones pineadas, mismo modelo/backend/turnos/timeouts, k≥3).
- 10-min: `kitty update`/check de versión; crash-report vs ADR-PANIC-STRATEGY (hook-before-abort o silencio declarado); telemetría opt-in explícito con política; i18n declarado fuera (criterio BGE-M3/E2E-GPU).
- W9 Release: SBOM firmada + provenance + changelog + anuncio comunidad. AC: artefactos verificables (`cosign verify-blob` + `slsa-verifier`) + anuncio solo con claims reproducidos (manifiestos citados).
- Nunca: vetos 1-8 + EBA-bench + BoundedNorm-engine + sleep-sin-QR + gates-en-bench.
- Mapa honesto: W0-W3 = paridad producto; W4-W6 = ventaja medible (latencia/VRAM); W7-W9 = credibilidad + lanzamiento. Techo: 8.2 interno, 10 con externa.
