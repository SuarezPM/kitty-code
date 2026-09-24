# BACKEND-DEFAULT-CRITERIA — cuándo `kitty-infer` podría ser default (Fase 3)

Decisión vigente: **Ollama sigue siendo el default. El switch a embebido
queda para un GO explícito del owner.** Este documento fija los criterios
medibles que deben cumplirse antes de proponerlo — no hay switch en este
carril.

## Criterios (todos, no la mayoría)

1. **Turnos completos embebidos en GPU**: MiniCPM5-2B y Qwen3-4B cierran
   `kitty run` end-to-end con `KITTY_BACKEND=kitty-infer` en RTX 2060S 8GB
   (hoy: verde, "forty-two", 41-117s, picos 3.1/4.8GB).
2. **K2-Horizon sin techo de tiempo**: el turno agente K2 debe completar en
   <10 min en 8GB o el spill mínimo debe documentarse como perfil soportado
   (hoy: estable sin OOM con spill-12 pero sin completar en 25 min).
3. **Paridad distribucional** ✅ VERDE (Gate 2, 2026-09-22): Qwen3-4B f16
   release-release mismo-GPU (kitty vs upstream d5ae0f1, prompt fijo,
   8 tokens top-5): argmax 8/8 (100%), set 8/8 (100%), Δargmax med
   0.000011 máx 0.015027 (<0.1); única diferencia 1 swap de cola r4/r5 en
   pos1. Umbral explícito: argmax 100% + Δargmax<0.1 + set 100%.
   Evidencias: /tmp/kitty-logs/ours-qwen3-f16.json vs
   upstream-qwen3-f16.json. K2 excluido por alcance (determinismo, no
   paridad — sin baseline upstream).
4. **Bench comparativo**: misma suite polyglot/tb2 en ambos backends con el
   diseño de `EMBEDDED-BENCH-SPEC.md`; el embebido no debe perder >5pp de
   pass-rate ni romper los gates existentes.
   Nota Gate C (2026-09-23): el 2/15-vs-3/15 qwen3 (infer-vs-ollama) queda
   NO-MEDIDO como paridad — runs únicos n=15, Δ±1 dentro del ruido; no citar
   como wash (absorbido por qwen3-ollama k>=3 @120s = {4,4,7} en Fase 3).
   Nota D1 (gate bench-k>=3, 2026-09-23): minicpm5-infer k>=3 @150s = {0,0,0}
   (+k4 0/15) es ROJO-solo-con-placement-GPU (k1: 4543MiB pico, util GPU 68%);
   k2-k4 CPU-spill (`Layers 0-41: cpu` del auto-map pese a GPU libre y ollama
   vacio) quedan NO-MEDIDOS como paridad GPU — 0/15 medido como conducta CPU.
   qwen3-infer k>=3 @180s = {0,0,0}, todo CPU-spill. Manifiestos (13):
   `benches/ab/manifests/manifest-20260923T*-f3-*.json`.
5. **Errores accionables**: todo fallo de `KittyInferClient` clasifica con
   hint (OOM/KV/tokenizer/modelo ausente); cero pánicos escapando al CLI en
   los paths probados (hoy: `expect` del scheduler paged en prompt largo
   con CTX corto — mitigado con CTX mínimo documentado, pendiente endurecer
   a error amable).
6. **Docs al día**: tabla CTX-mínimo (apéndice serving en `engine/F0-RECON.md`)
   y este archivo actualizados con los números del humo que propone el cambio.
7. **Robustez en runs encadenados**: entre runs GPU consecutivos se verifica
   contexto sano (`nvidia-smi` + cola del log previo sin `CUDA_ERROR_*`); un
   crash CUDA envenena al siguiente run en silencio (Gate C).
8. **Paridad de envelope VRAM**: comparar backends bajo la misma envolvente
   (pico + deriva VRAM); un Δ medido con distinta ocupación no es paridad.

## Recomendación explícita

NO proponer el switch hasta que (2) y (4) estén verdes: el riesgo
concentrado es K2 en 8GB (spill CPU-bound) y la ausencia de bench comparativo.
MiniCPM5/Qwen3 están listos como opt-in (`KITTY_BACKEND=kitty-infer`) y esa
es la recomendación operativa hoy: embebido para modelos ≤4B verificados,
Ollama para todo lo demás.
