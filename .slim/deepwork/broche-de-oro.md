# Deepwork — broche de oro (kitty-infer + Kitty-Code)

## Goal
Cerrar con broche de oro: (1) turno K2 embebido GPU, (2) paridad mismo-GPU vs
upstream, (3) pequeñeces con dueño + decisión del interruptor default.

## Estado heredado (no re-derivar; referencias, no contenido)
- Plan vivo: `/home/thelinconx/Proyectos/kitty-infer/F0-RECON.md` (fases F0–F5,
  robo nexus, apéndices parche #3909 / F5 / follow-ups D3–D8).
- Commits en origen: kitty-infer `cbca414`, Kitty-Code `6f6a31a` (árboles limpios).
- Evidencias: `/tmp/kitty-logs/` (CUDA), `/tmp/kitty-logs-upstream/` (baseline CPU),
  `/tmp/kitty-logs-cpu/` (CPU-vs-CPU bit-igual 3/3), `/tmp/kitty-logs-upstream-cuda/ATTEMPT.txt`.
- Modelos: `~/.cache/kitty/models/{MiniCPM5-2B,Qwen3-4B}-Q4_K_M.gguf`,
  `K2-Horizon-4B-BF16.gguf` (9.5GB) + tokenizers.
- Infra: contenedor `nvidia/cuda:12.9.1-devel-ubuntu24.04` + runtime NVIDIA OK;
  host glibc 2.44 incompatible con build CUDA nativo.

## Research aceptado (para que oracle no re-descubra)
- `candle 35d7ae7` post-#3558 colisiona en CUDA≥12.2 sm_75; PRs #3601 (revert) y
  #3909 (gate por toolkit) OPEN; backport local en
  `docs/patches/candle-compatibility-cuh-3909.patch` (kitty-infer).
- Upstream `d5ae0f1` pinea la misma rev 35d7ae7 (corrección sobre 27f20fe).
- qwen2.5-coder-7B: NaN solo F16-CUDA → override F32 en `infer_policy`.
- CTX=4096 provoca panic scheduler con prompt agente → CTX=8192 embebido.

## Plan de fases (orden, dueño, gate)
1. **F1-turno-K2** — fixer (reuse fix-2). Scope: contenedor + `kitty run`
   K2 9.5GB+spill. Gate: exit 0 + output + pico VRAM + evidencias.
2. **F2-paridad-CUDA** — fast-generic (reuse fas-4). Scope: clon upstream +
   contenedor + /tmp. Gate: veredicto numérico o documentado-sin-cierre
   (time-box ~90min). Secuencial tras F1 por exclusividad GPU.
3. **F3-pequeñeces** — fixer (código, reuse fix-2) + librarian (watch PR #3909,
   solo web) en paralelo, scopes disjuntos. Scope: tool_call_id, bench sobre
   infer, doc CTX-mínimo, tracking parche, criterios + recomendación del
   interruptor default (el switch solo con GO explícito). Gate: checks + tests + docs.
4. **Remediación única** — batch de hallazgos oracle accionables de las 3 gates;
   validación enfocada; re-review solo si cambia decisión/riesgo.
5. **Cierre** — commits por boundary, validación final, resumen.

## Presupuesto oracle
3 gates iniciales + máx 2 re-reviews c/u (tope 9, esperado 3–4). Intentos
numerados en cada prompt ("Gate N — review attempt X of 3").

## Estado
- [x] Setup deepwork (ignores verificados, este fichero, plan en docs/)
- [x] Overview mostrado al usuario — GO end-to-end recibido
- [x] Fase 1 ejecutada (veredicto PARCIAL: envelope hardware, fixes en alcance)
- [x] Gate 1 @oracle (ora-1, attempt 1/3): PASS-con-deuda — PARCIAL legítimo (envelope HW); deudas a remediación única: D1 paréntesis+alloc+test-negativo, D3 clamp-capas+precheck-por-capas (HOLD como API pública → knob interno ≤12), D5 log-thinking+capas+ctx, D4/D7 tabla VRAM por perfil en F0-RECON
- [x] Soak K2 (fas-4, 94min): OOM-EN-COMPLETION en decode, deriva +1536 → envelope REVISADO (spill-12 no estable en CTX largo)
- [x] Retry embebido (fas-4, 40min): ENVELOPE-FINAL — pico 6731, deriva +674, exit 124 sin OOM ni texto; gate Fase 1 NO-verde por física; fixes a commit
- [>] Commit Fase 1 (fas-3, ambos repos) + Fase 2 paridad-CUDA en contenedor (fas-4) en paralelo
- [x] Watch PRs #3909/#3601 (lib-2): ambos OPEN, backport idéntico al propuesto → MANTENER parche (Fase 3 input listo)
- [ ] Segundo mandato autónomo AFK registrado: cerrar todo end-to-end sin pedir GO
- [ ] Fase 2 / gate / remediación
- [ ] Fase 3 / gate / remediación
- [ ] Remediación única + validación (+ re-reviews condicionales)
- [ ] Commits finales + resumen de sesión
