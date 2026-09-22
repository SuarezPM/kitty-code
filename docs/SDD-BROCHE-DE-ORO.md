# SDD — Broche de oro (kitty-infer + Kitty-Code)

## WHAT (objetivo y alcance)
Cerrar los pendientes verificables del motor propio y su embebido:
1. Turno K2 embebido GPU (`kitty run` completo, 9.5GB + spill honesto).
2. Paridad mismo-GPU contra upstream `d5ae0f1` (gap con ruta clara; vale
   documentado-sin-cierre si el time-box no alcanza).
3. Pequeñeces con dueño: `tool_call_id` multi-herramienta, `bench` sobre el
   motor nuevo, doc de CTX-mínimo embebido, tracking de devolución del parche
   #3909, y criterios + recomendación del interruptor default
   (el switch solo con GO explícito del usuario).

Fuera de alcance: multimodal/MoVA-36B (refusal por diseño), re-arquitecturas,
cambios de default sin GO.

## Estado de partida (evidencia, no claims)
- kitty-infer `cbca414`, Kitty-Code `6f6a31a`, árboles limpios e in-sync.
- Humo engine CUDA verde (MiniCPM5/Qwen3/K2-F16); turnos embebidos "forty-two"
  verdes en GPU; paridad CPU-vs-CPU bit-igual 3/3.
- Gaps conocidos: turno K2 embebido pendiente; paridad mismo-device pendiente;
  follow-ups D3–D8 documentados en `F0-RECON.md`.

## HOW (fases, dueño, gate)
### Fase 1 — Turno K2 embebido GPU
Dueño: fixer (sesión fix-2 reutilizada). Scope: contenedor CUDA + `kitty run`
con `KITTY_BACKEND=kitty-infer` y GGUF K2 (spill 12 GPU + resto CPU, CTX=8192).
**Gate:** exit 0 + texto generado + pico VRAM + logs en `/tmp/kitty-logs/`.
*Por qué esta fase primero: es el único pendiente del espectro 2B→9B+K2 y
desbloquea el commit F5-final.*

### Fase 2 — Paridad mismo-GPU vs upstream
Dueño: fast-generic (sesión fas-4 reutilizada). Scope: clon upstream +
contenedor + `/tmp` (ningún repo kitty). Secuencial tras Fase 1 (GPU exclusiva).
Time-box ~90 min; vale cierre documentado.
**Gate:** veredicto numérico PARIDAD/DIVERGENCIA o informe de techo.
*Por qué después: no toca código kitty; corre en la misma GPU y contaría
recursos con la Fase 1.*

### Fase 3 — Pequeñeces con dueño + decisión default
Dueños en paralelo (disjuntos): fixer (código: `tool_call_id`, `bench` sobre
infer, doc CTX-mínimo) + librarian (solo web: estado PRs #3601/#3909 upstream).
El interruptor default se entrega como criterios + recomendación, sin switch.
**Gate:** checks + tests verdes + docs + recomendación explícita.
*Por qué al final: son independientes entre sí y del resultado numérico de
las fases 1–2; se benefician del árbol ya verde.*

### Remediación y cierre
- Gate `@oracle` tras cada fase (3 iniciales; re-reviews máx 2 por gate, solo si
  cambia decisión/riesgo).
- Una única pasada de remediación con los hallazgos accionables + validación
  enfocada.
- Commit por boundary válido + push (sin --force) + resumen de sesión.

## Riesgos
- K2 9.5GB BF16 al límite de 8GB VRAM → spill ya probado a nivel engine;
  si el turno embebido OOMea, el fallback es documentar el perfil exacto que sí
  cabe (no forzar).
- Upstream-CUDA puede topar con otro techo de toolchain → cierre documentado
  aceptado de antemano.
- `bench` sobre infer puede requerir diseño (no solo wiring) → si excede el
  scope, se entrega como spec + spike, no como bench completo silencioso.
