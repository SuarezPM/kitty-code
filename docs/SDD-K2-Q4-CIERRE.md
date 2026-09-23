# SDD — K2-Q4 cierre (kitty-infer + Kitty-Code)

## WHAT
Cerrar los pendientes con el quant oficial IFM (`K2-Horizon-4B-Q4_K_M.gguf`,
esperado ~2.2GB vs 9.5GB BF16): turno K2 embebido completo, bench + decisión
default, y (solo si hace falta) KV-offload. Fondo: watch del parche (sin acción).

## Estado de partida
- K2-BF16: engine verde determinista; turno embebido ENVELOPE-FINAL por física
  (prefill 24 capas CPU, deriva VRAM en CTX largo). Commits `8a704ab`/`4679607`.
- Q4 ≈ 2.2GB cabe entero en 8GB con KV → prefill en GPU (minutos, no horas).

## HOW
### Fase A — K2-Q4 download + smoke + turno (fixer, reuse fix-2)
1. Descarga a `~/.cache/kitty/models/` + verifica tamaño/sha + arch=`k2_horizon`
   y block_count contra el BF16 conocido (ojo: fichero dice "4B", modelo es 3.7B).
2. Smoke engine mismo protocolo F5 (F16, f16a/f16b determinismo, top-5 Ferrox).
3. Turno embebido `kitty run` (CTX según tabla, spill auto — con Q4 no debería
   necesitarlo) + paridad spot vs corrida BF16.
**Gate:** turno exit 0 + output + pico VRAM + evidencias `/tmp/kitty-logs/k2q4-*`.

### Fase B — KV-offload (CONDICIONAL, solo si A no cierra)
Paged-attention en mapeo mixto GPU/CPU. **Gate:** turno exit 0 o envelope
documentado con números. No existe como fase activa hasta veredicto de A.

### Fase C — Bench completo + decisión default (fixer)
Bench sobre infer con la metodología del spec (sin tocar thresholds/semillas),
pass-rate medido, y veredicto del interruptor contra BACKEND-DEFAULT-CRITERIA.md
(switch solo con GO explícito).
**Gate:** números + recomendación explícita.

### Remediación y cierre
- Gate `@oracle` tras cada fase activa (re-reviews máx 2/gate, solo si cambia
  decisión/riesgo); una remediación única; commits por boundary + resumen.

## Riesgos
- El Q4 puede traer template/tokenizer distintos al BF16 → verificar antes del humo.
- Si el quant está corrupto o es otro checkpoint, fallback: cuantizar BF16 local
  (ISQ/UQFF en árbol) en vez de bloquearse.
- Bench puede revelar gap real vs Ollama → se reporta, no se esconde.
