# Ventana GPU serial — 2026-10-09 — protocolo

Lane única GPU. Una celda cada vez, `ollama stop` entre series, 1 modelo residente.

## Entorno fijado (pre-serie, verificado)
- GPU: NVIDIA GeForce RTX 2060 SUPER 8GB (8192 MiB)
- CPU: AMD Ryzen 5 3600 6C/12T, RAM 46Gi
- Ollama: 0.40.1, servicio systemd con:
  - `OLLAMA_KV_CACHE_TYPE=q8_0` (kv-cache.conf + override.conf)
  - `OLLAMA_FLASH_ATTENTION=1`
  - `OLLAMA_CONTEXT_LENGTH=16384` (ctx justo)
  - `OLLAMA_HOST=0.0.0.0`
- HEAD: 5d8090d357840e96ee3240375432e5cacbb151b7
- Árbol: dirty pre-existente de otras lanes (ver manifiesto), stash vacío.
  NO commit, NO toques en crates/ — solo scripts nuevos bajo bench/.
- Kitty: 0.1.0 (`target/debug/kitty`, `target/debug/kitty-cli`)
- little-coder: 1.17.0 en /home/thelinconx/Proyectos/Repos/little-coder

## W6 — matriz VRAM
- Q4 default (salvo desviación declarada), ctx 16384 (servidor), KV q8_0 (servidor), flash-attn on.
- GPU-layers: las que Ollama offloadee por medición (`ollama ps`: PROCESSOR 100% GPU vs CPU/GPU).
- 1 modelo residente: verificar `ollama ps` vacío antes de cada celda, `ollama stop <modelo>` después.
- Por celda: baseline nvidia-smi → `ollama run <modelo> "Responde solo: OK"` (1 inferencia corta, fuerza carga+1 turno)
  → `ollama ps` (SIZE/PROCESSOR/CONTEXT) → nvidia-smi pico → `ollama stop`.
- VRAM pico modelo ≈ usado_pico - baseline_desktop. OOM sí/no observado (fallo carga o inferencia).
- Script: `measure-vram.sh` (serie, no paralelo).

## tb2-8 (marcado latencia W4)
- Justicia: mismo modelo `ornith:9b` / backend ollama / turnos (max_iter 20, hops 0) / timeouts (celda 60s ornith-ollama, default 120).
- Kitty: `./target/debug/kitty bench tb2 --limit 8 --no-memory -o <out>` (profile auto, context 16384, thinking 2048).
- Little: pi RPC `ollama/ornith:9b` con mismo prompt-set tb2-homemade (8 prompts), timeout 60s/ejercicio, temp 0.6.
- k≥3 por agente (3 reps). Manifiesto con `gates_off` (router/knowledge off en bench) + HEAD.
- Métrica: p50/p99 latencia por ejercicio + pass-rate. Comparativa honesta con asimetría declarada
  (Kitty guard+sandbox activos = producto; little accept-all sin gates).

## W7 goldens con modelo
- Determinismo por semilla: trazas, no bits. Misma celda 2× con misma semilla → comparar
  secuencia de tool-calls/verdict (no bytes GPU).
- E2E-CI con GPU queda HELD (sin runner GPU en CI).

## Manifiestos
- `bench/gpu-ventana-2026-10-09/manifest-*.json`: BenchManifest por serie (N, versiones, k, hardware, gates_off, HEAD, árbol).
- `benches/ab/manifests/`: copia de manifiestos tb2 para linaje con serie previa.
