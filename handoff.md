# Handoff — retomar mañana (2026-09-09, cierre de sesión)

## Dónde estamos

Deepwork `prove-harness-2026-09`: **Fases 0–4 cerradas, HOLD antes de Fase 5**.
7 commits, Gates A–E (A/B/C con re-review), todo reconciliado, ningún agente vivo.

- HEAD: `b1e7000` (`feat(cli,ci): TB2.1 evidence-grade guards + Phase E close`)
  ← `726253c` (Gate B tests) ← `9e19301` (descope mutación) ← `68dcd8f`
  (codemap) ← `e59f8c2` (Fase C) ← `aa424fc` (killer test) ← `36cf2dc` (pins).
- Árbol: limpio salvo untracked conocidos (`handoff.md`, 3 manifiestos
  A/B + 2 armA huérfanos + 3 marcadores VOID en `benches/ab/`).
- Primer par citable: `20260909T155333Z` — empate 14/14, gate válido,
  manifiesto limpio. Criterion: 1.78µs / 1.31µs.
- Oracle smoke TB2.1: 25/25 PASS (infra). Harbor v0.22.0 instalado.
- Descopes registrados: mutación full (OOM ×2, queda killer test +
  logs en `results/`), ClawEval (mismatch upstream). GPQA bloqueado
  (falta approval + token HF).

## Para reanudar (en este orden)

1. Leer este handoff + `.slim/deepwork/prove-harness-2026-09.md` (estado
   vivo) + `openspec/changes/prove-harness-2026-09/tasks.md` (Fase 5).
2. Verificar entorno: `ollama list` (ornith:9b + ornith-1.5:9b),
   `harbor --version` (0.22.0), `git status` (solo untracked conocidos),
   binario `target/release/kitty-cli` más nuevo que el último commit con
   código (`run_ab.sh` NO reconstruye si existe — binario obsoleto =
   correr código viejo en silencio).
3. Pedir al usuario las 3 decisiones del HOLD (GO por suites, GPQA,
   escisión SWE-bench) — están en el mensaje de HOLD del historial.
4. Con GO: pre-registro 5.1 con esquema EXTENDIDO (fingerprints
   daemon/modelo, digest modelo, chat_template sha — deuda task 0.2),
   lanzar detached + watcher (nunca lanes longevos para runs; redirigir
   salida a fichero o mueren por contexto).
5. Gate entre suites; 5.2 exige pre-reg sin `7131e43` (pin retirado) y
   con rev-6/sha registrados.

## Lecciones operativas (no repetir)

- Lanes con runs de >30 min mueren en este box: lanzar detached + watcher
  ligero (cuidado: el watcher NO debe contarse a sí mismo en el pgrep).
- No co-planificar mutantes + GPU: OOM probado con 46 GB.
- Pares a medias → marcador VOID in-situ, jamás emparejar retroactivo.
- `run_ab.sh` rehúsa árbol sucio: commitear docs del change antes.
- pgrep se automachea: usar truco `[k]` siempre.

## Secretos

- `OPENROUTER_API_KEY` vive SOLO en el transcript del chat 2026-09-09
  (jamás en disco). Si se revive ClawEval, **re-pedirla**; pasarla solo
  en memoria, nunca citarla ni guardarla (ver memoria Engram
  "Descoped mutants..." y "HOLD before Fase 5" para el resto).
- `OPENROUTER_API_KEY`/`HF_TOKEN` ausentes del entorno: ClawEval 4.3 y
  GPQA 4.2 siguen bloqueados hasta que el usuario los aporte.

## Estado anterior (2026-09-03)

Remediación auditoría archivada (21/21, 6.8→8.3), mutation hashline
100%, resultado guard pendiente entonces (ver historial Engram ids
268-272). Superado por lo de arriba; se conserva por trazabilidad.

---

# Handoff — 2026-09-25, infer-readiness + cierre (HEAD 7d7b591, pusheado)

## Dónde estamos

Re-auditoría Nº2 **7.9/10** + plan SDD-82-10 PASS + infer-readiness MiniCPM5/K2 (Gates 1-3 PASS). Commits `9eacc08` (plan) + `7d7b591` (smokes + celda K2 310 + maratón pre-reg + evidencia 6/6). Score 7.9 firme; 8.2/10 con listas cerradas.

## Maratón: readiness por modelo

- **ornith 1.5 9B**: Ollama listo, celda 60s. Sin cambios.
- **MiniCPM5**: Ollama + infer (sidecar 9894271 verificado, smoke timeout-path honesto).
- **K2**: infer Connected + carga real (`resident ready`, primera vez sin escupir), celda 310, sha256 oficial == etag. GO-carga SÍ, GO-ejercicio lo dirá la GPU.
- Pre-reg: `benches/ab/manifests/manifest-20260925T050048Z-marathon-prereg.json` (matrix/seeds[42]/k=3/sha). **Falta fijar N de polyglot antes de correr** (R2 pendiente de ejecución).

## Para reanudar (en este orden)

1. Leer `.slim/deepwork/infer-minicpm5-k2.md` + `docs/SDD-PLAN-INFER-MK.md` + pre-reg de arriba.
2. Exports: `MODELS=~/.cache/kitty/models TOK=~/.cache/kitty/tokenizers`; `ollama stop` residentes + `nvidia-smi` antes de cada smoke infer (1 proceso = 1 modelo).
3. Fijar N polyglot en la pre-reg y correr la maratón (GPU/release para GO-ejercicio).

## Pendiente declarado

T-E1.6 lints, mutants ≥80 (57.8% parcial), wire compilable in-tree, HU-8 en tag, E2E-CI, BGE-M3/rerank offline, holdout-runner, nextest-swap. CI: 5 rojos preexistentes documentados (0 delta con este push). Puerta al 10: auditoría externa (humano).
