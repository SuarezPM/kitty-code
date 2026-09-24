# SDD — Upgrade definitivo + integración mono-repo

Origen: evidencia exp-2 (dos repos, path-dep frágil) + research lib-3 (12 fuentes)
cristalizados por ora-1. Estado: plan aprobado pendiente de 3 decisiones.

## 1. Decisión de integración: A — workspace único

`kitty-infer` se muda dentro de Kitty-Code (`engine/`); un solo lock, un solo CI,
cero costura externa. Descartados: C (no elimina el defecto raíz) y B (fricción
submodule por el mismo resultado con más pasos).

| Paso | Acción | Riesgo |
|---|---|---|
| M0 | Congelar HEADs verdes; tag `pre-mono` en ambos | ninguno |
| M1 | `git subtree add --prefix=engine` (histórico) o squash | el remoto propio de kitty-infer deja de ser origen |
| M2 | 7 crates a `workspace.members`; path-deps `../../../` → `engine/` | regenerar `Cargo.lock` una vez |
| M3 | Features `kitty-infer[-cuda]` gatillan `mistralrs/cuda`; colapsa stub | call-sites `main.rs`/`bench`, mecánico |
| M4 | CI unificada, matriz cpu/cuda, cache único | tiempos de build CUDA |
| M5 | `NOTICE` MIT al raíz; F0-RECON, codemaps, Dockerfiles | scripts/docs |
| M6 | Archivar kitty-infer como mirror de solo-lectura | operativo |

Corte: M1–M3 atómicos en rama; `cargo check --workspace` + suites (94/505/83+…)
verdes antes de M4.

## 2. Fases de upgrade (dueño, gate, esfuerzo)

| # | Fase | Dueño | Gate | Esf. |
|---|---|---|---|---|
| 1 | Timeouts por modelo/backend (`timeouts.json`: 2B≥150s, 4B≥180s) | fixer | ningún cap uniforme de 30s en logs | S |
| 2 | Warm-resident (preload o health sin destruir; nada post-crash) | fixer | 1er turno < cap con margen ≥3×; cero 139 en serie de 4 | S |
| 3 | Bench k≥3 + manifiestos persistidos (nada en `/tmp`) | fast-generic | 3 corridas/celda, Δ documentado | M |
| 4 | Migración mono-repo (paralela a 1–3) | fixer | clon limpio verifica todo sin rutas externas | L |
| 5 | Q8 K2 + historial truncado (sin ladder, sin BF16-first) | fixer | turno e2e coherente ventana ≥40min; top-5 vs BF16 | M |
| 6 | KV-FP8 (+fallback) + K2-turno (verificar sm_75 primero) | fixer | sin degeneración vs umbral BF16 | M |
| 7 | Acta robustez + VRAM-paridad en criterios | fast-generic | doc con líneas Gate C D6 | S |

1→3 y 2→3 por dependencia; 4 sin dependencia runtime; 6 tras 5 en K2.

## 3. Criterios del interruptor (revisados)

(1) turnos + envelope VRAM a favor; (2) K2-**Q8** coherente <60min (Q4 excluido);
(3) paridad sin cambio; (4) `>5pp sobre k≥3 warm` (hoy no-medido, no rojo);
(5) + cero SIGSEGV/DEINITIALIZED; (6) + actas; **(7) NUEVO:** build mono-repo
desde clon limpio. Los caps son contrato del bench, no del harness. Sin GO
explícito no hay switch.

## 4. Decisiones del usuario (pendientes)

1. Historial git M1: **squash + mirror `pre-mono` legado (DECIDIDO)**.
2. Destino repo kitty-infer: **archivar como mirror de solo-lectura tras M6 (DECIDIDO)**.
3. KV-FP8: **(a) gateado por compute-cap con fallback q8_0 (default tomado)**.
4. Timebox K2-largo: **≥40min vale (DECIDIDO)**.
5. Naming: **mantener `mistralrs-*` (default tomado)**.

## 5. Qué NO hacer

No re-correr bench con caps 30s; no quant-ladder; no tocar thresholds del bench;
no quedarse en C; no switch sin GO; no reabrir gates (paridad, F2/HOLD, NO-SWITCH);
no polyglot/qwen25 como gates; no feature nueva en la migración.

## 6. M6 — preparación de archivado (NO ejecutado en GitHub)

Estado: tag local `mono-mirror-point` en kitty-infer HEAD (punto espejo
post-M1; local, sin push). El repo origen queda intacto como respaldo
(workflows CUDA originales incluidos) hasta el archivado.

Pasos para archivar tras 1 release del mono-repo (otro carril, con GO):
1. Verificar que `main` del mono-repo compila + CI verde post-merge.
2. En GitHub: settings del repo kitty-infer → Danger Zone → Archive repository.
3. Avisar a contribuidores (el mirror queda de solo-lectura; PRs/issues van al
   mono-repo). NO borrar el repo (decisión 2: mirror, no borrado).
4. Prohibido: `gh repo archive` sin GO explícito; cambiar remotos del mono-repo.
