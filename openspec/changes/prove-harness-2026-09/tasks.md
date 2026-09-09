# Tasks: prove-harness-2026-09

## 0. Fase 0 — Metodología y pins (sin cómputo pesado)

- [ ] 0.1 Fijar versiones en la spec `official-methodology`: commit de `polyglot-benchmark`, versión de `harbor` + dataset `terminal-bench-2-1`, versiones de `swebench` + OpenHands + `chat_template.jinja` de ornith, versión de Ollama; registrar de dónde sale cada pin.
- [ ] 0.2 Extender el esquema de manifiesto `benches/ab/` a los 4 suites (campos por ejercicio para polyglot: outcome/attempt/tests_touched; `-k` y trajectorias para TB2.1; run-id para SWE-bench).
- [ ] 0.3 Gate Fase 0: revisión del diseño contra las honesty rules (un revisor confirma que ningún task permite tocar el entorno de bench).

## 1. Fase 1 — Mutación del guard (DESCOPED 2026-09-09: full run inviable en este box — dos OOM kills; se conserva evidencia parcial)

- [x] 1.2 Añadir el test asesino de `strip_heredoc_bodies`: `cat > f << EOF\na\nb\nEOF\necho hi` debe dar el strip exacto (mata el `+=`→`*=`); segundo caso con dos heredocs en un comando. (Hecho, commit `aa424fc`.)
- [ ] 1.1 Full mutant run — DESCOPED: dos lanzamientos muertos por OOM del sistema (logs `results/mutants-guard-20260909T142104Z.log` y `...150009Z.log`); no relanzar en este box.
- [ ] 1.3 Ratchet >70% — DESCOPED con 1.1; evidencia parcial archivada (~70/337 puntuados + cluster MISSED en aritmética de `split_command_chain` como trabajo futuro documentado).
- [ ] 1.4 Gate Fase 1 → Gate B′ ligero (killer test + evidencia parcial + descope), sin informe de mutación completo.

## 2. Fase 2 — Release con firma + higiene de recall (código, sin benches)

- [ ] 2.1 Crear `.github/workflows/release.yml`: trigger por tag, `permissions: id-token: write + contents: write`, `cargo build --release -p kitty-cli`, SBOM adjunto al release.
- [ ] 2.2 Añadir firma keyless (`sigstore/cosign-installer`, `cosign sign --yes` + bundle Rekor) y `actions/attest-build-provenance`; documentar verificación en `SECURITY.md`.
- [ ] 2.3 Migrar recall/save de `EpisodeStore` en `bench/run.rs`, `bench/polyglot.rs`, `bench/gpqa.rs` (vía `bench/mod.rs:371`) al patrón take/`spawn_blocking`/restore de `agent/response.rs`; sin cambio de concurrencia.
- [ ] 2.4 Gate Fase 2: suite completa + clippy estricto + dry-run del workflow (actionlint o lectura cruzada si no hay remoto).

## 3. Fase 3 — Medición ligera A/B (pre-registrada)

- [ ] 3.1 Pre-registrar manifiesto (modelos, N=15, suite polyglot homemade, thinking 2048, profile medium, interleave) y ejecutar `./benches/ab/run_ab.sh <A> <B>`; publicar el par solo si pasa el gate (errored ≤10%).
- [ ] 3.2 Correr `cargo bench -p kitty-harness` (criterion del parser) y archivar la línea base.
- [ ] 3.3 Gate Fase 3: par A/B válido archivado con manifiesto previo a resultados; documentar que mide modelo-vs-modelo, no ablación por palanca.

## 4. Fase 4 — Desbloquear loaders (artefactos + smokes, sin full runs)

- [ ] 4.1 TB2.1: fijar dataset (`-d terminal-bench/terminal-bench-2-1` o pin local `-p` verificado), añadir N-selection al wrapper, correr oracle smoke `-k5`.
- [ ] 4.2 GPQA: materializar `~/.cache/kitty/gpqa/diamond.jsonl` + sidecar `gpqa_manifest.json` (pasos de `gpqa.rs:353-369`); emitir una pre-registración válida de prueba (gated, sin invocar modelo si falta).
- [ ] 4.3 ClawEval: instalar `claw-eval`, fijar `OPENROUTER_API_KEY`, `data/fixtures`, verificar imagen `claw-eval-agent:latest`; `is_ready` en verde.
- [ ] 4.4 Gate Fase 4: los tres preflights en verde; ningún full run todavía.

## 5. Fase 5 — Benches pesados, en orden (cada uno pre-registrado y con gate)

- [ ] 5.1 Polyglot-225 ×2 runs con ornith 1.5 9B (órdenes distintos, temp 0.3, ctx 32768, thinking 2048): reportar media±SD, split por intento, auditoría de tests tocados. (~20h/run).
- [ ] 5.2 TB2.1 89×5 (`terminus-2`, temp 1.0/top_p 1.0, trajectorias publicadas) + oracle `-k5` previo.
- [ ] 5.3 GPQA Diamond 198 con pre-registro válido.
- [ ] 5.4 SWE-bench-verified 500 (OpenHands fijado + eval oficial, red apagada, historial podado, `preds.json`+logs publicados). Si la máquina no lo alberga, se escinde a su propio change sin bloquear 5.1–5.3.
- [ ] 5.5 Gate Fase 5: cada suite con manifiesto previo, tablas separadas por grading, medias (nunca best-of-k).

## 6. Fase 6 — Publicación honesta

- [ ] 6.1 Actualizar `AUDIT.md`/`CNAS.md`/`docs/FLOOR-BENCHMARK.md` con las tasas reales, método de grading por métrica y comparables correctos (little-coder 45.56% solo contra polyglot-225 nativo; B1 homemade nunca contra tasas oficiales).
- [ ] 6.2 Archivar el change con sync de las 5 capabilities a `openspec/specs/`.
