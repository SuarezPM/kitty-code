# Roadmap: Patrones Robables (steal-list de los propios repos del usuario)

**Fecha:** 2026-08-21 · **Plan:** f6-p0-quickwins-nightbench todo 10 · **Estado:** Must-Have priorizado
**Fuente:** repos del propio usuario (Apohara-Catalyst, apohara-codesearch, HUDvendor/apohara-vouch,
HUDvendor/apohara-probanza). Cada patrón fue verificado con grep antes de citar — sin citas de memoria.
**Regla de adopción:** solo patrones ya batallados en producción propia; cero dependencias nuevas sin
gate (`cargo deny` + `cargo audit`) y cero abstracción para un solo uso.

## Prioridad P0 — desbloquean P0s actuales o cierran deuda de seguridad

### 1. sqlite-vec `OnceLock` auto_extension unificado
- **Origen:** `apohara-codesearch/crates/apohara-indexer/src/storage.rs:79`
  (`ensure_vec_extension_registered`, guard `OnceLock` en :80) — copia gemela en
  `Apohara-Catalyst/crates/apohara-indexer/src/storage.rs:61`.
- **Qué resuelve:** registrar `sqlite3_vec_init` como auto-extension UNA vez por proceso vía
  `sqlite3_auto_extension` (idempotente por puntero de función, blindado con `OnceLock` para evitar el
  roundtrip FFI en cada `open_db`). Es `pub` a propósito: crates downstream registran la extensión sin
  heredar el schema del indexer.
- **Aterrizaje en Kitty:** `crates/kitty-memory/src/episodic.rs` — si se añade búsqueda vectorial vec0,
  la registración vive en UN punto exportado y `episodic`/futuros stores la reutilizan. Hoy el riesgo es
  doble-registración silenciosa al sumar un segundo store SQLite.

### 2. blake3 `FeatureHashEmbedder` 384D fallback zero-deps
- **Origen:** `Apohara-Catalyst/crates/apohara-indexer/src/embeddings.rs:21` (`feature_hash_embed`,
  signed feature-hashing con L2-normalize); envuelto como backend en
  `apohara-codesearch/crates/apohara-indexer/src/embedder.rs:78` (`FeatureHashEmbedder`),
  `EMBED_DIM = 384` en `storage.rs:44`.
- **Qué resuelve:** embedding determinista (~0 RAM, sin descarga de modelo, sin ONNX runtime) que sirve
  como piso de calidad cuando fastembed no está (CI, arranque frío, máquina sin modelos).
- **Aterrizaje en Kitty:** `crates/kitty-memory/src/embedding.rs` hoy depende de fastembed opt-in;
  este fallback elimina el camino "sin feature → sin dense ranking" y hace el recall híbrido testeable
  en CI sin descargar ~88 MB.

### 3. Chunker tree-sitter 9 lenguajes, ventana 60 / overlap 15
- **Origen:** `apohara-codesearch/crates/apohara-indexer/src/chunker.rs:31-33`
  (`WINDOW_LINES = 60`, `WINDOW_OVERLAP = 15`, stride = 60−15); los 9 backends declarados en
  `crates/apohara-indexer/Cargo.toml:47-56` (rust, typescript, python, go, bash, java, c, ruby, cpp).
- **Qué resuelve:** troceado consciente de sintaxis con solape fijo — los chunks caben en contexto y
  ningún símbolo queda cortado por la mitad de una ventana.
- **Aterrizaje en Kitty:** ingesta de conocimiento/repo-scale memory: mismos números 60/15 (medidos,
  no inventados) sobre `tree-sitter`; empezar con rust+python+typescript y crecer a los 9.

## Prioridad P1 — correctitud e higiene de datos

### 4. Embedder con `id()`/`dim()` versionados persistidos en meta
- **Origen:** `apohara-codesearch/crates/apohara-indexer/src/embedder.rs:46-53`
  (trait `Embedder`: `fn id()` :50, `fn dim()` :52; contrato documentado en :42-44),
  id estable `"feature-hash-v1"` en :74; la tabla `meta` rechaza mezclar embeddings de backends
  distintos (`schema::read_embedder_meta`).
- **Qué resuelve:** migración de modelo de embeddings sin corromper índices viejos — el índice sabe
  QUIÉN generó cada vector y con qué ancho, y se niega a mezclar.
- **Aterrizaje en Kitty:** `kitty-memory` guarda BLOBs 384D de MiniLM; al sumar el fallback blake3 del
  punto 2 (u otro modelo) hace falta exactamente este `meta.embedder_id/dim` o el RRF comparará
  peras con manzanas.

### 5. Codec comprimido estilo TurboVec `codec_v8` (cuando aterrice)
- **Origen real (honesto):** NO existe implementación local verificable todavía — solo referencias de
  diseño: `HUDvendor/apohara-vouch/.claude/rules/performance.md:35` ("turbovec index (100 PRs,
  1536-dim, 4-bit) | ~40 MB") y `HUDvendor/apohara-probanza/core/docs/threat-model.md:352,380`
  (invariante INV-15 de `codec_v8` + link upstream `SuarezPM/Apohara_Context_Forge`). El nombre
  "Dekanus" del plan no corresponde a ningún repo local — se corrige aquí la atribución.
- **Qué aporta cuando exista:** vectores 4-bit (~8x menos RAM que f32) con invariantes testeadas
  (herencia pre-RoPE) — relevante solo si el índice de Kitty supera ~10⁵ chunks.
- **Decisión:** en lista de vigilancia, NO en backlog de implementación. Robar el patrón de invariantes
  (`tests/test_codec_v8.py::test_v8_inherits_pre_rope_invariant`) cuando el código sea importable.

### 6. Compresión de prompt LLMLingua-2, chunks de 160 palabras
- **Origen:** `HUDvendor/apohara-vouch/crates/themis-compressor/src/classifier.rs:41`
  (`DEFAULT_CHUNK_SIZE_WORDS: usize = 160` ≈ 290 tokens, bajo el cap 512 de xlm-roberta),
  port documentado en `src/lib.rs:3` y `classifier.rs:1-30` (force-tokens `.!? ,\n`, rate-based selection).
- **Qué resuelve:** recortar contexto compartido (~−44% medido en Context Forge) manteniendo fronteras
  de oración — crítico para meter recall episódico + knowledge cards en la ventana 16K de ornith.
- **Aterrizaje en Kitty:** compactor de `crates/kitty-harness/src/context.rs` — el puerto Rust es
  determinista y sin deps neurales (scoring por longitud de palabra como stand-in documentado);
  swap a modelo real después sin romper callers.

## Prioridad P2 — supply-chain y auditoría

### 7. `deny.toml` license allowlist + cargo-audit + attestations
- **Origen:** `apohara-codesearch/deny.toml:26` ([licenses] permissive-only: MIT/Apache-2.0/BSD/ISC/
  Zlib/Unicode/MPL-2.0…, cualquier otra licencia FALLA el gate); `cargo-audit` pinned en
  `.github/workflows/ci.yml:97-110`; attestations de release en `.github/workflows/release.yml:160-161`;
  Scorecard semanal en `.github/workflows/scorecard.yml`.
- **Qué resuelve:** ninguna dependencia copyleft-fuerte o vulnerable entra sin revisión explícita.
- **Estado en Kitty:** `deny.toml` + `cargo deny check` ya existen (W5); falta espejar el job
  cargo-audit pinned y las attestations de release. Copiar verbatim de codesearch.

### 8. Ledger JSONL append-only con `O_NOFOLLOW` + `fchmod 0600`
- **Origen:** `Apohara-Catalyst/crates/apohara-audit/src/lib.rs:170` (`open_with_0600`),
  `O_NOFOLLOW` en :186 (anti symlink-swap), `fchmod(fd)` en :200 (cierra la carrera
  create→chmod contra swap de inode; chmod por path quedó documentado como el bug anterior).
- **Qué resuelve:** log de auditoría que un atacante local no puede redirigir ni pre-permisar.
- **Aterrizaje en Kitty:** ledger de decisiones de herramientas (qué Bash/Edit ejecutó el agente y con
  qué veredicto de Guard) en `~/.kitty/audit-*.jsonl` — mismo patrón rotación diaria UTC + size-based.

### 9. Reaper DAG de tareas con liveness propio
- **Origen:** `Apohara-Catalyst/crates/apohara-dispatch/src/task_graph.rs:256`
  (`reap_stale_claims`, doc US-F1.1/F2.1 en :238-252) + detección de staleness en
  `crates/apohara-dispatch/src/claim.rs:479` (`claim_is_stale`: TTL + heartbeat + dead-PID probe).
- **Qué resuelve:** claims muertos no estrangulan el grafo ni strandan tareas posteriores; el reaper
  re-verifica staleness atómicamente antes de liberar.
- **Aterrizaje en Kitty:** si el roadmap multi-agente/sesiones paralelas aterriza, el registro de
  sesiones activas necesita exactamente esto; hasta entonces es patrón de referencia, no tarea.

## Explícitamente NO priorizado

- **wasmtime / plugins WASM como merge gate** — el audit lo rechaza: plugins estáticos
  (`PluginRegistry` sin dlopen, ya implementado en `crates/kitty-harness/src/plugin.rs`) son más
  seguros y suficientes. No reabrir.
- **Entrenar/fine-tunear modelo, motor own-inference para v0.1** — ver `docs/VISION-ROADMAP.md`
  (keep Ollama; own engine solo si Ornith-397B MoE sparsity + FWHT 3-bit + SelectivePolicy +
  layer_streaming lo exigen).

## Orden de robo recomendado

1 → 2 → 4 juntos (vector search honesta en kitty-memory: extensión única + fallback zero-deps +
meta versionada son un solo cambio coherente), luego 3 (chunker para ingesta), 7 (supply-chain, barato),
8 (ledger), 6 (compresión de prompt cuando la ventana apriete), 5 y 9 en vigilancia.
