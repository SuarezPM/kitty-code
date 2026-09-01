# f6-remaining-tracks-opt-order - Work Plan

## TL;DR (For humans)

**What you'll get:** La máquina lista para benchmarks oficiales (Docker + harbor verificados con smoke de 5 tareas), un sandbox que contiene comandos peligrosos, memoria que recuerda mejor que antes, opción para escalar a un modelo grande solo si falla, un sistema para que terceros añadan herramientas sin tocar el núcleo, una terminal interactiva usable, y —solo si hace falta— corrección gramatical para modelos pequeños, todo medido al final con los benchmarks oficiales de 225 y 89 tareas publicados con honestidad.

**Why this approach:** Instalamos primero lo que bloquea (Docker/harbor), luego aseguramos (sandbox) y hacemos más inteligente la memoria antes de añadir extensiones (cloud/plugin). La terminal va después porque necesita que el motor esté estable. La gramática es condicional (solo si el modelo pequeño falla >5% de validez) y los benchmarks largos van al final para medir el harness definitivo una sola vez, no dos.

**What it will NOT do:** No hará push a GitHub ni configurará CI/OpenSSF/scorecard/SBOM (lo dejamos para el final como pediste), no tocará los datasets oficiales vendorizados, no añadirá gramática si el modelo ya es >95% válido, y no comparará benchmarks caseros con oficiales como si fueran iguales.

**Effort:** Large — ~3-5 semanas pared (bench full 15-24h+3-8h domina), resto ~6-9 días de código
**Risk:** Medium - long-run bench frágil a timeouts Docker, sandbox depende de syscalls Linux, GBNF condicional puede requerir build FFI
**Decisions to sanity-check:** Orden W0→W5, GO gate GBNF <95% tiny, Option A llama-cpp-2 in-process vs B daemon, embeddings fastembed/ort vs rten bloqueado

Your next move: Ejecuta con Hephaestus: `$start-work f6-remaining-tracks-opt-order` (o `--worktree` si quieres PR). No toques código a mano antes.

---

> TL;DR (machine): Large effort, Medium risk, 7 waves (W0-W5 + F) 27 impl + 4 verifiers =31, infra->safety->memory->extensibility->tui->gbnf-conditional->bench-full

## Scope
### Must have
- W0 Infra: Docker en CachyOS (`sudo pacman -S docker && sudo systemctl enable --now docker`), harbor vía `uv tool install harbor` (bins harbor/hb/hr), smoke `harbor run -p bench/official/terminal-bench-2 -a oracle --k 5` 5/5 pass verifica bench/official correcto
- W1 Sandbox seccomp/Landlock: nuevo `crates/kitty-harness/src/sandbox.rs`, integrado en `tools.rs` Bash wrapper, política FS whitelist + net deny + OOM/segfault contenidos, criterio "ningún bypass conocido sobrevive suite guard"
- W1 Memory BM25+RRF: crate `bm25` + embeddings `fastembed` (o `ort`) + RRF manual, migración `episodic.rs` store, flag `--semantic-memory`, A/B recall semántico supera Jaccard en tareas recurrentes medible
- W2 Cloud fallback opt-in: `CloudClient: ChatClient` genérico (env `KITTY_CLOUD_API_KEY`), `AgentConfig.cloud_fallback` bool + `--cloud-fallback`, escalada solo en fallo con costo documentado en `docs/AUDIT.md` y log
- W2 Plugin system: `PluginRegistry` trait + loader (cargo `features` manifest + registro estático, no dlopen inseguro), `ToolRegistry::register_plugin` hereda Guard, terceros registran tools sin tocar core
- W3 TUI interactivo: `crates/kitty-cli/src/tui.rs` prompt real + historial + scroll + status bar live (ctx%, duration) + streaming `tokio::mpsc` desde `Agent::run` events, feature `tui` mantenido
- W4 GBNF conditional: probe `kitty bench polyglot --profile tiny --limit 50` mide validity; si <95% GO implementa `LlamaCppClient: ChatClient` con `llama-cpp-2@0.1.154` `LlamaSampler::grammar_lazy` + `json_schema_to_grammar` + triggers, si ≥95% NO-GO mantiene spike post-hoc documentado, A/B validity 88%→>95%
- W5 Full benchmarks oficiales: `polyglot 225` 15-24h + `TB2 89` 3-8h con modelo `oracle` (NO smollm2/qwen3), `bench/official` Docker, publish `docs/AUDIT.md` columna Consumer vs Vendor + `docs/FLOOR-BENCHMARK.md` + `bench_*.json` reproducibles, grading honesto etiquetado

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Push a GitHub (`git remote add origin`, `git push`), CI `ci.yml`, OpenSSF Scorecard/Badge/SBOM/releases firmados — defer explícito fuera de este plan
- Enviar `grammar` + `tools` juntos en `/api/chat` (HTTP 400 `server-common.cpp:1289`) — usar `grammar_lazy` con triggers si GO
- Añadir `llama-cpp-2` a `Cargo.toml` antes del GO gate — spike mantiene `Cargo.toml:68` defer comment
- Editar `bench/official/*` datasets vendored (solo leer, nunca escribir)
- Comparar `KittyQuiz-225`/`KittyShell-89` string-match como Aider Polyglot/TB2 unit-test — prohibido por `AGENTS.md` anti-pattern
- `unwrap/expect` en prod, `clone` injustificado, módulos >250 LOC sin split (`guard.rs` 1444L + `agent.rs` 1177L ya candidatos split, no empeorar)
- Fake `TS` compilation o `DO NOT EDIT` headers hand-written
- Plugin que bypass Guard o rompa `hashline.rs` BLAKE3 optimistic concurrency

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after (TDD opcional para sandbox/memory) + framework `cargo test --workspace` (247 tests hoy) + `cargo clippy --all-targets --all-features -- -D warnings` + `cargo fmt --check`
- Evidence: `.omo/evidence/f6-remaining-tracks-opt-order/task-<N>.log` y `.omo/evidence/f6-remaining-tracks-opt-order/bench-*.json` (attemptDir = ` .omo/evidence/f6-remaining-tracks-opt-order/` fuera de ulw-loop, o `currentAttemptDir` si dentro)
- Per-todo QA: happy + failure con comando exacto y path evidencia; bench usa `harbor run` + `kitty bench` + `jq` asserts; sandbox usa `cargo test -p kitty-harness guard,sandbox`; memory usa A/B script `benches/ab_recall.py`
- Final wave: 4 verifiers paralelos, todos deben APPROVE antes de handoff

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 0 — Infra desbloqueo (3 todos, 0.5-1h, sin código): Docker + harbor + smoke
- Wave 1 — Safety + Memory (6 todos, 2-3 días): sandbox + BM25/embedding/RRF (paralelizables tras diseño sandbox)
- Wave 2 — Extensibility (5 todos, 1.5-2 días): cloud fallback + plugin (tras W1, sandbox debe estar para no escapar)
- Wave 3 — UX (4 todos, 1-1.5 días): TUI interactivo (tras W2, core estable)
- Wave 4 — Grammar conditional (5 todos, 1-3 días si GO, 0.5d si NO-GO): probe tiny → decisión → impl si GO
- Wave 5 — Benchmarks oficiales + publish (4 todos, 18-32h pared): polyglot 225 + TB2 89 + AUDIT.md
- Wave 6 — Final verification (4 todos paralelos)

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 2,3 | - |
| 2 | 1 | 3,19 | - |
| 3 | 2 | 4,24,25 | - |
| 4 | 3 | 5,6 | 7 |
| 5 | 4 | 6 | 7,8 |
| 6 | 5 | 11,12,24 | 7,8 |
| 7 | 4 | 8,9 | 4,5 |
| 8 | 7 | 9 | 4,5 |
| 9 | 8 | 10,15 | 4 |
| 10 | 9 | 11 | 12,13 |
| 11 | 10 | 15,26 | 12,13,14 |
| 12 | 6 | 13 | 10,11 |
| 13 | 12 | 14 | 10,11 |
| 14 | 13 | 15,26 | 10,11 |
| 15 | 9,11,14 | 16 | - |
| 16 | 15 | 17 | - |
| 17 | 16 | 18,26 | - |
| 18 | 17 | 26 | - |
| 19 | 2 | 20 | 4-14 |
| 20 | 19 | 21,22 | - |
| 21 | 20 | 22,23 | - |
| 22 | 21 | 23 | - |
| 23 | 22 | 24,26 | - |
| 24 | 3,23 | 26 | 25 |
| 25 | 3 | 26 | 24 |
| 26 | 24,25 | 27 | - |
| 27 | 26 | F1-F4 | - |

> Conditional: si Todo 20 = NO-GO entonces arista 20→24 bypass 21-23 (21-23 Skipped, no bloquea W5). Worker debe interpretar 20 como gate y no esperar 23 en NO-GO.

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. W0: Instalar y verificar Docker en CachyOS
  What to do / Must NOT do: Ejecutar `sudo pacman -S --needed docker` + `sudo systemctl enable --now docker` + `sudo usermod -aG docker $USER` (si aplica) + verificar `docker info` y `docker run --rm hello-world`. Must NOT instalar podman como sustituto sin verificar harbor compat, Must NOT dejar daemon sin enable (reboot rompe bench).
  Parallelization: Wave 0 | Blocked by: - | Blocks: 2,3,19
  References (executor has NO interview context - be exhaustive): bench/official/README.md:28-54 ADR TB2 smoke deferred, docs/PLAN-EJECUCION.md:322-327 F6.2 budget gate >45.56%/ >9.2%, docs/STRATEGY.md:87 gate <50% re-encuadrar
  Acceptance criteria (agent-executable): `docker info` retorna 0 + `docker run --rm hello-world` contiene "Hello from Docker!" + `systemctl is-active docker` == active
  QA scenarios (name the exact tool + invocation): happy: `docker info 2>&1 | head -20` evidencia task-1.log contiene Server Version; failure: sin daemon `docker info` debe fallar y plan lo detecta antes de harbor
  Commit: N | infra: no commit (infra local)

- [ ] 2. W0: Instalar harbor CLI y verificar datasets oficiales
  What to do / Must NOT do: `uv tool install harbor` (bins harbor/hb/hr) + `harbor --help` + `harbor dataset list --legacy | grep aider-polyglot` + verificar `bench/official/terminal-bench-2/task.toml` 89 tasks + `bench/official/aider-harbor/polyglot_*` 225 tasks + `bench/official/polyglot-benchmark/*/exercises/practice` 225. Must NOT usar `harbor download aider/aider-polyglot@1.0` con prefix (legacy registry requiere sin org), Must NOT modificar `bench/official/*`.
  Parallelization: Wave 0 | Blocked by: 1 | Blocks: 3,19
  References: bench/official/README.md:39-54 harbor gotcha legacy registry, bench/official/README.md:10-14 Contents table, bench/official/terminal-bench-2/task.toml dataset
  Acceptance criteria: `harbor --version` 0 + `harbor dataset list --legacy 2>&1 | grep -q aider-polyglot` + `ls bench/official/terminal-bench-2 | wc -l` >=89 + `ls bench/official/aider-harbor | wc -l` >=200
  QA scenarios: happy: `harbor --help | grep -q "run"` + `ls bench/official/aider-harbor/polyglot_python_* | head -1`; failure: `harbor download aider/aider-polyglot@1.0` debe fallar (prefix) y doc lo menciona
  Commit: N | infra: no commit

- [ ] 3. W0: Smoke harbor run oracle k5 verifica bench/official correcto
  What to do / Must NOT do: `harbor run -p bench/official/terminal-bench-2 -a oracle --k 5 --output /tmp/harbor-smoke.json` (5 tasks oracle) + verificar 5/5 pass + `jq .summary /tmp/harbor-smoke.json`. Must NOT correr full 89 aquí (solo smoke), Must NOT usar smollm2/qwen3 (gate <50% demo-only).
  Parallelization: Wave 0 | Blocked by: 2 | Blocks: 4,24,25
  References: bench/official/README.md:28-37 ADR TB2 smoke deferred 5-task oracle, docs/PLAN-EJECUCION.md:324 Gate >9.2% TB2, docs/FLOOR-BENCHMARK.md:54 NO-GO sub-2B
  Acceptance criteria: `/tmp/harbor-smoke.json` existe + `jq .passed` ==5 + `harbor run` exit 0 + evidencia `.omo/evidence/f6-remaining-tracks-opt-order/task-3-smoke.log` con 5/5
  QA scenarios: happy: smoke 5/5 pass; failure: sin Docker daemon debe abortar con mensaje claro "docker not available" y no marcar verde
  Commit: Y | feat(bench): harbor smoke k5 verifies official datasets

- [ ] 4. W1: Diseñar sandbox seccomp/Landlock interface y política
  What to do / Must NOT do: Crear `crates/kitty-harness/src/sandbox.rs` con `SandboxConfig { fs_allow: Vec<PathBuf>, net_deny: bool, mem_limit_mb: u32, cpu_timeout_secs: u32 }` + `Sandbox::apply()` que envuelve `tokio::process::Command` para Bash. Definir FS whitelist (workspace + /tmp), net deny default, seccomp filter block execve peligrosos fuera de policy. Must NOT usar Docker como sandbox (harbor ya usa Docker para bench, sandbox es in-process Landlock/seccomp), Must NOT romper `guard.rs` existing checks.
  Parallelization: Wave 1 | Blocked by: 3 | Blocks: 5,6
  References: docs/PLAN-EJECUCION.md:349 Sandbox backlog criterio "ningún bypass sobrevive suite guard; OOM/segfault no escapan", crates/kitty-harness/src/guard.rs:824-912 check_command layers, crates/kitty-harness/src/tools.rs:352-435 Bash tokio Command kill_on_drop, docs/STRATEGY.md:22 foso seguridad
  Acceptance criteria: `crates/kitty-harness/src/sandbox.rs` existe + `cargo test -p kitty-harness --lib sandbox` al menos 2 tests compilan + `cargo clippy -p kitty-harness -- -D warnings` limpio
  QA scenarios: happy: `cargo test -p kitty-harness sandbox::tests::allow_workspace_write` pass; failure: intento `sandbox.apply` con path fuera de whitelist debe retornar `SandboxDenied`
  Commit: Y | feat(harness): sandbox Landlock/seccomp interface + policy

- [ ] 5. W1: Implementar sandbox enforcement integrado en ToolRegistry Bash
  What to do / Must NOT do: Modificar `tools.rs::execute_bash` para llamar `Sandbox::apply` antes de spawn, aplicar Landlock FS rules + seccomp filter (crate `landlock` o `seccomp` / `apohara-agentguard` si disponible), limitar mem/cpu, capturar OOM/segfault como `ToolResult { success:false, output:"sandbox: OOM killed" }` no panic. Mantener `kill_on_drop(true)` + `timeout` 30s. Must NOT permitir `eval` ni `cat > file` bypass (guard ya block), Must NOT aumentar LOC >250 sin split.
  Parallelization: Wave 1 | Blocked by: 4 | Blocks: 6
  References: crates/kitty-harness/src/tools.rs:310-324 write-vs-edit invariant double-block, tools.rs:352 Bash timeout 30s, guard.rs:753-808 detect_write_targets, guard.rs:56-62 INTERPRETERS + 69 ALWAYS_BLOCKED eval
  Acceptance criteria: `cargo test -p kitty-harness --lib sandbox` 5+ tests + `cargo test -p kitty-harness guard` 57 tests siguen verdes + `kitty-cli` Bash `echo hello` bajo sandbox retorna success
  QA scenarios: happy: `Bash sandbox allow echo hello` ok; failure: `Bash sandbox deny rm -rf /` blocked por Guard antes de sandbox + `cat > /etc/passwd` denied por Landlock even if guard missed
  Commit: Y | feat(harness): sandbox enforcement in Bash tool

- [ ] 6. W1: Tests regresión sandbox + guard suite sin bypass
  What to do / Must NOT do: Añadir tests en `sandbox.rs` y `guard.rs` cubriendo: `echo $(rm -rf /)` Block, `cat <(rm -rf /)` Block, `eval "rm -rf /"` Block, `sh -c 'echo $(rm -rf /)'` Block, `tar -czf backup.tar.gz dir/` Allow, OOM `stress --vm 1 --vm-bytes 1G` contained, segfault `kill -11` contained. Verificar `cargo test -p kitty-harness guard sandbox` 100% verde. Must NOT dejar bypass conocido verde.
  Parallelization: Wave 1 | Blocked by: 5 | Blocks: 11,12,24
  References: docs/PLAN-EJECUCION.md:118-131 F1.2 tests regresión guard, guard.rs:824-912 check_command, tools.rs:352 sandbox apply
  Acceptance criteria: `cargo test -p kitty-harness --lib` todos pass + nuevo `sandbox::tests::oom_contained` pass + `guard::tests::sh_backup_allowed` pass
  QA scenarios: happy: 3 bypass cases Block; failure: `find . -name '*.tmp' -exec rm {} \;` documentado Allow o Warn explícito, no Block inesperado
  Commit: Y | test(harness): sandbox+guard regression suite exhaustive

- [ ] 7. W1: Memory BM25 lexical (reemplazo Jaccard pre-embedding)
  What to do / Must NOT do: Añadir crate `bm25 = "0.2"` (o `rank-bm25`) a `kitty-memory`, implementar `Bm25Index` en `episodic.rs` junto a Jaccard existente, `recall_bm25(query, k=50)` tokenize igual `67-73` lowercase alfanum, IDF + TF scoring, ranking BM25 antes de RRF. Mantener `EpisodeStore::open` + `MAX_RECALL_CANDIDATES 50`. Must NOT romper `compute_id`/`compute_hash` BLAKE3 existentes, Must NOT cambiar schema DB sin migración.
  Parallelization: Wave 1 | Blocked by: 4 | Blocks: 8,9
  References: crates/kitty-memory/src/episodic.rs:64-89 tokenize+similarity Jaccard 77-89, episodic.rs:168-200 recall_similar ranks Jaccard+recency, Cargo.toml:42 blake3 +44 rusqlite bundled, docs/PLAN-EJECUCION.md:350 MemScenes criterio recall semántico supera Jaccard
  Acceptance criteria: `cargo test -p kitty-memory` pass + `recall_bm25("fix auth bug")` retorna episodios relevantes con score > Jaccard en dataset `benches/recall_fixture.json`
  QA scenarios: happy: BM25 recall top-3 contiene episodio con "auth" vs Jaccard; failure: query sin overlap retorna fallback recent (no panic)
  Commit: Y | feat(memory): BM25 lexical recall

- [ ] 8. W1: Memory embedding dense con fastembed/ort (desbloqueo rten/Gemma)
  What to do / Must NOT do: Elegir `fastembed = "4"` (ONNX pure Rust, modelos `AllMiniLML6V2` ~80MB) o `ort` + `bge-small`, descargar modelo a `~/.kitty/models/`, implementar `EmbeddingStore` que computa embedding para `goal`+`outcome` y guarda vector en SQLite `episodes` nueva columna `embedding BLOB` (migración `ALTER TABLE`), cosine similarity. Documentar `rten` bloqueado int8 incompat `episodic.rs:166` y por qué se eligió fastembed. Must NOT bundled modelo en repo (gitignore `*.onnx`, `*.bin`), Must NOT bloquear `tiny` profile (auto --no-memory sigue).
  Parallelization: Wave 1 | Blocked by: 7 | Blocks: 9
  References: crates/kitty-memory/src/episodic.rs:165-167 NOTE lexical not semantic blocked rten/Gemma, crates/kitty-cli/src/main.rs:399-402 auto-disable tiny memory, docs/STRATEGY.md knowledge 5 cards scoring word1.0/bigram2.0
  Acceptance criteria: `cargo test -p kitty-memory --features embedding` pass + `embedding("hello world")` len 384 + `cargo clippy -- -D warnings` limpio
  QA scenarios: happy: embedding cosine "fix auth" vs "auth bug" >0.8; failure: sin modelo descargado retorna error claro "run kitty memory download" no panic, sin internet fallback a BM25
  Commit: Y | feat(memory): dense embedding fastembed + migration

- [ ] 9. W1: Memory RRF fusion + A/B supera Jaccard + integración Agent
  What to do / Must NOT do: Implementar `reciprocal_rank_fusion(bm25_rank, embedding_rank, k=60)` + `recall_hybrid(query, k=3)` que fusiona BM25 + dense, guarda en `Agent::recall_episodes` (652-678) inyecta top-3 como `[Past Episodes]` tail, `save_episode` 681-692 guarda embedding. Añadir `AgentConfig.semantic_memory: bool` (default false, opt-in `--semantic-memory`) y `benches/ab_recall.py` que mide recall@3 Jaccard vs hybrid en 20 episodios fixture. Criterio: hybrid supera Jaccard en A/B. Must NOT activar semantic para `tiny` (4K window), Must NOT duplicar token budget.
  Parallelization: Wave 1 | Blocked by: 8 | Blocks: 10,15
  References: crates/kitty-harness/src/agent.rs:652-678 recall_episodes tail inject, agent.rs:211-217 memory_store open ~/.kitty/episodes.db, agent.rs:125-126 EpisodeStore field, kitty-memory/src/episodic.rs:92-112 EpisodeStore open index
  Acceptance criteria: `cargo test -p kitty-memory hybrid` RRF pass + `benches/ab_recall.py --limit 20` muestra hybrid win rate >60% vs Jaccard + `kitty run --semantic-memory` inject funciona
  QA scenarios: happy: `recall_hybrid("fix auth")` top-1 es episodio auth; failure: hybrid con 0 candidatos fallback recent no crash
  Commit: Y | feat(memory): RRF hybrid + Agent integration + A/B

- [ ] 10. W2: Cloud fallback CloudClient: ChatClient genérico
  What to do / Must NOT do: Crear `crates/kitty-runtime/src/cloud.rs` con `CloudClient<C: ChatClient>` o `CloudClient { api_key, base_url, model_frontier }` que implementa `ChatClient` trait `client.rs:19` vía `reqwest` rustls, lee `KITTY_CLOUD_API_KEY` env, timeout 120s, mapea `ToolDefinition` a provider frontier (OpenAI-compatible `/v1/chat/completions`). Cost metering log `cost_usd` por call. Must NOT hardcode vendor (genérico), Must NOT log API key.
  Parallelization: Wave 2 | Blocked by: 9 | Blocks: 11
  References: crates/kitty-runtime/src/client.rs:19-34 ChatClient trait Pin<Box Future>> Send+Sync, client.rs:177 OllamaClient POST /api/chat + /api/tokenize, agent.rs:23-55 AgentConfig model/profile, docs/PLAN-EJECUCION.md:351 Cloud fallback criterio escalada solo en fallo con costo
  Acceptance criteria: `cargo test -p kitty-runtime cloud` MockCloud pass + `CloudClient::new_from_env` sin key retorna error claro + `cargo clippy -- -D warnings` limpio
  QA scenarios: happy: MockCloud chat retorna tool_calls válidos; failure: sin `KITTY_CLOUD_API_KEY` debe error "set KITTY_CLOUD_API_KEY" no panic, timeout 120s handled
  Commit: Y | feat(runtime): CloudClient generic frontier

- [ ] 11. W2: Agent fallback orchestration opt-in solo en fallo
  What to do / Must NOT do: Añadir `AgentConfig.cloud_fallback: bool` default false + CLI `--cloud-fallback` (`kitty-cli/src/main.rs:132`), en `agent.rs::run` si local exhaust `max_iterations` o `ToolLoop` o 3 `empty` seguidos, entonces reintenta 1 vez con `CloudClient` (si flag y key), log `tracing::warn!("cloud fallback triggered: reason={}")` + costo. Documentar costo en `docs/AUDIT.md` sección Cloud. Must NOT escalar silencioso sin flag, Must NOT loop infinito cloud (máx 1 fallback por task).
  Parallelization: Wave 2 | Blocked by: 10 | Blocks: 15,26
  References: crates/kitty-harness/src/agent.rs:296 run loop 20 iterations compaction, agent.rs:113-185 new_inner, kitty-cli/src/main.rs:132 Cli BenchAb, docs/PLAN-EJECUCION.md:351 costo documentado
  Acceptance criteria: `cargo test -p kitty-harness agent::tests::cloud_fallback_on_exhaustion` pass + `kitty run --cloud-fallback` sin key logea warning y no crash + con MockCloud fallback retorna success
  QA scenarios: happy: local fail -> cloud success; failure: local success no debe trigger cloud (costo 0), cloud fail debe retornar error final con ambos logs
  Commit: Y | feat(harness): cloud fallback orchestration opt-in

- [ ] 12. W2: Plugin system PluginRegistry trait + loader
  What to do / Must NOT do: Crear `crates/kitty-harness/src/plugin.rs` con `trait Plugin { fn name(&self)->&str; fn tools(&self)->Vec<ToolDefinition>; fn execute(&self, name:&str, args:Value)->Result<ToolResult> }` + `PluginRegistry { plugins: Vec<Box<dyn Plugin>> }` + loader `load_from_manifest("plugins.toml")` o `features = ["plugin-foo"]` estático. Registrar sin tocar core: `ToolRegistry::register_plugin`. Must NOT usar `dlopen`/`libloading` inseguro sin sandbox, Must NOT romper `ToolRegistry` existente `tools.rs:58`.
  Parallelization: Wave 2 | Blocked by: 6 | Blocks: 13
  References: crates/kitty-harness/src/tools.rs:58 ToolRegistry dispatch Read/Edit/Write/Bash/Glob/Grep, docs/PLAN-EJECUCION.md:352 Plugin criterio terceros sin tocar core, docs/STRATEGY.md:54-55 plugin deferred YAGNI hasta MCP, crates/kitty-cli/src/mcp.rs:17 MCP server real (no plugin)
  Acceptance criteria: `plugin.rs` existe + `PluginRegistry::register` test con dummy plugin retorna tool list + `cargo test -p kitty-harness plugin` pass
  QA scenarios: happy: dummy plugin `HelloTool` registra y ejecuta; failure: plugin con nombre colisión debe error "tool already exists" no overwrite
  Commit: Y | feat(harness): PluginRegistry trait + loader

- [ ] 13. W2: Plugin ToolRegistry integración + Guard inheritance
  What to do / Must NOT do: Modificar `tools.rs::execute` para delegar a `PluginRegistry` si tool no es core, heredar `Guard::check_command` para plugins tipo Bash, aislar plugin errors como `ToolResult success:false` no abort loop (F2.2 pattern `agent.rs:307` match execute). Añadir `plugin` docs `docs/PLUGIN.md`. Must NOT permitir plugin `eval` bypass, Must NOT duplicar `Agent::new` dedup `agent.rs:113`.
  Parallelization: Wave 2 | Blocked by: 12 | Blocks: 14
  References: crates/kitty-harness/src/tools.rs:310-324 double-block write, guard.rs:824 check_command layers, agent.rs:307 registry.execute match Ok/Err
  Acceptance criteria: `cargo test -p kitty-harness plugin_integration` plugin tool hereda guard block + `cargo clippy -- -D warnings` limpio
  QA scenarios: happy: plugin tool `Read` like funciona; failure: plugin `Bash rm -rf /` block por Guard igual que core
  Commit: Y | feat(harness): plugin ToolRegistry integration + Guard

- [ ] 14. W2: Plugin tests aislamiento + docs
  What to do / Must NOT do: Tests: plugin registra 2 tools, core no modificado, `cargo test` plugin suite 3 cases + doc `docs/PLUGIN.md` con ejemplo `plugins.toml` y `cargo features`. Verificar `cargo test --workspace` 247+ nuevos verdes. Must NOT añadir dependencia WASM sin feature gate.
  Parallelization: Wave 2 | Blocked by: 13 | Blocks: 15,26
  References: docs/PLAN-EJECUCION.md:352 criterio terceros pueden registrar tools sin tocar core
  Acceptance criteria: `docs/PLUGIN.md` existe + 2 plugin tests pass + `cargo test --workspace` verde
  QA scenarios: happy: third-party plugin sin tocar `tools.rs` core; failure: plugin que panica debe capturarse como ToolResult failure no crash harness
  Commit: Y | docs: plugin system guide + tests

- [ ] 15. W3: TUI prompt input + historial buffer
  What to do / Must NOT do: Extender `tui.rs:59-91` ui() chunk prompt de placeholder `>` a `tui_input::Input` o `ratatui::widgets::Paragraph` con `crossterm` key handling, buffer `Vec<String> history`, `history_idx`, manejo `Up/Down` navega historial, `Enter` submit. Mantener `status bar` 94-147 colores red>80 yellow>60. Must NOT romper feature gate `#[cfg(feature="tui")]`, Must NOT bloquear hilo UI.
  Parallelization: Wave 3 | Blocked by: 9,11,14 | Blocks: 16
  References: crates/kitty-cli/src/tui.rs:1-193 display-only feature-gated, tui.rs:25-56 StatusInfo + context_pct, main.rs:3-4 cfg feature tui +127 Commands::Tui
  Acceptance criteria: `cargo test -p kitty-cli --features tui` pass + `cargo build --features tui` ok + manual `kitty tui` type "hello" aparece en input
  QA scenarios: happy: type "fix bug" Enter -> input cleared + history push; failure: input vacío Enter no debe submit (no crash)
  Commit: Y | feat(tui): prompt input + history

- [ ] 16. W3: TUI scroll + status bar live + conversación
  What to do / Must NOT do: Añadir `conversation: Vec<Message>` render en chunk middle con `List` scrollable (`scroll_offset`), `PgUp/PgDn` scroll, status bar `render_status_bar` actualiza `context_pct()` live + `session_duration()`. Must NOT hardcode altura 3 rows, usar `Layout` constraints.
  Parallelization: Wave 3 | Blocked by: 15 | Blocks: 17
  References: crates/kitty-cli/src/tui.rs:59-91 ui 3 chunks, tui.rs:94-147 render_status_bar ctx color, tui.rs:150-193 run_tui crossterm alternate screen 100ms poll
  Acceptance criteria: TUI muestra 2 mensajes + scroll `j/k` funciona + status bar verde->amarillo->rojo según ctx
  QA scenarios: happy: 10 mensajes scroll offset 5 visible; failure: 0 mensajes muestra "No messages yet" placeholder sin panic
  Commit: Y | feat(tui): scroll + live status

- [ ] 17. W3: TUI streaming integración tokio mpsc con Agent::run
  What to do / Must NOT do: Modificar `tui::run_tui` para crear `tokio::mpsc::channel(100)` + spawn `Agent::run` en task, enviar `AgentEvent` (agent.rs:64) via channel a UI loop `tokio::select!` 100ms poll + `crossterm::event::poll`. Render streaming `thinking` + `tool_calls` + `tool_results` incremental. `r` reset limpia channel. Must NOT bloquear UI en `Agent::run` sync (usar spawn), Must NOT perder `AgentEvent` existente o eliminarlo (F4.3).
  Parallelization: Wave 3 | Blocked by: 16 | Blocks: 18,26
  References: crates/kitty-harness/src/agent.rs:64 AgentEvent, agent.rs:296 run loop, tui.rs:170-193 run_app 100ms poll q/Esc/r, agent.rs:448/475 tail-injection
  Acceptance criteria: TUI `kitty tui` + prompt "write hello.py" muestra streaming tool calls sin freeze + `cargo test --features tui` pass
  QA scenarios: happy: agent emite tool_call -> TUI lista muestra "Tool: Write hello.py" en <200ms; failure: agent error -> TUI muestra "Tool failed: ..." no crash, q/Esc sale
  Commit: Y | feat(tui): streaming mpsc Agent integration

- [ ] 18. W3: TUI polish + tests feature-gated
  What to do / Must NOT do: Añadir `#[cfg(test)]` para TUI StatusInfo unit tests `context_pct`, añadir `cargo test --features tui` 3 tests, verificar `cargo clippy --features tui -- -D warnings` limpio, doc `docs/TUI.md` usage. Must NOT dejar `TODO` placeholder `tui.rs:78`.
  Parallelization: Wave 3 | Blocked by: 17 | Blocks: 26
  References: crates/kitty-cli/src/tui.rs:193 run_app
  Acceptance criteria: 3 TUI tests verdes + clippy tui limpio + `docs/TUI.md` existe
  QA scenarios: happy: `cargo test --features tui` 3 pass; failure: sin feature `tui` build no incluye tui.rs (no error)
  Commit: Y | test(tui): polish + docs

- [ ] 19. W4: Tiny probe mide tool-call validity <95% gate
  What to do / Must NOT do: Correr `kitty bench polyglot --limit 50 --profile tiny --exercises benches/exercises_polyglot.json --out /tmp/tiny_probe.json` + script `benches/measure_validity.py` que parsea `tool_calls` validity via `GrammarConfig::validate` (strict filter rate) y calcula % valid. Guardar `/tmp/tiny_probe_validity.json`. Must NOT usar `oracle` para este probe (debe ser `tiny`/`smollm2:1.7b` o `qwen3:4b`), Must NOT modificar `grammar.rs` antes de gate.
  Parallelization: Wave 4 | Blocked by: 2 | Blocks: 20
  References: docs/STRATEGY.md:112-113 GO when tiny validity <95% after prompt/router tuning, crates/kitty-harness/src/model_profile.rs:32 TINY 4K/5iter/4tools/no-native/30s/0.7, crates/kitty-harness/src/grammar.rs:79 validate, docs/FLOOR-BENCHMARK.md:54 53% floor <50%
  Acceptance criteria: `/tmp/tiny_probe_validity.json` contiene `valid_pct` numérico + `benches/measure_validity.py` existe + `cargo test --workspace` no afectado
  QA scenarios: happy: probe retorna valid_pct ej 88% (<95% => GO); failure: si probe >=95% => documenta NO-GO y no implementa llama-cpp-2
  Commit: Y | chore(bench): tiny validity probe 50

- [ ] 20. W4: Decisión GO/NO-GO documentada ADR
  What to do / Must NOT do: Escribir `docs/ADR-GBNF-GO.md` con veredicto GO o NO-GO basado en `valid_pct` de Todo 19, citar `docs/STRATEGY.md:104` F6.4 ADR-005 y `test_gbnf_spike_documented` 354-360. Si NO-GO, mantener spike post-hoc y saltar 21-23 con justificación; si GO, justificar Option A vs B. Must NOT decidir sin evidencia `/tmp/tiny_probe_validity.json`.
  Parallelization: Wave 4 | Blocked by: 19 | Blocks: 21,22
  References: docs/STRATEGY.md:104-145 F6.4 GBNF spike ADR, crates/kitty-runtime/src/grammar.rs:6-8 SPIKE defer header
  Acceptance criteria: `docs/ADR-GBNF-GO.md` existe con `Verdict: GO|NO-GO` + `valid_pct` + razonamiento A vs B + `cargo test` still green
  QA scenarios: happy: GO doc lista tradeoffs A 3-5d vs B 1-2d; failure: doc sin valid_pct debe fallar review F1
  Commit: Y | docs: ADR GBNF GO/NO-GO verdict

- [ ] 21. W4: Si GO — implementar LlamaCppClient: ChatClient con llama-cpp-2 0.1.154
  What to do / Must NOT do: Añadir `llama-cpp-2 = "0.1.154"` a `[workspace.dependencies]` y `kitty-runtime` `Cargo.toml` (solo si GO), implementar `crates/kitty-runtime/src/llama.rs` con `struct LlamaCppClient { model_path: PathBuf, sampler: LlamaSampler }` que impl `ChatClient` `client.rs:19`: `chat` usa `LlamaSampler::grammar_lazy` + `json_schema_to_grammar` por `ToolDefinition`, carga GGUF `Q4_K_M` si existe, maneja `Send/Sync` con `Arc<Mutex>`. No daemon. Must NOT enviar `grammar` + `tools` juntos en HTTP path (solo sampler local), Must NOT bloquear si `model_path` no existe (fallback a Ollama).
  Parallelization: Wave 4 | Blocked by: 20 | Blocks: 22,23
  References: docs/STRATEGY.md:118 LlamaSampler::grammar/grammar_lazy, Cargo.toml:68-75 F6.4 defer comment not to add now, crates/kitty-runtime/src/client.rs:19 trait + 362 MockClient Arc<Mutex>, crates/kitty-runtime/src/grammar.rs:44 to_gbnf +151 GrammarClient
  Acceptance criteria: `cargo build -p kitty-runtime --features llama` compila + `LlamaCppClient::new("~/.kitty/models/ornith.gguf")` no panic + `cargo test -p kitty-runtime llama` 2 tests pass (si NO-GO este todo se marca Skipped con doc)
  QA scenarios: happy: LlamaCppClient chat con 2 tools retorna tool_call válido con sampler; failure: sin GGUF debe error "model not found, fallback to Ollama" no crash
  Commit: Y | feat(runtime): LlamaCppClient grammar_lazy (GO only, else Skipped)

- [ ] 22. W4: Wire GrammarConfig -> sampler triggers sin HTTP 400 pitfall
  What to do / Must NOT do: En `llama.rs` convertir `GrammarConfig::to_gbnf()` a `json_schema_to_grammar` por tool, crear `LlamaSampler::grammar_lazy(triggers=["{\"name\""])` que activa gramática solo en tool-call prefix (PR #9639), no enviar `tools` array + `grammar` string juntos. En `agent.rs::new_inner` 191-200 si `enable_grammar && profile==tiny` usa `LlamaCppClient` inner envuelto. Mantener `GrammarClient` post-hoc como fallback. Must NOT violar `server-common.cpp:1289` pitfall.
  Parallelization: Wave 4 | Blocked by: 21 | Blocks: 23
  References: docs/STRATEGY.md:128 pitfall HTTP 400, docs/STRATEGY.md:132 table A vs B, agent.rs:191-200 auto-wrap GrammarClient, grammar.rs:44 to_gbnf
  Acceptance criteria: `agent.rs` tiny profile usa LlamaCppClient + grammar_lazy triggers + `cargo test -p kitty-harness agent::tests::grammar_lazy_no_http400` pass
  QA scenarios: happy: tiny prompt genera tool_call con grammar_lazy sin 400; failure: tool_call con nombre inválido filtrado por sampler no llega a registry (0 hallucinated tools)
  Commit: Y | feat(harness): wire grammar_lazy triggers (GO only)

- [ ] 23. W4: A/B validity bench tiny 50 con/sin grammar target >95%
  What to do / Must NOT do: Correr `benches/ab_gbnf.py --limit 50 --profile tiny` comparando `spike post-hoc` vs `llama grammar_lazy` medir `tool_call_validity` % (GrammarConfig validate) y `single-file success` string-match. Target `>95% validity` (STRATEGY §6 90% general, F6.4 95% sub-2B). Guardar `benches/ab_gbnf_result.json`. Si NO-GO, este todo valida que spike mantiene >95% sin bind. Must NOT reclamar sub-2B agent si benchmark muestra <95% con grammar (honestidad).
  Parallelization: Wave 4 | Blocked by: 22 | Blocks: 24,26
  References: docs/STRATEGY.md:96 Métricas éxito tool-call validity >90%, docs/STRATEGY.md:112 target >95% para sub-2B, docs/FLOOR-BENCHMARK.md:54 NO-GO <50%
  Acceptance criteria: `benches/ab_gbnf_result.json` con `with_grammar_valid_pct` >95 si GO, o `post_hoc_valid_pct` >=95 si NO-GO + doc en `docs/FLOOR-BENCHMARK.md` update
  QA scenarios: happy: with_grammar 98% vs post_hoc 88% => GO validado; failure: with_grammar <95% debe documentar limitación y no afirmar >95%
  Commit: Y | bench: A/B GBNF validity >95% (GO) or spike validated (NO-GO)

- [ ] 24. W5: Full Aider Polyglot 225 oficial con oracle 15-24h
  What to do / Must NOT do: `harbor run -p bench/official/aider-harbor -a kitty --model oracle --k 225` o `bench/official/aider-harness/benchmark/benchmark.py` con `kitty-cli` harness (Docker), timeout por task 60s (profile medium), guardar `bench_polyglot_225_oracle.json` con `{passed, total, rate_pct, latency_ms}` + `harbor logs`. Grading por unit tests Exercism (no string-match). Must NOT usar `smollm2:1.7b` (gate <50% demo), Must NOT editar `bench/official/*`, Must NOT comparar homemade 96.44% como oficial.
  Parallelization: Wave 5 | Blocked by: 3,23 | Blocks: 26
  References: bench/official/README.md:12-13 polyglot-benchmark 225 cpp26 go39 java47 js49 py34 rust30, docs/PLAN-EJECUCION.md:324 Gate >45.56%, docs/FLOOR-BENCHMARK.md:8 grading string-match NOT comparable
  Acceptance criteria: `bench_polyglot_225_oracle.json` existe con `total==225` + `rate_pct` numérico + `harbor run` logs en `.omo/evidence/f6-remaining-tracks-opt-order/task-24-polyglot.log`
  QA scenarios: happy: 225 tasks completan 15-24h sin Docker daemon crash; failure: task timeout individual no aborta batch (retry false, continúa), OOM contained por sandbox no escapa
  Commit: Y | bench: official polyglot 225 oracle full

- [ ] 25. W5: Full Terminal-Bench 2.0 Harbor 89 con oracle 3-8h
  What to do / Must NOT do: `harbor run -p bench/official/terminal-bench-2 -a kitty --model oracle --k 89` con Docker per-task `environment/Dockerfile`, guardar `bench_tb2_89_oracle.json`. Gate >9.2%. Must NOT correr sin Docker, Must NOT confundir homemade `benches/exercises_tb2.json` (8 tasks) con oficial 89.
  Parallelization: Wave 5 | Blocked by: 3 | Blocks: 26
  References: bench/official/README.md:10 terminal-bench-2 89 tasks instruction.md solution tests task.toml, docs/PLAN-EJECUCION.md:325 Gate >9.2%
  Acceptance criteria: `bench_tb2_89_oracle.json` con `total==89` + logs `.omo/evidence/task-25-tb2.log` + `rate_pct` >9.2 o documentado negativo honesto
  QA scenarios: happy: 89 tasks 3-8h pass >9.2%; failure: 0/89 debe publicarse como negativo honesto en AUDIT.md, no ocultar
  Commit: Y | bench: official TB2 89 oracle full

- [ ] 26. W5: Publish AUDIT.md Consumer vs Vendor + docs honesta
  What to do / Must NOT do: Actualizar `docs/AUDIT.md` (crear si no existe, buscar en `docs/TECHNICAL_AUDIT_REPORT.md`) con sección `## Benchmarks oficiales (oracle)` tabla `Consumer (oracle) vs Vendor (Aider harness)` con `polyglot 225 oracle %` y `TB2 89 oracle %` + `Consumer vs Vendor` delta + nota `homemade 96.44% string-match NOT comparable`. Actualizar `docs/FLOOR-BENCHMARK.md` con oracle results + `CNAS.md` culture tag. Must NOT fabricar números, Must NOT borrar historia F5.
  Parallelization: Wave 5 | Blocked by: 24,25,11,14,18,23 | Blocks: 27
  References: docs/PLAN-EJECUCION.md:326 publish en AUDIT.md columna Consumer vs Vendor, docs/TECHNICAL_AUDIT_REPORT.md, benches/baseline.json 96.44, docs/FLOOR-BENCHMARK.md gate
  Acceptance criteria: `docs/AUDIT.md` contiene `oracle` + `Consumer vs Vendor` + `bench_polyglot_225_oracle.json` rate + `bench_tb2_89_oracle.json` rate + `cargo test` no roto por docs
  QA scenarios: happy: `grep -q "Consumer vs Vendor" docs/AUDIT.md`; failure: si bench falló, AUDIT.md debe decir "negative result: X% below gate" no vacío
  Commit: Y | docs: AUDIT.md Consumer vs Vendor publish

- [ ] 27. W5: Docs finales + codemap + cargo verde + CHANGELOG
  What to do / Must NOT do: Actualizar `codemap.md` + `crates/*/codemap.md` + `docs/STRATEGY.md` F6.4 ADR final + `CHANGELOG.md` Keep a Changelog `## [Unreleased] - F6 remaining tracks`, correr `cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings` + `cargo test --workspace` 247+ tests verdes + `cargo build --release` + `cargo deny check` (advisories). Commit final. Must NOT dejar `bench_*.json` en root (mover a `benches/` o `.omo/evidence/`), Must NOT push a GitHub (defer).
  Parallelization: Wave 5 | Blocked by: 26 | Blocks: F1-F4
  References: AGENTS.md toolchain stable rustfmt+clippy -D warnings, docs/STRATEGY.md F6.4 ADR defer, docs/PLAN-EJECUCION.md:35 Definition of Done global cargo test verde clippy limpio commit + codemap/AUDIT
  Acceptance criteria: `cargo test --workspace` verde + `cargo clippy -- -D warnings` 0 + `cargo fmt --check` ok + `codemap.md` menciona sandbox/memory/cloud/plugin/tui/gbnf/bench
  QA scenarios: happy: `cargo test --workspace 2>&1 | tail -1` contiene "passed"; failure: clippy warnings debe bloquear commit (fix antes)
  Commit: Y | chore: codemap + CHANGELOG + fmt/clippy green

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  What to check: Cada todo referencia path:línea real, acceptance ejecutable (`cargo test`, `harbor run`, `jq`), QA happy+failure con evidencia `.omo/evidence/...`, dependency matrix coherente, scope IN/OUT respetado (no push/OpenSSF, no grammar+tools 400, no edit bench/official), GBNF conditional con gate documentado.
  Evidence: `.omo/plans/f6-remaining-tracks-opt-order.md` audit log `.omo/evidence/f6-remaining-tracks-opt-order/F1.log`
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F2. Code quality review
  What to check: `cargo clippy -- -D warnings` 0, `cargo fmt --check` ok, módulos <250 LOC ideal (sandbox.rs nuevo no >500), sin unwrap/expect en prod, sin clone injustificado, workspace resolver 2, shared dependencies, Conventional Commits `feat(*):` + DCO `Signed-off-by` si se pushea (no en este plan).
  Evidence: `cargo clippy` output + `cargo fmt --check` + `tokei` LOC `.omo/evidence/F2.log`
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F3. Real manual QA
  What to check: Ejecutar realmente `docker run --rm hello-world`, `harbor run --k 1` smoke, `kitty bench polyglot --limit 3 --profile medium` 3/3 pass, `kitty tui` abre sin crash (q sale), `kitty run --semantic-memory "fix bug"` inject funciona, `kitty run --cloud-fallback` sin key warn no crash. No grep-only QA.
  Evidence: comandos reales + logs `.omo/evidence/F3.log` + screenshots TUI si aplica
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F4. Scope fidelity
  What to check: Todo 1-27 entregado, push/OpenSSF defer respetado (no `SECURITY.md`/`scorecard.yml` en este plan), GBNF no añade dep si NO-GO, bench oficial publica Consumer vs Vendor honesto, no compara homemade 96.44% como oficial, TUI no rompe `kitty run` CLI existente.
  Evidence: `git log --oneline` 10+ commits F6 + `docs/AUDIT.md` Consumer vs Vendor + `bench_*_oracle.json` reproducibles `.omo/evidence/F4.log`
  Verdict: APPROVE | CHANGES_REQUESTED

## Commit strategy
- 1 commit por todo con `feat(harness):` / `feat(memory):` / `feat(runtime):` / `feat(tui):` / `feat(bench):` / `docs:` + DCO `Signed-off-by` si se requiere push (defer aquí). W0 infra no commit (local). W1 6 commits sandbox+memory, W2 5 commits cloud+plugin, W3 4 commits tui, W4 5 commits gbnf (GO) o 2 docs si NO-GO, W5 4 commits bench+docs. Total 15-20 commits. Nunca squash que oculte gate; bench full en commit separado con `bench_polyglot_225_oracle.json` adjunto.

## Success criteria
- Docker + harbor instalados y smoke 5/5 pass (`/tmp/harbor-smoke.json` 5 passed)
- Sandbox: `cargo test -p kitty-harness guard,sandbox` 100% + ningún bypass `echo $(rm -rf /)` sobrevive, OOM/segfault contenidos como ToolResult failure
- Memory: `benches/ab_recall.py` hybrid win >60% vs Jaccard, `recall_hybrid` top-3 relevante, `--semantic-memory` opt-in funciona para medium/large no tiny
- Cloud: `CloudClient` trait genérico + `kitty run --cloud-fallback` escalada solo en fallo con costo logeado, sin key error claro
- Plugin: `PluginRegistry` terceros registran tools sin tocar core, 2+ plugin tests verdes, `docs/PLUGIN.md` existe
- TUI: `cargo build --features tui` ok, `kitty tui` prompt + historial + scroll + streaming mpsc sin freeze, 3 tests tui verdes
- GBNF: `docs/ADR-GBNF-GO.md` con veredicto + `benches/ab_gbnf_result.json` >95% si GO else spike validado, sin HTTP 400
- Bench oficial: `bench_polyglot_225_oracle.json` 225 tasks + `bench_tb2_89_oracle.json` 89 tasks, publicados en `docs/AUDIT.md` Consumer vs Vendor honesto, gates >45.56% y >9.2% evaluados (positivo o negativo honesto)
- Docs: `codemap.md` + `CHANGELOG.md` + `docs/FLOOR-BENCHMARK.md` actualizados, `cargo test --workspace` verde, `cargo clippy -D warnings` limpio, `cargo fmt --check` ok
- Final wave F1-F4 all APPROVE

