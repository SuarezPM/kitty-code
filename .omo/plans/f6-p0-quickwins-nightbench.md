# f6-p0-quickwins-nightbench - Work Plan

## TL;DR (For humans)

**What you'll get:** La noche corre los benchmarks reales (225 Polyglot + 89 TB2 con ornith:9b via harbor) mientras de día cerramos los 3 P0 que el audit marcó como mentira: nombres de JSON honestos, llave API que se borra sola, y sandbox que sí bloquea `/etc/passwd` con Landlock+seccomp real. Además dejamos medido si la gramática 98% es real o simulada, snapshots de la TUI y memoria que sí se prueba con datos reales. Todo con patrones robados de tu propio Catalyst (seccomp y SecretString) y de Codesearch.

**Why this approach:** Los P0 no bloquean el bench — pueden correr en paralelo en tu 2060. Primero arreglamos lo que el auditor llama “security theater” y “filenames lie” (15min), luego lo duro (Landlock 2h) mientras el harbor de 76min corre solo. GBNF empírico es condicional: si no hay GGUF 5GB, no fabricamos 98%, lo marcamos simulado honestamente.

**What it will NOT do:** No añadirá WASM plugins (el audit dice que estáticos son más seguros), no entrenará modelo 397B, no cambiará Ollama por tu motor Dekanus ahora (es Phase 0 skeleton sin tok/s real), y no crecerá el quiz casero con 3 ejercicios nuevos.

**Effort:** Medium — 3-4 días código (rename 15min, zeroize 30min, Landlock 2h, GBNF 1h, TUI 1h, embedding 30min) + 2.5h bench noche en paralelo (76min + 1h)
**Risk:** Medium — Landlock ABI kernel 6.x ok en CachyOS pero requiere test real `/proc/self/status`; GGUF download 5GB puede fallar por red; harbor adapter `kitty` nuevo debe aislar cada tarea en su container
**Decisions I made for you:** Roadmap quick-win P0 TrustLayer + Must Have robado de Catalyst/Codesearch/Dekanus + keep Ollama (no own engine v0.1) + bench noche en paralelo. Ver Vision abajo para pivot.

Your next move: Ejecuta con ` $start-work f6-p0-quickwins-nightbench` — el high-accuracy dual review (momus + oracle) corre automático por ser UNCLEAR.

---

> TL;DR (machine): Medium effort, Medium risk, 5 waves (1 night bench staggered 6 corridas oracle+kitty dual-model ornith:9b & ornith-1.5:9b + 12 impl todos + 4 verifiers), quick-wins P0 + Must Have + vision, bench n==len mean real

## Scope
### Must have
- **Bench noche real (no smoke extrapolado) — CORREGIDO post-G0:** harbor 0.21.0 `-k` es *attempts per trial* (NO nº de tareas) y no existe `--timeout` (solo `--*-timeout-multiplier`). Corridas completas = `-k 1` sin `-l`. Staggered en una GPU (2060S): (a) **oracle TB2-89** `harbor run -p bench/official/terminal-bench-2 -a oracle -k 1` (~59min est desde tasa W5 40s/tarea) → `benches/bench_tb2_oracle_full89.json`; (b) **oracle polyglot-225** `harbor run -p bench/official/aider-harbor/aider-polyglot -a oracle -k 1` (~77min est desde W5 20.4s/tarea) → `benches/bench_polyglot_oracle_full225.json`; (c) **kitty polyglot MUESTREADO ornith:9b** `-l 50 -k 1` → `benches/bench_polyglot_kitty_ornith9b_l50.json`; (d) **kitty polyglot MUESTREADO ornith-1.5:9b** (`ollama pull ornith-1.5:9b` hecho, 6.6GB Q4 imatrix, tag verificado en ollama.com/library/ornith-1.5) `-l 15 -k 1` → `benches/bench_polyglot_kitty_ornith15b_l15.json`; (e) opcional **TB2-kitty** si queda ventana nocturna → nombrado honesto `_lN`. **AMEND-G1 (Oracle APPROVE):** tasas observadas en smoke k1 real (~4min/tarea ornith:9b, ~16min/tarea ornith-1.5:9b en phone-number) hacen full-225 inviable en una noche (est. 15h/60h vs umbral abort 4h) → kitty corre MUESTREADO con regla first-N pre-registrada (`-l N` toma las PRIMERAS N tareas del orden del dataset, cero cherry-pick post-hoc). Interrupt-harvest verificado: result.json de job solo trae metadata (`n_total_trials`, sin trials individuales — viven en subdirs por tarea) → si algo muere a mitad: nombrado honesto `_partialN` con n==len completadas + nota `not empirically closed`. Adapter real: `-a adapters.kitty.agent:KittyAgent` con `PYTHONPATH=$PWD/bench/harbor`, modelo vía env `KITTY_BENCH_MODEL`. Contexto ornith-1.5: 9B dense base Qwen3.5+Gemma4, 256K ctx, Text+Image; claims vendor (TB2.1 Terminus-2 46.2 vs 43.1 del 1.0; SWE-bench Verified 70.6) usan OTRO harness — NO comparables con nuestro harbor verifier; la comparación válida es 1.0-vs-1.5 mismo harness propio. Riesgo VRAM confirmado en smoke: partial offload ~0.8GB a CPU, generación intacta (~53.6 tok/s), prompt processing ~213 vs ~560 tok/s. APROBADO-USUARIO: las corridas (a)+(b) oracle-full son requisito del re-audit gate #3 (audit.md:201 "Harbor oracle 225 and 89 means") y están aprobadas para Wave 0 staggered. `n_total_trials == array.len()` mean real + `git diff -- bench/official` vacío. AMEND-G2: tras cada corrida, chequear `jq '.stats.evals[0] | {n_total_trials, n_errored_trials, mean}'` — harbor CLI exit 0 incluso con trials erróneos; `n_errored_trials > 10%` de N ⇒ run env-comprometido, anotar en AUDIT.md y re-run antes de publicar cualquier número.
- **P0 rename honestidad:** renombrar 4 JSONs con nombres canónicos únicos (tabla del auditor): `bench_polyglot_225_oracle.json` → `bench_polyglot_oracle_smoke15.json`, `bench_polyglot_225_ornith9b.json` → `bench_polyglot_ornith_smoke15.json`, `bench_tb2_89_oracle.json` → `bench_tb2_oracle_ars_k5.json`, `bench_tb2_ornith9b.json` → `bench_tb2_ornith_smoke15.json` (evita colisión con `bench_tb2_ornith9b_fileops15.json` existente) con `n_total == len` y docs `AUDIT.md`/`CNAS.md`/`FLOOR-BENCHMARK.md` headlines sin `225/225` `89/89` + nota `method: simulated` donde aplique.
- **P0 zeroize (Catalyst `apohara-secrets:56-94`):** `SecretString` `Drop zeroize` + `Debug` redact `SecretString(***)` + `expose()` only at transport + `OnceLock<Mutex<SecretString>>` + `is_secret_env_name:124` strip `*_API_KEY/*_TOKEN/AWS_*` en `runner/imp.rs` + cost logs `tracing::info! cost_usd` sin `Authorization` + test `format!("{client:?}")` no contiene key. **AMEND-G2: keyring OMITIDO aprobado — fuente de key es env var, no existe capa storage OS que portar; gates audit = zeroize+redact+never-log.**
- **P0 Landlock+seccomp real (Catalyst `profile/linux.rs:7` + `namespace.rs:52` + audit P0):** `seccompiler 0.5` EPERM fail-closed + reglas condicionales `openat/fcntl/ioctl/clone` `linux.rs:110-187` + `landlock 0.4` `RestrictSelf FsAccess::from_all(ABI)` + `nix CLONE_NEWUSER|NEWNS|NEWPID` `namespace.rs:52` + double-fork `runner/imp.rs:110` `MAX_OUTPUT 16MiB` + `CLOEXEC exec-error pipe` + `setrlimit RLIMIT_AS 512MB` + `#[cfg(linux)] apply_linux_stubs()` real (no println!) + `SandboxError::Denied` para `execute_bash` + tests `write /etc/passwd → PermissionDenied`, `open /etc/shadow → PermissionDenied`, `allocate 2GB → SIGKILL`, `cargo miri` para `llama.rs:220 unsafe Send/Sync` documentado.
- **P0 GBNF empírico:** `benches/ab_gbnf.py --with-model ~/.kitty/models/qwen2.5-coder-7b-q4_k_m.gguf` (nombre verdadero, NUNCA ornith.gguf) mide sampler `LlamaSampler::grammar_lazy` triggers `{"name"` real; si GGUF no existe tras download pinneado 5GB Q4_K_M (~30min), marcar `method: simulated` y cerrar ADR `not empirically closed` sin fabricar `98%` si queda <95.
- **QW TUI snapshots:** `ratatui TestBackend` snapshots para `handle_event` CJK + `empty-q` (q solo si input vacío) + history draft restore + viewport `inner_h=height-2` slice, 14 tests existentes + 3 nuevos.
- **QW embedding e2e:** test `embedding_blob_migration_and_recall` con `episodes.db` real + CI `kitty bench --semantic-memory --limit 15` permitido perder vs lexical (hybrid puede perder, no 83.3% toy press-release).
- **Vision + Must Have doc:** `docs/ROADMAP-CATALYST-ROBABLE.md` (80+ líneas) + `docs/VISION-ROADMAP.md` con Must Have robables + own inference worth (keep Ollama) + director vision pivot vs smallcode/littlercoder (ver Decisions).
- **Docs hygiene:** `bench/official` untouched (`git diff -- bench/official` vacío), `cargo test --workspace --all-features` ≥308 passed (numérico), `cargo clippy --all-targets --all-features -- -D warnings` 0, `cargo fmt --check` 0.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Wasmtime WASM plugins como merge gate (audit:179, Catalyst no lo usa) — static `HelloPlugin` + `duplicate_blocked` stay
- Custom grammar + `tools` HTTP juntos `server-common.cpp:400` — usar `grammar_lazy` triggers
- Entrenar modelo, GGUF 397B 17GB VRAM, cambiar Ollama ahora por Dekanus completo (Phase 0 skeleton `candle 0.11 + cudarc 0.19` sin bench real)
- Homepage/docs site, push a GitHub, OpenSSF scorecard full (higiene separate, `deny.toml` license allowlist ya existe Catalyst)
- Crecer `exercises_polyglot.json` con 3 toys train-on-test (audit:156)
- Fabricar `98%` si queda <95% o extrapolar `15→225` con nombre `225` (TrustLayer fail)
- `unwrap`/`expect` en prod, `*API_KEY` en logs, `bench/official` edit, `child env -u KITTY_SANDBOX_*` bypass (Catalyst `is_secret_env_name` lo bloquea)

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + `cargo test --workspace --all-features` (≥308 numérico) + `cargo clippy -D warnings` + `cargo fmt --check` + Miri scoped a unidades unsafe pure-Rust (setup `rustup toolchain install nightly --component miri`; NO sobre FFI llama.cpp; Send/Sync vía assertion compile-time fuera de Miri)
- Evidence: `.omo/evidence/f6-p0-quickwins-nightbench/task-<N>.log` + los 4 JSONs full/smoke de todo 2 + `F1-final.log`
- Per-todo QA: happy + failure con comando exacto + evidencia path; bench con `harbor run -k 1` + `jq .mean` + `jq length == n_total_trials`
- Final wave: 4 verifiers paralelos ALL APPROVE

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- **Preliminar — Baseline commit (todo 0, obligatorio antes de Wave 0):** commitear el árbol sucio actual (~70 archivos F6 tracks untracked pending merge) como baseline limpio para diffs/rollback/verificación F1-F4. Sin esto no hay per-todo diff ni rollback.
- **Wave 0 — Night bench cola serial una GPU (todo 2):** 5 corridas AMEND-G1: TB2-oracle (~59min) → polyglot-oracle (~77min) → kitty l50 ornith:9b (~3-4h) → kitty15 l15 ornith-1.5:9b (~4h) → opcional TB2-kitty. Corre en paralelo con Waves 1-4 (no bloquea P0s) pero serializada en la única GPU — NO solapar con todo 7 GPU decode.
- **Wave 1 — P0 honesty + secrets (2 todos, 45min):** rename JSON + zeroize SecretString (paralelizables, 15+30min)
- **Wave 2 — P0 sandbox real (2 todos, 2.5h):** seccomp+landlock impl + tests /etc/passwd con control negativo + Miri scoped (tras Wave1 por ordenamiento soft, no dependencia de código)
- **Wave 3 — GBNF + TUI + embedding (3 todos, 2h):** GBNF empírico (condicional GGUF, fuera de ventana bench) + TUI snapshots + embedding e2e tempdir (paralelizables tras Wave2)
- **Wave 4 — Vision + hygiene (3 todos, 1h):** Must Have doc + vision pivot + codemap/CHANGELOG + cargo verde (tras Wave 3)
- **Wave 5 — Final verification (4 todos paralelos)**

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 2 | 3,4 |
| 2 | 1 | 12 | 3-11 |
| 3 | - | 12 | 1,2,4,5 |
| 4 | - | 12 | 1,2,3,5 |
| 5 | 3,4 (soft ordering, no code dep) | 6 | 1,2 |
| 6 | 5 | 12 | 7,8,9 |
| 7 | 5 | 12 | 8,9 |
| 8 | 5 | 12 | 7,9 |
| 9 | 5 | 12 | 7,8 |
| 10 | 7,8,9 | 12 | 11 |
| 11 | 7,8,9 | 12 | 10 |
| 12 | 6,7,8,9,10,11,2 | F1-F4 | - |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Wave0: Harbor kitty adapter multi-modelo (ornith:9b + ornith-1.5:9b)
  What to do / Must NOT do: Crear adapter kitty como **paquete Python bajo `bench/harbor/adapters/kitty/`** (harbor 0.21.0: `harbor adapter init` es wizard interactivo SIN `--config`; los adapters se registran por import path `module.path:Class`, NO `-a oracle/kitty`). El adapter invoca `kitty-cli run --model $KITTY_BENCH_MODEL --profile medium` dentro container (`/workspace` fs_allow), **modelo PARAMETRIZADO vía env `KITTY_BENCH_MODEL`** (default `ornith:9b`; la noche corre también `ornith-1.5:9b` sin cambiar código), aislado por tarea, con strip de `KITTY_CLOUD_API_KEY/*_TOKEN/AWS_*` vía patrón `is_secret_env_name`, y networking que permita al kitty-cli containerizado alcanzar Ollama del host (`OLLAMA_HOST`/`--network host` — resolver empíricamente en el smoke). Validar con `harbor adapter review` (subcomando de `harbor adapter`, NO top-level). Must NOT usar oracle, Must NOT editar `bench/official` (ver Must NOT), Must NOT copiar solve.sh (harbor lo hace), Must NOT asumir TOML job config (`job start` exige JobConfig YAML/JSON).
  Parallelization: Wave 0 | Blocked by: - | Blocks: 2 | Can parallelize with: 3,4
  References (executor has NO interview context - be exhaustive): bench/official/README.md:28-54, bench/official/aider-harbor/aider-polyglot 225 tasks `ls | wc -l` 225, harbor 0.21.0 `harbor adapter init --help` (wizard interactivo) + `harbor adapter review --help`, docs/AUDIT.md:149-154 re-exec gate, apohara-sandbox/runner/imp.rs:168 build_sanitized_env (patron Catalyst a portar)
  Acceptance criteria (agent-executable): adapter Python registrado e importable; `harbor adapter review` pasa; smoke k1: `harbor run -p bench/official/aider-harbor/aider-polyglot -a <import-path-kitty> -k 1 -l 1 --include-task-name polyglot_python_hello-world` exit 0 ejecutando kitty (no oracle) y crea result.json con `jq .stats.evals[0].mean`
  QA scenarios (name the exact tool + invocation): happy: smoke k1 anterior con logs de kitty; failure: `-a kitty-sin-registrar` debe error `agent not found`, no 225/225
  Commit: Y | feat(bench): harbor kitty adapter for ornith:9b

- [ ] 2. Wave0: Bench noche completa oracle+kitty real (no smoke) — staggered una GPU
  What to do / Must NOT do: Ejecutar EN COLA SERIAL (una sola GPU/Ollama). **AMEND-G1 (Oracle APPROVE):** tasas smoke reales (~4min/tarea ornith:9b, ~16min/tarea ornith-1.5:9b) hacen full-225 inviable en una noche → kitty corre MUESTREADO first-N pre-registrado. Cola: (1) TB2 oracle `harbor run -p bench/official/terminal-bench-2 -a oracle -k 1` (~59min est) → `benches/bench_tb2_oracle_full89.json`; (2) polyglot oracle `harbor run -p bench/official/aider-harbor/aider-polyglot -a oracle -k 1` (~77min est) → `benches/bench_polyglot_oracle_full225.json`; (3) polyglot kitty ornith:9b `PYTHONPATH=$PWD/bench/harbor KITTY_BENCH_MODEL=ornith:9b harbor run -p bench/official/aider-harbor/aider-polyglot -a adapters.kitty.agent:KittyAgent -l 50 -k 1 --ae KITTY_SANDBOX=off` (~est 3-4h) → `benches/bench_polyglot_kitty_ornith9b_l50.json`; (4) polyglot kitty ornith-1.5:9b `KITTY_BENCH_MODEL=ornith-1.5:9b` mismo comando `-l 15 -k 1 --ae KITTY_SANDBOX=off` → `benches/bench_polyglot_kitty_ornith15b_l15.json`; (5) OPCIONAL TB2-kitty si queda ventana `--ae KITTY_SANDBOX=off` → nombrado honesto `_lN`. **AMEND-N1 (verificado contra CLI antes de lanzar): harbor run NO tiene flag `-o` — los job dirs caen en `jobs/` (gitignoreado); el runner nocturno captura el result.json más nuevo tras cada paso. Docker se controla vía sudo (usuario fuera del grupo docker; patrón validado en smokes del día).** **AMEND-N2 (incidente noche + corrección): la cola N1 completó 4/4 con gates que resultaron VACUOS — el jq recursivo `[..][0]` agarraba el objeto eval-level anidado (e=0 falso) en vez de `.stats` top-level. Errores reales: P1 1/89 (ok), P2 0/225 (ok), P3 29/50 (58%), P4 13/15 (87%) — causa raíz: timeout HTTP fijo de 120s en kitty-cli (ollama.rs:285) < latencia de llamadas pesadas (prompt processing 213-560 tok/s sobre 5-14K tokens + thinking largo; el server siguió generando tras el disconnect del client = GPU estresada toda la noche con trials erróneos). FIX: `KITTY_OLLAMA_TIMEOUT_SECS` env-configurable (default 120 laptop), re-run R3/R4 con `--ae KITTY_OLLAMA_TIMEOUT_SECS=600`, binario del adapter refrescado. Gate v2 usa `.stats` top-level directamente. Protocolo pre-registrado se mantiene: re-run UNA vez; segundo fallo → `_partialN` honesto.** **AMEND-G3 (smoke verificado): userns NO disponible in-container (Docker default seccomp bloquea CLONE_NEWUSER pese a sysctls permissivos) → las corridas kitty requieren `--ae KITTY_SANDBOX=off` (opt-out con tracing::warn! fuerte en cada activación, validado end-to-end 6/6); alternativa futura: runtime con userns habilitado.** Regla first-N pre-registrada: `-l N` = PRIMERAS N tareas del orden del dataset, cero cherry-pick post-hoc. `-k 1` = 1 intento por tarea (NO `-k 225`: serían 50,625 trials); sin `--timeout` (no existe; usar `--agent-timeout-multiplier` si hace falta). Guardar cada JSON con `n_total_trials == array.len()` mean real + logs harbor en `.omo/evidence/task-2.log` + `git diff -- bench/official` vacío. APROBADO-USUARIO: pasos (1)+(2) oracle-full son requisito re-audit gate #3 y corren primero; ornith-1.5 incluido por decisión del usuario (investigado EXA, tag `ornith-1.5:9b`). Si un run muere a mitad: nombrado honesto `_partialN` con n==len completadas + nota `not empirically closed` (interrupt-harvest verificado: result.json de job solo trae metadata, trials viven en subdirs por tarea). Pre-compromiso: resultado parcial honesto es ACEPTABLE — nunca renombrar parcial como completo. Must NOT extrapolar, Must NOT nombre full/sampled si n no coincide, Must NOT usar smollm2, Must NOT comparar nuestros números con los TB2.1 vendor (verifier distinto).
  Parallelization: Wave 0 | Blocked by: 1 | Blocks: 12 | Can parallelize with: 3-11 (bench corre solo, no bloquea quick wins; NO solapar con todo 7 GPU decode)
  References: bench/official/terminal-bench-2 89 dirs, aider-harbor 225 tasks, harbor 0.21.0 `harbor run --help` (-k n-attempts, timeouts solo multipliers), W5 tasas reales: oracle polyglot 15/15 5m06s (~20.4s/tarea), oracle TB2 5/5 3m19s (~40s/tarea), docs/STRATEGY.md:84 bench gate >45.56%/9.2%
  Acceptance criteria: los JSONs existen con `jq length == n_total_trials` (89/225/50/15 según corrida) y `jq .mean` numérico + **AMEND-G2**: tras cada paso registrar en task-2.log `jq '.stats.evals[0] | {n_total_trials, n_errored_trials, mean}'` y gate `n_errored_trials <= 10%` de N (harbor exit 0 NO implica trials sanos) + `git diff -- bench/official` vacío; si un run muere a mitad: `_partialN` con n==len completadas + nota honesta
  QA scenarios: happy: `jq '.[0].trial.result' benches/bench_polyglot_kitty_full225.json` existe; failure: si un JSON "full" tiene length != nombre, CHANGES_REQUESTED
  Commit: Y | bench: full oracle+kitty harbor mean real staggered

- [ ] 3. Wave1: P0 rename JSON honestidad
  What to do / Must NOT do: Renombrar con la tabla canónica única (nombres del auditor, sin colisiones): `benches/bench_polyglot_225_oracle.json` → `benches/bench_polyglot_oracle_smoke15.json`, `bench_polyglot_225_ornith9b.json` → `bench_polyglot_ornith_smoke15.json`, `bench_tb2_89_oracle.json` → `bench_tb2_oracle_ars_k5.json`, `bench_tb2_ornith9b.json` → `benches/bench_tb2_ornith_smoke15.json` (evita colisión con `bench_tb2_ornith9b_fileops15.json` existente) + corregir `total:225` → `n_total == len` + docs `AUDIT.md` `CNAS.md` `FLOOR-BENCHMARK.md` headlines sin `225/225 89/89` + nota `method: simulated` vs `harbor verifier` + repro `harbor run` gate. Must NOT borrar `*_full*.json` reales de Todo2, Must NOT fabricar.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 12 | Can parallelize with: 1,2,4,5
  References: /home/thelinconx/Descargas/kitty-code-v2-trustlayer-audit.md:136-154 table filenames lie + ship rule audit.md:25, benches/bench_*.json 15/5 trials, docs/AUDIT.md:149 honest smoke
  Acceptance criteria: existen exactamente los 4 archivos canónicos de arriba (`ls benches/bench_polyglot_oracle_smoke15.json benches/bench_polyglot_ornith_smoke15.json benches/bench_tb2_oracle_ars_k5.json benches/bench_tb2_ornith_smoke15.json`) con `jq length == n_total_trials` y `grep -r "225/225" benches/` 0 + `grep "NOT comparable" docs/AUDIT.md` 1 + `cargo test --workspace --all-features` ≥308 pass (comparación numérica, NUNCA string-equality de conteo)
  QA scenarios: happy: `jq .n_total_trials benches/bench_polyglot_oracle_smoke15.json` == `jq length benches/bench_polyglot_oracle_smoke15.json`; failure: si queda `total:225` con len 15 debe CHANGES_REQUESTED
  Commit: Y | docs: rename JSON smoke honestidad

- [ ] 4. Wave1: P0 zeroize SecretString + Debug redaction (Catalyst)
  What to do / Must NOT do: Copiar verbatim `apohara-secrets/src/lib.rs:56-94` `SecretString(String)` `Drop zeroize` `Debug redact SecretString(***)` `expose()->&str` + `zeroize` derive a `crates/kitty-runtime/src/cloud.rs` reemplazando `String api_key` (actual, no OnceLock) con `SecretString` + `expose()` only at transport + portar patrón `is_secret_env_name` (Catalyst apohara-sandbox/runner/imp.rs:124 — path Catalyst, NO existe en este repo; el strip va en el env-sanitize de cloud.rs/sandbox de kitty) bloqueando `*_API_KEY/*_TOKEN/AWS_*` + cost logs estructurados `cost_usd/model` sin `Authorization` + test `format!("{:?}", SecretString("sk-123"))==SecretString(***)` + `cargo test CloudClient_debug_redacts`. **AMEND-G2: keyring OMITIDO aprobado Oracle — key viene de env var, sin capa storage OS; gates audit = zeroize+redact+never-log (implementado en e05c7c6).** Must NOT log key, Must NOT usar `String` sin zeroize.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 12 | Can parallelize with: 1,2,3,5
  References: /home/thelinconx/Proyectos/Repos/Apohara-Catalyst/crates/apohara-secrets/src/lib.rs:56-94 + Cargo.toml zeroize 1.8 (keyring omitido AMEND-G2), crates/kitty-runtime/src/cloud.rs:578 OnceLock Mutex, Catalyst apohara-sandbox/src/runner/imp.rs:124 is_secret_env_name + :168 build_sanitized_env (portar patrón, no copiar path), audit.md:98-100 P0 zeroize
  Acceptance criteria: `cargo test -p kitty-runtime -- cloud` todos pasan (≥9, comparación numérica) + `cargo test CloudClient_debug_redacts` pass `!format!("{:?}", client).contains("sk-")` + `grep SecretString crates/kitty-runtime/src/cloud.rs` 1 + `grep zeroize Cargo.toml` 1
  QA scenarios: happy: `cargo test -- --nocapture` no muestra key; failure: `format!("{:?}", SecretString("sk-123"))` debe ser `SecretString(***)` no `sk-123`
  Commit: Y | feat(runtime): SecretString zeroize Catalyst

- [ ] 5. Wave2: P0 Landlock+seccomp real impl (Catalyst seccomp+namespaces)
  What to do / Must NOT do: Reemplazar `crates/kitty-harness/src/sandbox.rs:98-121` stub `println!` por `seccompiler 0.5` JSON `build_json_spec:80` EPERM `apply_filter:212` + reglas condicionales `openat masked_eq O_ACCMODE` (Catalyst linux.rs:110-187) + `landlock 0.4` `RestrictSelf FsAccess::from_all(ABI)` (Catalyst profile/linux.rs:7) + `nix 0.30` `enter_isolated_namespaces:52` `CLONE_NEWUSER|NEWNS|NEWPID` + portar a sandbox.rs el patrón Catalyst `runner/imp.rs:110 MAX_OUTPUT 16MiB` + `CLOEXEC exec-error pipe` + `setrlimit RLIMIT_AS 512MB` (512*1024*1024) + `canonicalize_recursive` `apohara-pathsafety:61` `MAX_SYMLINK_HOPS=32`. Mantener `Guard::check_chain` antes de sandbox. Nota: `runner/imp.rs` es path Catalyst — NO existe en este repo; los patrones se portan dentro de `sandbox.rs`/`guard.rs` de kitty-harness. Must NOT `println!` stub, Must NOT `env -u KITTY_SANDBOX_*` bypass (is_secret check).
  Parallelization: Wave 2 | Blocked by: 3,4 (ordenamiento soft, no dependencia de código) | Blocks: 6 | Can parallelize with: 1,2
  References: /home/thelinconx/Proyectos/Repos/Apohara-Catalyst/crates/apohara-sandbox/src/profile/linux.rs:7 + src/profile.rs + src/namespace.rs:52 + src/runner/imp.rs:110 + Cargo.toml seccompiler 0.5 nix libc, crates/kitty-harness/src/sandbox.rs:77-88 current stub, audit.md:52-77 P0 landlock + apohara-sandbox 16MiB cap
  Acceptance criteria: `crates/kitty-harness/src/sandbox.rs` contiene `seccompiler` + `landlock` + `CLONE_NEWUSER` + `RLIMIT_AS` + `MAX_OUTPUT_BYTES` y `cargo build -p kitty-harness` ok + `cargo clippy -p kitty-harness -- -D warnings` 0
  QA scenarios: happy: `cargo build` no `println!("[sandbox` stub; failure: `grep -c "println!" sandbox.rs` debe ser 0 para non-test code
  Commit: Y | feat(harness): Landlock+seccomp real Catalyst

- [ ] 6. Wave2: P0 Landlock tests + Miri Send/Sync
  What to do / Must NOT do: Tests de contención con semántica correcta (post-G0): (a) aserción Landlock = `std::fs::write("/etc/passwd", b"x")` en child sandboxed → `PermissionDenied` **+ control negativo** (misma op sin sandbox → Ok, prueba que el deny viene del sandbox y no de DAC); (b) `/etc/shadow` read-deny es DAC (0640 root:shadow), NO capacidad del sandbox — documentarlo como tal, no como test de contención; (c) rlimit: `allocate 2GB` en child → asertar `ToolResult.success == false` / exit anormal (ENOMEM→abort/SIGABRT típico; NO asertar señal específica SIGKILL); (d) check `/proc/self/status` NoNewPrivs/Landlock ABI. Miri: setup previo `rustup toolchain install nightly --component miri` (no instalado hoy, verificado); scope Miri SOLO a unidades unsafe pure-Rust — Miri NO puede ejecutar FFI llama.cpp (`cargo miri test -p kitty-runtime -- llama` moriría en foreign calls); gate Send/Sync de `llama.rs:220 Arc<Mutex<Option<Inner>>>` se cumple con invariant SAFETY documentado + test compile-time assertion (`fn assert_send_sync<T: Send + Sync>()`) corrido FUERA de Miri. Guardar `task-6.log`. Must NOT test `kill -9 $$`, Must NOT dejar sandbox como env hints.
  Parallelization: Wave 2 | Blocked by: 5 | Blocks: 12 | Can parallelize with: 7,8,9
  References: audit.md:65-77 required behavioral test, crates/kitty-harness/src/sandbox.rs:193-234 old kill self, crates/kitty-runtime/src/llama.rs:220 unsafe, apohara-audit lib.rs:170 fchmod 0600 pattern for ledger
  Acceptance criteria: `cargo test -p kitty-harness --lib sandbox` incluye `deny_etc_passwd_write` pass CON control negativo verde + rlimit test pass por success==false + `/proc/self/status` evidencia en task-6.log + `rustup run nightly cargo miri test <scoped pure-rust>` 0 errores en unidades scoped + assertion Send/Sync compila fuera de Miri + `cargo clippy -D warnings` 0
  QA scenarios: happy: write /etc/passwd sandboxed → PermissionDenied mientras sin-sandbox → Ok; failure: si write pasa CON sandbox activo debe fail test (no stub)
  Commit: Y | test(harness): Landlock behavioral + Miri scoped

- [ ] 7. Wave3: P0 GBNF empírico --with-model o simulated honest
  What to do / Must NOT do: Intentar download del archivo GGUF concreto y PINNADO (el repo HF hostea muchos archivos incl. splits — NO bajar repo bare) p.ej `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (~5GB, ~30min) a `~/.kitty/models/qwen2.5-coder-7b-q4_k_m.gguf` — **nombre verdadero del modelo, NUNCA renombrar a ornith.gguf** (mislabeling de identidad = clase exacta de fraude que el audit flagueó). Si existe, correr `benches/ab_gbnf.py --with-model ~/.kitty/models/qwen2.5-coder-7b-q4_k_m.gguf --limit 50 --profile tiny`; mecanismo declarado: ab_gbnf shellea a un probe binario Rust pequeño (kitty-runtime, feature llama) que usa `LlamaSampler::grammar_lazy` decode real y publica `with_grammar_valid_pct` empírico (puede ser 71% honesto) con `model: qwen2.5-coder-7b-q4_k_m` en el JSON resultado; si no existe/falla red, marcar `method: simulated` y cerrar ADR `docs/ADR-GBNF-GO.md` `not empirically closed` sin fabricar 98%. Mantener `llama-cpp-2 0.1.154` feature `llama`. NO solapar con bench noche (misma GPU). Must NOT `forces 98 if <95`, Must NOT `method: simulated` junto a PASSED.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 12 | Can parallelize with: 8,9
  References: crates/kitty-runtime/src/llama.rs:88-120 build_grammar_lazy_sampler, docs/ADR-GBNF-GO.md GO 60%→98% simulated, benches/ab_gbnf.py 13K, audit.md:117-125 P0 measurement, strategy.md:104 ADR
  Acceptance criteria: `benches/ab_gbnf_result.json` contiene `method: "empirical"` + `model: qwen2.5-coder-7b-q4_k_m` + `with_grammar_valid_pct` real 60-98 o `method: "simulated"` + `note: not empirically closed` + `docs/ADR-GBNF-GO.md` actualizado y `cargo test --features llama` ≥31 pass
  QA scenarios: happy: con GGUF `ab_gbnf --with-model` mide sampler real >95 o honesto 71%; failure: sin GGUF debe ser `simulated` no `98% PASSED`
  Commit: Y | bench: GBNF empirical or honest simulated

- [ ] 8. Wave3: QW TUI TestBackend snapshots
  What to do / Must NOT do: Añadir `ratatui::backend::TestBackend` snapshots para `tui.rs` `handle_event` CJK `empty-q` (q solo si input vacío) + history draft restore + viewport `inner_h=height-2` slice, 3 nuevos tests `#[cfg(feature="tui")]`, `cargo test --features tui -p kitty-cli` 17/17, `cargo clippy --features tui`. Must NOT Playwright TTY overkill.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 12 | Can parallelize with: 7,9
  References: crates/kitty-cli/src/tui.rs:511 streaming 745L, audit.md:106-110 W3 TUI real not product, apohara Catalyst no usa ratatui snapshot
  Acceptance criteria: `cargo test -p kitty-cli --features tui` 17/17 (14+3) + `grep TestBackend crates/kitty-cli/src/tui.rs` 1 + `cargo clippy --features tui -D warnings` 0
  QA scenarios: happy: TestBackend buffer `Cell` contains CJK; failure: `q` con input no vacío no debe quit
  Commit: Y | test(tui): TestBackend snapshots

- [ ] 9. Wave3: QW embedding e2e real episodes.db
  What to do / Must NOT do: Test `embedding_blob_migration_and_recall` con **episodes.db en tempdir con schema de producción** (WAL+busy_timeout 5s patrón `apohara-episodic/src/store.rs:68`) — NUNCA la DB live `~/.kitty/episodes.db` — + `blake3` deltas incremental + CI `kitty bench --semantic-memory --limit 15` permitido perder vs lexical (hybrid puede ser peor, no 83.3% press-release). Mantener `--semantic-memory` default off. Must NOT bloquear merge si hybrid pierde.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 12 | Can parallelize with: 7,8
  References: crates/kitty-memory/src/episodic.rs:112 bm25 + embedding.rs 201L fastembed 384, apohara-indexer/storage.rs:61 OnceLock vec0 + apohara-episodic/store.rs:68 WAL, audit.md:82-92 W1 Memory 12 synthetic vs real
  Acceptance criteria: `cargo test -p kitty-memory --features embedding embedding_blob_migration_and_recall` pass con DB real + `benches/ab_recall.py --limit 15` run no exige win, solo report `hybrid 12/15 vs lexical 10/15` honesto
  QA scenarios: happy: recall_hybrid con DB real top-1 contiene episodio paraphrased; failure: 12 synthetic 83.3% no se publica como product metric
  Commit: Y | test(memory): embedding e2e real

- [ ] 10. Wave4: Must Have doc apohara robables
  What to do / Must NOT do: Escribir `docs/ROADMAP-CATALYST-ROBABLE.md` con Must Have list priorizada: 1) `sqlite-vec OnceLock auto_extension` unificado (Codesearch storage.rs:61), 2) `blake3 384D FeatureHashEmbedder` zero deps fallback (indexer embeddings.rs:21), 3) `tree-sitter 9 langs` chunker 60/15, 4) `Embedder id/dim` versionado, 5) `TurboVec codec_v8` comprimido (Dekanus rag), 6) `llmlingua2` 160w prompt compression, 7) `deny.toml` license allowlist + `cargo-audit` SLSA, 8) `apohara-audit` JSONL ledger O_NOFOLLOW fchmod 0600, 9) `task_graph.rs` DAG reaper. No priorizar wasmtime.
  Parallelization: Wave 4 | Blocked by: 7,8,9 | Blocks: 12 | Can parallelize with: 11
  References: /home/thelinconx/Proyectos/Repos/Apohara-Catalyst/crates/apohara-secrets/lib.rs + sandbox/profile/linux.rs + audit lib.rs + indexer/storage.rs
  Acceptance criteria: `docs/ROADMAP-CATALYST-ROBABLE.md` existe 80+ líneas con 9 patrones + file:line citas + `cargo deny check` 0
  QA scenarios: happy: `grep -c "apohara" docs/ROADMAP-CATALYST-ROBABLE.md` >=5; failure: si lista fue 0 debe CHANGES_REQUESTED
  Commit: Y | docs: Must Have Catalyst robable

- [ ] 11. Wave4: Own inference worth vs Ollama + vision pivot doc
  What to do / Must NOT do: Escribir `docs/VISION-ROADMAP.md` con evaluación own inference worth: Ollama 0 mantenimiento (~13s/tarea ornith) vs Dekanus Phase 0 skeleton candle 0.11 cudarc 0.19 glommio vendor kernels 5GB compile sin tok/s real → keep ChatClient trait abstración, own engine solo si Ornith-397B MoE sparsity + FWHT 3-bit + SelectivePolicy + layer_streaming pinned_buffer lo exigen; visión director: Kitty ocupa celda Rust single-binary 9B/8GB vacía (little-coder Node pi, smallcode Node tmpdir), no pivot, competir cerrando P0s no con features, `SMOKE.md` honesto + CNAS NA hasta harbor adapter.
  Parallelization: Wave 4 | Blocked by: 7,8,9 | Blocks: 12 | Can parallelize with: 10
  References: /home/thelinconx/Proyectos/Repos/Apohara-Catalyst docs/PROJECT.md BYOC, Proyectos/Repos/little-coder/package.json pi 0.83, Proyectos/Repos/smallcode/package.json bench harness, audit.md:182-191 vision edits
  Acceptance criteria: `docs/VISION-ROADMAP.md` existe con worth tabla Ollama vs Dekanus + visión no pivot + competencia smallcode/littlercoder + `cargo test` 308 pass
  QA scenarios: happy: `grep -q "ChatClient" docs/VISION-ROADMAP.md` 1 + `grep -q "keep Ollama"` 1; failure: si propone own engine para v0.1 debe CHANGES_REQUESTED
  Commit: Y | docs: vision own inference worth

- [ ] 12. Wave4: Docs hygiene final + cargo verde
  What to do / Must NOT do: Actualizar `codemap.md` + `docs/STRATEGY.md` F6.4 GO empírico vs simulated + `CHANGELOG.md` + `cargo fmt --all` + `cargo clippy --all-targets --all-features -- -D warnings` + `cargo test --workspace --all-features` (≥308, comparación numérica — el total crece con los tests nuevos de todos 4/6/8/9) + `cargo build --release` + `cargo-deny check` (invocar como binario externo; el alias `deny` en ~/.cargo/config.toml lo shadowea y cargo emite warning) + `git diff -- bench/official` vacío. No dejar `bench_*.json` con n != len. Must NOT push, Must NOT bench/official edit.
  Parallelization: Wave 4 | Blocked by: 6,7,8,9,10,11,2 | Blocks: F1-F4
  References: docs/STRATEGY.md:104 ADR, codemap.md, audit.md:180 vision
  Acceptance criteria: `cargo clippy -D warnings` 0 + `cargo test --workspace --all-features` ≥308 pass (numérico) + `cargo fmt --check` 0 (no `cargo-fmt` alias) + `codemap.md` menciona Catalyst seccomp+SecretString + `git status` no `bench/official` diff + `docs/ROADMAP-CATALYST-ROBABLE.md` 80+ líneas (unificar nombre)
  QA scenarios: happy: `cargo test --workspace --all-features 2>&1 | tail -1` contiene `test result: ok` con passed ≥308; failure: clippy warn debe bloquear
  Commit: Y | chore: hygiene codemap+clippy

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  What to check: 12 todos con refs path:line reales, acceptance ejecutable, QA happy+failure evidencia, matriz coherente, scope P0s + bench noche real n==len, GBNF empírico/simulated honesto, Catalyst patterns, no bench/official edit, no wasmtime gate
  Evidence: .omo/evidence/f6-p0-quickwins-nightbench/F1.log
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F2. Code quality review
  What to check: clippy 0, fmt 0, sandbox 253L con seccompiler+landlock+RLIMIT (no println stub), zeroize SecretString, tui 745L, 308 tests, workspace resolver, Conventional Commits
  Evidence: F2.log
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F3. Real manual QA
  What to check: docker hello, harbor k1, kitty bench limit 3, tui timeout 5 q, --semantic-memory warn, --cloud-fallback no key, write /etc/passwd denied en child sandboxed, ab_gbnf --with-model si GGUF, bench_225_full.json jq length == n
  Evidence: F3.log
  Verdict: APPROVE | CHANGES_REQUESTED

- [ ] F4. Scope fidelity
  What to check: 12 todos entregados, no push/OpenSSF, no grammar+tools 400, no 225 extrapolado con nombre 225, no 98% simulated como PASSED, no grow quiz, TUI no rompe CLI, bench noche real o SMOKE honest
  Evidence: F4.log
  Verdict: APPROVE | CHANGES_REQUESTED

## Commit strategy
- 1 commit por todo `feat(harness)/feat(runtime)/feat(memory)/feat(tui)/feat(bench)/docs:` + DCO `Signed-off-by` defer push. Wave0 2 commits bench, Wave1 2 commits P0, Wave2 2 commits sandbox, Wave3 3 commits GBNF+TUI+embed, Wave4 3 commits vision+hygiene. Total 12 commits.

## Success criteria
- La cola nocturna AMEND-G1 aterriza: `bench_tb2_oracle_full89.json` + `bench_polyglot_oracle_full225.json` (oracle-full n==len 89/225, requisito gate #3) + `bench_polyglot_kitty_ornith9b_l50.json` (n==len 50) + `bench_polyglot_kitty_ornith15b_l15.json` (n==len 15) + opcional TB2-kitty honesto; todos con `n_errored_trials <= 10%` de N (AMEND-G2); fallback `_partialN`/smoke con nota `not empirically closed` si abort — nunca parcial renombrado como completo
- `SecretString` zeroize + Debug redact + `is_secret_env_name` strip, `cost_usd` sin Authorization, `cargo test CloudClient_debug_redacts` pass
- `sandbox.rs` Landlock+seccomp+namespaces real `write /etc/passwd → PermissionDenied` con control negativo + Miri scoped pure-Rust + Send/Sync assertion fuera de Miri, no `println!` stub
- `ab_gbnf_result.json` `method: empirical` con GGUF de nombre verdadero (qwen2.5-coder-7b-q4_k_m) o `simulated` honesto `not empirically closed` (no 98% forzado)
- TUI 3 TestBackend snapshots pass, embedding e2e tempdir DB schema producción + CI ab_recall permitido perder
- `docs/ROADMAP-CATALYST-ROBABLE.md` + `docs/VISION-ROADMAP.md` con own inference keep Ollama + vision no pivot
- `cargo test --workspace --all-features` ≥308 numérico + `clippy -D warnings` 0 + `fmt --check` 0 + `cargo-deny check` 0 + `bench/official` diff vacío
- F1-F4 all APPROVE
