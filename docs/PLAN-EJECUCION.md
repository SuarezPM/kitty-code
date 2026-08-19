# Plan de Ejecución — Kitty Code: Hardening + Roadmap

**Fecha:** 2026-08-19
**Autor:** Análisis técnico de sesión (hallazgos verificados: 104 tests OK, clippy limpio)
**Alcance:** Corregir bugs críticos y altos, robustez operativa, calidad de código, unificar
benchmarks, higiene del repo (rename + git + OpenSSF) y avanzar el roadmap declarado.

---

## 0. Resumen ejecutivo

| # | Fase | Prioridad | Esfuerzo | Depende de |
|---|------|-----------|----------|------------|
| F0 | Higiene y fundación (rename dir + git + LICENSE + .gitignore) | **P0** | ~1-2 h | — |
| F1 | Seguridad: bypasses de la guarda de Bash + timeout + tests | **P1** | ~4-6 h | F0 |
| F2 | Bugs del core: hashline, errores de tool, budget, tool-loop, retry | **P2** | ~1-2 días | F1 |
| F3 | Robustez operativa: Read truncado, Grep/Glob con .gitignore | **P3** | ~4-6 h | F2 |
| F4 | Calidad y deuda: dedup, código muerto, mock + tests de agent.rs, flags CLI, defaults | **P4** | ~1-2 días | F3 |
| F5 | Benchmarks unificados sobre el harness (225/89) + re-run honesto | **P5** | ~2-3 días | F4 |
| F6 | Roadmap: quick wins → benchmarks oficiales → OpenSSF → llama.cpp → largo plazo | **P6** | semanas | F5 |

**Principio de orden:** primero *versionar* (F0) → luego *no romper nada más* (F1 seguridad) →
luego *que el loop sea correcto* (F2) → *robusto* (F3) → *probable* (F4) → *medible de verdad*
(F5) → *roadmap* (F6). Los benchmarks antes de F2 medirían un harness roto; por eso van después.

**Definition of Done global:** cada fase termina con `cargo test --workspace` verde, clippy
limpio (`-D warnings`), commit en git y actualización de `codemap.md`/`AUDIT.md` cuando aplique.

---

## F0 — Higiene y fundación (P0)

> La renombra del directorio rompe el cwd de cualquier sesión en curso. **Hacerla la primera
> operación de una sesión nueva**, antes de tocar código, y verificar después.

### F0.1 Renombrar el directorio `Qwen3.8-27B` → `Kitty-Code`
- [x] `mv /home/thelinconx/Proyectos/Qwen3.8-27B /home/thelinconx/Proyectos/Kitty-Code` (programado al
      final de la ejecución para no romper el cwd de la sesión; las referencias ya apuntan al path nuevo)
- [ ] Actualizar referencias absolutas al path viejo:
  - `Modelfile` (`FROM /home/thelinconx/Proyectos/...`), `AGENTS.md`, `codemap.md`,
    `docs/TECHNICAL_AUDIT_REPORT.md` (l.7), `docs/PLAN-SUCESOR-SMALLCODE.md`, `.slim/deepwork/*`.
  - `grep -rn "Qwen3.8-27B" --include="*.md" --include="Modelfile" --include="*.toml" .`
- [ ] **Riesgo:** cualquier shell abierta con cwd viejo queda huérfana. Verificar con `pwd` y
  `cargo build` tras el rename.

### F0.2 `git init` + rama `main` + baseline
- [ ] `git init -b main` y **primer commit de baseline ANTES de ningún cambio funcional**:
      `git add -A && git commit -m "baseline: kitty-code v0.1.0 (104 tests)"`.
- [ ] `git config user.name/user.email` locales.
- [ ] Subir a GitHub (crear repo `SuarezPM/kitty-code`, `git remote add origin` + push) —
      **prerrequisito** para toda la parte CI de F6.3.

### F0.3 `.gitignore` completo (el GGUF de 17.9 GB NO debe entrar)
- [ ] Añadir a `.gitignore` (hoy solo cubre `target/`, `.slim/`, swp):
  ```gitignore
  *.gguf
  *.bin
  *.safetensors
  __pycache__/
  *.pyc
  *.db
  *.db-*
  bench_*.json
  week0_*.json
  *.log
  .opencode/node_modules/
  ```
- [ ] Verificar antes del commit: `git status --porcelain | grep -i gguf` → vacío.
- [x] **Decisión (usuario):** borrar el GGUF `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.9 GB) — hecho.
      Modelo canónico `ornith:9b`. `Modelfile` actualizado.

### F0.4 `LICENSE` (MIT, como declara Cargo.toml)
- [ ] Crear `LICENSE` MIT con holder (preguntar al usuario: nombre personal u organización).
- [ ] Añadir `license = "MIT"` + `license-file` en el workspace si aplica.

### F0.5 Baseline verificado
- [ ] `cargo test --workspace` → 104 OK · `cargo clippy --all-targets -- -D warnings` → limpio.
- [ ] Commit `F0 done`.

---

## F1 — Seguridad: cerrar bypasses de la guarda de Bash (P1)

**Archivo:** `crates/kitty-harness/src/guard.rs` (+ `tools.rs` para el timeout).

### F1.1 Regla de oro: ningún comando se permite "a ciegas"
El defecto raíz es `check_command` (l.144-146): `SAFE_OUTPUT_COMMANDS` → `Allow` sin mirar el
resto. Los argumentos de `echo`/`cat`/`grep`/`printf`/`tee`/etc. **sí se ejecutan** vía
sustitución de comandos. Fix:

1. **Pre-scan de sustitución de comandos en el comando RAW** (antes de cualquier otra lógica):
   - Si contiene `$(` , `` ` `` (backtick) o process substitution `<( ` / `>(` →
     **escaneo tipo intérprete sobre el RAW** (no `strip_quotes`), porque ahí hay ejecución.
   - No confundir con `$VAR`/`$3`/`${x}` sin `(`: esos son expansión de variable, no ejecución
     (pero `$((...))` aritmética y `${x:=$(cmd)}` sí ejecutan → el detector busca `$(` y `` ` ``).
2. **Añadir a `INTERPRETERS`:** `eval`, `source`, `.` (dot), `env`, `xargs` con banderas
   (o tratarlos con escaneo raw siempre). Para `eval` en concreto, evaluar **hard-block**
   (raro que un agente de código necesite `eval`); si se decide permitir, escaneo raw.
3. **Refinar el escaneo de intérpretes** para eliminar el falso positivo conocido
   (quick win del AUDIT): cuando un script/argumento contiene `rm -rf /` como **texto literal**
   (comentario, string) dentro de `sh -c`, el patrón no debe matchear dentro de string literals
   de `-c`/`-e` cuando el comando es un script embebido que solo lo imprime. Regla práctica:
   para intérpretes, bloquear patrones peligrosos salvo que estén dentro de comillas simples/dobles
   que el shell no interpretaría en posición de ejecución — validar con el caso `sh-backup`.
4. **Defensa en profundidad (post-F5):** el plan original pedía sandbox seccomp/Landlock
   (crate `apohara-agentguard`); `vendor/` está vacío. Dejar como item de backlog F6.5 con
   criterio de aceptación: "ningún bypass conocido sobrevive a la suite de guard".

### F1.2 Tests de regresión (cada caso = 1 test en `guard.rs`)
| Caso | Esperado |
|------|----------|
| `echo $(rm -rf /)` | Block |
| `printf "$(rm -rf /)"` | Block |
| `` echo `rm -rf /` `` | Block |
| `cat <(rm -rf /)` | Block |
| `eval "rm -rf /"` | Block |
| `sh -c 'echo $(rm -rf /)'` | Block |
| `echo "cost: $5"` | Allow (variable, no sustitución) |
| `echo 'rm -rf / is dangerous'` | Allow (texto literal, sin `$(`) |
| `tar -czf backup.tar.gz dir/` | Allow (caso sh-backup) |
| `find . -name '*.tmp' -exec rm {} \;` | Allow o Warn documentado (decisión explícita) |

### F1.3 Timeout de Bash (evita `sleep 10000` colgando al agente)
- [ ] En `tools.rs::execute_bash`, reemplazar `std::process::Command::output()` por
      `tokio::process::Command` con `kill_on_drop(true)` envuelto en
      `tokio::time::timeout(config.bash_timeout_secs, ...)`.
- [ ] Nuevo campo `bash_timeout_secs` en `AgentConfig` (default 60s) + flag CLI `--bash-timeout`.
- [ ] Timeout → `ToolResult { success: false, output: "command timed out after Ns", retry: false }`
      (nunca error fatal; ver F2.2).

**Verificación F1:** `cargo test -p kitty-harness guard` (todos los casos nuevos + los 13
existentes) · clippy limpio · commit `F1 done`.

---

## F2 — Bugs del core del loop (P2)

### F2.1 🔴 Hashline: el path primario de Edit está roto end-to-end
**Problema:** `Read` muestra 8 chars del hash (`tools.rs` l.111); `Edit` compara contra el hash
completo de 64 chars (`hashline.rs` l.139) → nunca coincide → cae al fallback, que además busca
por **contenido nuevo** (`tools.rs` l.196) → en un reemplazo típico falla con `retry`.

**Diseño recomendado (protocolo `{path, line, hash, new}`):** el modelo ya ve `LINE:HASH|content`
en Read; darle la línea como target elimina la ambigüedad de líneas duplicadas (líneas idénticas
tienen hashes idénticos — el diseño actual tampoco las maneja):
- [ ] `Read` mantiene el prefijo corto (8 chars) pero la salida ya incluye `line_num`.
- [ ] `Edit` acepta `line` (número) + `hash` (prefijo 8 chars, validación de longitud ≥ 6):
      1. Re-lee el archivo (verificación de concurrencia contra el snapshot: si el archivo
         cambió → `FileChanged` y mensaje "re-read and retry").
      2. Verifica que `lines[line-1].hash` empiece por el prefijo; si no → error auto-saneable
         que devuelve las etiquetas actuales (`LINE:HASH|content`) para que el modelo reintente.
      3. Reemplaza la línea y re-hashlea el snapshot.
- [ ] Eliminar `fallback_search_replace` (heurística por contenido) o reducirlo a "búsqueda por
      prefijo en todo el archivo si `line` está fuera de rango".
- [ ] Tests: round-trip `Read → Edit con prefijo 8 chars → aplicado`; línea duplicada
      desambiguada por `line`; archivo cambiado → rechazo; prefijo ambiguo → error claro.
- [ ] Test de integración end-to-end del registro (`tools.rs`) que use el formato real de Read.

> Alternativa descartada: exponer el hash completo (64 chars) infla el contexto y no resuelve
> las líneas duplicadas.

### F2.2 🔴 Los errores de tool abortan la sesión
- [ ] En `agent.rs` l.307 y l.368, reemplazar `self.registry.execute(tc)?` por manejo explícito:
  ```rust
  match self.registry.execute(tc) {
      Ok(result) => { /* push tool result (existe) */ }
      Err(e) => {
          let msg = format!("Tool '{name}' failed: {e}");
          self.push_tool_message(&msg).await; // success: false
      }
  }
  ```
- [ ] Los mensajes `tool` con error van **inmediatamente después** del `assistant` con
      `tool_calls` (orden válido para Ollama). Ver F2.5.
- [ ] Nunca un `ToolError` propaga fuera de `run()` salvo fallos de I/O inesperados.
- [ ] Test de integración: modelo que llama `Edit` sin leer → el loop continúa y el modelo
      (stub) se corrige.

### F2.3 🟠 El presupuesto de contexto cuenta dos veces
**Problema:** `push_message` (l.703) consume heurístico (chars/4) y además el prompt (l.214),
outputs de tool (l.320, l.375) y texto final (l.389) se consumen con conteo real → ~2x.
A la vez, preflight (system) y el summary de compactación no se consumen.

**Fix (un único punto de contabilidad):**
- [ ] `push_message` se vuelve `async` y es el **único** consumidor: cuenta 1 vez por mensaje
      con `count_tokens` real (fallback heurístico si el server falla). Eliminar los `consume`
      sueltos de l.214/320/375/389.
- [ ] El mensaje system de preflight (l.538) y el `[Context Summary]` (l.693) también pasan por
      `push_message` (o un `consume` explícito).
- [ ] `trigger_compaction`: `freed`/`needed` calculados con el mismo método que el consumo
      (real vía `count_tokens` del texto viejo/sumario, fallback heurístico).
- [ ] Test unitario del contador: secuencia de pushes con un stub de conteo → total exacto.

### F2.4 🟠 Tool-loop solo se detecta dentro de una misma respuesta
- [ ] `QualityMonitor`: la ventana se mantiene **entre turnos** (quitar el `clear()` de
      `next_turn` o mover la gestión a `check_response`).
- [ ] Identificar el bucle por **nombre + argumentos** (hash del JSON de args), no solo nombre.
- [ ] Reset de la ventana cuando: hay respuesta de texto puro, o aparece una tool distinta.
- [ ] Corregir el mensaje correctivo para que refleje args reales.
- [ ] Test: 3 llamadas iguales en 3 respuestas seguidas (una por turno) → ToolLoop detectado;
      misma tool con args distintos → no loop.

### F2.5 🟠 Orden de mensajes inválido ante tool alucinada
- [ ] Hoy: `assistant(tool_calls)` → `user(correctivo)` → `tool(resultado)` (agent.rs l.280-283).
- [ ] Fix: ejecutar todas las tools primero, pushear todos los `tool` results, y **después**
      el mensaje `user` correctivo (o inyectar el correctivo como contenido del propio
      resultado de tool). El orden de Ollama exige: assistant(tool_calls) → tool* → siguiente.

### F2.6 🟠 `retry_without_thinking` descarta tool_calls
- [ ] En vez de devolver el texto crudo (l.549-582), marcar `thinking_disabled = true` y
      **continuar el loop** con esa respuesta (procesando tool_calls y quality checks), con un
      flag para evitar recursión infinita (máx. 1 reintento sin thinking por turno).

**Verificación F2:** tests nuevos por bug + integración con mock (ver F4.2) · `cargo test` ·
clippy · commit `F2 done`.

---

## F3 — Robustez operativa (P3)

| ID | Item | Fix propuesto | Archivos |
|----|------|---------------|----------|
| F3.1 | `Read` sin límite vuelca archivos gigantes al contexto | Cap de líneas configurable (default 300) con `... (N líneas omitidas — usa Grep)`; archivos grandes → sugerir Grep. Tree-sitter signatures quedan en backlog F6. | `tools.rs` |
| F3.2 | `Grep`/`Glob` recorren `target/`, `node_modules`, `.git` | Sustituir `walkdir` por el crate estándar **`ignore`** (`WalkBuilder`), que respeta `.gitignore` y excluye `.git`/`target` por defecto; añadir `ignore = "0.4"` (workspace) y quitar `walkdir` si queda sin uso. | `tools.rs`, `Cargo.toml` |
| F3.3 | Bash sin timeout | Hecho en F1.3. | — |
| F3.4 | Falso positivo guarda `sh-backup` | Cubierto en F1.1.3 con test de regresión `tar -czf ...`. | `guard.rs` |

**Verificación F3:** tests de Grep con tree `target/` y `.gitignore`; Read con archivo > cap;
`cargo test` · clippy · commit `F3 done`.

---

## F4 — Calidad de código y deuda técnica (P4)

### F4.1 Deduplicación de constructores
- [ ] `Agent::new` y `Agent::with_tools` (l.113-185) → un `fn new_inner(client, config, tools)`
      privado; ambos públicos delegan. (~70 LOC menos.)

### F4.2 Probabilidad: mock del cliente + tests de `agent.rs` (hoy 0 tests)
- [ ] Definir en `kitty-runtime` un trait `ChatClient` (`async_trait` o trait nativo):
      `chat`, `count_tokens`, `health_check`. `OllamaClient` lo implementa.
- [ ] `Agent` recibe `Box<dyn ChatClient>` (o `Arc`) — se mantiene el runtime actual, se gana
      testabilidad.
- [ ] `MockClient` en `#[cfg(test)]`/`dev-dependencies` con respuestas guionizadas
      (secuencia: tool_call → error de tool → texto final).
- [ ] Tests de integración del loop:
  - tarea simple que termina en texto;
  - tool error → el loop continúa (F2.2);
  - compactación se dispara al 80% y el halt guard corta;
  - tool-loop entre turnos (F2.4);
  - mensaje `user` correctivo aparece tras los `tool` results (F2.5);
  - episode se guarda como fail cuando se agotan `max_iterations`.

### F4.3 Código muerto
- [ ] `AgentEvent` (agent.rs l.64): **emitirlo** vía `tracing`/callback o eliminarlo.
- [ ] `CompactionResult` (context.rs l.20), `ParseError` (parser.rs l.15),
      `QualityError` (quality_monitor.rs l.16): eliminar.
- [ ] Variantes de `RuntimeError` sin uso (`ModelNotFound`, `ConnectionRefused`,
      `UnexpectedResponse`, `ToolExecution`): eliminar o usar.
- [ ] `crates/codemap.md` es un placeholder sin rellenar: completarlo en esta fase.

### F4.4 CLI expone los knobs reales
- [ ] Flags: `--knowledge` (enable), `--knowledge-dir`, `--context-window`,
      `--temperature`, `--top-p`, `--max-iterations`, `--bash-timeout` (F1.3), `--no-memory` (existe).
- [ ] Hoy `enable_knowledge=false` por defecto y sin flag → la feature de knowledge injection
      **nunca se activa en `kitty run`**; con el flag queda usable y testeable.

### F4.5 Defaults consistentes de modelo
- [ ] Un solo modelo canónico. Recomendación: `ornith:9b` (el de los docs/AUDIT).
      `AgentConfig::default()` (hoy `qwen3:8b`), CLI (`ornith:9b`) y `Modelfile`
      (`Qwen3.8-27B-UD-Q4_K_XL`) deben coincidir o documentar la matriz.
- [ ] Decidir el destino del GGUF de 27B (F0.3) y alinear `Modelfile`/README.

**Verificación F4:** `cargo test` (incl. nuevos tests de agent con mock) · clippy `-D warnings` ·
`cargo build --release` · commit `F4 done`.

---

## F5 — Benchmarks unificados sobre el harness (P5)

> **Problema de honestidad:** el 96.44% (225) y 12.36% (89) salen de scripts Python que hablan
> directo con Ollama (solo Write+Bash), **sin pasar por el harness Rust** — mientras la narrativa
> es "el harness es el producto". El binario solo embebe 15+8 ejercicios.

- [ ] **F5.1** Exportar los 225 ejercicios de `benches/run_polyglot_225.py` y los 89 de
      `run_tb2_89.py` a datos JSON (`benches/exercises_polyglot.json`, `exercises_tb2.json`):
      `{id, lang, prompt, check_strings[]}`.
- [ ] **F5.2** El binario carga el JSON y corre cada ejercicio con el **harness completo**
      (guardas + hashline + quality + budget + memory opcional), mismo tempdir/timeout/grading
      (string-match, documentado).
- [ ] **F5.3** Re-run honesto (2 runs 225 + 1 run 89) → números NUEVOS del harness en
      `bench_*_harness.json`.
- [ ] **F5.4** Actualizar `AUDIT.md`/`CNAS.md`/`codemap.md`: números del harness, fecha,
      nota explícita de que ahora sí mide el harness.
- [ ] **F5.5 (opcional, hacia Polyglot real)** modo de grading por **unit tests** en tempdir
      (compilar/ejecutar con toolchains del lenguaje) como primer paso del benchmark oficial.

**Verificación F5:** `kitty bench polyglot --limit 15 --json` usa el harness · 225/89 corren
desde JSON · resultados reproducibles · commit `F5 done`.

---

## F6 — Roadmap del repo

### F6.1 Quick wins pendientes (del AUDIT §9)
- [ ] **Thinking budget A/B:** modo `kitty bench ab --thinking 1024,2048,4096 --limit 50`
      (n=50 por condición) → elegir el mejor y fijarlo como default.
- [ ] **Más ejercicios:** list comprehensions, decorators con args, async/await (Python);
      ampliar Rust/Go apuntando a los 5 fallos consistentes (impl traits, generics).
- [ ] **Knowledge cards por lenguaje** (Rust traits y Go generics ya existen en `knowledge/`):
      añadir cards y activar `--knowledge` por defecto en bench tras A/B.

### F6.2 Benchmarks oficiales (la prioridad #1 del propio handoff)
- [ ] **Aider Polyglot real** — Exercism 225, grading por unit tests (Docker), 2 intentos.
      Budget declarado: 15-24 h. Gate: > 45.56%. Requiere el harness estable (F2-F5).
- [ ] **Terminal-Bench 2.0 real** — Harbor containers, 89 tasks. Budget: 3-8 h. Gate: > 9.2%.
- [ ] Publicar resultados en `AUDIT.md` con la columna Consumer vs Vendor (CNAS).

### F6.3 OpenSSF Scorecard ≥ 7/10 y Best Practices Badge Silver
- [ ] `SECURITY.md`, `CONTRIBUTING.md`, `CODEOWNERS`, issue templates.
- [ ] `rust-toolchain.toml` (pin de toolchain estable + política MSRV) y `deny.toml`
      (advisories, licenses, bans) + `cargo-deny` en CI.
- [ ] CI GitHub Actions (`ci.yml`): `cargo fmt --check`, `clippy -D warnings`, `cargo test`,
      `cargo deny check`, `cargo audit`; pins de actions **por SHA**; permisos mínimos
      (`permissions: contents: read`).
- [ ] CodeQL + dependency-review + gitleaks (secret scanning).
- [ ] `scorecard.yml` (OpenSSF Scorecard action) y badge.
- [ ] SBOM en releases (`cargo-sbom` o `cyclonedx-bom`) + releases firmados/signed provenance.
- [ ] Renovate/Dependabot para dependencias; changelog (`CHANGELOG.md`) + semver.

### F6.4 Motor de inferencia
- [ ] Adapter **llama.cpp** con grammar enforcement (GBNF) para tool-calls tras F4.2
      (el trait `ChatClient` ya abstrae el runtime → solo hay que implementar el adapter).
      Opción A: bindgen; Opción B: candle. Objetivo: > 95% tool-call validity y sin daemon.
- [ ] A/B tool-call validity con/sin grammar (target: 88% → 98% según el plan).

### F6.5 Backlog largo plazo (con criterios de aceptación, no fechas)
| Item | Criterio de aceptación |
|------|------------------------|
| Sandbox seccomp/Landlock (backlog de F1.1.4) | Ningún bypass conocido sobrevive la suite de guard; OOM/segfault no escapan del sandbox |
| MemScenes (lifecycle EverMemOS) + BM25/embeddings + RRF | recall semántico supera al Jaccard en tareas recurrentes (A/B medible) |
| Cloud fallback opt-in | Escalada a modelo frontier solo en fallo, con costo documentado |
| Plugin system | Terceros pueden registrar tools sin tocar el core |
| MCP server | Otros agentes consumen Kitty Code como tool MCP |
| TUI interactivo | Prompt real, scroll, historial (hoy es display de estado) |

---

## 7. Orden de ejecución sugerido y verificación por fase

```bash
# F0: rename (sesión nueva) → git init → .gitignore → LICENSE → commit baseline
# F1: cargo test -p kitty-harness guard && cargo clippy --all-targets -- -D warnings
# F2: cargo test --workspace && cargo clippy --all-targets -- -D warnings
# F3: cargo test --workspace
# F4: cargo test --workspace && cargo build --release
# F5: kitty bench polyglot --limit 15 --json && python3 benches/run_polyglot_225.py (harness JSON)
# F6: por roadmap, cada item con su verificación
```

Cada fase = 1+ commits con mensaje `F{n} done`, tests verdes y docs actualizadas
(`codemap.md` honesto: registrar qué se corrigió, como ya hace el repo).

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|--------|-------|------------|
| Rename del dir rompe cwd de sesiones activas | Alta | Hacerlo primero en sesión nueva; verificar `pwd`/build |
| GGUF 17.9 GB entra al repo por descuido | Media | `.gitignore` ANTES del primer `git add`; verificación con grep |
| Prefijo de hash 8 chars colisiona (líneas duplicadas) | Media | Diseño F2.1 usa `line + hash` → sin ambigüedad |
| Over-blocking de la guarda rompe casos legítimos (sh-backup) | Media | Suite de regresión F1.2 con casos benignos explícitos |
| Benchmarks oficiales consumen mucho tiempo (15-24 h) | Alta | Hacerlos después de F5 (harness estable); gate explícito; publicar negativo si falla |
| CI requiere repo en GitHub | Alta | F0.2 incluye push; sin remote, CI queda bloqueado → avisar al usuario |
| `count_tokens` por mensaje añade latencia (1 RTT/mensaje) | Media | Fallback heurístico ya existe; medir en F5; opción `--no-real-tokenize` |

---

## 9. Decisiones que necesito del usuario (antes de ejecutar)

1. **Holder del LICENSE MIT** (nombre personal / org) y confirmación de **renombrar** el dir.
2. **Destino del GGUF de 27B** (mover fuera / borrar / dejar ignorado) y **modelo canónico**
   (`ornith:9b` recomendado).
3. **GitHub repo**: ¿crear `SuarezPM/kitty-code` y pushear? (necesario para CI/OpenSSF).
4. **F1.1**: ¿hard-block `eval` siempre (recomendado) o permitir con escaneo raw?
5. **Alcance de esta tanda**: ¿ejecutar F0→F2 en esta sesión y dejar F3+ para la siguiente,
   o todo en orden hasta donde alcance el tiempo?

---

*Plan basado en hallazgos verificados en el código (guard.rs, tools.rs, hashline.rs, agent.rs,
quality_monitor.rs, context.rs) y en el roadmap declarado por AUDIT.md, CNAS.md,
docs/TECHNICAL_AUDIT_REPORT.md y .slim/deepwork/handoff.md.*
