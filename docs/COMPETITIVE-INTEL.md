# Competitive Intelligence — little-coder & smallcode

**Fecha:** 2026-08-19 · **Fuentes:** recon de código (explorer), PRs/issues + EXA (librarian), research previa (lib-1/lib-2/ora-1). Ver `.slim/deepwork/competitive-intel.md` para el contexto aceptado.

---

## 1. little-coder (itayinbarr) — TS/Node sobre `pi`

**Qué es:** wrapper fino sobre el runtime `pi` (`@earendil-works/pi-coding-agent`). Todo el comportamiento "small-model" vive en ~30 extensiones bajo `.pi/extensions/`; el loop, TUI y compactación son la caja negra de pi. No es un fork: re-invoca pi con `--no-extensions` y parchea su fuente en runtime (`bin/little-coder.mjs:142-155`).

### Hace bien (robar)
| Patrón | Dónde | Por qué importa |
|--------|-------|-----------------|
| **Write-vs-Edit invariant + guard de redirects de shell** | `write-guard/index.ts:159-206` | Bloquea Write sobre archivos existentes (fuerza Edit), reescribe `/foo.md`→cwd, rechaza device names de Windows, y espeja la guarda en `cat > file`. Dispara en `tool_call` (atrapa el write nativo de pi). Es el invariante que impulsa el score Polyglot. |
| **Inyección tail-message (preserva KV cache)** | `_shared/inject.ts:14-24` (issue #73) | Inyecta guía por-turno al final de la conversación, no en system prompt. |
| **Thinking-budget cap + nudge a implementar** | `thinking-budget/index.ts:138-170` | Cuenta tokens de thinking en `message_update`; al exceder fuerza level "off" + `ctx.abort()`. Evita crash de stale-ctx. |
| **Skill/knowledge cards por turno** | `skill-inject` + `knowledge-inject` | Selección 3-prioridades (error-recovery > recency > intent), budget-capped, cacheado, dedup. Scoring word=1.0/bigram=2.0. |
| **Parser de output robusto** | `output-parser/parser.ts` | `repairJson`, `firstBalancedObject`, recupera fenced/`<tool_call>`/bare/LFM2 "Pythonic" (`parseLiquidToolCalls:324`). |
| **Quality monitor** | `quality-monitor/quality.ts:26-76` | Empty/hallucinated/loop/malformed con awareness de cambio de estado (issue #81). |
| **Permission-gate segment-wise** | `permission-gate/index.ts:67-72` | Juzga cada segmento de la cadena, rechaza cualquier write-redirect, y **nombra el remedio** (no solo "refused"). |
| **Background shells event-driven** | `bg-shell/index.ts` | `ShellStart` con wake por `match`/`silence`/`every_n_matches`, kill por process-group + watchdog SIGTERM→SIGKILL, reaping en shutdown. |
| **Sub-agentes aislados read-only** | `subagent/spawn.ts` | Tools gateadas por env (`SUBCODER_ALLOWED_TOOLS:50`), watchdog retry, concurrency cap. |
| **Context-watchdog mid-run** | `context-watchdog/index.ts` | Compactación al 80% con loop-guard (issue #68). |
| **Per-phase model selection** | `phase-model/phases.ts` | Plan con modelo grande, actúa con modelo pequeño. |
| **Checkpoint, read-guard, evidence, benchmark-profiles, llama-cpp-provider** | varios | Read-guard trima reads gigantes; profiles por modelo (temp/thinking/turn budgets); probe de `n_ctx` en vivo. |

### Hace mal (superficie de ataque)
- **Benchmark harness incompleto:** `benchmarks/aider_polyglot.py:85-101` solo trae el descriptor Python; go/rust/cpp/js/java están stubbed ("copy them verbatim"). El headline **78.67% no es reproducible** del harness publicado sin copia manual — gap de honestidad.
- **Loop = caja negra de pi:** compactación mid-run (#68) y thinking-budget son aproximaciones/wrappers; los fixes profundos pertenecen a pi. Parchea pi en runtime — frágil y acoplado a versión.
- **Bloat de runtime:** Node ≥22.19; cada sub-coder es un **proceso Node completo** que relanza el launcher (`spawn.ts:303`); Playwright es dependencia dura.
- **Supuesto single-file:** el write-guard "no reescribir archivos enteros" es un invariante duro tuneado para modelos pequeños; la reescritura `/foo.md`→cwd asume paths raíz de un segmento.
- **Evidence en memoria** (se pierde en shutdown; solo sobrevive vía `evidence-compact`).
- **Fabricación residual** en deep-research (CVE IDs falsos, arxiv con fecha futura) — reconocido, sin fix (`deep_research_assessment.md:44`).
- **Scores absolutos modestos:** TB 2.0 24.6% (rank 120), GAIA 40%.

### Quejas recurrentes de usuarios (issues)
- **Compaction frágil = su dolor #1:** #59 (overflow mid-run), #68 ("Nothing to compact" sesión muerta), #109/#108/#91 (compaction rota, crashes de sub-coder).
- **Bypasses de seguridad:** #70 (ShellSession saltó write-guard — fixed v1.12.0), #107 (semicolon tras `/dev/null` evade device exemption), #94 ("SAFE_PREFIXES fácilmente bypasseable por el modelo").
- **Fiabilidad de tool-calls por formato de modelo:** #42 (LFM2.5 pythonic), #34 (GLM PascalCase hallucination), #96 (false-positive wrong-tool, abierto), #92 (Qwen 3.6-27b).
- **Fricción Node:** #30 ("redo without node.js?" — won't fix, estructural), #22 (glob necesita Node 22.19+), #31 (npm deprecation), #27 (bun update bug).
- **Sub-agentes:** #93 (no read-only), #83, #108.

### Desarrollo activo (PRs)
#105 Plan Mode persiste plan aprobado + espera `/implement`; #102 fixes orphaned sub-coders/dropped tool calls/dead checkpoint; #90 permission prompt; #62 publicar como paquete `pi` (modularización); #39 subagent rework; #38 thinking budget 4096→8192; #49 TUI truncation. Flagship: KV-cache-safe tail injection (#73) y context-watchdog (#59).

**Salud:** muy sana — ~2.4k⭐, ~1k descargas npm/semana, releases casi diarios, owner con respuestas root-cause profundas. Riesgo: dependencia pesada de pi, wrapper complejo, compaction aún inestable.

---

## 2. smallcode (Doorman11991) — JS/Node monolito

**Qué es:** CLI Node de proceso único. `bin/smallcode.js` (3,329 líneas) contiene el loop (`runAgentLoop` :587; tool loop :976). Schemas + routing en `bin/tools.js` y `src/tools/two_stage_router.js`. Clasificador regex determinista (`src/compiled/tool_router.js`) pre-filtra schemas antes de la llamada LLM. `bin/governor.js` hace verificación/hard-fail/decompose. Bench real: `bench/harness.js` spawna el binario por tarea y `bench/diff.js` da veredicto con exit-code.

### Hace bien (robar)
| Patrón | Dónde | Por qué importa |
|--------|-------|-----------------|
| **2-stage tool routing** | `two_stage_router.js`, `tools.js:66-113` | Envía solo un selector de categoría de ~200 tokens primero (en ctx ≤16k), luego el schema filtrado — ahorra ~800 tokens/turno. Clasificador determinista zero-token. **Affirmation guard** (`smallcode.js:806-831`) evita que "yes/ok" quite las write tools. |
| **Parser forgiving** | `smallcode.js:1271-1320` | JSON.parse → `repairToolCall` → extracción regex de `path`/`content` para `write_file`. Multi-formato (JSON/YAML/XML/Hermes/lfm2) + fallback `reasoning_content`. |
| **Patch-first + auto-verify** | `tools.js:17`, `governor.js:72-126` | Search-replace quirúrgico, luego `node --check`/`py_compile`/`go build` tras cada write. Read-before-write guard. |
| **Early-stop + quality monitor** | `early_stop.js`, `quality_monitor.js` | Detecta loops de repetición, patch spirals, regresión de greeting, tool names alucinados; cap de 2 correcciones. |
| **Tool dedup** | `src/tools/dedup.js` | Ventana deslizante read-only + set idempotente por turno — mata loops de modelos pequeños. |
| **Hybrid search offline** | `hybrid_search.js` | BM25 + hashed-vector fusion, chunking centrado en símbolos, sin descargar modelo. |
| **Contract / DoD** | `contract.js`, `tools.js:29-34` | Aserciones declarativas que el agente no puede saltarse ("done"-guard). |
| **Bench + diff con exit-code** | `harness.js`, `diff.js` | Runs reales, veredicto CI (0 improved / 1 regressed / 2 noise) — disciplina measure-first. |
| **Extras:** trust-decay, retry con temperatura adaptativa, router de modelos adaptativo, snapshot/auto-rollback, knowledge injection, bootstrap detection, auto-discovery de test-runner. |

### Hace mal (superficie de ataque)
- **Monolito sin TS real:** `bin/smallcode.js` 3,329 líneas; ~36k LOC JS. README afirma MarrowScript "compila a 1400+ líneas de TypeScript", pero `src/compiled/` es JS escrito a mano con headers "DO NOT EDIT" — ARCHITECTURE.md admite que parte es hand-written. **Engañoso.**
- **Benchmarks deshonestos:** la tabla "87/100 single-file success" de COMPARISON.md está **fabricada** — el harness real solo tiene 19 polyglot + 5 smoke + 10 tool-use; las columnas OpenCode/Pi son "community estimates", no corridas por ellos. El % headline no es reproducible del harness.
- **Verify débil:** bench chequea *contenido* del archivo (`content.includes('a + b')`), no corrección de ejecución — "pass" ≠ código correcto (etiquetado honestamente como file-based).
- **"Semantic" search = bag-of-words hasheado**, no embeddings reales (los comentarios lo admiten).
- **Sin memoria episódica** (grep no encontró ninguna) pese al brief — solo long-term tipada + working memory.
- **Fragilidad:** `try/catch` generalizado que traga errores (degradación elegante esconde bugs); features gateadas tras env vars/deps opcionales; `verifyCode` shella `npx tsc`/`go build` (lento, dependiente de toolchain). Solo sesión única.

### Quejas recurrentes de usuarios (issues)
- **TUI es un bug-farm (blessed alt-screen):** #78 (freeze + 100% CPU tras el primer prompt; solo `--classic` funciona), #84 (flicker Gnome), #93 (navegación teclado), #96 (right-click Linux Mint), #98 (doble-tap teclas).
- **Memory leak:** #82 (miles de procesos `smallcode.js --mcp` spawnados — sin atender).
- **Context/timeout:** #100 (emergency compaction/SIGINT), #79 (write >11kb falla en llama.cpp), #76 (timeout 300s, fix solo por config).
- **Regresión:** #80 (`/provider` ausente en v1.6.0).
- **UX/docs:** #102 "Where PROVIDER?", #101 "Where is PLAN?".

### Desarrollo activo (PRs)
#97 "Consolidated features: agents, live TUI feed, skills/memory/evolver" merged Aug 6 — **sin release tag desde v1.6.0**. PRs #103–#110 son todos `chore(deps)` abiertos (hono, fast-uri, body-parser) — maintainer en modo crucero. Features reales previas: #75 hybrid search, #69 TDD harness, #42 contract/DoD, #39 SSRF guard hardening, #28 plugin system.

**Salud:** enfriándose. Último release May 31; último push Jul 22 solo deps; 19 issues abiertos, varios serios (#82, #78) sin fix. Feature grande de Aug-6 sin release → pipeline de releases estancado. Riesgo de maintainer único.

---

## 3. Gaps & oportunidades (para un harness Rust)

1. **Binario único sin Node** — ambos son Node; usuarios piden nativo explícitamente (#30). Rust gana install, footprint, cold-start, y evita el infierno de versiones de Node (#22).
2. **TUI frágil** — el TUI blessed de smallcode es un bug-farm (#78/#84/#93/#96/#98). Un TUI `ratatui` en Rust (memory-safe, sin proceso por render) es una victoria directa de calidad.
3. **Higiene de procesos/memoria** — el leak de procesos MCP de smallcode (#82) es un problema de spawn de Node; Rust gestiona el ciclo de vida limpiamente.
4. **Compaction/contexto sin resolver en ambos** — little-coder aún pelea overflow mid-run (#109/#108/#91); smallcode no tiene compaction robusta (#100/#79). Un harness Rust con contexto KV-cache-aware + tail-injection (robar #73) + watchdog mid-run es un diferenciador.
5. **Modelo de seguridad/derechos débil en ambos** — bypasses de guarda en little-coder (#70/#107/#94), SSRF en smallcode (#39). Construir un permission-gate estricto y testeado + detección de shell-write desde el día uno.
6. **Parser de tool-calls robusto** — ambos sangran esfuerzo en quirks de formato de modelo (#42/#34/#96). Un parser forgiving multi-formato es table stakes.
7. **Gap multi-file / codebase grande** — smallcode auto-reporta 46% multi-file en COMPARISON.md, pero esa fuente está desacreditada (ver §2) y no hay review externa que lo confirme. La evidencia real del gap: smallcode es solo sesión única y sin memoria episódica; little-coder asume single-file (write-guard). **Cautela estratégica:** el fallo multi-file es en parte un piso de capacidad del modelo (Kitty Code: 12.36% en tareas encadenadas con 9B; sub-2B lo empeora). La apertura honesta es single-file + tier sub-2B + seguridad/compaction — no indexación de codebase.
8. **MCP / plugin ecosystem** — ambos competidores tienen historia MCP/plugin: smallcode envía un servidor MCP (`mcp.json`, `src/tools/mcp_client.js`) y little-coder cabalga el ecosistema de extensiones de pi. Kitty Code no tiene ninguno (solo backlog F6.5). Para un "harness" en 2026, MCP es table stakes — la Fase 2 debe decidir: ship MCP server/client o depriorizar explícitamente con razonamiento.

## 4. Qué robar (resumen consolidado)

**De little-coder:** write-vs-edit invariant + guard de redirects de shell, tail-message injection (KV-cache-safe), thinking-budget cap + nudge, skill/knowledge cards por turno con 3-prioridades, parser malformed-output, background shells event-driven, per-phase model selection, context-watchdog mid-run, permission-gate segment-wise que nombra el remedio, sub-agente aislado por allow-list.

**De smallcode:** 2-stage tool routing con affirmation guard, parser forgiving en capas (JSON→repair→regex), patch-first + auto-verify, early-stop + quality monitor, tool dedup (ventana deslizante + set idempotente), hybrid search offline (BM25 + hashed-vector), contract/DoD done-guard, bench harness con exit-code, trust-decay + retry con temperatura adaptativa, snapshot/auto-rollback.

**Evitar:** parchear pi en runtime, sub-coder = proceso Node completo, benchmark harness incompleto (construir uno reproducible desde el día uno), la narrativa "compiled MarrowScript" (es JS a mano), los números de comparación inflados, verify por contenido.

## 5. Implicación preliminar para Kitty Code (alimenta Fase 2)

Para ser **no-trivial** y ganarles, Kitty Code debe combinar: (a) lo que ya tiene (hashline, guarda de seguridad, quality monitor, budget, memoria episódica, honestidad de benchmarks) + (b) los patrones robables que faltan (2-stage routing, write-vs-edit invariant, shell-redirect guard, args-aware tool dedup, contract/DoD, bench reproducible con exit-code) + (c) los diferenciadores estructurales que ellos no pueden copiar fácilmente (Rust binario único, tier sub-2B, permission-gate estricto, compaction robusta).

**Marco honesto (revisión Oracle Gate 1):** el diferenciador debe anclarse en el **piso real**: fiabilidad single-file + tier sub-2B + seguridad + compaction robusta. El multi-file pertenece al tier de modelos grandes, no al claim principal — posicionarlo como apertura arriesga sobre-invertir en indexación que ningún modelo pequeño puede explotar.

**Nota de commoditización:** little-coder ya envía perfiles por modelo (`benchmark-profiles`). Los model profiles de Kitty Code deben diferenciarse (Rust-native, grammar-enforced, tier sub-2B), no ser "nosotros también tenemos profiles".

**Decisión pendiente para Fase 2:** postura MCP/plugin (ship vs. depriorizar con razonamiento).

La síntesis estratégica completa es la Fase 2.
