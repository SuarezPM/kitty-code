---
slug: f6-remaining-tracks-opt-order
status: approved
intent: clear
review_required: true
plan_path: .omo/plans/f6-remaining-tracks-opt-order.md
plan_sha256: 347lines-v2-fixed-approved
review_round_id: rr-20260820-002
pending-action: done - plan approved both lanes round2
review:
  momus:
    status: approved
    workspace_root: /home/thelinconx/Proyectos/Kitty-Code
    target: .omo/plans/f6-remaining-tracks-opt-order.md
    round_id: rr-20260820-002
    plan_sha256: 347lines-v2
    launch_id: momus-002
    session: ses_fde1b0bd0ffeEjIHJVethpPOIi
    result: APPROVE - 6/6 fixes verified, decision-complete
  independent:
    status: approved
    workspace_root: /home/thelinconx/Proyectos/Kitty-Code
    target: .omo/plans/f6-remaining-tracks-opt-order.md
    round_id: rr-20260820-002
    plan_sha256: 347lines-v2
    launch_id: oracle-002
    session: ses_fde1a875dffeP72vLPDJ5tTpIh
    result: APPROVE - 5/5 fixes verified, template/scope/GBNF/order PASS
round1:
  momus-001: APPROVE non-blocking (ses_fde1fb0d2ffegkWMC6Ps8Q7FnI)
  oracle-001: CHANGES_REQUESTED 5 fixes (ses_fde1cdaecffeUhYkk8zwQ1uh6c) -> fixed
next: handoff to Hephaestus $start-work f6-remaining-tracks-opt-order
approach: Orden óptimo en 6 olas: W0 infra Docker/harbor + smoke k5, W1 sandbox seccomp/Landlock + MemScenes BM25+RRF en paralelo, W2 cloud fallback + plugin system, W3 TUI interactivo, W4 GBNF conditional GO (tiny validity <95% -> llama-cpp-2 0.1.154 grammar_lazy), W5 full benchmarks oficiales polyglot 225 + TB2 89 con oracle y publish AUDIT.md Consumer vs Vendor. Push/OpenSSF explicitamente defer al final fuera de este plan.
---

# Draft: f6-remaining-tracks-opt-order

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| F6.2-infra | Docker + harbor instalados, smoke k5 verifica bench/official correcto | active | bench/official/README.md:28-54, docs/PLAN-EJECUCION.md:322-327 |
| F6.2-full | Polyglot 225 (15-24h) + TB2 89 (3-8h) con oracle, AUDIT.md Consumer vs Vendor | active | bench/official/README.md:10-14, docs/FLOOR-BENCHMARK.md |
| F6.4-GBNF | Spike post-hoc validado -> conditional decode-time llama-cpp-2 grammar_lazy si tiny <95% | active | crates/kitty-runtime/src/grammar.rs:1-8, docs/STRATEGY.md:104-145, Cargo.toml:68 |
| F6.5-sandbox | seccomp/Landlock nuevo sandbox.rs, ningún bypass sobrevive suite guard | active | crates/kitty-harness/src/guard.rs:1-1444, tools.rs:352, docs/PLAN-EJECUCION.md:349 |
| F6.5-memscenes | BM25+embeddings+RRF supera Jaccard en A/B | active | crates/kitty-memory/src/episodic.rs:165-200, docs/PLAN-EJECUCION.md:350 |
| F6.5-cloud | Cloud fallback opt-in, escalada solo en fallo con costo documentado | active | crates/kitty-runtime/src/client.rs:19, agent.rs:23-55 |
| F6.5-plugin | Plugin registry terceros sin tocar core | active | crates/kitty-harness/src/tools.rs:58, docs/STRATEGY.md:54-55 |
| F6.5-tui | TUI interactivo prompt real scroll historial (hoy display-only) | active | crates/kitty-cli/src/tui.rs:1-193, Cargo.toml ratatui 0.29 |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Modelo para F6.2 full | oracle (no smollm2/qwen3) | smollm2 floor 53% <50% demo-only per FLOOR-BENCHMARK.md:54, user explícito "no smollm2" | sí |
| GBNF GO gate | tiny validity <95% tras router tuning, si no NO-GO keep spike | docs/STRATEGY.md:112,113, agent.rs:89 tiny enable_grammar | sí |
| GBNF Option | A llama-cpp-2 0.1.154 LlamaSampler::grammar_lazy (recomendada) vs B llama-server daemon fallback | STRATEGY.md:131-135, server-common.cpp:1289 pitfall | sí |
| Sandbox impl | Nuevo crate module sandbox.rs con seccomp Landlock via apohara-agentguard o crate seccomp, no Docker | guard.rs ya 1444L split candidate, docs/PLAN-EJECUCION.md:114 vendor vacío | sí |
| Embeddings | fastembed/ort + BM25 crate, no rten (bloqueado int8) | episodic.rs:166 blocked, docs/PLAN-EJECUCION.md:350 | sí |
| Cloud provider | Trait CloudClient: ChatClient genérico, no pin a vendor, opt-in flag --cloud-fallback | No existe cloud code, reversible | sí |
| TUI stack | Seguir ratatui 0.29 + crossterm 0.28, streaming agent.run | tui.rs ya usa ese stack | sí |
| Orden olas | W0 infra -> W1 sandbox+memory -> W2 cloud+plugin -> W3 tui -> W4 gbnf conditional -> W5 full bench | Desacopla long-run 15-24h al final, GBNF depende de tiny probe, seguridad antes de cloud | sí |

## Findings (cited - path:lines)
- F6.2 infra clonada 225 polyglot + 89 TB2 Harbor pero Docker NO instalado (CachyOS), harbor no instalado, smoke k5 deferred: bench/official/README.md:28-54, docs/PLAN-EJECUCION.md:322-327
- Homemade benches 96.44% string-match NOT comparable a Aider Polyglot, honestamente etiquetados: bench/official/README.md:15-26, benches/exercises_polyglot.json, FLOOR-BENCHMARK.md:8-11
- F6.4 GBNF spike genera GBNF string (to_gbnf 44-69) pero solo post-hoc validate (79-143) + GrammarClient strict filter (151-219), header SPIKE defer 6-8, Cargo defer comments 68-75: crates/kitty-runtime/src/grammar.rs:1-398, docs/STRATEGY.md:104-145
- Pitfall HTTP 400 si grammar + tools juntos server-common.cpp:1289, solución grammar_lazy PR#9639: docs/STRATEGY.md:128
- Guard real 1444L interpreter-aware + 18 BLOCKED +13 RISKY + ALWAYS_BLOCKED eval 69, shell-write detect 753-808, tools Bash 30s timeout kill_on_drop: guard.rs:824-912, tools.rs:352-435
- ChatClient trait Send+Sync Pin<Box Future>> real + MockClient Arc<Mutex<VecDeque>> + OllamaClient 300s timeout /api/chat /api/tokenize: client.rs:19-362
- ModelProfile tiny 4K/5iter/4tools/no-native/30s/0.7 medium 16K/1024/20iter/6tools etc: model_profile.rs:32-66, agent.rs:50-91 with_profile enable_grammar tiny
- 2-stage ToolRouter real Read/Write/Run/Search/Plan affirmation guard yes|ok|sure|go|proceed: router.rs:15-310, agent.rs:50-51 use_routing false default
- Memory lexical Jaccard real 50 candidates tokenize lowercase + similarity, recall_similar ranks Jaccard+recency, episodic.rs 92-200, blocked embedding rten/Gemma int8 165-167
- MCP server real stdio JSON-RPC 6 tools inherits Guard: crates/kitty-cli/src/mcp.rs:17-182
- TUI display-only feature-gated 193L status bar + placeholder No messages yet: tui.rs:1-193, main.rs:3-4,127,497
- Sandbox/plugin/cloud-fallback 0 hits grep, backlog F6.5 criteria: docs/PLAN-EJECUCION.md:349-354, STRATEGY.md:54-55

## Decisions (with rationale)
- Defer push + OpenSSF al final explícito per user: no CI/scorecard/SBOM en este plan
- Orden óptimo elegido por Prometheus: infra desacopla blocker Docker, seguridad (sandbox) antes de cloud (no bypass), memoria antes de cloud (escalada con contexto), plugin después de MCP estable, TUI último UX tras core estable, GBNF conditional tras tiny probe y antes de full bench (GBNF afecta validity, bench mide final), full bench al final por duración 15-24h+3-8h
- Option A llama-cpp-2 preferida sobre B daemon porque Ollama ya es HTTP daemon fragmentaría runtime, in-process tighter control: STRATEGY.md:138
- Embeddings: elegir fastembed/ort sobre rten bloqueado, BM25 crate bm25 + RRF manual
- Cloud fallback opt-in con doc costo, no auto-escalada silenciosa
- TUI mantener ratatui, no migrar a otro stack

## Scope IN
- W0: sudo pacman -S docker + enable --now + harbor via uv tool install harbor + smoke harbor run -a oracle --k 5 verifica bench/official correcto
- W1: Sandbox seccomp/Landlock nuevo módulo, suite guard sin bypass, OOM/segfault contenidos
- W1: MemScenes BM25+RRF: BM25 lexical + embedding dense + RRF + migración store, A/B supera Jaccard
- W2: Cloud fallback opt-in: CloudClient: ChatClient + Agent fallback loop + CLI --cloud-fallback + costo doc
- W2: Plugin system: PluginRegistry trait + loader (cargo features / WASM / dlopen) terceros registran tools sin core
- W3: TUI interactivo: prompt real + scroll + historial + streaming agent.run, mantener feature tui
- W4: GBNF conditional: medir tiny validity <95% -> si GO implementar LlamaCppClient: ChatClient con LlamaSampler::grammar_lazy + json_schema_to_grammar, si NO-GO mantener spike post-hoc documentado
- W5: Full benchmarks: polyglot 225 + TB2 89 con oracle, publish AUDIT.md columna Consumer vs Vendor, CNAS honesta, resultados reproducibles
- Tests + docs actualizados, clippy -D warnings, cargo test --workspace verde por fase

## Scope OUT (Must NOT have)
- Push a GitHub / git remote add origin + CI OpenSSF Scorecard/Best Practices Badge/SBOM/releases firmados (defer explícito per user al final)
- Fake TS compilation o DO NOT EDIT headers (anti-pattern COMPETITIVE-INTEL)
- Comparar KittyQuiz-225/KittyShell-89 string-match como Aider Polyglot/TB2 oficial (prohibido AGENTS.md)
- Enviar grammar + tools juntos HTTP 400 (server-common.cpp:1289) — debe usar grammar_lazy si GO
- Añadir llama-cpp-2 dep antes de GO gate (Cargo defer)
- Editar bench/official/* datasets (Docker vendor, no tocar)
- unwrap/expect en prod, clone injustificado, módulos >250 LOC sin split
- Plugin que toque guard bypass o rompa hashline BLAKE3

## Open questions
- Ninguna bloqueante: todas resolvibles por research (ordering delegado por user "Orden optimo que tu creas mejor"). Si TUI streaming requiere refactor Agent::run streaming, ¿polling 100ms vs tokio channel? Default: tokio channel mpsc.
- Embeddings model choice: fastembed (ONNX) vs ort — default fastembed por pure Rust.

## Approval gate
status: approved 2026-08-20
approved_by: user "Apruebo, escribe el plan para cristalizarlo"
next: execution via Hephaestus $start-work f6-remaining-tracks-opt-order
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
