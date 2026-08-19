# Deep Interview Spec: Kitty Code

## Metadata
- Interview ID: kitty-code-2026-08-18
- Rounds: 10
- Final Ambiguity Score: 14.9%
- Type: greenfield
- Generated: 2026-08-18
- Threshold: 20%
- Threshold Source: default
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.298 |
| Constraint Clarity | 0.85 | 0.25 | 0.213 |
| Success Criteria | 0.85 | 0.25 | 0.213 |
| Context Clarity | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.851** |
| **Ambiguity** | | | **14.9%** |

## Topology
| Component | Status | Description |
|-----------|--------|-------------|
| Agent Loop | active | Core cycle: preflight → LLM → parse → gate → exec → verify |
| Edit System | active | Hashline primary + search-replace fallback |
| Context Management | active | Tail injection + tree-sitter signatures + stdout truncation |
| Infrastructure | active | Sandbox, pathsafety, audit, episodic memory (from Apohara) |

## Goal

Kitty Code es un coding agent local en Rust, un solo binario, optimizado para ornith:9B en RTX 2060S 8GB. Mide su performance con CNAS (Consumer-Normalized Agentic Score): Aider Polyglot, Terminal-Bench 2.0, y smoke SWE-Verified. Publica números reales en AUDIT.md con columna Consumer vs Vendor.

## Constraints
- Hardware: RTX 2060S 8GB + 48GB RAM + Ryzen 5 3600
- Model: ornith:9b (Q4_K_M, Ollama 0.32.14)
- Context window: medir en semana 0 (8K/16K/32K)
- Temperature: 0.6 default (oficial), 1.0 para benchmarks repro
- Binary: <15 MB stripped, <30 MB RSS, <50 ms boot
- No cloud-first, no multi-CLI orchestration, no Node/Python runtime
- 4 crates + vendor de Apohara (auditado, no copy-paste)
- Memory v0: episodic blake3, cero VRAM adicional

## Non-Goals
- Orquestar CLIs de terceros (Apohara ya cubre BYOC)
- TUI en Fase 0 (CLI + RPC primero)
- EverMemOS completo (post-MVP si v0 demuestra olvido)
- Multi-agent o plugin system (post-MVP)
- Cloud fallback (post-MVP)

## Acceptance Criteria
- [ ] Polyglot ≥45% con ornith:9B en 8GB (superar little-coder)
- [ ] TB 2.0 ≥18% en 8GB (meta intermedia)
- [ ] Tool-call validity ≥95%
- [ ] Edit apply success (hashline) ≥80%
- [ ] Binario <15 MB, <30 MB RSS, <50 ms boot
- [ ] AUDIT.md publicado con 🟢 medido / 🟡 self-reported / 🔬 no corrido
- [ ] VRAM usage <7.5 GB (0.5 GB headroom mínimo)

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Harness debe ser pesado (inyectar guidance) | Perplexity: Ornith ya internalizó scaffold | Elegido harness pesado (usuario decidió) |
| 2-stage router ahorra ~800 tokens/turno | Perplexity: puede causar distribution shift | Elegido 2-stage router (usuario decidió) |
| Temp 0.3-0.5 es mejor | Perplexity: Ornith entrenado a 0.6-1.0 | Temp 0.6 default, A/B en semana 0 |
| 11 crates necesarios | Perplexity: over-engineering | 4 crates + vendor Apohara |
| rten + Gemma para embeddings | Perplexity: compite por VRAM con ornith | blake3 feature-hashing, cero VRAM |
| RTK como binario externo | Perplexity: rompe "un solo binario" | Truncado nativo, no RTK |
| EverMemOS para memoria | Perplexity: paper conversacional, no coding loop | Episodic blake3, cero VRAM |

## Technical Context

### Archivo de edicion (hashline)
- Cada linea se etiqueta al leer: `LINE:HASH|content`
- Edit por hash, no por texto
- Si el archivo cambió, apply se rechaza antes de corromper
- Search-replace como fallback para archivos no ledos

### Context management
- Prefijo inmutable: system prompt + tool schemas (byte-idéntico toda la sesión)
- Tail injection: skills, knowledge, TODO, errores → al final de la conversación
- Tree-sitter signatures para archivos >N lineas
- Truncado nativo de stdout (no RTK)
- Watchdog 80% + loop guard (de little-coder)
- Context window: medir en semana 0

### Tools (10, 2-stage router)
1. Read (hashline + recorte a 30 lineas si no cabe)
2. Edit (hashline apply; search-replace como fallback)
3. Write (solo archivos nuevos)
4. Bash (whitelist + juicio por cadena + bloqueo de redirects)
5. Glob
6. Grep
7. ShellStart (wake_on, sin polling)
8. Test / Cargo / runner detectado en bootstrap
9. WebSearch
10. WebFetch

### Runtime
- Default: Ollama 0.32.14, ornith:9b
- Flash Attention + q8_0 KV cache
- keep_alive=30m
- llama.cpp como adapter B para GBNF y medir GDN checkpoint

### Memory v0
- Episodic blake3 + sqlite-vec (de Apohara)
- Al cerrar sesion: Episode { goal, files_touched, test_verdict, outcome }
- Al abrir: recall_for_goal inyecta ≤3 episodios en la cola
- Cero VRAM adicional

### Infrastructure (vendor de Apohara)
- apohara-sandbox: seccomp-bpf fail-closed, 3 tiers
- apohara-pathsafety: symlink-escape detection
- apohara-episodic: cross-session memory
- apohara-audit: JSONL 0600 append-only

## Interview Transcript

### Round 1: Harness Thesis
**Q:** Cuando ornith:9B genera su propio plan interno, ¿el harness debe delegar la orquestación o inyectar guidance?
**A:** Harness pesado — inyectar TODO anchors, knowledge cards, plan guidance.
**Ambiguity:** 50.5%

### Round 2: Edit Primitive
**Q:** ¿Cuál es la primitiva de edición primaria?
**A:** Hashline + fallback (hashline para archivos ya ledos, search-replace como fallback).
**Ambiguity:** 45.9%

### Round 3: Tool Set
**Q:** ¿Cuántas tools y cómo se gestionan?
**A:** 2-stage router.
**Ambiguity:** 39.7%

### Round 4: Context Window
**Q:** ¿Cuál es el context window real de ornith en 8GB?
**A:** Medir en semana 0 (8K/16K/32K).
**Ambiguity:** 35.1%

### Round 5: Metrics Framework
**Q:** ¿CNAS como framework de métricas?
**A:** Sí, CNAS de Perplexity (Polyglot + TB + SWE-V).
**Ambiguity:** 27.5%

### Round 6: Memory Strategy
**Q:** ¿Cuál es la estrategia de memoria?
**A:** Episodic blake3 (MVP). blake3 + sqlite-vec, cero VRAM.
**Ambiguity:** 25.1%

### Round 7: Tool Composition
**Q:** ¿Qué tools exactas?
**A:** 8 base + web search + web fetch = 10 tools.
**Ambiguity:** 23.4%

### Round 8: Apohara Extraction
**Q:** ¿Extraer de Apohara o from scratch?
**A:** Extraer 4 crates (sandbox, pathsafety, episodic, audit) pero auditar, vendorizar, optimizar y adaptar.
**Ambiguity:** 21.2%

### Round 9: Context Strategy
**Q:** ¿Estrategia completa de context management?
**A:** Tail injection + tree-sitter signatures + truncado nativo de stdout.
**Ambiguity:** 16.5% ✅

### Round 10: Naming
**Q:** ¿Cómo se llama el proyecto?
**A:** Kitty Code.
**Ambiguity:** 14.9% ✅
