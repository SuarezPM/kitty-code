# Plan Definitivo: Sucesor Espiritual de SmallCode

## Nombre: **OrnithForge** (o el que prefieras)

---

## 0. Resumen Ejecutivo

**Objetivo**: Crear un coding agent local optimizado para ornith:9B en RTX 2060S 8GB, combinando lo mejor de:
- **SmallCode** (patterns de agent: 2-stage routing, patch-first, TODO planning)
- **Context Forge / DeKanus** (safety kernel, compresión, AUDIT.md culture)
- **RTK** (shell output compression, hook-based integration)
- **EverMemOS** (memory lifecycle: MemCells → MemScenes → Reconstructive Recollection)
- **LeanCTX** (tree-sitter signatures, content-addressed recovery)

**Decisión clave: FROM SCRATCH en Rust** (no fork de SmallCode)

Razones:
1. SmallCode = JS monolith de 3.3k líneas, sin TypeScript, sin modularidad real
2. DeKanus ya tiene workspace Rust de 8 crates como base
3. RTK (76k stars) es Rust y se integra como binario externo
4. Rust permite compilar a binario único sin dependencias de runtime
5. Los patterns valiosos de SmallCode se pueden reimplementar en ~2k LOC

---

## 1. Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    OrnithForge CLI                      │
│  (interfaz TUI simple, sin mouse tracking, sin flicker) │
├─────────────────────────────────────────────────────────┤
│                   Agent Core                            │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │ 2-Stage      │ │ Tool Router  │ │ Forgiving      │  │
│  │ Tool Router  │ │ (Rust impl)  │ │ JSON Parser    │  │
│  └──────────────┘ └──────────────┘ └────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │ TODO Planner │ │ Patch Engine │ │ Context Budget │  │
│  │ (anchor      │ │ (search+     │ │ (cap tool      │  │
│  │  injection)  │ │  replace)    │ │  results 4k)   │  │
│  └──────────────┘ └──────────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Context Compression Layer                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │ RTK Binary (shell output) + Tree-sitter Signatures│  │
│  │ + Content-Addressed Recovery (inspired by LeanCTX)│  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│               Memory Lifecycle (EverMemOS)              │
│  Phase I: Episodic Trace Formation (MemCells)           │
│  Phase II: Semantic Consolidation (MemScenes)           │
│  Phase III: Reconstructive Recollection (hybrid RRF)    │
├─────────────────────────────────────────────────────────┤
│              Inference Runtime                          │
│  Ollama API (localhost:11434) + grammar enforcement     │
│  Flash Attention + q8_0 KV cache                       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Estructura de Crates

```
ornithforge/
├── Cargo.toml              # workspace
├── crates/
│   ├── of-agent/           # Core agent loop, tool dispatch, state machine
│   ├── of-router/          # 2-stage tool routing (Rust impl de SmallCode pattern)
│   ├── of-parser/          # Forgiving multi-format tool call parser
│   ├── of-patch/           # Search-and-replace patch engine
│   ├── of-context/         # Context budget, compression, assembly
│   ├── of-memory/          # MemCell/MemScene lifecycle (EverMemOS-inspired)
│   ├── of-retrieve/        # Hybrid retrieval: BM25 + embeddings + RRF
│   ├── of-embed/           # Local embeddings (rten + Gemma MRL 256d)
│   ├── of-tui/             # Simple TUI (ratatui, no mouse tracking)
│   ├── of-config/          # Configuration, .env loading, provider auth
│   └── of-audit/           # AUDIT.md ledger, honesty badges (from Context Forge)
├── tools/                  # Built-in tool implementations
├── knowledge/              # Injected context files (cheat sheets, patterns)
├── tests/                  # Integration tests + harness
└── rtk/                    # RTK submodule or binary reference
```

---

## 3. Módulos Clave (detallado)

### 3.1 Agent Core (`of-agent`)
- **Agent loop**: pre-work → LLM call → parse → execute → post-work
- **Pre-work**: regex clarifier (zero tokens), @file expansion, git diff injection
- **Bootstrap detection**: escanea workspace en primer turno, inyecta build commands
- **Per-turn state**: TODO anchor re-injected cada turno ("ACTIVE PLAN step 3/5")
- **Snapshot + auto-rollback**: checkpoint por turno, revert en validación fallida

### 3.2 2-Stage Tool Router (`of-router`)
- **Stage 1**: Weighted regex scoring → clasifica categoría (read/write/search/run/plan)
- **Stage 2**: Inyecta solo schemas de la categoría ganadora
- **Ahorro**: ~800 tokens por turno vs enviar todas las herramientas
- **Affirmation guard**: "yes"/"ok" no re-clasifica away de write tools
- **Key insight de SmallCode**: "Sending all 20 tool definitions every single time is wasteful for small context windows"

### 3.3 Forgiving Parser (`of-parser`)
- Parsea: JSON, YAML, XML, Hermes, Liquid AI, plain text
- Auto-repair: wrong param names, type mismatches
- Fallback a reasoning_content cuando content está vacío (reasoning models)
- **Key insight de SmallCode**: small models producen tool calls malformed, hay que repararlos

### 3.4 Patch Engine (`of-patch`)
- **Search-and-replace** como primitiva primaria (NO full-file rewrite)
- Small models no pueden reproducir archivos enteros confiablemente
- Parches quirúrgicos = más confiables y más baratos en tokens
- **Read-before-write guard**: rechaza writes a archivos no ledos este turno

### 3.5 Context Budget (`of-context`)
- Tool results capped a 4k chars
- Mid-turn eviction (evict least relevant si contexto > 80%)
- Semantic compression (resumir, no truncar)
- **Key insight de EXA research**: keep system prompt < 500 tokens para 9B models

### 3.6 Memory Lifecycle (`of-memory`) — EverMemOS-inspired
- **MemCell** = (episode, atomic_facts, foresight, metadata)
- **MemScene** = (cells, centroid, last_timestamp) — clustering online
- **Reconstructive Recollection**: hybrid BM25 + embeddings + RRF
- **Foresight filtering**: descarta expired foresight signals
- **Agentic verification**: sufficiency check antes de pasar al modelo
- **Key insight de EverMemOS**: "memory should be a lifecycle, not a database"

### 3.7 Retrieval (`of-retrieve`)
- **BM25**: Tantivy (Rust, ya probado en apo-mem)
- **Embeddings**: rten + Gemma MRL 256d (ya probado en spike S2)
- **RRF fusion**: Reciprocal Rank Fusion de BM25 + dense
- **Relevance floor**: no pasar contexto si score < threshold
- **Abstención**: si no hay suficiente evidencia, decir "no sé" en vez de inventar

### 3.8 TUI (`of-tui`)
- **ratatui** (Rust TUI framework, probado, estable)
- Sin mouse tracking (SmallCode tenía flickering issues)
- Status bar: tokens usados, model info, TODO progress
- Simple, sin fullscreen alternate buffer (evita flickering)

### 3.9 Audit Ledger (`of-audit`)
- AUDIT.md público con números reales (de Context Forge)
- check_audity.sh CI guard
- Honest-scope badges: 🟢 verified, 🟡 self-reported, 🔬 not yet tested
- **Key insight de Context Forge**: "Running the experiment honestly and reporting the negative IS the product"

---

## 4. Optimizaciones para Ornith:9B en 2060S 8GB

### VRAM Budget
| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (Q4_K_M) | ~5.6 GB | ornith:9b default |
| KV cache (8K ctx, q8_0) | ~1.2 GB | Flash Attention enabled |
| CUDA overhead | ~0.5 GB | Driver, allocator |
| **Total** | **~7.3 GB** | Leaves 0.7 GB headroom |

### Ollama Config
```bash
OLLAMA_FLASH_ATTENTION=1      # Ya en systemd
OLLAMA_KV_CACHE_TYPE=q8_0     # Ya en systemd
OLLAMA_KEEP_ALIVE=30m         # Evitar reload entre prompts
OLLAMA_CONTEXT_LENGTH=8192    # No over-allocate
```

### Agent Optimizations
1. **System prompt < 500 tokens** — ornith:9B se ahoga con prompts largos
2. **Grammar enforcement** — llama.cpp `--grammar` para tool calls (88% → 98% valid)
3. **Temperature 0.3-0.5** — NO 0 (loops más a temp 0)
4. **Schema re-injection cada 10 turns** — previene schema drift
5. **Max 10-15 tools por contexto** — accuracy cae > 90% tools
6. **Thinking budget cap** — limitar reasoning tokens por call
7. **Retry budget: 3 max** — exponential backoff, no loops infinitos

### Context Compression (de papers 2026)
1. **RTK shell hooks** — 60-90% reducción en output de comandos
2. **Tree-sitter signatures** — leer solo function signatures, no archivos enteros
3. **Content-addressed recovery** — comprimir pero poder recuperar originals
4. **Prefix caching** — num_keep para system prompt en KV cache
5. **ContextBudget** — budget-aware inference state

---

## 5. Fases de Implementación

### Fase 0: Fundación (1-2 semanas)
- [ ] Workspace Rust con crates iniciales
- [ ] `of-config`: .env loading, provider auth, Modelfile generation
- [ ] `of-parser`: forgiving multi-format tool call parser
- [ ] `of-router`: 2-stage tool routing (Rust impl)
- [ ] Test harness básico (5-10 tests)

### Fase 1: Agent Core (2-3 semanas)
- [ ] `of-agent`: agent loop, pre-work, post-work
- [ ] `of-patch`: search-and-replace engine
- [ ] `of-context`: budget engine, tool result capping
- [ ] TODO planner con anchor injection
- [ ] Snapshot + auto-rollback
- [ ] 20-30 tests unitarios

### Fase 2: Context Compression (1-2 semanas)
- [ ] RTK integration (shell hooks, binary reference)
- [ ] Tree-sitter signature extraction (inspired by LeanCTX)
- [ ] Content-addressed recovery (minimal impl)
- [ ] Bootstrap detection (workspace scan)
- [ ] Knowledge injection (keyword overlap)

### Fase 3: Memory Lifecycle (2-3 semanas)
- [ ] `of-memory`: MemCell creation (episodic trace formation)
- [ ] MemScene clustering (online, cosine similarity)
- [ ] `of-retrieve`: BM25 (Tantivy) + embeddings (rten)
- [ ] RRF fusion
- [ ] Reconstructive recollection
- [ ] Foresight filtering

### Fase 4: TUI + Polish (1-2 semanas)
- [ ] `of-tui`: ratatui-based, simple, no flicker
- [ ] Model escalation (opt-in cloud fallback)
- [ ] Compound tools (read_and_patch, create_and_run)
- [ ] Per-tool trust decay (Bayesian scoring)
- [ ] Integration tests completas

### Fase 5: Optimization + Audit (1 semana)
- [ ] AUDIT.md ledger
- [ ] Benchmark harness (SWE-bench, Terminal-Bench)
- [ ] VRAM profiling, latency optimization
- [ ] Grammar enforcement tuning
- [ ] Documentation

---

## 6. Decisiones de Diseño

### Fork vs From Scratch
**FROM SCRATCH en Rust.**
- SmallCode patterns se reimplementan en ~2k LOC
- El monolith JS de 3.3k líneas no es mantenible
- Rust = binario único, sin dependencias de runtime
- DeKanus workspace como base (8 crates pre-estructurados)

### Lenguaje: Rust
- Ya tenés experiencia (apo-mem v2 en Rust)
- RTK es Rust, Tantivy es Rust, rten es Rust
- Binario único, sin node_modules, sin npm
- Performance nativa para context compression

### Ollama vs llama.cpp directo
**Ollama como default**, con opción de llama.cpp directo.
- Ollama ya funciona, flash attention, KV cache quant
- llama.cpp para grammar enforcement (si Ollama no lo soporta bien)
- La capa de inferencia debe ser swappable

### Memoria: EverMemOS vs Simple
**EverMemOS-inspired pero simplificado.**
- MemCells + MemScenes = sí, el lifecycle es valioso
- Agentic verification = sí, previene hallucination
- Foresight filtering = sí, para sesiones largas
- Profile evolution = post-MVP (complejo, no crítico para coding)

### TUI: ratatui vs bubbletea vs crossterm directo
**ratatui** — es el estándar Rust, bien documentado, sin flickering.

---

## 7. Comparativa con Competencia

| Feature | SmallCode | OpenCode | OrnithForge (plan) |
|---------|-----------|----------|-------------------|
| Lenguaje | JS (monolith) | TypeScript | Rust |
| Modelos | 8B-35B | Cloud only | 9B local (ornith) |
| Context compression | RTK hooks | None | RTK + tree-sitter + CCR |
| Tool routing | 2-stage regex | All tools | 2-stage regex |
| Memory | SQLite FTS5 | Plugin | EverMemOS lifecycle |
| TUI | Fullscreen (flicker) | Terminal | Ratatui (no flicker) |
| Safety | None | None | AUDIT.md ledger |
| Escalation | Cloud fallback | Cloud native | Opt-in cloud fallback |
| Multi-file success | 46% | ~75% | Target: 60%+ |

---

## 8. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Ornith:9B no soporta tool calling bien | Media | Grammar enforcement, forgiving parser, schema re-injection |
| VRAM insuficiente para contexto largo | Baja | q8_0 KV cache, 8K context cap, prefix caching |
| Rust compilation time lento | Media | Incremental builds, cargo-nextest, parallel compilation |
| EverMemOS too complex for MVP | Media | Simplificar a MemCells only, MemScenes post-MVP |
| RTK binary not found | Baja | Graceful degradation, shell output truncation fallback |
| Tree-sitter grammars weigh too much | Baja | Only include 5-10 languages initially |

---

## 9. Métricas de Éxito

| Métrica | Target | Mínimo |
|---------|--------|--------|
| Generation speed | >50 tok/s | >40 tok/s |
| Single-file success | >85% | >75% |
| Multi-file success | >60% | >50% |
| Tool call validity | >95% | >85% |
| Context compression | >50% | >30% |
| Time to first response | <3s | <5s |
| VRAM usage | <7.5 GB | <8 GB |

---

## 10. Tech Stack Final

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Rust | Performance, binary distribution, ecosystem |
| Inference | Ollama API | Already working, flash attention, KV cache |
| Embeddings | rten + Gemma MRL 256d | Local, fast, proven in spike S2 |
| Search | Tantivy | Rust-native, BM25, proven in apo-mem |
| TUI | ratatui | Standard Rust TUI, no flicker |
| Shell compression | RTK binary | 76k stars, proven, hook-based |
| Code analysis | tree-sitter | 26 languages, structural understanding |
| Config | TOML | Standard Rust, human-readable |
| Testing | cargo-nextest | Fast, parallel Rust tests |
| CI | GitHub Actions | Standard, with AUDIT.md generation |

---

*Plan generado: 2026-08-18*
*Basado en research de 4 agents paralelos + paper EverMemOS (ACL 2026)*
