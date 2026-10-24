# Kitty Code

A **local coding agent** written in Rust, designed to run small models (e.g.
`ornith:9b`, Q4_K_M, ~5.6 GB) on consumer GPUs (tested on an RTX 2060 Super 8 GB)
via the [Ollama](https://ollama.com) HTTP API.

The value is not the model — it is the **harness**: the tool-call loop, safety
guards, hashline-based file edits, quality monitor, context budget, knowledge
injection, and episodic memory wrapped around a small local LLM.

## Quickstart (3 comandos a primer run funcionando)

```bash
./install.sh                        # 1. instala `kittycode` en ~/.local/bin (+ symlinks compat `kitty`, `kitty-cli`)
kittycode doctor                    # 2. diagnostica Ollama/modelo/VRAM/disco; si falta el modelo propone el `ollama pull <id>` exacto
kittycode run "escribe hola mundo"  # 3. primer run (añade `--plan` para dry-run sin edits)
```

Sin subcomando, `kittycode` abre la TUI interactiva (feature `tui`, por
defecto). Requisitos: Linux x86_64, [Ollama](https://ollama.com) sirviendo en
`http://localhost:11434`. Sin red para descargar el binario de release,
`./install.sh --from-source` compila con cargo. Detalle de flags en
`./install.sh --help`.

## Features

- **Native tool calling** via Ollama's `tools` parameter (the model emits
  structured tool calls, not parsed text).
- **Tools:** `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep`.
- **Hashline edits:** files are tagged with per-line BLAKE3 hashes; edits target
  a line hash and are rejected if the file changed on disk since it was read
  (optimistic concurrency).
- **Safety guard:** an interpreter-aware shell guard blocks dangerous commands
  (e.g. `rm -rf /`, and payloads hidden inside `sh -c '…'`, `python3 -c '…'`,
  `awk 'BEGIN{system(…)}'`).
- **Quality monitor:** detects empty responses, tool loops, and hallucinated
  tools.
- **Context budget:** tracks token usage (real token counts via Ollama's
  `/api/tokenize` when available, heuristic fallback otherwise) and compacts the
  conversation when near the limit.
- **Episodic memory:** past episodes are stored in SQLite and recalled by
  lexical similarity to help with recurring tasks.
- **Knowledge injection:** markdown "cards" scored against the prompt and
  injected as context.

## Build & run

```bash
cargo build --release -p kitty-cli
ollama pull ornith:9b
./target/release/kittycode run "Write a Rust function that reverses a string"
```

Bare `kittycode` (sin subcomando) abre la TUI interactiva. Otros comandos:
`run` (con `--plan`/`--seed`), `info` (status + tool list), `doctor`
(Ollama/modelo/VRAM/disco con remediación `ollama pull` exacta), `skills`
(lista SKILL.md + XML), `undo` (revierte el último edit del journal),
`mcp-client` (lista/llama tools en un servidor MCP externo),
`bench polyglot|tb2|tb2-official|demo-12|ppc|gpqa|claweval` (suites; ver
alcance honesto abajo), `bench-ab` (A/B intercalado con Wilson CI), `mcp`
(servidor MCP stdio con las 6 tools tras el Guard) y `tui` (atajo explícito
a la TUI; equivale al bare sin subcomando).

La TUI es feature `tui` y viene **por defecto** (`[features] default =
["tui"]`): `cargo run -p kitty-cli -- tui` funciona sin flags extra, y
`cargo build --no-default-features` la excluye (entonces `kittycode tui` no
existe y el bare sin subcomando lo dice).
Embedded-backend `count_tokens` es un heurístico honesto (`len/4`, sin
cargar el modelo por diseño).

## Project layout

| Crate | Responsibility |
|-------|----------------|
| `kitty-runtime` | Ollama HTTP client + tool wire types + cloud/llama fallbacks |
| `kitty-harness` | Agent loop (`agent/`), tools + hashline edits, parser, guard (`guard/`), sandbox (Landlock+seccomp), quality monitor, budgets, config, knowledge, bootstrap |
| `kitty-memory`  | Episodic memory (BLAKE3 + SQLite, BM25 with cached index) |
| `kitty-cli`     | CLI, benchmark suites (`bench/`), MCP server, TUI |
| `engine/`       | Fork mistral.rs como backend embebido GGUF (upstream MIT: `engine/NOTICE` + `engine/LICENSE`); `mistralrs-cli` dormant |

See [`codemap.md`](./codemap.md) for the full architecture map.

## Honesty notes

- `kittycode bench` suites:
  - `polyglot` (dataset oficial polyglot-benchmark, grading por unit-test),
    `tb2-official` (Terminal-Bench 2.1 oficial vía Harbor CLI + Docker),
    `gpqa`/`gpqa_diamond` (gated, requieren `--pre-registration`), `claweval`
    (requiere `OPENROUTER_API_KEY` + claw-eval CLI).
  - `demo-12` (suite fija v1.0.0: 12 ejercicios caseros, 1 rep/invocación, la
    serie exige **k≥3** invocaciones con manifiestos; exit 2 = inconcluso
    hasta completar la serie) y `tb2` (8 ejercicios **caseros** locales, no el
    Terminal-Bench oficial). No son comparables con benchmarks oficiales ni
    publicables con una sola rep.
   - `ppc` (Harness² offline, k=1 fijo, `--limit ≤3` con default efectivo 3;
     k=1 ⇒ siempre NoSignal (n tareas ≠ k reps); exit 2 si k<3 o n<3,
     exit 1 si ledger sucio o CI solapa; tripwire `KITTY_RUN_LEAKED` ⇒
     exit 3 sin lección ni resume).
- Bench con una sola rep (`bench-ab` reps=1, `demo-12`/`ppc` de una
  invocación) es inconcluso por diseño: el signal solo vale con serie k≥3,
  ledger limpio y Wilson separado.
- This is a research harness, not a production system.

## License

MIT
