# Kitty Code

A **local coding agent** written in Rust, designed to run small models (e.g.
`ornith:9b`, Q4_K_M, ~5.6 GB) on consumer GPUs (tested on an RTX 2060 Super 8 GB)
via the [Ollama](https://ollama.com) HTTP API.

The value is not the model — it is the **harness**: the tool-call loop, safety
guards, hashline-based file edits, quality monitor, context budget, knowledge
injection, and episodic memory wrapped around a small local LLM.

## Quickstart (3 comandos a primer run funcionando)

```bash
./install.sh                        # 1. instala `kitty` en ~/.local/bin (+ symlink `kitty-cli`)
kitty doctor                        # 2. diagnostica Ollama/modelo/VRAM/disco; si falta el modelo propone el `ollama pull <id>` exacto
kitty run "escribe hola mundo"      # 3. primer run (añade `--plan` para dry-run sin edits)
```

Requisitos: Linux x86_64, [Ollama](https://ollama.com) sirviendo en
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
./target/release/kitty run "Write a Rust function that reverses a string"
```

Other commands: `info` (status + tool list), `doctor` (Ollama/model/VRAM/disk
probe with exact-pull remediation), `skills` (discovered SKILL.md list +
prompt XML), `undo` (revert last journaled edit), `mcp-client` (list/call
tools on an external MCP server child process), `bench polyglot|tb2|tb2-official|gpqa|claweval`
(benchmark suites; gpqa/claweval are gated, offline-first approximations),
`bench-ab` (interleaved A/B with Wilson CI), `mcp` (stdio MCP server exposing
the 6 tools via the Guard), and `tui` (status display, opt-in feature).

TUI is not on by default: build/run it with `cargo run -p kitty-cli --bin kitty --features tui -- tui`.
Embedded-backend `count_tokens` is an honest heuristic (`len/4`, no model load by design).

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

- Benchmarks shipped in this repo (`KittyQuiz-225`, `KittyShell-89`) are
  **homemade** quizzes graded by string matching. They are **not** the official
  Aider Polyglot or Terminal-Bench 2.0 benchmarks and are not comparable to
  them. See [`AUDIT.md`](./AUDIT.md) and [`CNAS.md`](./CNAS.md).
- This is a research harness, not a production system.

## License

MIT
