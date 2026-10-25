# Little Harbor Agent Adapter

Runs [little-coder](https://github.com/itayinbarr/little-coder) headless as the
solving agent inside Harbor task containers, against a local Ollama server on
the host. Mirror of `../kitty/` (same gateway detection, env sanitization, and
log layout, with `KITTY_` → `LITTLE_` renames).

## Usage

```bash
# from the Kitty-Code repo root
PYTHONPATH="$PWD/bench/harbor" harbor run \
  -p bench/official/aider-harbor/aider-polyglot \
  -a adapters.little.agent:LittleAgent \
  -k 1
```

- `-a adapters.little.agent:LittleAgent` — import path (harbor resolves custom
  agents by import path, not by name).
- `-k 1` — one attempt per trial. A full run is `-k 1` with no `-l`.

### Model parameterization

The model is selected at run time through the environment — no code changes:

```bash
LITTLE_BENCH_MODEL=ornith-1.5:9b harbor run ...   # default
LITTLE_BENCH_MODEL=minicpm5:2b harbor run ...
LITTLE_BENCH_MODEL=ollama/qwen3.5 harbor run ...  # provider/id used as-is
```

| Env var                  | Default            | Purpose                                              |
|--------------------------|--------------------|------------------------------------------------------|
| `LITTLE_BENCH_MODEL`     | `ornith-1.5:9b`    | Model id; bare ids become `ollama/<id>` for `--model`|
| `LITTLE_BENCH_THINKING`  | unset (pi default) | Thinking level (`off\|minimal\|low\|medium\|high…`)   |
| `LITTLE_OLLAMA_URL`      | auto (gateway IP)  | Explicit Ollama URL override                         |
| `LITTLE_OLLAMA_TIMEOUT_SECS` | unset          | Forwarded + logged (no CLI flag consumes it yet)     |
| `LITTLE_BENCH_VERSION`   | `latest`           | npm version pin (`little-coder@<version>`)            |
| `LITTLE_BENCH_NODE_VERSION` | `22.23.2`       | Node tarball version when container Node is too old  |

## Install strategy (node + npm, no binary)

little-coder is a Node.js launcher (`bin/little-coder.mjs`) over the bundled
pi runtime — there is no static binary to cross-build and upload (unlike
`kitty-cli` + `build-binary.sh`). `install()` therefore:

1. Ensures `curl bash git ca-certificates xz tar` via harbor's
   `ensure_system_dependencies`.
2. Checks `node --version` against the >= 22.19.0 preflight; when missing or
   too old (Ubuntu 22.04 apt ships Node 12), downloads the official
   `nodejs.org/dist/v<LITTLE_BENCH_NODE_VERSION>/node-v…-linux-x64.tar.xz`
   into `/usr/local`.
3. Runs `npm install -g little-coder[@<LITTLE_BENCH_VERSION>]` and verifies
   with `little-coder --no-update-check --version` (reports the bundled pi
   version, e.g. `0.83.0`) plus `npm ls -g little-coder`.

`bun` is not required: the launcher resolves both npm and bun install layouts,
and the npm layout is what `install()` produces. The little-coder checkout
itself is never uploaded or modified (read-only + execution scope).

## Expose host Ollama to containers

Same as the kitty adapter (Ollama binds `127.0.0.1` by default, which
containers cannot reach; harbor exposes no `--network host`):

```bash
# /etc/systemd/system/ollama.service.d/override.conf gets:
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
sudo systemctl daemon-reload && sudo systemctl restart ollama

# allow docker bridge subnets to reach Ollama (ufw)
sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp
```

At run time the agent autodetects the container's default gateway IP
(`/proc/net/route` fallback included), normalizes to a `/v1` base URL, and
exports `OLLAMA_BASE_URL` — the env knob little-coder's
`llama-cpp-provider` extension reads as the per-provider `baseUrl` override.
Set `LITTLE_OLLAMA_URL` to bypass detection.

## Invocation inside the container

```bash
export OLLAMA_BASE_URL="$BASE"          # http://<gateway>:11434/v1
export LITTLE_CODER_NO_CTX_PROBE=1      # skip llamacpp /props probe
little-coder --no-update-check --model "ollama/$LITTLE_BENCH_MODEL" \
  [--thinking "$LITTLE_BENCH_THINKING"] -p "<instruction>"
```

Verified live on 2026-10-24 (little-coder 1.17.0, Ollama local):

```bash
node bin/little-coder.mjs --no-update-check --model ollama/ornith-1.5:9b \
  -p "crea /tmp/little-smoke/ok.txt con el texto ok"
# Warning: Model "ornith-1.5:9b" not found for provider "ollama". Using custom model id.
# Creado. El archivo `/tmp/little-smoke/ok.txt` contiene el texto `ok`.
# EXIT:0 — /tmp/little-smoke/ok.txt exists, 2 bytes, content `ok` (no newline)
```

Notes:

- `--model` is the only flag (no `-m` shorthand exists). The
  `Using custom model id` warning is expected for ids absent from the shipped
  `models.json` and does not affect the exit code.
- A `LITTLE_BENCH_MODEL` value already containing `/` is passed through
  untouched, so other providers (`llamacpp/…`, `lmstudio/…`) work if their
  `*_BASE_URL` is reachable.
- `--no-update-check` skips the npm registry poll on every trial launch.
- Console output is tee'd to `/logs/agent/little.txt`; resolved
  model/model-ref/URLs are written to `/logs/agent/little-env.txt`.

## Environment sanitization

Env forwarded to the little-coder process is filtered with the Catalyst
`is_secret_env_name` pattern (`*_API_KEY`, `*_TOKEN`, `*_SECRET`,
`AWS_*`, provider prefixes, etc.), and the same predicate is applied inside
the container shell to unset any secret-looking variables the image or
orchestrator injected — so trials can never leak host credentials.

## Sandbox boundary

Outer Docker boundary only; no Landlock/seccomp layer is configured here
(little-coder executes task commands via its bash tool inside the container).
