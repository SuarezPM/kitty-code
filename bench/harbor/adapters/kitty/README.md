# Kitty Harbor Agent Adapter

Runs [Kitty Code](https://github.com/SuarezPM/kitty-code) (`kitty-cli`) as the
solving agent inside Harbor task containers, against a local Ollama server on
the host.

## Usage

```bash
# from the Kitty-Code repo root
PYTHONPATH="$PWD/bench/harbor" harbor run \
  -p bench/official/aider-harbor/aider-polyglot \
  -a adapters.kitty.agent:KittyAgent \
  -k 1
```

- `-a adapters.kitty.agent:KittyAgent` — import path (harbor resolves custom
  agents by import path, not by name).
- `-k 1` — one attempt per trial. A full run is `-k 1` with no `-l`.

### Model parameterization

The model is selected at run time through the environment — no code changes:

```bash
KITTY_BENCH_MODEL=ornith:9b harbor run ...    # default
KITTY_BENCH_MODEL=ornith-1.5:9b harbor run ...
```

| Env var             | Default            | Purpose                                    |
|---------------------|--------------------|--------------------------------------------|
| `KITTY_BENCH_MODEL` | `ornith:9b`        | Model passed as `--model` to kitty-cli     |
| `KITTY_BENCH_PROFILE` | `medium`         | Profile preset (`tiny\|medium\|large`)     |
| `KITTY_OLLAMA_URL`  | auto (gateway IP)  | Explicit Ollama URL override               |
| `KITTY_BENCH_BINARY`| adapter `bin/` dir | Host path of the kitty-cli binary to upload|

## Terminal-Bench 2.0 (protocolo k=3)

Comando exacto para correr 1 tarea de TB2.0 con Kitty (smoke test; la serie
completa la corre el pilot):

```bash
# desde la raíz del repo Kitty-Code
PYTHONPATH="$PWD/bench/harbor" KITTY_BENCH_MODEL=minicpm5:2b harbor run \
  -d terminal-bench@2.0 \
  -a adapters.kitty.agent:KittyAgent \
  -m minicpm5:2b \
  -l 1 -k 3 \
  -n 1 \
  --ae KITTY_BENCH_MODEL=minicpm5:2b \
  --ae KITTY_SANDBOX=off \
  --ae KITTY_OLLAMA_TIMEOUT_SECS=900 \
  --agent-timeout-multiplier 3
```

Flags:

- `-d terminal-bench@2.0` — dataset TB2.0 del registry de Harbor.
- `-a adapters.kitty.agent:KittyAgent` — import path del adapter (igual que
  en la sección Usage).
- `-m minicpm5:2b` — model name para Harbor; el modelo efectivo lo gobierna
  `KITTY_BENCH_MODEL` (el adapter lo pasa como `--model` a kitty-cli).
- `-l 1` — una sola tarea (smoke test). Sin `-l`, corre el dataset completo.
- `-k 3` — 3 intentos por trial (protocolo TB2.0 k=3).
- `-n 1` — un trial a la vez (los modelos locales 2B/9B no dan para
  concurrencia; además evita saturar la GPU del host).
- `--ae KITTY_BENCH_MODEL=minicpm5:2b` — modelo efectivo dentro del
  contenedor (requerido; `--ae` aterriza en `_extra_env`, no en
  `os.environ`). El `KITTY_BENCH_MODEL=` del prefijo es cinturón y
  tirantes para el mismo valor.
- Variante Ornith: sustituir `minicpm5:2b` por `ornith-1.5:9b` en `-m`,
  en el prefijo y en `--ae` (el default del adapter sigue siendo
  `ornith:9b`).

### Timeouts recomendados

- `--ae KITTY_OLLAMA_TIMEOUT_SECS=900` — techo por petición a Ollama; los
  modelos pequeños locales (2B) razonan lento en tareas largas de
  terminal. Sin esto, respuestas truncadas se registran como vacías.
- `--agent-timeout-multiplier 3` — multiplica el timeout de la fase de
  agente de Harbor (precedente: `bench/harbor/rerun-v3.sh` usa esta misma
  combinación para `ornith:9b`).
- Si el timeout salta en build de imagen o verificación (no en el agente),
  añadir `--timeout-multiplier 2` (o 3) en lugar de seguir subiendo solo el
  de agente.

### Requisito de red (resumen)

Imprescindible antes de cualquier corrida TB2.0 — detalle en
`One-time setup` §2:

1. Ollama escuchando en `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`, no solo
   `127.0.0.1`) + `OLLAMA_CONTEXT_LENGTH=16384` (kitty-cli nunca envía
   `num_ctx`; sin esto el servidor trunca el prompt y el modelo devuelve
   respuestas vacías).
2. Firewall abierto del bridge Docker hacia el host:
   `sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp`.
3. El adapter autodetecta el gateway del bridge (`ip route` /
   `/proc/net/route`) y llama a `http://<gateway>:11434`. Solo para
   depurar la red, forzar con `--ae KITTY_OLLAMA_URL=http://<ip>:11434`.

## One-time setup

### 1. Build the container-compatible binary

Task images are Ubuntu 22.04 based (glibc 2.35); a host-built binary fails
with `GLIBC_2.39 not found`. Build once inside an ubuntu:22.04 container:

```bash
sudo -u "$USER" -g docker bash bench/harbor/adapters/kitty/build-binary.sh
```

This produces `bench/harbor/adapters/kitty/bin/kitty-cli`, which the agent
uploads into every trial container via `environment.upload_file` and installs
at `/usr/local/bin/kitty-cli` (harbor agents have no bind-mount API).

### 2. Expose host Ollama to containers

Ollama binds `127.0.0.1` by default, which containers cannot reach. Harbor
does not expose `--network host` or compose `extra_hosts`, so instead:

```bash
# /etc/systemd/system/ollama.service.d/override.conf gets:
Environment="OLLAMA_HOST=0.0.0.0"
# kitty-cli never sends num_ctx, so the server default governs the context
# window. Without this, long benchmark prompts are silently truncated and
# the model returns empty responses:
Environment="OLLAMA_CONTEXT_LENGTH=16384"
sudo systemctl daemon-reload && sudo systemctl restart ollama

# allow docker bridge subnets to reach Ollama (ufw)
sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp
```

At run time the agent autodetects the container's default gateway IP
(`/proc/net/route` fallback included) and calls
`http://<gateway>:11434`. Set `KITTY_OLLAMA_URL` to bypass detection.

## Invocation inside the container

```
kitty-cli --url "$URL" --model "$KITTY_BENCH_MODEL" run --profile medium "<instruction>"
```

Note: `--model` is a top-level flag in kitty-cli and must precede the `run`
subcommand.

Console output is tee'd to `/logs/agent/kitty.txt`; resolved model/profile/URL
are written to `/logs/agent/kitty-env.txt`.

## Environment sanitization

Env forwarded to the kitty process is filtered with the Catalyst
`is_secret_env_name` pattern (`*_API_KEY`, `*_TOKEN`, `*_SECRET`,
`AWS_*`, provider prefixes, etc.), and the same predicate is applied inside
the container shell to unset any secret-looking variables the image or
orchestrator injected — notably `KITTY_CLOUD_API_KEY`, so trials can never
trigger paid cloud fallback.

## Sandbox boundary

Bash children are Landlock/seccomp-contained by default; harbor bench runs set
`KITTY_SANDBOX=off` because the outer Docker boundary is active; opt-out is
loud and per-run.
