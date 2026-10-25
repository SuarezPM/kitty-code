"""LittleAgent — Harbor adapter that solves tasks with little-coder.

Runs ``little-coder`` headless inside the task container against the host's
Ollama instance. Mirror of ``bench/harbor/adapters/kitty/agent.py``. Design
notes:

- **Delivery**: little-coder is a Node.js launcher (``bin/little-coder.mjs``)
  over the bundled pi runtime — there is no single static binary to upload
  like ``kitty-cli``. ``install()`` therefore installs a Node 22 runtime
  (official tarball, only when missing/too old — Ubuntu 22.04 apt ships a
  Node far below the required >= 22.19.0) and then ``npm install -g
  little-coder`` inside the container. No host repo files are uploaded; the
  little-coder checkout itself is never modified.
- **Networking**: Ollama listens on the host; containers reach it through the
  docker bridge gateway IP (requires host Ollama bound beyond loopback, e.g.
  ``OLLAMA_HOST=0.0.0.0``, plus a firewall allow rule for docker subnets).
  The container shell resolves the gateway and exports ``OLLAMA_BASE_URL``
  (the env knob little-coder's ``llama-cpp-provider`` extension reads as a
  per-provider ``baseUrl`` override). Set ``LITTLE_OLLAMA_URL`` to skip
  gateway autodetection.
- **Model parameterization**: ``LITTLE_BENCH_MODEL`` (default
  ``ornith-1.5:9b``) selects the model at run time — no code changes between
  runs. A bare id (``ornith-1.5:9b``) is sent as ``ollama/<id>``; a value that
  already contains ``/`` (``ollama/foo``, ``llamacpp/bar``) is used as-is.
  Unknown ids for a known provider resolve through pi's custom-model
  fallback (clones the provider's base entry, warns
  ``Using custom model id``) — verified live, not just code-read.
- **Verified headless command** (smoke 2026-10-24, exit 0, file created)::

    little-coder --no-update-check --model ollama/ornith-1.5:9b \\
      -p "crea /tmp/little-smoke/ok.txt con el texto ok"

  Note: ``--model`` is the only flag (no ``-m`` shorthand exists); the
  ``Using custom model id`` warning on stderr is expected for ids absent
  from the shipped ``models.json`` and does not affect the exit code.
"""

from __future__ import annotations

import shlex
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .env_sanitizer import build_shell_secret_unset_snippet, sanitize_env

AGENT_NAME = "little"
DEFAULT_MODEL = "ornith-1.5:9b"
DEFAULT_THINKING = ""
DEFAULT_VERSION = "latest"
DEFAULT_NODE_VERSION = "22.23.2"

MODEL_ENV_VAR = "LITTLE_BENCH_MODEL"
THINKING_ENV_VAR = "LITTLE_BENCH_THINKING"
VERSION_ENV_VAR = "LITTLE_BENCH_VERSION"
NODE_VERSION_ENV_VAR = "LITTLE_BENCH_NODE_VERSION"
OLLAMA_URL_ENV_VAR = "LITTLE_OLLAMA_URL"
TIMEOUT_ENV_VAR = "LITTLE_OLLAMA_TIMEOUT_SECS"

# Env knob read by little-coder's llama-cpp-provider extension as the
# per-provider baseUrl override. Exported inside the container shell.
PROVIDER_BASE_URL_ENV = "OLLAMA_BASE_URL"

CONTAINER_LOG_PATH = "/logs/agent/little.txt"
CONTAINER_ENV_LOG_PATH = "/logs/agent/little-env.txt"

_DEFAULT_GATEWAY_PORT = 11434
_MIN_NODE_MAJOR, _MIN_NODE_MINOR, _MIN_NODE_PATCH = 22, 19, 0


class LittleAgent(BaseInstalledAgent):
    """Installs and runs little-coder headless inside the task container."""

    @staticmethod
    @override
    def name() -> str:
        return AGENT_NAME

    @override
    def get_version_command(self) -> str:
        # Reports the bundled pi runtime version (little-coder forwards
        # --version to pi); the npm package version is logged at install time.
        return "little-coder --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(
            environment,
            ("curl", "bash", "git", "ca_certificates", "xz", "tar"),
        )
        node_version = self._get_env(NODE_VERSION_ENV_VAR) or DEFAULT_NODE_VERSION
        version = self._get_env(VERSION_ENV_VAR) or DEFAULT_VERSION
        package = (
            "little-coder" if version == "latest" else f"little-coder@{version}"
        )
        self.logger.info("Ensuring Node %s and package %s", node_version, package)
        await self.exec_as_root(
            environment,
            command=(
                "set -e; "
                "need_node=0; "
                "if ! command -v node >/dev/null 2>&1; then need_node=1; else "
                "cur=$(node -p 'process.versions.node' 2>/dev/null || echo 0.0.0); "
                f"if ! node -e 'const [M,m,p]=process.versions.node.split(\\\".\\\").map(Number);"
                f"process.exit(M>{_MIN_NODE_MAJOR}||(M=={_MIN_NODE_MAJOR}&&(m>{_MIN_NODE_MINOR}||"
                f"(m=={_MIN_NODE_MINOR}&&p>={_MIN_NODE_PATCH}))))?0:1' "
                "2>/dev/null; then need_node=1; fi; fi; "
                "if [ \"$need_node\" = 1 ]; then "
                f"curl -fsSL https://nodejs.org/dist/v{node_version}/"
                f"node-v{node_version}-linux-x64.tar.xz -o /tmp/node.tar.xz && "
                "tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && "
                "rm -f /tmp/node.tar.xz; fi; "
                f"npm install -g {shlex.quote(package)} && "
                "little-coder --no-update-check --version && "
                f"npm ls -g {shlex.quote(package)} --depth=0"
            ),
        )

    def _gateway_detection_snippet(self) -> str:
        """Bash that resolves the docker bridge gateway IP (host) address."""
        return (
            "GW=$(ip route show default 2>/dev/null | awk 'NR==1{print $3}'); "
            'if [ -z "$GW" ]; then '
            "hex=$(awk '$2==\"00000000\"{print $3; exit}' /proc/net/route); "
            'GW=$(printf "%d.%d.%d.%d" "0x${hex:6:2}" "0x${hex:4:2}" '
            '"0x${hex:2:2}" "0x${hex:0:2}"); fi; '
            f'URL="http://$GW:{_DEFAULT_GATEWAY_PORT}"'
        )

    def _model_ref_snippet(self) -> str:
        """Bash that maps $MODEL to a pi model ref (prefix ollama/ if bare)."""
        return (
            'case "$MODEL" in */*) MODEL_REF="$MODEL";; '
            '"") MODEL_REF="ollama/' + DEFAULT_MODEL + '";; '
            '*) MODEL_REF="ollama/$MODEL";; esac'
        )

    def _build_run_command(self, instruction: str) -> str:
        quoted_instruction = shlex.quote(instruction)
        url_resolution = (
            f'if [ -n "${OLLAMA_URL_ENV_VAR}" ]; then '
            f'URL="${OLLAMA_URL_ENV_VAR}"; else '
            f"{self._gateway_detection_snippet()}; fi; "
            # Normalize to a /v1 base URL for the provider override.
            'URL="${URL%/}"; '
            'case "$URL" in */v1) BASE="$URL";; *) BASE="$URL/v1";; esac; '
            f'export {PROVIDER_BASE_URL_ENV}="$BASE"; '
            f"{self._model_ref_snippet()}; "
            'if [ -n "${' + THINKING_ENV_VAR + '}" ]; then '
            f'THINKING_ARGS="--thinking ${{{THINKING_ENV_VAR}}}"; else '
            'THINKING_ARGS=""; fi'
        )
        env_log = (
            "printf 'model=%s\\nmodel_ref=%s\\nollama_url=%s\\nbase_url=%s\\n"
            "thinking=%s\\ntimeout_secs=%s\\n' "
            f"\"$MODEL\" \"$MODEL_REF\" \"$URL\" \"$BASE\" "
            f"\"${{{THINKING_ENV_VAR}:-unset}}\" "
            f"\"${{{TIMEOUT_ENV_VAR}:-unset}}\" "
            f"> {CONTAINER_ENV_LOG_PATH}"
        )
        return (
            # Explicit pipefail so a non-zero little-coder exit propagates
            # through `| tee` instead of being masked by tee's exit status.
            # Harbor's BaseInstalledAgent._exec already sets pipefail on the
            # outer shell; this keeps the invariant if the command is ever
            # executed outside that wrapper.
            "set -o pipefail; "
            f"{build_shell_secret_unset_snippet()}; "
            f"MODEL=${{{MODEL_ENV_VAR}}}; "
            f"{url_resolution}; "
            f"{env_log}; "
            "export LITTLE_CODER_NO_CTX_PROBE=1; "
            "little-coder --no-update-check --model \"$MODEL_REF\" "
            "$THINKING_ARGS "
            f"-p {quoted_instruction} 2>&1 | tee {CONTAINER_LOG_PATH}"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self._get_env(MODEL_ENV_VAR) or DEFAULT_MODEL
        thinking = self._get_env(THINKING_ENV_VAR) or DEFAULT_THINKING

        # Read bench-control vars from harbor's env sources (--ae lands in
        # _extra_env, NOT os.environ) and re-add the prefix that
        # _get_env_prefixed strips. Falls back to os.environ via _env_sources.
        little_env = {
            f"LITTLE_{k}": v for k, v in self._get_env_prefixed("LITTLE_").items()
        }
        forwarded_env: dict[str, str] = sanitize_env(
            {
                **little_env,
                MODEL_ENV_VAR: model,
                **self.extra_env,
            }
        )
        if thinking:
            forwarded_env[THINKING_ENV_VAR] = thinking
        explicit_url = self._get_env(OLLAMA_URL_ENV_VAR)
        if explicit_url:
            forwarded_env[OLLAMA_URL_ENV_VAR] = explicit_url
        timeout = self._get_env(TIMEOUT_ENV_VAR)
        if timeout:
            forwarded_env[TIMEOUT_ENV_VAR] = timeout

        metadata: dict[str, Any] = {
            "model": model,
            "thinking": thinking or "unset(pi default)",
            "ollama_url": explicit_url or "auto(gateway)",
        }

        try:
            await self.exec_as_agent(
                environment,
                command=self._build_run_command(instruction),
                env=forwarded_env,
            )
        finally:
            context.metadata = metadata
