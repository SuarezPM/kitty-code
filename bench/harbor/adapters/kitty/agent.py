"""KittyAgent — Harbor adapter that solves tasks with Kitty Code.

Runs ``kitty-cli`` inside the task container against the host's Ollama
instance. Design notes:

- **Binary delivery**: task images are Ubuntu 22.04 based (glibc 2.35), so a
  host-built binary fails with ``GLIBC_2.39 not found``. Use
  ``build-binary.sh`` once to produce a jammy-compatible binary at
  ``bench/harbor/adapters/kitty/bin/kitty-cli``; it is copied into each
  container via ``environment.upload_file`` (harbor has no bind-mount API for
  agents).
- **Networking**: Ollama listens on the host; containers reach it through the
  docker bridge gateway IP (requires host Ollama bound beyond loopback, e.g.
  ``OLLAMA_HOST=0.0.0.0``, plus a firewall allow rule for docker subnets).
  Set ``KITTY_OLLAMA_URL`` to skip gateway autodetection.
- **Model parameterization**: ``KITTY_BENCH_MODEL`` (default ``ornith:9b``)
  selects the model at run time — no code changes between runs.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .env_sanitizer import build_shell_secret_unset_snippet, sanitize_env

AGENT_NAME = "kitty"
DEFAULT_MODEL = "ornith:9b"
DEFAULT_PROFILE = "medium"

MODEL_ENV_VAR = "KITTY_BENCH_MODEL"
PROFILE_ENV_VAR = "KITTY_BENCH_PROFILE"
OLLAMA_URL_ENV_VAR = "KITTY_OLLAMA_URL"
BINARY_ENV_VAR = "KITTY_BENCH_BINARY"

CONTAINER_BINARY_PATH = "/usr/local/bin/kitty-cli"
CONTAINER_LOG_PATH = "/logs/agent/kitty.txt"
CONTAINER_ENV_LOG_PATH = "/logs/agent/kitty-env.txt"

_DEFAULT_GATEWAY_PORT = 11434


class KittyAgent(BaseInstalledAgent):
    """Installs and runs the kitty-cli binary inside the task container."""

    @staticmethod
    @override
    def name() -> str:
        return AGENT_NAME

    @override
    def get_version_command(self) -> str:
        return f"{CONTAINER_BINARY_PATH} --version"

    def _resolve_host_binary(self) -> Path:
        """Locate the kitty-cli binary to upload into the container."""
        candidates: list[Path] = []
        override_path = self._get_env(BINARY_ENV_VAR)
        if override_path:
            candidates.append(Path(override_path).expanduser())
        adapter_root = Path(__file__).resolve().parent
        candidates.append(adapter_root / "bin" / "kitty-cli")
        # Host-native build (only works in containers whose glibc is new
        # enough; kept as last-resort fallback).
        candidates.append(
            adapter_root.parents[3] / "target" / "release" / "kitty-cli"
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "kitty-cli binary not found. Build it once with "
            "`bash bench/harbor/adapters/kitty/build-binary.sh` or point "
            f"{BINARY_ENV_VAR} at an existing glibc<=2.35 build. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        host_binary = self._resolve_host_binary()
        self.logger.info("Uploading kitty binary %s", host_binary)
        await environment.upload_file(host_binary, CONTAINER_BINARY_PATH)
        await self.exec_as_root(
            environment,
            command=f"chmod 755 {shlex.quote(CONTAINER_BINARY_PATH)}",
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

    def _build_run_command(self, instruction: str) -> str:
        quoted_instruction = shlex.quote(instruction)
        url_resolution = (
            f'if [ -n "${OLLAMA_URL_ENV_VAR}" ]; then '
            f'URL="${OLLAMA_URL_ENV_VAR}"; else '
            f"{self._gateway_detection_snippet()}; fi"
        )
        env_log = (
            "printf 'model=%s\\nprofile=%s\\nollama_url=%s\\n' "
            f"\"${{{MODEL_ENV_VAR}}}\" \"${{{PROFILE_ENV_VAR}}}\" \"$URL\" "
            f"> {CONTAINER_ENV_LOG_PATH}"
        )
        return (
            # Explicit pipefail so a non-zero kitty-cli exit propagates
            # through `| tee` instead of being masked by tee's exit status.
            # Harbor's BaseInstalledAgent._exec already sets pipefail on the
            # outer shell; this keeps the invariant if the command is ever
            # executed outside that wrapper.
            "set -o pipefail; "
            f"{build_shell_secret_unset_snippet()}; "
            f"{url_resolution}; "
            f"{env_log}; "
            f"{CONTAINER_BINARY_PATH} "
            f"--url \"$URL\" "
            f"--model \"${{{MODEL_ENV_VAR}}}\" "
            f"run --profile \"${{{PROFILE_ENV_VAR}}}\" "
            f"{quoted_instruction} 2>&1 | tee {CONTAINER_LOG_PATH}"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self._get_env(MODEL_ENV_VAR) or DEFAULT_MODEL
        profile = self._get_env(PROFILE_ENV_VAR) or DEFAULT_PROFILE

        forwarded_env: dict[str, str] = sanitize_env(
            {
                MODEL_ENV_VAR: model,
                PROFILE_ENV_VAR: profile,
                **self.extra_env,
            }
        )
        explicit_url = self._get_env(OLLAMA_URL_ENV_VAR)
        if explicit_url:
            forwarded_env[OLLAMA_URL_ENV_VAR] = explicit_url

        metadata: dict[str, Any] = {
            "model": model,
            "profile": profile,
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
