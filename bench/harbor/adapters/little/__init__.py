"""Harbor agent adapter that runs little-coder headless as the solving
agent inside Harbor task containers.

Import path (run harbor with ``PYTHONPATH=<repo>/bench/harbor``)::

    harbor run -p <tasks> -a adapters.little.agent:LittleAgent -k 1

The model is parameterized through the ``LITTLE_BENCH_MODEL`` environment
variable on the host (default ``ornith-1.5:9b``); no code changes are needed
to switch models.
"""

from .agent import LittleAgent
from .env_sanitizer import is_secret_env_name, sanitize_env

__all__ = ["LittleAgent", "is_secret_env_name", "sanitize_env"]
