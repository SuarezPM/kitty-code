"""Harbor agent adapter that runs Kitty Code (``kitty-cli``) as the solving
agent inside Harbor task containers.

Import path (run harbor with ``PYTHONPATH=<repo>/bench/harbor``)::

    harbor run -p <tasks> -a adapters.kitty.agent:KittyAgent -k 1

The model is parameterized through the ``KITTY_BENCH_MODEL`` environment
variable on the host (default ``ornith:9b``); no code changes are needed to
switch models.
"""

from .agent import KittyAgent
from .env_sanitizer import is_secret_env_name, sanitize_env

__all__ = ["KittyAgent", "is_secret_env_name", "sanitize_env"]
