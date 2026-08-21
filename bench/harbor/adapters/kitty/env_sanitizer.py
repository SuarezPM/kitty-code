"""Sanitization of environment variables forwarded to the kitty process.

Ported from Apohara-Catalyst
(`crates/apohara-sandbox/src/runner/imp.rs:is_secret_env_name`) so benchmark
containers never receive credentials from the host environment — most notably
``KITTY_CLOUD_API_KEY``, which would enable paid cloud fallback inside trials.
"""

from __future__ import annotations

import re

# Uppercased-name suffixes that always mark a secret.
_SECRET_SUFFIXES: tuple[str, ...] = (
    "_API_KEY",
    "_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
)

# Uppercased-name prefixes for known credential-bearing providers.
_SECRET_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_",
    "OPENAI_",
    "GROQ_",
    "TOGETHER_",
    "MISTRAL_",
    "OPENROUTER_",
    "GEMINI_",
    "GOOGLE_",
    "COHERE_",
    "XAI_",
    "DEEPSEEK_",
    "PERPLEXITY_",
    "FIREWORKS_",
    "PINECONE_",
    "VOYAGE_",
    "REPLICATE_",
    "HUGGINGFACE_",
    "AWS_",
    "GCP_",
    "GCLOUD_",
    "AZURE_",
    "CLOUDFLARE_",
    "DIGITALOCEAN_",
    "VERCEL_",
    "NETLIFY_",
    "HEROKU_",
    "RENDER_",
    "SUPABASE_",
    "STRIPE_",
    "SENTRY_",
    "CIRCLE_",
    "GITLAB_",
    "BITBUCKET_",
)

# Exact names that are secrets despite not matching a prefix/suffix.
_SECRET_EXACT: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "FLY_API_TOKEN",
        "RAILWAY_TOKEN",
        "NPM_TOKEN",
        "DATABASE_URL",
        "MONGODB_URI",
        "MYSQL_URL",
        "POSTGRES_URL",
        "REDIS_URL",
        "NOTION_TOKEN",
        "SLACK_TOKEN",
        "SLACK_BOT_TOKEN",
        "DISCORD_TOKEN",
        "TELEGRAM_TOKEN",
        "LINEAR_API_KEY",
    }
)


def is_secret_env_name(name: str) -> bool:
    """Return True when *name* looks like a credential and must be stripped."""
    upper = name.upper()
    if any(upper.startswith(prefix) for prefix in _SECRET_PREFIXES):
        return True
    if any(upper.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return True
    return upper in _SECRET_EXACT


def sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of *env* with every secret-looking variable removed."""
    return {k: v for k, v in env.items() if not is_secret_env_name(k)}


def build_shell_secret_unset_snippet() -> str:
    """Build a bash snippet that unsets secret-looking vars already present in
    the container's own environment (image ENV, compose-injected vars, etc.).

    Uses the same predicate as :func:`is_secret_env_name`, compiled to a
    POSIX extended regex evaluated inside the container shell.
    """
    alternatives = [f"^{re.escape(p)}" for p in _SECRET_PREFIXES]
    alternatives += [f"{re.escape(s)}$" for s in _SECRET_SUFFIXES]
    alternatives += [f"^{re.escape(name)}$" for name in sorted(_SECRET_EXACT)]
    pattern = "|".join(alternatives)
    return (
        f"_kitty_secrets=$(env | sed 's/=.*//' | grep -E '{pattern}' || true); "
        "if [ -n \"$_kitty_secrets\" ]; then unset $_kitty_secrets; fi; "
        "unset _kitty_secrets"
    )
