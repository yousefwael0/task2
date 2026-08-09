"""
Owns the one piece of backend-wide mutable state: the Groq API key,
settable at runtime from the sidebar's "Groq API key" field.

Deliberately kept separate from llm_client.py, which stays a pure,
side-effect-free factory module (see its docstring) — this is the one
file allowed to mutate process state, and it owns exactly one value.

Design choice: the override writes straight into os.environ, so
llm_client._require_api_key() (which reads GROQ_API_KEY from the
environment) picks it up with no changes on its end. Unlike model or
temperature — chosen per research run, passed through RunRequest —
the API key is treated as one shared credential for the whole running
backend process, matching how it's actually set: once, via .env or
the sidebar, not per run.
"""

import os


def get_active_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def set_override_key(new_key: str) -> None:
    if new_key:
        os.environ["GROQ_API_KEY"] = new_key


def has_key() -> bool:
    return bool(get_active_key())


def masked_key() -> str:
    key = get_active_key()
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
