"""
Groq client factory and chat helper.

Nothing in this module runs on import — no network calls, no prompts,
no env mutation. Every function here is called explicitly, at a known
point in the backend's lifecycle:

    - get_available_models() -> called once, in FastAPI's lifespan startup
    - build_llm(model, temperature) -> called per /run request, using
      that request's sidebar selections
    - groq_chat(...) -> called by individual agent nodes

This keeps the module safe to import from tests, scripts, or the agent
package without side effects, and avoids a shared mutable "active model"
that concurrent requests could stomp on.
"""

import os

from groq import Groq
from langchain_groq import ChatGroq

from agent.config import DEFAULT_MODEL

FALLBACK_MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY isn't set. Caught at the API boundary,
    not swallowed here — a server should fail loudly and early, not
    prompt interactively for a key it can't get from a terminal."""


def get_api_key():
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise MissingAPIKeyError
    return key


def get_available_models() -> list[str]:
    """Fetch and sort model IDs from the Groq API.

    Called once at backend startup. Falls back to a static list if the
    API call fails, so a transient network issue doesn't prevent the
    server from starting.
    """
    try:
        client = Groq(api_key=get_api_key())
        models = client.models.list()
        return sorted(model.id for model in models.data)
    except Exception as exc:
        print(f"[llm_client] Failed to fetch models dynamically: {exc}")
        return FALLBACK_MODELS


def resolve_default_model(available_models: list[str]) -> str:
    """Pick a safe default from an available-models list.

    Falls back to the first available model rather than a hardcoded
    index, which breaks on short lists.
    """
    if DEFAULT_MODEL in available_models:
        return DEFAULT_MODEL
    if available_models:
        return available_models[0]
    return DEFAULT_MODEL


def build_llm(model: str, temperature: float) -> ChatGroq:
    """Construct a LangChain ChatGroq client for one request.

    Called fresh per /run call with that request's sidebar-selected
    model and temperature — never shared or mutated globally, so
    concurrent sessions with different selections can't collide.
    """
    return ChatGroq(model=model, temperature=temperature)


def groq_chat(
    client: Groq,
    model: str,
    temperature: float,
    system: str,
    user: str,
) -> tuple[str, int]:
    """One raw Groq call (bypassing LangChain). Returns (text, tokens_used).

    Used where a node wants direct control over the request rather than
    going through the LangChain wrapper.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens
