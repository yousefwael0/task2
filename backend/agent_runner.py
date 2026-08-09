"""
Runs the compiled LangGraph research agent and yields SSE-shaped events.

Replaces backend/placeholder_agent.py now that agent/graph.py exists.
Same async generator interface, same {"node", "output"} event shape —
routes.py only needed its import and one function call swapped, not
its SSE-encoding logic.
"""

from groq import Groq

from agent.graph import build_graph
from backend import runtime_config, session_store

# Compiled once, when this module is first imported (i.e. once per
# process, not once per request) — compiling a StateGraph isn't free,
# and nothing about the compiled graph is request-specific. Building
# it doesn't need a Groq key (it's pure structure: nodes and edges),
# so this is safe even before one's configured.
_graph = build_graph()


async def run(session_id: str, goal: str, model: str, temperature: float):
    if not runtime_config.has_key():
        # Yielded as an event, not raised — an exception raised mid-async-
        # generator after a StreamingResponse has already started sending
        # can leave the client with a broken, silently-truncated stream
        # instead of a readable message. See frontend's fallback node
        # handling for how this surfaces in the UI.
        yield {
            "node": "error",
            "output": {"message": "GROQ_API_KEY is not set on the backend."},
        }
        return

    client = Groq(api_key=runtime_config.get_active_key())
    retriever = session_store.get(session_id)  # None if nothing uploaded this session

    config = {
        "configurable": {
            "client": client,
            "model": model,
            "temperature": temperature,
            "retriever": retriever,
        }
    }

    # Seed every key ResearchState declares — LangGraph doesn't default
    # missing keys, and agent/nodes.py reads several of these with plain
    # indexing (state["goal"], state["retry_count"], ...) on the very
    # first pass, before any node has had a chance to write them.
    initial_state = {
        "goal": goal,
        "tasks": [],
        "findings": [],
        "critique": "",
        "quality_score": 0.0,
        "retry_count": 0,
        "report": "",
        "tokens_used": 0,
    }

    async for event in _graph.astream(initial_state, config=config):
        node_name, output = next(iter(event.items()))
        yield {"node": node_name, "output": output}
