"""
In-memory retriever store, keyed by the frontend's session_id.

A plain dict is the entire implementation, deliberately. This backend
runs as a single process for the course submission — no need for
Redis or a real session backend. Retrievers are lost on server
restart, which is fine for a demo run; don't add persistence here
unless the assignment actually asks for cross-restart sessions.
"""

from agent.retrieval import Retriever

_retrievers: dict[str, Retriever] = {}


def get(session_id: str) -> Retriever | None:
    return _retrievers.get(session_id)


def set(session_id: str, retriever: Retriever) -> None:
    _retrievers[session_id] = retriever
