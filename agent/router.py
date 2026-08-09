"""
The conditional edge that runs after the Critic and decides whether the
graph approves the current draft or loops back to re-plan.

route_after_critic returns one of two node names: "reporting" (approve
— either the score passed, or the retry cap was hit) or
"increment_retry" (loop back).

increment_retry is a one-line bookkeeping node, not the Planner itself.
LangGraph conditional edges can only choose a destination — they can't
also return a state update — so bumping retry_count needs an actual
node sitting on the retry path. graph.py wires increment_retry -> planner
as a normal edge right after it.
"""

from agent.config import MAX_RETRIES, QUALITY_THRESHOLD
from agent.state import ResearchState


def route_after_critic(state: ResearchState) -> str:
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "reporting"

    if state["retry_count"] >= MAX_RETRIES:
        # Retry cap reached with a still-failing score — approve anyway so
        # the graph terminates instead of looping forever. reporter()
        # checks quality_score against QUALITY_THRESHOLD itself and flags
        # this exact case in the rendered report.
        return "reporting"

    return "increment_retry"


def increment_retry(state: ResearchState) -> dict:
    return {"retry_count": state["retry_count"] + 1}
