from langgraph.graph import END, StateGraph

from agent.nodes import critic, planner, reporter, researcher
from agent.router import increment_retry, route_after_critic
from agent.state import ResearchState


def build_graph():
    """Compile once (e.g. at backend startup) and reuse across requests —
    compiling is not free, and nothing about the compiled graph is
    request-specific. Per-request values (model, temperature, retriever)
    go through RunnableConfig at .astream() time, not through this
    function — see backend/placeholder_agent.py's swap-plan comment."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("critic", critic)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("reporting", reporter)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"increment_retry": "increment_retry", "reporting": "reporting"},
    )

    graph.add_edge("increment_retry", "planner")
    graph.add_edge("reporting", END)

    return graph.compile()
