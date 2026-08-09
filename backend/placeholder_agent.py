"""
Stand-in for the real LangGraph agent stream.

TODO(agent): once agent/graph.py exposes build_graph(), replace the
body of run() with something like:

    from agent.graph import build_graph
    from backend import session_store

    _graph = build_graph()  # compile once at import time, not per-request

    async def run(session_id, goal, model, temperature):
        config = {
            "configurable": {
                "model": model,
                "temperature": temperature,
                "retriever": session_store.get(session_id),
            }
        }
        initial_state = {"goal": goal, "critique": "", "retry_count": 0, "tokens_used": 0}
        async for event in _graph.astream(initial_state, config=config):
            node_name, output = next(iter(event.items()))
            yield {"node": node_name, "output": output}

Nothing calling run() needs to change when that swap happens — same
async generator interface, same {"node", "output"} event shape.

Hardcoded below as a run that deliberately fails its first quality
check and retries once, matching the exact sequence the frontend was
built and rehearsed against — so the retry UI is already verified
end-to-end over real HTTP before the real agent exists.
"""

import asyncio


async def run(session_id: str, goal: str, model: str, temperature: float):
    events = [
        {
            "node": "planner",
            "output": {
                "tasks": [
                    "Survey background on the topic",
                    "Identify recent developments",
                    "Compare leading approaches",
                ],
                "tokens_used": 310,
            },
        },
        {
            "node": "researcher",
            "output": {
                "findings": [
                    {
                        "task": "Survey background on the topic",
                        "content": "Placeholder finding: background context.",
                        "source": "placeholder.pdf",
                    },
                    {
                        "task": "Identify recent developments",
                        "content": "Placeholder finding: partial recent coverage.",
                        "source": "web search",
                    },
                ],
                "tokens_used": 640,
            },
        },
        {
            "node": "critic",
            "output": {
                "quality_score": 0.55,
                "critique": "Recent developments are only partially covered, and no comparison across approaches was done.",
                "tokens_used": 180,
            },
        },
        {"node": "decision", "output": {"next": "planner", "retry_count": 1}},
        {
            "node": "planner",
            "output": {
                "tasks": [
                    "Find 2025-2026 developments specifically",
                    "Directly compare top 2 approaches",
                    "Re-verify background claims",
                ],
                "tokens_used": 290,
            },
        },
        {
            "node": "researcher",
            "output": {
                "findings": [
                    {
                        "task": "Find 2025-2026 developments specifically",
                        "content": "Placeholder finding: recent development detail.",
                        "source": "placeholder.pdf",
                    },
                    {
                        "task": "Directly compare top 2 approaches",
                        "content": "Placeholder finding: comparison detail.",
                        "source": "web search",
                    },
                ],
                "tokens_used": 710,
            },
        },
        {
            "node": "critic",
            "output": {
                "quality_score": 0.86,
                "critique": "Coverage is now sufficient across background, recent developments, and comparison.",
                "tokens_used": 175,
            },
        },
        {"node": "decision", "output": {"next": "reporting", "retry_count": 1}},
        {
            "node": "reporting",
            "output": {
                "report": {
                    "title": f"Research report: {goal}",
                    "summary": "Placeholder summary tying together the findings above.",
                    "sections": [
                        {
                            "heading": "Background",
                            "content": "Placeholder background section content.",
                        },
                        {
                            "heading": "Recent developments",
                            "content": "Placeholder recent-developments section content.",
                        },
                        {
                            "heading": "Comparison",
                            "content": "Placeholder comparison section content.",
                        },
                    ],
                    "sources": ["placeholder.pdf", "web search"],
                },
                "tokens_used": 420,
            },
        },
    ]
    for event in events:
        await asyncio.sleep(0.4)
        yield event
