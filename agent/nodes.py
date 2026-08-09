#agent/nodes.py
import json

from groq import Groq
from langchain_core.runnables import RunnableConfig

from agent.state import FinalReport, ResearchState
from backend.llm_client import groq_chat

PLANNER_SYSTEM_PROMPT = """You are the Planner in a research pipeline: Planner -> Researcher -> Critic -> Decision -> Reporting.

Your job: break a research goal into concrete, ordered research tasks for the Researcher to execute.

Rules:
- Produce 3 to 5 tasks. Each task must be specific and actionable (something a researcher could search or look up directly) — not a restatement of the goal.
- Order tasks logically: foundational/background tasks before narrower or comparative ones.
- If you are given critique from a previous attempt, you are REVISING a rejected plan, not writing a fresh one. Your new tasks must visibly target the specific gaps named in the critique. Do not repeat a task from the previous plan unless the critique explicitly says that task was fine — reuse without change is a failure.
- Output ONLY valid JSON, no prose before or after, in exactly this shape:
{"tasks": ["task 1", "task 2", "task 3"]}
"""


def build_planner_user_prompt(goal: str, previous_critique: str) -> str:
    if previous_critique:
        return (
            f"Research goal: {goal}\n\n"
            f"This is a RETRY. The previous plan was rejected. Critique from the Critic:\n"
            f"{previous_critique}\n\n"
            "Revise the plan so it directly addresses every gap named above."
        )
    return f"Research goal: {goal}\n\nThis is the first attempt. Produce the initial task list."


def planner(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    client: Groq = cfg["client"]
    model: str = cfg["model"]
    temperature: float = cfg["temperature"]

    previous_critique = state.get("critique", "")
    user_prompt = build_planner_user_prompt(state["goal"], previous_critique)

    text, tokens = groq_chat(
        client, model, temperature, PLANNER_SYSTEM_PROMPT, user_prompt
    )

    try:
        tasks = json.loads(text)["tasks"]
    except (json.JSONDecodeError, KeyError):
        tasks = [
            line.strip("-• ").strip() for line in text.splitlines() if line.strip()
        ]

    return {"tasks": tasks, "tokens_used": tokens}


def researcher(state: ResearchState):
    # TODO research each task in the list of tasks and return findings in state.findings[]
    pass


def critic(state: ResearchState):
    # TODO critique the findings against the sources and return a state.critique string and a state.quality_score
    pass


def reporter(state: ResearchState):
    # TODO return the final resarch findings in a FinalReport Object
    report = FinalReport()
    return report
