import json

from groq import Groq
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agent.config import MAX_RETRIES, QUALITY_THRESHOLD
from agent.retrieval import retrieve
from agent.state import FinalReport, Finding, ResearchState
from backend.llm_client import groq_chat


def _llm_config(config: RunnableConfig) -> tuple[Groq, str, float]:
    """Every node needs the same three things out of config["configurable"]
    — pulled into one helper instead of repeating it four times."""
    cfg = config["configurable"]
    return cfg["client"], cfg["model"], cfg["temperature"]


# ---------------------------------------------------------------- planner --

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
    client, model, temperature = _llm_config(config)

    previous_critique = state.get("critique", "")
    user_prompt = build_planner_user_prompt(state["goal"], previous_critique)
    text, tokens = groq_chat(
        client, model, temperature, PLANNER_SYSTEM_PROMPT, user_prompt
    )

    try:
        tasks = json.loads(text)["tasks"]
    except (json.JSONDecodeError, KeyError):
        # A malformed response shouldn't kill the graph mid-run — degrade
        # to a line-based split instead of crashing.
        tasks = [
            line.strip("-• ").strip() for line in text.splitlines() if line.strip()
        ]

    return {"tasks": tasks, "tokens_used": tokens}


# -------------------------------------------------------------- researcher --

RAG_SYSTEM_PROMPT = """You are the Researcher in a research pipeline. You are
given a research task and excerpts retrieved from the user's uploaded
documents via similarity search.

Rules:
- Answer the task using ONLY the provided excerpts — do not draw on outside
  or general knowledge to fill gaps.
- Similarity search always returns *something*, even when nothing relevant
  exists in the documents, since it ranks by distance, not by a relevance
  threshold. So judge first whether the excerpts actually address the task
  before answering from them.
- If the excerpts do not meaningfully address the task, say so explicitly
  and plainly (e.g. "The provided sources do not cover this task.") — do
  not invent an answer from general knowledge to compensate.
- Be concise: 3 to 5 sentences.
"""

RESEARCH_FALLBACK_SYSTEM_PROMPT = """You are a research assistant. No source
documents are available for this task — answer from your own knowledge only.

Rules:
- Be concise and factual: 3 to 5 sentences.
- Do not fabricate specific statistics, dates, or citations you're not
  confident about.
- Write plainly — no preamble like "Based on my knowledge".
"""


def _build_rag_user_prompt(task: str, retrieved: list[Finding]) -> str:
    excerpts = "\n\n".join(
        f"[Excerpt from {f['source']}]\n{f['content']}" for f in retrieved
    )
    return f"Research task: {task}\n\nRetrieved excerpts:\n{excerpts}"


def researcher(state: ResearchState, config: RunnableConfig) -> dict:
    client, model, temperature = _llm_config(config)
    retriever = config["configurable"].get("retriever")  # None if no docs uploaded

    findings: list[Finding] = []
    total_tokens = 0

    for task in state["tasks"]:
        retrieved = retrieve(retriever, task) if retriever is not None else []

        if retrieved:
            # A retriever exists and returned chunks — similarity search
            # always returns *something* once the index is non-empty, a low
            # score doesn't mean an empty list. Ground the LLM's answer in
            # those chunks explicitly rather than using them as the finding
            # verbatim (no synthesis) or letting the model quietly fall back
            # to general knowledge if they're not actually relevant.
            text, tokens = groq_chat(
                client,
                model,
                temperature,
                RAG_SYSTEM_PROMPT,
                _build_rag_user_prompt(task, retrieved),
            )
            source_label = ", ".join(sorted({f["source"] for f in retrieved}))
        else:
            # No documents were uploaded for this session, or the index
            # came back genuinely empty. General knowledge, clearly labeled
            # as unverified rather than presented as sourced.
            text, tokens = groq_chat(
                client,
                model,
                temperature,
                RESEARCH_FALLBACK_SYSTEM_PROMPT,
                f"Research task: {task}",
            )
            source_label = "model knowledge (unverified)"

        total_tokens += tokens
        findings.append(Finding(task=task, content=text.strip(), source=source_label))

    return {"findings": findings, "tokens_used": total_tokens}


# ------------------------------------------------------------------ critic --

CRITIC_SYSTEM_PROMPT = """You are the Critic in a research pipeline. Evaluate whether the findings
gathered actually satisfy the research goal and cover every task.

Rules:
- quality_score is a float from 0.0 to 1.0: how well the findings cover the
  goal, are specific (not vague), and are consistent with each other.
- critique must name concrete gaps ("no data on X", "Task 3 wasn't addressed",
  "findings are too generic to be useful") — not generic praise or generic
  criticism. This critique is fed directly back into the Planner on a retry,
  so vague critique produces a vague re-plan.
- Output ONLY valid JSON, no prose before or after, in exactly this shape:
{"quality_score": 0.0, "critique": "..."}
"""


def critic(state: ResearchState, config: RunnableConfig) -> dict:
    client, model, temperature = _llm_config(config)

    findings_text = "\n".join(
        f"- [{f['task']}] ({f['source']}): {f['content']}" for f in state["findings"]
    )
    user_prompt = (
        f"Research goal: {state['goal']}\n\n"
        f"Tasks:\n" + "\n".join(f"- {t}" for t in state["tasks"]) + "\n\n"
        f"Findings:\n{findings_text}"
    )

    text, tokens = groq_chat(
        client, model, temperature, CRITIC_SYSTEM_PROMPT, user_prompt
    )

    try:
        parsed = json.loads(text)
        quality_score = float(parsed["quality_score"])
        critique = parsed["critique"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Fail conservative: an unparseable critic response counts as a
        # failing score, so the retry guardrail (not a silent bad pass)
        # decides what happens next.
        quality_score = 0.0
        critique = "Critic response could not be parsed as valid JSON; treating this pass as failing."

    return {"quality_score": quality_score, "critique": critique, "tokens_used": tokens}


# --------------------------------------------------------------- reporting --

REPORTER_SYSTEM_PROMPT = """You are the Reporting agent in a research pipeline. Given a research goal,
the findings gathered, and the Critic's final assessment, write a structured
summary.

Rules:
- key_findings must contain at least one item — short, specific, standalone
  statements, not restatements of the task list.
- risks should name real gaps or uncertainties in the research (e.g. missing
  recent data, single-source claims) — not generic disclaimers.
- Output ONLY valid JSON, no prose before or after, in exactly this shape:
{"summary": "2-4 sentence overview", "key_findings": ["...", "..."], "risks": ["...", "..."]}
"""


def _confidence_from_score(score: float) -> str:
    if score >= QUALITY_THRESHOLD:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def render_report_markdown(report: FinalReport, below_threshold: bool) -> str:
    lines = [f"# Research report: {report.goal}", ""]

    if below_threshold:
        lines += [
            f"> ⚠️ Accepted after reaching the retry limit ({MAX_RETRIES} retries), "
            f"with a quality score below the {QUALITY_THRESHOLD} threshold.",
            "",
        ]

    lines += [report.summary, "", "## Key findings"]
    lines += [f"- {kf}" for kf in report.key_findings]

    lines += ["", "## Risks / open gaps"]
    lines += [f"- {r}" for r in report.risks] or ["- None noted."]

    lines += [
        "",
        f"## Confidence: {report.confidence}",
        f"*(iteration {report.iterations})*",
    ]

    lines += ["", "## Sources"]
    lines += [f"- {s}" for s in report.sources] or ["- None."]

    return "\n".join(lines)


def reporter(state: ResearchState, config: RunnableConfig) -> dict:
    client, model, temperature = _llm_config(config)

    findings_text = "\n".join(
        f"- [{f['task']}] {f['content']} (source: {f['source']})"
        for f in state["findings"]
    )
    user_prompt = (
        f"Goal: {state['goal']}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Critic's final assessment (score {state['quality_score']:.2f}): {state['critique']}"
    )

    text, tokens = groq_chat(
        client, model, temperature, REPORTER_SYSTEM_PROMPT, user_prompt
    )

    sources = sorted({f["source"] for f in state["findings"]})
    iterations = state["retry_count"] + 1
    confidence = _confidence_from_score(state["quality_score"])

    try:
        parsed = json.loads(text)
        report = FinalReport(
            goal=state["goal"],
            summary=parsed["summary"],
            key_findings=parsed.get("key_findings")
            or [f["content"][:200] for f in state["findings"][:3]],
            risks=parsed.get("risks", []),
            sources=sources,
            iterations=iterations,
            confidence=confidence,
        )
    except (json.JSONDecodeError, KeyError, ValidationError):
        # Defensive fallback so a malformed LLM response never crashes the
        # graph on its last node — build a minimal but valid report
        # straight from the raw findings instead of failing the whole run.
        report = FinalReport(
            goal=state["goal"],
            summary="Automated summary unavailable; showing raw findings below.",
            key_findings=[f["content"][:200] for f in state["findings"][:5]]
            or ["No findings were gathered."],
            risks=[
                "Report generation failed to parse a structured summary from the model."
            ],
            sources=sources,
            iterations=iterations,
            confidence=confidence,
        )

    below_threshold = state["quality_score"] < QUALITY_THRESHOLD
    markdown = render_report_markdown(report, below_threshold)

    return {"report": markdown, "tokens_used": tokens}
