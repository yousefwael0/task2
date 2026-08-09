import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Finding(TypedDict):
    """One retrieved-or-generated piece of evidence for a single task.
    Canonical definition — agent/retrieval.py imports this rather than
    defining its own, so there's one shape, not two that can drift."""

    task: str
    content: str
    source: str


class ResearchState(TypedDict):
    goal: str  # the user's research objective
    tasks: list[str]  # produced by the Planner
    findings: list[Finding]  # produced by the Researcher
    critique: str  # the Critic's written feedback / gaps
    quality_score: float  # 0.0 - 1.0, produced by the Critic
    retry_count: int  # incremented on every loop back
    report: str  # final structured output, rendered to markdown after
    # validating through FinalReport — see nodes.reporter
    tokens_used: Annotated[int, operator.add]  # summed across every node call,
    # including retries — see agent/nodes.py


class FinalReport(BaseModel):
    goal: str
    summary: str
    key_findings: list[str] = Field(min_length=1)
    risks: list[str]
    sources: list[str]
    iterations: int
    confidence: str = Field(description="high | medium | low")
