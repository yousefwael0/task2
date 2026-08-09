# agent/state.py
import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchState(TypedDict):
    goal: str  # the user's research objective
    tasks: list[str]  # produced by the Planner
    findings: list[str]  # produced by the Researcher
    sources: list[str]  # grounded sources produced by the Researcher
    critique: str  # the Critic's written feedback / gaps
    quality_score: float  # 0.0 - 1.0, produced by the Critic
    retry_count: int  # incremented on every loop back
    report: str  # final structured output
    tokens_used: Annotated[int, operator.add]


class FinalReport(BaseModel):
    goal: str
    summary: str
    key_findings: list[str] = Field(min_length=1)
    risks: list[str]
    sources: list[str]
    iterations: int
    confidence: str = Field(description="high | medium | low")
