"""Structured outcomes for agent executions."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.metrics import AgentMetrics


class AgentStatus(str, Enum):
    """Whether an agent produced a usable analysis."""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


class AgentRunResult(BaseModel):
    """Result of one agent run, including a payload when parsing succeeded."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: AgentStatus
    output: Any = None
    error: Optional[str] = None
    metrics: AgentMetrics


class GroundingSource(BaseModel):
    """A source returned by search grounding."""

    title: str = ""
    uri: str = ""


def merge_status(
    left: Optional[dict[str, str]], right: Optional[dict[str, str]]
) -> dict[str, str]:
    """Merge per-agent status maps written by parallel graph nodes."""
    return {**(left or {}), **(right or {})}
