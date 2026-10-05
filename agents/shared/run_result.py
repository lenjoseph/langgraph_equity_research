"""Turn an LLM call into an AgentRunResult."""

import time
from typing import Optional

from agents.shared.agent_utils import LLMCall
from models.metrics import AgentMetrics
from models.result import AgentRunResult, AgentStatus


def finalize_llm_call(
    agent_name: str,
    model: str,
    started: float,
    call: LLMCall,
    token_budget: Optional[int],
) -> AgentRunResult:
    """Attach latency and budget flags without dropping a parsed payload."""
    budget_exceeded = bool(token_budget and call.usage.total_tokens > token_budget)
    if call.error and "budget" in call.error.lower():
        budget_exceeded = True
    metrics = AgentMetrics(
        agent_name=agent_name,
        latency_ms=(time.perf_counter() - started) * 1000,
        token_usage=call.usage,
        model=model,
        budget_exceeded=budget_exceeded,
    )
    status = call.status
    if call.result is None and status == AgentStatus.OK:
        status = AgentStatus.FAILED
    elif budget_exceeded and status == AgentStatus.OK and call.result is not None:
        status = AgentStatus.DEGRADED
    return AgentRunResult(
        status=status,
        output=call.result,
        error=call.error,
        metrics=metrics,
    )
