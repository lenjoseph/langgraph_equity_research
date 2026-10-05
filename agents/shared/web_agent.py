"""Search-grounded research agents."""

import time

from agents.shared.agent_utils import invoke_llm_with_metrics
from agents.shared.grounding import apply_grounding_filter
from agents.shared.llm_models import get_google_llm
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import AgentTokenConfig
from models.result import AgentRunResult


def run_search_grounded_agent(
    *,
    agent_name: str,
    model: str,
    prompt: str,
    output_schema: type,
    token_config: AgentTokenConfig,
) -> AgentRunResult:
    """Call Gemini with search grounding and keep only citations that match returned chunks."""
    started = time.perf_counter()
    llm = get_google_llm(
        model=model,
        temperature=0.0,
        with_search_grounding=True,
        max_tokens=token_config.max_output_tokens,
    )
    call = invoke_llm_with_metrics(
        llm,
        prompt,
        output_schema,
        token_budget=token_config.token_budget,
    )
    call = apply_grounding_filter(call)
    return finalize_llm_call(
        agent_name, model, started, call, token_config.token_budget
    )
