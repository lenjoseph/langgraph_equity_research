import time
from typing import Optional

import dotenv

from agents.aggregation.prompt import research_aggregation_prompt
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.result import AgentRunResult
from models.state import EquityResearchState


dotenv.load_dotenv()

AGENT_NAME = "aggregation"


def _enum_text(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def specialist_research_block(state) -> str:
    """Format every specialist analysis so later revisions stay grounded."""
    duration = _enum_text(state.trade_duration)
    direction = _enum_text(state.trade_direction)
    return (
        f"Ticker: {state.ticker}\n"
        f"Trade Duration: {duration}\n"
        f"Trade Direction: {direction}\n\n"
        f"Fundamental Analysis:\n{state.fundamental_sentiment}\n\n"
        f"Technical Analysis:\n{state.technical_sentiment}\n\n"
        f"Macro Analysis:\n{state.macro_sentiment}\n\n"
        f"Peer Analysis:\n{state.peer_sentiment}\n\n"
        f"Industry Analysis:\n{state.industry_sentiment}\n\n"
        f"Headline Analysis:\n{state.headline_sentiment}\n\n"
        f"SEC Filings Analysis:\n{state.filings_sentiment}\n\n"
    )


def build_aggregation_prompt(state: EquityResearchState) -> str:
    """Build the aggregator prompt, keeping specialist research on every revision."""
    research = specialist_research_block(state)
    if state.feedback:
        return (
            f"{research_aggregation_prompt}\n\n"
            f"Specialist research to use as the only source of facts:\n\n{research}"
            f"Your previous synthesis:\n{state.combined_sentiment}\n\n"
            f"Revise that synthesis using this feedback:\n{state.feedback}\n"
        )
    return (
        f"{research_aggregation_prompt}\n\n"
        f"Aggregate the following equity research:\n\n{research}"
    )


def get_aggregated_sentiment(
    state: EquityResearchState,
    iteration: int = 1,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """
    Aggregate sentiment from all research agents.

    Args:
        state: The current equity research state
        iteration: The iteration number (1-based) for the aggregation loop
        token_config: Optional token configuration for this agent

    Returns:
        AgentRunResult whose output is the synthesis text when generation succeeded
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.aggregation
    model = LLM_MODELS["open_ai_smart"]
    prompt = build_aggregation_prompt(state)

    llm = get_openai_llm(
        model=model,
        temperature=0.2,
        max_tokens=config.max_output_tokens,
    )
    call = run_agent_with_tools(
        llm, prompt, track_tokens=True, token_budget=config.token_budget
    )
    result = finalize_llm_call(AGENT_NAME, model, start_time, call, config.token_budget)
    if iteration > 1:
        result.metrics.agent_name = f"{AGENT_NAME}_{iteration}"
    return result
