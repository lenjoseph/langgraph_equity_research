import time
from typing import Tuple

import dotenv

from agents.aggregation.prompt import research_aggregation_prompt
from models.state import EquityResearchState
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from models.metrics import AgentMetrics


dotenv.load_dotenv()

AGENT_NAME = "aggregation"


def get_aggregated_sentiment(
    state: EquityResearchState, iteration: int = 1
) -> Tuple[str, AgentMetrics]:
    """
    Aggregate sentiment from all research agents.

    Args:
        state: The current equity research state
        iteration: The iteration number (1-based) for the aggregation loop

    Returns:
        Tuple of (aggregated sentiment string, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]

    if state.feedback:
        prompt = f"Your original response: {state.combined_sentiment}. Revise your response based on this feedback: {state.feedback}"
    else:
        prompt = f"{research_aggregation_prompt}\n\nAggregate the following equity research:\n\n"
        prompt += f"Ticker: {state.ticker}\n"
        prompt += f"Trade Duration: {state.trade_duration.value}\n"
        prompt += f"Trade Direction: {state.trade_direction.value}"
        prompt += f"Fundamental Analysis:\n{state.fundamental_sentiment}\n\n"
        prompt += f"Technical Analysis:\n{state.technical_sentiment}\n\n"
        prompt += f"Macro Analysis:\n{state.macro_sentiment}\n\n"
        prompt += f"Peer Analysis:\n{state.peer_sentiment}\n\n"
        prompt += f"Industry Analysis:\n{state.industry_sentiment}\n\n"
        prompt += f"Headline Analysis:\n{state.headline_sentiment}\n\n"
        prompt += f"SEC Filings Analysis:\n{state.filings_sentiment}\n\n"

    llm = get_openai_llm(model=model, temperature=0.2)
    result, token_usage = run_agent_with_tools(llm, prompt, track_tokens=True)

    latency_ms = (time.perf_counter() - start_time) * 1000
    # Include iteration in agent name for tracking multiple loops
    agent_name = f"{AGENT_NAME}_{iteration}" if iteration > 1 else AGENT_NAME
    metrics = AgentMetrics(
        agent_name=agent_name,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
    )
    return result, metrics
