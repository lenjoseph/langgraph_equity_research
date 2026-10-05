import time
from typing import Any, Dict, Optional

import dotenv

from agents.fundamentals.prompt import fundamentals_research_prompt
from agents.fundamentals.tools import (
    get_earnings_and_financial_health,
    get_fundamentals_tool,
)
from agents.shared.agent_utils import invoke_llm_with_metrics, run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import FundamentalSentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "fundamental"


def get_fundamental_sentiment(
    ticker: str,
    cached_info: Optional[Dict[str, Any]] = None,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Generate fundamental sentiment analysis for a ticker."""
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.fundamental
    model = LLM_MODELS["open_ai_smart"]
    llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )

    if cached_info is not None:
        fundamentals_data = get_earnings_and_financial_health(
            ticker=ticker, cached_info=cached_info
        )
        prompt = f"{fundamentals_research_prompt}\n\n"
        prompt += f"Analyze the business fundamentals for ticker: {ticker}\n\n"
        prompt += (
            "Here is the fundamental data:\n"
            f"{fundamentals_data.model_dump_json(indent=2)}"
        )
        call = invoke_llm_with_metrics(
            llm, prompt, FundamentalSentimentOutput, token_budget=config.token_budget
        )
    else:
        prompt = (
            f"{fundamentals_research_prompt}\n\n"
            f"Analyze the business fundamentals for ticker: {ticker}"
        )
        call = run_agent_with_tools(
            llm,
            prompt,
            [get_fundamentals_tool],
            FundamentalSentimentOutput,
            track_tokens=True,
            token_budget=config.token_budget,
        )

    return finalize_llm_call(AGENT_NAME, model, start_time, call, config.token_budget)
