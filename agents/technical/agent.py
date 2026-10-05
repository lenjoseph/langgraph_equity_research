import time
from typing import Optional

import dotenv

from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from agents.technical.prompt import technical_research_prompt
from agents.technical.tools import get_technical_analysis_tool
from models.agent import TechnicalSentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "technical"


def get_technical_sentiment(
    ticker: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Generate technical sentiment analysis for a ticker."""
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.technical
    model = LLM_MODELS["open_ai_smart"]

    prompt = (
        f"{technical_research_prompt}\n\n"
        f"Analyze the technical indicators for ticker: {ticker}"
    )
    llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    call = run_agent_with_tools(
        llm,
        prompt,
        [get_technical_analysis_tool],
        TechnicalSentimentOutput,
        track_tokens=True,
        token_budget=config.token_budget,
    )
    return finalize_llm_call(AGENT_NAME, model, start_time, call, config.token_budget)
