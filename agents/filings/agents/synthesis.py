"""SEC Filings synthesis agent."""

import time
from typing import Optional

from agents.filings.prompts.synthesis_prompt import filings_synthesis_prompt
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from util.logger import get_logger
from models.agent import FilingsSentimentOutput
from models.metrics import AgentMetrics
from models.result import AgentRunResult, AgentStatus

logger = get_logger(__name__)

AGENT_NAME = "filings_synthesis"


def generate_filings_sentiment(
    ticker: str,
    context: Optional[str],
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """
    Generate sentiment analysis from SEC filings context.

    Args:
        ticker: Stock ticker symbol
        context: Retrieved context from SEC filings
        token_config: Optional token configuration for this agent

    Returns:
        Tuple of (FilingsSentimentOutput or None, AgentMetrics)
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.filings_synthesis
    model = LLM_MODELS["open_ai_smart"]

    if not context:
        logger.warning(f"No context provided for filings synthesis for {ticker}")
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            model=model,
        )
        return AgentRunResult(
            status=AgentStatus.DEGRADED,
            output=None,
            error="No SEC filings context retrieved.",
            metrics=metrics,
        )

    prompt = f"{filings_synthesis_prompt}\n\n{context}"

    # Get LLM and generate structured output
    llm = get_openai_llm(
        model=model,
        temperature=0.1,
        max_tokens=config.max_output_tokens,
    )

    call = invoke_llm_with_metrics(
        llm, prompt, FilingsSentimentOutput, token_budget=config.token_budget
    )
    return finalize_llm_call(AGENT_NAME, model, start_time, call, config.token_budget)
