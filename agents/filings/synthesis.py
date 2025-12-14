"""SEC Filings synthesis agent."""

import time
from typing import Optional, Tuple

from agents.filings.prompt import filings_research_prompt
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from util.logger import get_logger
from models.agent import FilingsSentimentOutput
from models.metrics import AgentMetrics, TokenUsage

logger = get_logger(__name__)

AGENT_NAME = "filings_synthesis"


def generate_filings_sentiment(
    ticker: str,
    context: str,
) -> Tuple[Optional[FilingsSentimentOutput], AgentMetrics]:
    """
    Generate sentiment analysis from SEC filings context.

    Args:
        ticker: Stock ticker symbol
        context: Retrieved context from SEC filings

    Returns:
        Tuple of (FilingsSentimentOutput or None, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]
    token_usage = TokenUsage()

    if not context:
        logger.warning(f"No context provided for filings synthesis for {ticker}")
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return None, metrics

    prompt = f"{filings_research_prompt}\n\n{context}"

    # Get LLM and generate structured output
    llm = get_openai_llm(model=model, temperature=0.1)

    try:
        result, token_usage = invoke_llm_with_metrics(
            llm, prompt, FilingsSentimentOutput
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return result, metrics
    except Exception as e:
        logger.error(f"Error generating filings sentiment: {e}", exc_info=True)
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return None, metrics
