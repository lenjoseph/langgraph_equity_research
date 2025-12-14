"""SEC Filings query builder agent.

Generates contextual search queries based on ticker, trade direction, and trade duration.
"""

import time
from typing import List, Tuple

from agents.filings.prompts.query_builder_prompt import query_builder_prompt
from agents.filings.util import _get_trade_direction_desc, _get_trade_duration_desc
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from util.logger import get_logger
from models.agent import QueryBuilderOutput
from models.metrics import AgentMetrics, TokenUsage
from models.state import TradeDuration, TradeDirection


logger = get_logger(__name__)

AGENT_NAME = "filings_query_builder"


def generate_search_queries(
    ticker: str,
    trade_direction: TradeDirection,
    trade_duration: TradeDuration,
) -> Tuple[List[str], AgentMetrics]:
    """
    Generate contextual search queries for SEC filings retrieval.

    Args:
        ticker: Stock ticker symbol
        trade_direction: Direction of the trade (long/short)
        trade_duration: Duration of the trade (day/swing/position)

    Returns:
        Tuple of (list of search queries, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_fast"]
    token_usage = TokenUsage()

    prompt = query_builder_prompt.format(
        ticker=ticker,
        trade_direction=trade_direction.value,
        trade_direction_desc=_get_trade_direction_desc(trade_direction),
        trade_duration=trade_duration.value,
        trade_duration_desc=_get_trade_duration_desc(trade_duration),
    )

    llm = get_openai_llm(model=model, temperature=0.3)

    try:
        result, token_usage = invoke_llm_with_metrics(llm, prompt, QueryBuilderOutput)
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )

        if result and result.search_queries:
            logger.info(
                f"Generated {len(result.search_queries)} search queries for {ticker}"
            )
            return result.search_queries, metrics
        else:
            logger.warning(f"No search queries generated for {ticker}, using defaults")
            return _get_default_queries(trade_direction), metrics

    except Exception as e:
        logger.error(f"Error generating search queries: {e}", exc_info=True)
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return _get_default_queries(trade_direction), metrics


def _get_default_queries(trade_direction: TradeDirection) -> List[str]:
    """Return default queries as fallback based on trade direction."""
    if trade_direction == TradeDirection.SHORT:
        return [
            "risk factors material risks",
            "debt obligations liquidity concerns",
            "competitive threats market challenges",
            "declining revenue margin pressure",
            "management turnover governance issues",
        ]
    else:
        return [
            "revenue growth trends performance",
            "competitive advantages market position",
            "management guidance positive outlook",
            "strategic initiatives growth drivers",
            "strong cash flow financial health",
        ]
