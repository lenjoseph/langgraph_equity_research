"""SEC Filings research agent."""

import time
from typing import Optional, Tuple

import dotenv

from agents.filings.prompt import filings_research_prompt
from agents.filings.tools import search_filings, FilingSearchResult
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from data.util.ingest_sec_filings import ingest_ticker_filings
from data.util.vector_store import collection_exists, get_collection_stats
from util.logger import get_logger
from models.agent import FilingsSentimentOutput
from models.metrics import AgentMetrics, TokenUsage

dotenv.load_dotenv()

logger = get_logger(__name__)

AGENT_NAME = "filings"

# Key topics to search for comprehensive analysis
SEARCH_TOPICS = [
    "risk factors material risks",
    "revenue growth trends performance",
    "competitive landscape market position",
    "management guidance outlook",
    "debt obligations liquidity",
]


def _gather_filing_context(
    ticker: str, top_k_per_topic: int = 3
) -> list[FilingSearchResult]:
    """
    Search filings for multiple topics to gather comprehensive context.

    Args:
        ticker: Stock ticker symbol
        top_k_per_topic: Number of results per search topic

    Returns:
        Deduplicated list of search results
    """
    all_results = []
    seen_texts = set()

    for topic in SEARCH_TOPICS:
        results = search_filings(
            ticker=ticker,
            query=topic,
            top_k=top_k_per_topic,
        )
        for result in results:
            # Deduplicate by text content (first 100 chars)
            text_key = result.text[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(result)

    # Sort by relevance score
    all_results.sort(key=lambda x: x.relevance_score, reverse=True)

    return all_results[:15]  # Limit total context


def get_filings_sentiment(
    ticker: str,
) -> Tuple[Optional[FilingsSentimentOutput], AgentMetrics]:
    """
    Generate sentiment analysis from SEC filings for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (FilingsSentimentOutput or None, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["open_ai_smart"]
    token_usage = TokenUsage()

    # Check if we have any filings
    if not collection_exists(ticker):
        logger.warning(f"No filings available for {ticker}")
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return None, metrics

    stats = get_collection_stats(ticker)
    if stats.get("document_count", 0) == 0:
        logger.warning(f"Empty filings collection for {ticker}")
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return None, metrics

    filing_results = _gather_filing_context(ticker)

    if not filing_results:
        logger.warning(f"No relevant filing excerpts found for {ticker}")
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = AgentMetrics(
            agent_name=AGENT_NAME,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=model,
        )
        return None, metrics

    context_parts = [f"SEC Filing excerpts for {ticker}:\n"]
    for result in filing_results:
        context_parts.append(
            f"\n[{result.filing_type} | {result.section} | {result.filing_date}]\n"
            f"{result.text}\n"
        )
    context = "".join(context_parts)

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
