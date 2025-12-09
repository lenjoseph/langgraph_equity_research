"""SEC Filings research agent."""

from typing import Optional

import dotenv

from agents.filings.prompt import filings_research_prompt
from agents.filings.tools import search_filings, FilingSearchResult
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from data.util.ingest_sec_filings import ingest_ticker_filings
from data.util.vector_store import collection_exists, get_collection_stats
from logger import get_logger
from models.agent import FilingsSentimentOutput

dotenv.load_dotenv()

logger = get_logger(__name__)

# Key topics to search for comprehensive analysis
SEARCH_TOPICS = [
    "risk factors material risks",
    "revenue growth trends performance",
    "competitive landscape market position",
    "management guidance outlook",
    "debt obligations liquidity",
]


def ensure_filings_ingested(ticker: str, years: int = 2) -> bool:
    """
    Ensure filings are ingested for a ticker.

    Returns True if ingestion was performed, False if filings already existed.
    """
    if collection_exists(ticker):
        stats = get_collection_stats(ticker)
        if stats.get("document_count", 0) > 0:
            logger.info(f"Filings already ingested for {ticker}: {stats}")
            return False

    logger.info(f"Ingesting filings for {ticker}...")
    result = ingest_ticker_filings(ticker, years=years)
    logger.info(f"Ingestion complete: {result}")
    return True


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


def get_filings_sentiment(ticker: str) -> Optional[FilingsSentimentOutput]:
    """
    Generate sentiment analysis from SEC filings for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        FilingsSentimentOutput or None if no filings available
    """
    # Ensure filings are ingested
    ensure_filings_ingested(ticker)

    # Check if we have any filings
    if not collection_exists(ticker):
        logger.warning(f"No filings available for {ticker}")
        return None

    stats = get_collection_stats(ticker)
    if stats.get("document_count", 0) == 0:
        logger.warning(f"Empty filings collection for {ticker}")
        return None

    # Gather context from filings
    filing_results = _gather_filing_context(ticker)

    if not filing_results:
        logger.warning(f"No relevant filing excerpts found for {ticker}")
        return None

    # Build context for LLM
    context_parts = [f"SEC Filing excerpts for {ticker}:\n"]
    for result in filing_results:
        context_parts.append(
            f"\n[{result.filing_type} | {result.section} | {result.filing_date}]\n"
            f"{result.text}\n"
        )
    context = "".join(context_parts)

    # Build prompt
    prompt = f"{filings_research_prompt}\n\n{context}"

    # Get LLM and generate structured output
    llm = get_openai_llm(model=LLM_MODELS["open_ai_smart"], temperature=0.1)
    structured_llm = llm.with_structured_output(FilingsSentimentOutput)

    try:
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error generating filings sentiment: {e}", exc_info=True)
        return None
