from typing import Optional

from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from pydantic_core.core_schema import arguments_schema

from data.util.vector_store import get_or_create_collection, collection_exists
from util.logger import get_logger

logger = get_logger(__name__)


class FilingSearchInput(BaseModel):
    ticker: str = Field(descripton="Stock ticker symbol")
    query: str = Field(description="Search query for finding relevant filing content")
    filing_types: list[str] = Field(
        default=["10-K", "10-Q", "8-K"],
        description="Types of filings to search (10-K, 10-Q, 8-K)",
    )
    sections: Optional[list[str]] = Field(
        default=None,
        description="Specific sections to search (e.g., risk_factors, mda)",
    )
    top_k: int = Field(default=5, description="Number of results to return")


class FilingSearchResult(BaseModel):
    text: str
    ticker: str
    filing_type: str
    section: str
    filing_date: str
    relevance_score: float


def search_filings(
    ticker: str,
    query: str,
    filing_types: list[str] = ["10-K", "10-Q", "8-K"],
    sections: Optional[list[str]] = None,
    top_k: int = 5,
) -> list[FilingSearchResult]:
    if not collection_exists(ticker):
        logger.warning(f"No filings collection for {ticker}")
        return []

    collection = get_or_create_collection(ticker)

    where_filter = None
    where_conditions = []

    if filing_types and len(filing_types) < 3:
        where_conditions.append({"filing_type": {"$in": filing_types}})

    if sections:
        where_conditions.append({"section": {"$in": sections}})

    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Error querying filings: {e}")
        return []

    search_results = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist

        search_results.append(
            FilingSearchResult(
                text=doc,
                ticker=meta.get("ticker", ticker),
                filing_type=meta.get("filing_type", "unknown"),
                section=meta.get("section", "unknown"),
                filing_date=meta.get("filing_date", "unknown"),
                relevance_score=round(similarity, 4),
            )
        )

    return search_results


def search_filings_tool_func(
    ticker: str,
    query: str,
    filing_types: list[str] = ["10-K", "10-Q", "8-K"],
    sections: Optional[list[str]] = None,
    top_k: int = 5,
) -> str:
    results = search_filings(ticker, query, filing_types, sections, top_k)

    if not results:
        return f"No relevant filings found for {ticker} matching query: {query}"

    output_parts = [
        f"Found {len(results)} relevant excerpts from {ticker} SEC filingsL\n"
    ]

    for i, result in enumerate(results, 1):
        output_parts.append(
            f"\n--- Result {i} ({result.filing_type}, {result.section}, {result.filing_date}) ---\n"
            f"Relevance: {result.relevance_score:.2%}\n"
            f"{result.text}\n"
        )

    return "".join(output_parts)


search_filings_tool = Tool(
    name="search_sec_filings",
    description=(
        "Search SEC filings (10-K, 10-Q, 8-K) for a stock ticker. "
        "Use this tool to find information about risk factors, business strategy, "
        "financial performance, management discussion, or material events. "
        "Provide a specific query about what you want to find."
    ),
    func=search_filings_tool_func,
    arguments_schema=FilingSearchInput,
)
