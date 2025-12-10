"""Ingestion pipeline for SEC filings."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from data.util.fetch_sec_filings import download_filing, fetch_filing_list
from data.util.filing_chunker import chunk_filing
from data.util.parse_sec_filing import parse_filing
from data.util.vector_store import (
    get_collection_stats,
    get_or_create_collection,
    collection_exists,
)
from data.util.embed_chunks import embed_chunks
from util.logger import get_logger
from models.agent import FilingMetadata

logger = get_logger(__name__)

FILINGS_CACHE_DIR = Path("data/filings")


def _get_cache_path(metadata: FilingMetadata) -> Path:
    """Get the cache file path for a filing."""
    ticker_dir = FILINGS_CACHE_DIR / metadata.ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{metadata.accession_number}.html"


def _load_cached_filing(metadata: FilingMetadata) -> Optional[str]:
    """Load a filing from cache if it exists."""
    cache_path = _get_cache_path(metadata)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def _save_filing_to_cache(metadata: FilingMetadata, content: str) -> None:
    """Save a filing to cache."""
    cache_path = _get_cache_path(metadata)
    cache_path.write_text(content, encoding="utf-8")


def _delete_cached_filing(metadata: FilingMetadata) -> None:
    """Delete a cached filing after successful embedding."""
    cache_path = _get_cache_path(metadata)
    if cache_path.exists():
        cache_path.unlink()
        logger.debug(f"Deleted cached filing: {cache_path}")

        ticker_dir = cache_path.parent
        if ticker_dir.exists() and not any(ticker_dir.iterdir()):
            ticker_dir.rmdir()
            logger.debug(f"Removed empty directory: {ticker_dir}")


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
    return True


def _get_ingested_accession_numbers(ticker: str) -> set[str]:
    """Get set of already-ingested accession numbers."""
    if not collection_exists(ticker):
        return set()

    collection = get_or_create_collection(ticker)

    results = collection.get(include=["metadatas"])
    accession_numbers = set()
    for meta in results.get("metadatas", []):
        if meta and "accession_number" in meta:
            accession_numbers.add(meta["accession_number"])

    return accession_numbers


def ingest_ticker_filings(
    ticker: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    years: int = 2,
    force: bool = False,
    keep_html: bool = False,
) -> dict:
    """
    Ingest SEC filings for a ticker into the vector store.

    Args:
        ticker: Stock ticker symbol
        filing_types: Types of filings to ingest
        years: Number of years of filings to fetch
        force: If True, re-ingest all filings even if already present
        keep_html: If True, retain HTML files after embedding (default: False to save disk space)

    Returns:
        Dict with ingestion statistics
    """
    stats = {"ingested": 0, "skipped": 0, "errors": 0, "chunks_created": 0}

    logger.info(
        f"Starting ingestion for {ticker} ({years} years, types: {filing_types})"
    )

    limit = years * 5  # Conservative estimate

    filings = fetch_filing_list(ticker, filing_types=filing_types, limit=limit)

    if not filings:
        logger.warning(f"No filings found for {ticker}")
        return stats

    cutoff_date = datetime.now() - timedelta(days=years * 365)
    filings = [
        f
        for f in filings
        if datetime.strptime(f.filing_date, "%Y-%m-%d") >= cutoff_date
    ]

    ingested = _get_ingested_accession_numbers(ticker) if not force else set()

    collection = get_or_create_collection(ticker)

    for filing in filings:
        if filing.accession_number in ingested:
            logger.debug(f"Skipping already-ingested filing: {filing.accession_number}")
            stats["skipped"] += 1
            continue

        try:
            raw_html = _load_cached_filing(filing)

            if raw_html is None:
                raw_html = download_filing(filing)
                if raw_html:
                    _save_filing_to_cache(filing, raw_html)

            if raw_html is None:
                logger.error(f"Failed to get filing content: {filing.accession_number}")
                stats["errors"] += 1
                continue

            sections = parse_filing(raw_html, filing.filing_type)

            chunks = chunk_filing(sections, filing)

            if chunks:
                embed_chunks(chunks, collection)
                stats["chunks_created"] += len(chunks)
                stats["ingested"] += 1
                logger.info(
                    f"Ingested {filing.filing_type} ({filing.filing_date}): "
                    f"{len(chunks)} chunks"
                )
                if not keep_html:
                    _delete_cached_filing(filing)
            else:
                logger.warning(
                    f"No chunks created from filing: {filing.accession_number}"
                )
                stats["errors"] += 1

        except Exception as e:
            logger.error(
                f"Error ingesting {filing.accession_number}: {e}", exc_info=True
            )
            stats["errors"] += 1

    logger.info(f"Ingestion complete for {ticker}: {stats}")
    return stats
