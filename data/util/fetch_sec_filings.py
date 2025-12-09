"""SEC EDGAR filing fetcher with rate limiting."""

import os
import time
from functools import lru_cache
from typing import Optional

import requests
from sec_edgar_api import EdgarClient

from logger import get_logger
from models.agent import FilingMetadata

logger = get_logger(__name__)


USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT")
if not USER_AGENT:
    raise ValueError(
        "SEC_EDGAR_USER_AGENT environment variable must be set (format: 'youremail@domain.extension')"
    )

# Rate limiting: SEC allows max 10 requests/second
REQUEST_DELAY = 0.15  # 150ms between requests

# SEC ticker-to-CIK mapping URL
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


@lru_cache(maxsize=1)
def _get_ticker_cik_map() -> dict[str, str]:
    """
    Fetch and cache the ticker-to-CIK mapping from SEC.

    Returns:
        Dict mapping uppercase ticker symbols to CIK strings
    """
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(TICKER_CIK_URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        ticker_map = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker and cik:
                ticker_map[ticker] = cik

        logger.info(f"Loaded {len(ticker_map)} ticker-to-CIK mappings from SEC")
        return ticker_map
    except Exception as e:
        logger.error(f"Failed to fetch ticker-CIK mapping: {e}")
        return {}


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Convert a ticker symbol to SEC CIK.

    Args:
        ticker: Stock ticker symbol

    Returns:
        CIK string (10 digits, zero-padded) or None if not found
    """
    ticker_map = _get_ticker_cik_map()
    return ticker_map.get(ticker.upper())


class SECFetcher:
    """Fetches SEC filings from EDGAR."""

    def __init__(self):
        self.client = EdgarClient(user_agent=USER_AGENT)
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def fetch_filing_list(
        self,
        ticker: str,
        filing_types: list[str] = ["10-K", "10-Q"],
        limit: int = 10,
    ) -> list[FilingMetadata]:
        """
        Fetch list of filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: Types of filings to fetch (10-K, 10-Q, 8-K)
            limit: Maximum number of filings to return

        Returns:
            List of FilingMetadata objects
        """
        self._rate_limit()

        # Convert ticker to CIK
        cik = _ticker_to_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return []

        try:
            submissions = self.client.get_submissions(cik=cik)
        except Exception as e:
            logger.error(f"Failed to fetch submissions for {ticker} (CIK: {cik}): {e}")
            return []

        filings = []
        recent_filings = submissions.get("filings", {}).get("recent", {})

        if not recent_filings:
            logger.warning(f"No filings found for {ticker}")
            return []

        forms = recent_filings.get("form", [])
        dates = recent_filings.get("filingDate", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        primary_documents = recent_filings.get("primaryDocument", [])

        cik = str(submissions.get("cik", "")).zfill(10)

        for i, form in enumerate(forms):
            if form in filing_types and len(filings) < limit:
                accession = accession_numbers[i].replace("-", "")
                doc = primary_documents[i]

                filings.append(
                    FilingMetadata(
                        ticker=ticker.upper(),
                        filing_type=form,
                        filing_date=dates[i],
                        accession_number=accession_numbers[i],
                        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}",
                    )
                )

        logger.info(f"Found {len(filings)} filings for {ticker}")
        return filings

    def download_filing(self, metadata: FilingMetadata) -> Optional[str]:
        """
        Download the raw content of a filing.

        Args:
            metadata: Filing metadata with URL

        Returns:
            Raw HTML/text content of the filing, or None if failed
        """
        self._rate_limit()

        headers = {"User-Agent": USER_AGENT}

        try:
            response = requests.get(metadata.url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to download filing {metadata.accession_number}: {e}")
            return None


# Singleton instance
_fetcher: Optional[SECFetcher] = None


def get_fetcher() -> SECFetcher:
    """Get or create the SEC fetcher singleton."""
    global _fetcher
    if _fetcher is None:
        _fetcher = SECFetcher()
    return _fetcher


def fetch_filing_list(
    ticker: str,
    filing_types: list[str] = ["10-K", "10-Q", "8-K"],
    limit: int = 10,
) -> list[FilingMetadata]:
    """Convenience function to fetch filing list."""
    return get_fetcher().fetch_filing_list(ticker, filing_types, limit)


def download_filing(metadata: FilingMetadata) -> Optional[str]:
    """Convenience function to download a filing."""
    return get_fetcher().download_filing(metadata)
