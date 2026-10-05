"""Filings refresh uses accession skips instead of a one-shot ingest."""

from datetime import datetime, timedelta
from unittest.mock import patch

from data.util.fetch_sec_filings import MissingSECIdentity, require_sec_user_agent
from data.util.ingest_sec_filings import (
    cap_filings,
    ensure_filings_ingested,
    warn_if_filings_stale,
)
from models.agent import FilingMetadata


def _filing(filing_type: str, filing_date: str, accession: str) -> FilingMetadata:
    return FilingMetadata(
        ticker="AAPL",
        filing_type=filing_type,
        filing_date=filing_date,
        accession_number=accession,
        url="https://example.test",
    )


def test_cap_filings_keeps_periodic_reports_and_limits_8k():
    filings = [_filing("10-K", "2026-01-01", "10k")]
    filings.extend(
        _filing("8-K", f"2026-01-{day:02d}", f"8k-{day}") for day in range(1, 13)
    )
    selected = cap_filings(filings, years=1)
    assert sum(item.filing_type == "10-K" for item in selected) == 1
    assert sum(item.filing_type == "8-K" for item in selected) == 8


def test_ensure_refreshes_even_when_documents_exist():
    with (
        patch("data.util.ingest_sec_filings.ingest_ticker_filings") as ingest,
        patch("data.util.ingest_sec_filings.get_collection_stats") as stats,
        patch("data.util.ingest_sec_filings.warn_if_filings_stale") as warn,
    ):
        stats.return_value = {"document_count": 4, "latest_filing_date": "2026-02-01"}
        assert ensure_filings_ingested("AAPL") is True
        ingest.assert_called_once()
        warn.assert_called_once()


def test_stale_filing_date_is_logged(caplog):
    old = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    with caplog.at_level("WARNING"):
        warn_if_filings_stale("AAPL", old)
    assert "older than one quarter" in caplog.text


def test_sec_identity_is_not_required_at_import(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_AGENT_KEY", raising=False)
    import data.util.fetch_sec_filings as fetch_module

    assert fetch_module.require_sec_user_agent is require_sec_user_agent
    try:
        require_sec_user_agent()
        raised = False
    except MissingSECIdentity:
        raised = True
    assert raised
