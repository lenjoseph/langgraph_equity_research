"""
CLI for ingesting SEC filings.
Usage: python scripts/ingest_filings.py AAPL --years 2
"""

import argparse
import sys
from pathlib import Path

from data.util import ingest_sec_filings

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC filings for a ticker")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--years", type=int, default=2, help="Number of years of filings (default: 2)"
    )
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        default=["10-K", "10-Q"],
        help="Filing types to ingest (default: 10-K 10-Q)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of existing filings",
    )

    args = parser.parse_args()

    print(f"Ingesting filings for {args.ticker.upper()}...")
    print(f"  Years: {args.years}")
    print(f"  Types: {args.types}")
    print(f"  Force: {args.force}")
    print()

    result = ingest_sec_filings(
        ticker=args.ticker.upper(),
        filing_types=args.types,
        years=args.years,
        force=args.force,
    )

    print()
    print("=" * 40)
    print("Ingestion Results:")
    print(f"  Ingested: {result['ingested']}")
    print(f"  Skipped:  {result['skipped']}")
    print(f"  Errors:   {result['errors']}")
    print(f"  Chunks:   {result['chunks_created']}")


if __name__ == "__main__":
    main()
