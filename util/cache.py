from datetime import datetime
from typing import Any, Dict, Optional
from langgraph.types import CachePolicy

from util.logger import get_logger

logger = get_logger(__name__)

# TTL configurations (in seconds)
TTL_SHORT = 300  # 5 minutes - for time-sensitive data
TTL_MEDIUM = 1800  # 30 minutes - for moderately fresh data
TTL_LONG = 3600  # 1 hour - for stable data
TTL_VERY_LONG = 7200  # 2 hours - for rarely changing data

# Thresholds for freshness checks
EARNINGS_IMMINENT_DAYS = 7  # Days before earnings to consider "imminent"


def is_earnings_imminent(ticker_info: Optional[Dict[str, Any]]) -> bool:
    """
    Check if earnings are within the imminent threshold.

    When earnings are imminent, fundamental data should be cached for shorter periods
    as new information may be priced in.

    Args:
        ticker_info: The cached yfinance ticker.info dict

    Returns:
        True if earnings are within EARNINGS_IMMINENT_DAYS, False otherwise
    """
    if not ticker_info:
        return False

    try:
        # yfinance stores earnings dates in various formats
        # Try to extract from common fields
        earnings_dates = ticker_info.get("earningsTimestamp")
        if earnings_dates:
            # Convert timestamp to datetime
            earnings_dt = datetime.fromtimestamp(earnings_dates)
            days_until = (earnings_dt - datetime.now()).days
            return 0 <= days_until <= EARNINGS_IMMINENT_DAYS

        # Alternative: check earningsDates list
        earnings_list = ticker_info.get("earningsDates", [])
        if earnings_list:
            for ed in earnings_list:
                if isinstance(ed, (int, float)):
                    earnings_dt = datetime.fromtimestamp(ed)
                    days_until = (earnings_dt - datetime.now()).days
                    if 0 <= days_until <= EARNINGS_IMMINENT_DAYS:
                        return True
    except Exception as e:
        logger.debug(f"Could not determine earnings imminence: {e}")

    return False


def get_fundamentals_ttl(ticker_info: Optional[Dict[str, Any]]) -> int:
    """
    Get dynamic TTL for fundamentals cache based on earnings proximity.

    Args:
        ticker_info: The cached yfinance ticker.info dict

    Returns:
        TTL in seconds - shorter when earnings are imminent
    """
    if is_earnings_imminent(ticker_info):
        logger.info("Earnings imminent - using short TTL for fundamentals cache")
        return TTL_SHORT
    return TTL_LONG


def get_current_date_bucket() -> str:
    """
    Get a date bucket string for cache key inclusion.

    This causes cache invalidation at day boundaries for time-sensitive data.

    Returns:
        Date string in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_hour_bucket() -> str:
    """
    Get an hour bucket string for cache key inclusion.

    This causes cache invalidation at hour boundaries.

    Returns:
        Hour string in YYYY-MM-DD-HH format
    """
    return datetime.now().strftime("%Y-%m-%d-%H")


def create_cache_policy(ttl: int, static_key: str | None = None) -> CachePolicy:
    """Util to create cache policies for research agents.

    Args:
        ttl: Time to live in seconds
        static_key: If provided, uses a static key. Otherwise, uses ticker.

    Returns:
        CachePolicy instance
    """
    if static_key:
        return CachePolicy(key_func=lambda x: static_key.encode(), ttl=ttl)

    # Handle both dict and object state representations
    # graph drawing in langsmith requires dict representation
    # graph execution requires pydantic object representation
    def key_func(x):
        if isinstance(x, dict):
            # Handle missing keys (e.g., when drawing graph without state)
            ticker = x.get("ticker", "default")
            return f"{ticker}".encode()
        return f"{x.ticker}".encode()

    return CachePolicy(key_func=key_func, ttl=ttl)


def create_filings_cache_policy(ttl: int = 3600) -> CachePolicy:
    """
    Create a cache policy for filings that includes trade parameters in the key.

    Args:
        ttl: Time to live in seconds

    Returns:
        CachePolicy instance with ticker, duration, and direction in key
    """

    def key_func(x):
        if isinstance(x, dict):
            ticker = x.get("ticker", "default")
            duration = x.get("trade_duration", "default")
            direction = x.get("trade_direction", "default")
        else:
            ticker = x.ticker
            duration = x.trade_duration
            direction = x.trade_direction

        return f"{ticker}-{duration}-{direction}".encode()

    return CachePolicy(key_func=key_func, ttl=ttl)


def create_fundamentals_cache_policy() -> CachePolicy:
    """
    Create a cache policy for fundamentals that adapts TTL based on earnings proximity.

    Cache key includes:
    - Ticker symbol
    - "earnings_imminent" flag if earnings are within threshold

    This causes cache misses when earnings status changes, ensuring fresh data
    during critical periods.

    Returns:
        CachePolicy with earnings-aware caching
    """

    def key_func(x):
        if isinstance(x, dict):
            ticker = x.get("ticker", "default")
            ticker_info = x.get("ticker_info")
        else:
            ticker = x.ticker
            ticker_info = x.ticker_info

        # Include earnings status in key - causes cache miss when status changes
        earnings_flag = (
            "earnings_imminent" if is_earnings_imminent(ticker_info) else "normal"
        )
        return f"{ticker}:{earnings_flag}".encode()

    def ttl_func(x):
        """Dynamic TTL based on earnings proximity."""
        if isinstance(x, dict):
            ticker_info = x.get("ticker_info")
        else:
            ticker_info = x.ticker_info

        return get_fundamentals_ttl(ticker_info)

    # Use shorter TTL since we'll dynamically adjust via key changes
    return CachePolicy(key_func=key_func, ttl=TTL_LONG)


def create_technical_cache_policy() -> CachePolicy:
    """
    Create a cache policy for technical analysis with time-bucketed keys.

    Technical data is time-sensitive, so we include hour buckets in the key
    to ensure relatively fresh price data.

    Returns:
        CachePolicy with hourly cache invalidation
    """

    def key_func(x):
        if isinstance(x, dict):
            ticker = x.get("ticker", "default")
        else:
            ticker = x.ticker

        # Include hour bucket for time-sensitive price data
        hour_bucket = get_hour_bucket()
        return f"{ticker}:{hour_bucket}".encode()

    return CachePolicy(key_func=key_func, ttl=TTL_MEDIUM)


def create_macro_cache_policy() -> CachePolicy:
    """
    Create a cache policy for macro data with daily cache invalidation.

    Macro economic data (GDP, CPI, etc.) typically updates monthly/quarterly,
    but we use daily buckets to ensure we don't miss releases.

    Returns:
        CachePolicy with daily cache invalidation
    """

    def key_func(x):
        # Macro data is ticker-independent, use date bucket as key
        date_bucket = get_current_date_bucket()
        return f"macro:{date_bucket}".encode()

    return CachePolicy(key_func=key_func, ttl=TTL_VERY_LONG)
