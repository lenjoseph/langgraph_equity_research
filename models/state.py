from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel


class TradeDuration(Enum):
    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"
    POSITION_TRADE = "position_trade"


class TradeDirection(Enum):
    SHORT = "short"
    LONG = "long"


class EquityResearchState(BaseModel):
    """State model for the equity research workflow."""

    ticker: str
    trade_duration: TradeDuration
    trade_direction: TradeDirection
    industry: Optional[str] = None
    business: Optional[str] = None
    fundamental_sentiment: Optional[str] = None
    technical_sentiment: Optional[str] = None
    macro_sentiment: Optional[str] = None
    industry_sentiment: Optional[str] = None
    peer_sentiment: Optional[str] = None
    headline_sentiment: Optional[str] = None
    filings_sentiment: Optional[str] = None
    combined_sentiment: Optional[str] = None
    compliant: bool = False
    feedback: Optional[str] = None
    is_ticker_valid: bool = False
    revision_iteration_count: int = 0
    ticker_info: Optional[Dict[str, Any]] = None  # Cached yfinance ticker.info
