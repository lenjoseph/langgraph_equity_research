from enum import Enum
from typing import Optional
from pydantic import BaseModel


class TradeDuration(Enum):
    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"
    POSITION_TRADE = "position_trade"


class EquityResearchState(BaseModel):
    """State model for the equity research workflow."""

    ticker: str
    trade_duration: TradeDuration
    fundamental_sentiment: Optional[str] = None
    technical_sentiment: Optional[str] = None
    macro_sentiment: Optional[str] = None
    industry_sentiment: Optional[str] = None
    headline_sentiment: Optional[str] = None
    combined_sentiment: Optional[str] = None
    compliant: bool = False
    feedback: Optional[str] = None
    is_ticker_valid: bool = False
    revision_iteration_count: int = 0
