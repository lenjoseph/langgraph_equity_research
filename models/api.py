from pydantic import BaseModel
from enum import Enum


class TradeDuration(str, Enum):
    """Enum for trade duration types."""

    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"
    POSITION_TRADE = "position_trade"


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""

    ticker: str
    trade_duration: TradeDuration
