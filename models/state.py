from pydantic import BaseModel
from models.api import TradeDuration


class EquityResearchState(BaseModel):
    """State model for the equity research workflow."""

    ticker: str
    trade_duration: TradeDuration
    fundamental_sentiment: str
    technical_sentiment: str
    macro_sentiment: str
    industry_sentiment: str
    headline_sentiment: str
    combined_sentiment: str
