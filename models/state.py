from typing import Optional
from pydantic import BaseModel


class EquityResearchState(BaseModel):
    """State model for the equity research workflow."""

    ticker: str
    fundamental_sentiment: Optional[str] = None
    technical_sentiment: Optional[str] = None
    macro_sentiment: Optional[str] = None
    industry_sentiment: Optional[str] = None
    headline_sentiment: Optional[str] = None
    combined_sentiment: Optional[str] = None
    is_ticker_valid: bool = False
