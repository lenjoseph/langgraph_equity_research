from pydantic import BaseModel, Field

from models.state import TradeDirection, TradeDuration

RESEARCH_DISCLAIMER = (
    "This output is research analysis for informational purposes only and is not investment advice."
)


class EquityResearchRequest(BaseModel):
    """Request model for equity research endpoint."""

    ticker: str
    trade_duration: TradeDuration
    trade_direction: TradeDirection
    token_preset: str = Field(
        default="standard",
        description="Token budget preset: unlimited, economy, standard, or premium",
    )
