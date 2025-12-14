from models.state import TradeDuration, TradeDirection


def _get_trade_direction_desc(direction: TradeDirection) -> str:
    """Get human-readable description of trade direction."""
    if direction == TradeDirection.SHORT:
        return "short - betting the stock will decline"
    return "long - betting the stock will rise"


def _get_trade_duration_desc(duration: TradeDuration) -> str:
    """Get human-readable description of trade duration."""
    descriptions = {
        TradeDuration.DAY_TRADE: "intraday position, held for hours",
        TradeDuration.SWING_TRADE: "short-term position, held for days to weeks",
        TradeDuration.POSITION_TRADE: "long-term position, held for weeks to months",
    }
    return descriptions.get(duration, "unknown duration")
