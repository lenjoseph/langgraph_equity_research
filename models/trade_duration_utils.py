"""Utility functions for trade duration conversions."""

from models.api import TradeDuration


def trade_duration_to_days(trade_duration: TradeDuration) -> int:
    """
    Convert trade duration enum to number of days.

    Args:
        trade_duration: Trade duration enum value

    Returns:
        Number of days corresponding to the trade duration
    """
    duration_mapping = {
        TradeDuration.DAY_TRADE: 1,
        TradeDuration.SWING_TRADE: 7,
        TradeDuration.POSITION_TRADE: 30,
    }
    return duration_mapping[trade_duration]


def trade_duration_to_label(trade_duration: TradeDuration) -> str:
    """
    Convert trade duration enum to human-readable label.

    Args:
        trade_duration: Trade duration enum value

    Returns:
        Human-readable label for the trade duration
    """
    label_mapping = {
        TradeDuration.DAY_TRADE: "Day Trade (1 day)",
        TradeDuration.SWING_TRADE: "Swing Trade (7 days)",
        TradeDuration.POSITION_TRADE: "Position Trade (30 days)",
    }
    return label_mapping[trade_duration]
