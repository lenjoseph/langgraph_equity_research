from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import Tool

from models.tools import TechnicalAnalysis, TechnicalAnalysisInput


# Default configuration for technical indicators
DEFAULT_PERIODS = {
    "sma_50": 50,
    "sma_200": 200,
    "rsi": 14,
    "stochastic": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb": 20,
    "interval": "1d",
    "history_period": "2y",  # Need at least 200 data points for SMA 200
}


def calculate_sma(data: pd.Series, length: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=length).mean()


def calculate_rsi(data: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"K": k, "D": d})


def calculate_macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return pd.DataFrame(
        {"MACD": macd, "Signal": macd_signal, "Histogram": macd_histogram}
    )


def calculate_bollinger_bands(
    data: pd.Series, length: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle = data.rolling(window=length).mean()
    std = data.rolling(window=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return pd.DataFrame({"Upper": upper, "Middle": middle, "Lower": lower})


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert to float, return default if NaN."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_signal(value: float, low_threshold: float, high_threshold: float) -> str:
    """Get signal based on value thresholds."""
    if pd.isna(value):
        return "unknown"
    if value < low_threshold:
        return "oversold"
    if value > high_threshold:
        return "overbought"
    return "neutral"


def safe_compare(val1: float, val2: float, default: int = 0) -> int:
    """
    Compare two values, return default if either is NaN.
    Returns: 1 if val1 > val2, -1 if val1 < val2, 0 otherwise.
    """
    if pd.isna(val1) or pd.isna(val2):
        return default
    if val1 > val2:
        return 1
    if val1 < val2:
        return -1
    return 0


def get_bollinger_signal(price: float, bb_upper: float, bb_lower: float) -> str:
    """Determine Bollinger Bands signal."""
    if pd.isna(bb_lower) or pd.isna(bb_upper):
        return "unknown"
    if price < bb_lower:
        return "oversold"
    if price > bb_upper:
        return "overbought"
    return "neutral"


def _add_technical_indicators(data: pd.DataFrame, periods: Dict[str, Any]) -> None:
    """Add technical indicators to data using specified periods in-place."""
    data["SMA_50"] = calculate_sma(data["Close"], periods["sma_50"])
    data["SMA_200"] = calculate_sma(data["Close"], periods["sma_200"])
    data["RSI"] = calculate_rsi(data["Close"], periods["rsi"])

    stoch = calculate_stochastic(
        data["High"], data["Low"], data["Close"], k_period=periods["stochastic"]
    )
    data["Stoch_K"] = stoch["K"]

    macd = calculate_macd(
        data["Close"],
        fast=periods["macd_fast"],
        slow=periods["macd_slow"],
        signal=periods["macd_signal"],
    )
    data["MACD"] = macd["MACD"]
    data["MACD_Signal"] = macd["Signal"]

    bb = calculate_bollinger_bands(data["Close"], periods["bb"])
    data["BB_Upper"] = bb["Upper"]
    data["BB_Lower"] = bb["Lower"]


def get_technical_analysis(ticker: str) -> TechnicalAnalysis:
    """
    Performs technical analysis on a stock with standard investment indicators.

    Args:
        ticker (str): Stock ticker (e.g., 'AAPL').

    Returns:
        TechnicalAnalysis: Structured output with indicators, signals, and sentiment.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(
            period=DEFAULT_PERIODS["history_period"],
            interval=DEFAULT_PERIODS["interval"],
        )

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Add technical indicators
        _add_technical_indicators(data, DEFAULT_PERIODS)

        # Current values (latest close)
        latest = data.iloc[-1]
        current_price = float(latest["Close"])

        # Calculate trends and signals
        sma_50_trend = safe_compare(current_price, latest["SMA_50"])
        sma_200_trend = safe_compare(current_price, latest["SMA_200"])
        macd_signal = safe_compare(latest["MACD"], latest["MACD_Signal"])

        # Overall sentiment score (-1 to 1)
        # Filter out 0 values that indicate missing/unknown data
        signals = [sma_50_trend, sma_200_trend, macd_signal]
        valid_signals = [s for s in signals if s != 0]
        overall_sentiment = (
            sum(valid_signals) / len(valid_signals) if valid_signals else 0.0
        )

        # Calculate Bollinger Band position
        bb_position = None
        bb_upper, bb_lower = latest["BB_Upper"], latest["BB_Lower"]

        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_position = safe_float((current_price - bb_lower) / bb_range * 100)

        return TechnicalAnalysis(
            ticker=ticker,
            current_price=current_price,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Momentum
            rsi=safe_float(latest["RSI"]),
            rsi_signal=get_signal(latest["RSI"], 30, 70),
            stoch_k=safe_float(latest["Stoch_K"]),
            stoch_signal=get_signal(latest["Stoch_K"], 20, 80),
            # Trend
            sma_50=safe_float(latest["SMA_50"]),
            sma_50_trend=sma_50_trend,
            sma_200=safe_float(latest["SMA_200"]),
            sma_200_trend=sma_200_trend,
            macd=safe_float(latest["MACD"]),
            macd_signal_value=macd_signal,
            # Volatility
            bb_position=bb_position,
            bb_signal=get_bollinger_signal(current_price, bb_upper, bb_lower),
            # Overall
            overall_sentiment=overall_sentiment,
        )

    except Exception as e:
        return TechnicalAnalysis(
            ticker=ticker,
            current_price=0.0,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rsi_signal="unknown",
            stoch_signal="unknown",
            sma_50_trend=0,
            sma_200_trend=0,
            macd_signal_value=0,
            bb_signal="unknown",
            overall_sentiment=-1.0,
            error=str(e),
        )


get_technical_analysis_tool = Tool(
    name="get_technical_analysis_tool",
    description="Use this tool to perform technical analysis on a stock. Provide the ticker. The tool calculates RSI, Moving Averages (50/200 day), MACD, Stochastic Oscillator, and Bollinger Bands. Returns structured data with buy/sell signals and overall sentiment.",
    func=get_technical_analysis,
    args_schema=TechnicalAnalysisInput,
)
