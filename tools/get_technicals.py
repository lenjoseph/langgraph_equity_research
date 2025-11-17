from langchain_core.tools import Tool
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TechnicalAnalysis(BaseModel):
    """Technical analysis data for a stock ticker."""

    ticker: str = Field(..., description="Stock ticker")
    current_price: float = Field(..., description="Current stock price")
    analysis_date: str = Field(
        ..., description="Date and time of analysis in YYYY-MM-DD HH:MM:SS format"
    )

    # Mid-term indicators (daily data)
    mid_rsi: Optional[float] = Field(None, description="14-day Relative Strength Index")
    mid_rsi_signal: str = Field(
        ..., description="RSI signal: oversold, neutral, overbought, or unknown"
    )
    mid_sma_50: Optional[float] = Field(
        None, description="50-day Simple Moving Average"
    )
    mid_sma_trend: int = Field(
        ..., description="Price vs 50 SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    mid_stoch_k: Optional[float] = Field(None, description="Stochastic %K oscillator")
    mid_stoch_signal: str = Field(
        ..., description="Stochastic signal: oversold, neutral, overbought, or unknown"
    )

    # Macro-term indicators (weekly data)
    macro_sma_200: Optional[float] = Field(
        None, description="200-period Simple Moving Average (weekly)"
    )
    macro_sma_trend: int = Field(
        ..., description="Price vs 200 SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    macro_macd: Optional[float] = Field(None, description="MACD line value (weekly)")
    macro_macd_signal: int = Field(
        ..., description="MACD vs Signal: 1 (bullish), -1 (bearish), 0 (unknown)"
    )
    macro_bb_position: Optional[float] = Field(
        None, description="Position within Bollinger Bands as percentage (0-100)"
    )
    macro_bb_signal: str = Field(
        ...,
        description="Bollinger Bands signal: oversold, neutral, overbought, or unknown",
    )

    # Overall assessment
    overall_sentiment: float = Field(
        ..., description="Overall sentiment score from -1 (bearish) to 1 (bullish)"
    )

    error: Optional[str] = Field(None, description="Error message if analysis failed")


class TechnicalAnalysisInput(BaseModel):
    """Input schema for technical analysis tool."""

    ticker: str = Field(..., description="Stock ticker (e.g., 'AAPL', 'MSFT')")
    period: str = Field(
        default="2y", description="Data period for analysis (e.g., '1y', '2y', '5y')"
    )


def calculate_sma(data: pd.Series, length: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=length).mean()


def calculate_rsi(data: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"K": k, "D": d})


def calculate_macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
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
    """Calculate Bollinger Bands"""
    middle = data.rolling(window=length).mean()
    std = data.rolling(window=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return pd.DataFrame({"Upper": upper, "Middle": middle, "Lower": lower})


def safe_float(value, default=None):
    """Convert to float, return default if NaN"""
    if pd.isna(value):
        return default
    return float(value)


def get_signal(value, low_threshold, high_threshold, default="unknown"):
    """Get signal, return default if NaN"""
    if pd.isna(value):
        return default
    if value < low_threshold:
        return "oversold"
    elif value > high_threshold:
        return "overbought"
    else:
        return "neutral"


def safe_compare(val1, val2, default=0):
    """Compare two values, return default if either is NaN"""
    if pd.isna(val1) or pd.isna(val2):
        return default
    return 1 if val1 > val2 else -1


def get_bollinger_signal(price, bb_upper, bb_lower):
    """Determine Bollinger Bands signal"""
    if pd.isna(bb_lower) or pd.isna(bb_upper):
        return "unknown"
    if price < bb_lower:
        return "oversold"
    elif price > bb_upper:
        return "overbought"
    else:
        return "neutral"


def add_daily_indicators(data: pd.DataFrame):
    """Add mid-term technical indicators to daily data"""
    data["SMA_50"] = calculate_sma(data["Close"], 50)
    data["RSI"] = calculate_rsi(data["Close"], 14)
    stoch = calculate_stochastic(data["High"], data["Low"], data["Close"])
    data["Stoch_K"] = stoch["K"]


def add_weekly_indicators(data: pd.DataFrame):
    """Add macro-term technical indicators to weekly data"""
    data["SMA_200"] = calculate_sma(data["Close"], 200)
    macd = calculate_macd(data["Close"])
    data["MACD"] = macd["MACD"]
    data["MACD_Signal"] = macd["Signal"]
    bb = calculate_bollinger_bands(data["Close"], 20)
    data["BB_Upper"] = bb["Upper"]
    data["BB_Lower"] = bb["Lower"]
    data["BB_Mid"] = bb["Middle"]


def get_technical_analysis(ticker: str, period: str = "2y") -> TechnicalAnalysis:
    """
    Performs macro and mid-term technical analysis on a stock using yfinance.

    Args:
        ticker (str): Stock ticker (e.g., 'AAPL').
        period (str): Data period (e.g., '2y' for 2 years).

    Returns:
        TechnicalAnalysis: Structured output with indicators, signals, and sentiment.
    """
    try:
        stock = yf.Ticker(ticker)
        daily_data = stock.history(period=period, interval="1d")
        weekly_data = stock.history(period=period, interval="1wk")

        if daily_data.empty or weekly_data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Add technical indicators
        add_daily_indicators(daily_data)
        add_weekly_indicators(weekly_data)

        # Current values (latest close)
        latest_daily = daily_data.iloc[-1]
        latest_weekly = weekly_data.iloc[-1]
        current_price = latest_daily["Close"]

        # Overall sentiment score (-1 to 1)
        # Filter out 0 values that indicate missing/unknown data
        mid_sma_trend = safe_compare(current_price, latest_daily["SMA_50"])
        macro_sma_trend = safe_compare(current_price, latest_weekly["SMA_200"])
        macro_macd_signal = safe_compare(
            latest_weekly["MACD"], latest_weekly["MACD_Signal"]
        )

        signals = [mid_sma_trend, macro_sma_trend, macro_macd_signal]
        valid_signals = [s for s in signals if s != 0]
        if valid_signals:
            overall_sentiment = sum(valid_signals) / len(valid_signals)
        else:
            overall_sentiment = 0.0  # Neutral if no valid signals

        return TechnicalAnalysis(
            ticker=ticker,
            current_price=float(current_price),
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Mid-term
            mid_rsi=safe_float(latest_daily["RSI"]),
            mid_rsi_signal=get_signal(latest_daily["RSI"], 30, 70),
            mid_sma_50=safe_float(latest_daily["SMA_50"]),
            mid_sma_trend=mid_sma_trend,
            mid_stoch_k=safe_float(latest_daily["Stoch_K"]),
            mid_stoch_signal=get_signal(latest_daily["Stoch_K"], 20, 80),
            # Macro-term
            macro_sma_200=safe_float(latest_weekly["SMA_200"]),
            macro_sma_trend=macro_sma_trend,
            macro_macd=safe_float(latest_weekly["MACD"]),
            macro_macd_signal=macro_macd_signal,
            macro_bb_position=safe_float(
                (current_price - latest_weekly["BB_Lower"])
                / (latest_weekly["BB_Upper"] - latest_weekly["BB_Lower"])
                * 100
            ),
            macro_bb_signal=get_bollinger_signal(
                current_price, latest_weekly["BB_Upper"], latest_weekly["BB_Lower"]
            ),
            # Overall
            overall_sentiment=overall_sentiment,
        )

    except Exception as e:
        return TechnicalAnalysis(
            ticker=ticker,
            current_price=0.0,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mid_rsi_signal="unknown",
            mid_sma_trend=0,
            mid_stoch_signal="unknown",
            macro_sma_trend=0,
            macro_macd_signal=0,
            macro_bb_signal="unknown",
            overall_sentiment=-1.0,
            error=str(e),
        )


get_technical_analysis_tool = Tool(
    name="get_technical_analysis_tool",
    description="Use this tool to perform technical analysis on a stock. Analyzes mid-term (daily) and macro-term (weekly) indicators including RSI, Moving Averages, MACD, Stochastic Oscillator, and Bollinger Bands. Returns structured data with buy/sell signals and overall sentiment.",
    func=get_technical_analysis,
    args_schema=TechnicalAnalysisInput,
)
