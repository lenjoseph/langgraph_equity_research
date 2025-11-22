from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


# Fundamentals Models
class FundamentalsData(BaseModel):
    """Fundamental analysis data for a stock ticker."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Company full name")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")

    # Earnings data
    earnings: Optional[Dict[str, Any]] = Field(
        None, description="Earnings data (annual and quarterly)"
    )

    # Financial statements
    balance_sheet: Optional[Dict[str, Any]] = Field(
        None, description="Balance sheet data (annual and quarterly)"
    )
    income_statement: Optional[Dict[str, Any]] = Field(
        None, description="Income statement data (annual and quarterly)"
    )
    cash_flow: Optional[Dict[str, Any]] = Field(
        None, description="Cash flow statement (annual and quarterly)"
    )

    # Financial ratios
    ratios: Optional[Dict[str, Any]] = Field(
        None,
        description="Key financial ratios (P/E, ROE, debt-to-equity, margins, etc.)",
    )

    # Valuation metrics
    valuation_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Valuation metrics for DCF and comparable analysis (market cap, EV, beta, etc.)",
    )

    # Price data
    current_price: Optional[float] = Field(None, description="Current stock price")
    target_price: Optional[Dict[str, Any]] = Field(
        None, description="Analyst target prices (mean, high, low)"
    )

    # Error handling
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    message: Optional[str] = Field(None, description="Additional error context")


class FundamentalsInput(BaseModel):
    """Input schema for fundamentals analysis tool."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")


# Technical Analysis Models
class TechnicalAnalysis(BaseModel):
    """Technical analysis data for a stock ticker."""

    ticker: str = Field(..., description="Stock ticker")
    current_price: float = Field(..., description="Current stock price")
    analysis_date: str = Field(
        ..., description="Date and time of analysis in YYYY-MM-DD HH:MM:SS format"
    )

    # Short-term indicators (calibrated to ~0.5x trade duration)
    short_rsi: Optional[float] = Field(
        None, description="Short-term Relative Strength Index"
    )
    short_rsi_signal: str = Field(
        ..., description="RSI signal: oversold, neutral, overbought, or unknown"
    )
    short_sma: Optional[float] = Field(
        None, description="Short-term Simple Moving Average"
    )
    short_sma_trend: int = Field(
        ..., description="Price vs short SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    short_stoch_k: Optional[float] = Field(None, description="Stochastic %K oscillator")
    short_stoch_signal: str = Field(
        ..., description="Stochastic signal: oversold, neutral, overbought, or unknown"
    )

    # Long-term indicators (calibrated to ~3-4x trade duration)
    long_sma: Optional[float] = Field(
        None, description="Long-term Simple Moving Average"
    )
    long_sma_trend: int = Field(
        ..., description="Price vs long SMA: 1 (above), -1 (below), 0 (unknown)"
    )
    macd: Optional[float] = Field(None, description="MACD line value")
    macd_signal_value: int = Field(
        ..., description="MACD vs Signal: 1 (bullish), -1 (bearish), 0 (unknown)"
    )
    bb_position: Optional[float] = Field(
        None, description="Position within Bollinger Bands as percentage (0-100)"
    )
    bb_signal: str = Field(
        ...,
        description="Bollinger Bands signal: oversold, neutral, overbought, or unknown",
    )

    # Period information
    rsi_period: Optional[int] = Field(None, description="RSI calculation period used")
    short_sma_period: Optional[int] = Field(None, description="Short SMA period used")
    long_sma_period: Optional[int] = Field(None, description="Long SMA period used")
    stochastic_period: Optional[int] = Field(None, description="Stochastic period used")
    bb_period: Optional[int] = Field(None, description="Bollinger Bands period used")

    # Overall assessment
    overall_sentiment: float = Field(
        ..., description="Overall sentiment score from -1 (bearish) to 1 (bullish)"
    )

    error: Optional[str] = Field(None, description="Error message if analysis failed")


class TechnicalAnalysisInput(BaseModel):
    """Input schema for technical analysis tool."""

    ticker: str = Field(..., description="Stock ticker (e.g., 'AAPL', 'MSFT')")
    trade_duration: str = Field(
        ...,
        description="Trade duration type. Must be one of: 'day_trade' (intraday), 'swing_trade' (7 days), or 'position_trade' (30 days)",
    )


# Macro Data Models
class HistoricalDataPoint(BaseModel):
    """A single historical data point with date and value."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: float = Field(..., description="The value at this date")


class IndicatorData(BaseModel):
    """Data for a single macroeconomic indicator."""

    latest_value: float = Field(..., description="Most recent value")
    latest_date: str = Field(
        ..., description="Date of most recent value in YYYY-MM-DD format"
    )
    historical_data: list[HistoricalDataPoint] = Field(
        ..., description="Historical time series data"
    )
    quarterly_change_pct_points: Optional[float] = Field(
        None,
        description="Quarterly change in percentage points (for rate-based indicators)",
    )
    monthly_change_percent: Optional[float] = Field(
        None, description="Monthly percentage change (for index-based indicators)"
    )
    yoy_inflation_rate: Optional[float] = Field(
        None, description="Year-over-year inflation rate (CPI only)"
    )
    error: Optional[str] = Field(
        None, description="Error message if data retrieval failed"
    )


class MacroDataResponse(BaseModel):
    """Response containing macroeconomic data from FRED."""

    timestamp: str = Field(..., description="Timestamp when data was retrieved")
    data: dict[str, IndicatorData] = Field(
        ..., description="Dictionary of indicator names to their data"
    )
    error: Optional[str] = Field(
        None, description="Error message if overall retrieval failed"
    )


class MacroDataInput(BaseModel):
    """Input schema for the macro data tool. No parameters required."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Options Data Models
class OptionsDataInput(BaseModel):
    """Input schema for options analysis tool."""

    ticker_symbol: str = Field(
        ..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
    )
    trade_duration: str = Field(
        ...,
        description="Trade duration type. Must be one of: 'day_trade', 'swing_trade', or 'position_trade'. The tool will select the most relevant expiration date based on this timeframe.",
    )
    min_oi: int = Field(
        100,
        description="Minimum open interest threshold to filter out illiquid contracts",
    )
