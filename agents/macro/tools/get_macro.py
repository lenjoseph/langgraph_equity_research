import pandas_datareader as pdr
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Optional
from pydantic import BaseModel, Field


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


def get_macro_data(period_days: int = 365) -> MacroDataResponse:
    """
    Retrieve the most recent macroeconomic data from FRED (Federal Reserve Economic Data).

    Args:
        period_days: Number of days of historical data to retrieve (default: 365)

    Returns:
        MacroDataResponse containing:
        - timestamp: When the data was retrieved
        - data: Dictionary of IndicatorData objects with keys:
            * gdp_growth: GDP Growth Rate (quarterly)
            * inflation_cpi: Consumer Price Index with YoY inflation rate
            * consumer_sentiment: University of Michigan Consumer Sentiment Index
        - error: Optional error message if retrieval failed

        Each IndicatorData contains:
        - latest_value: Most recent value
        - latest_date: Date of most recent value
        - historical_data: List of HistoricalDataPoint objects
        - Optional change metrics (quarterly_change_pct_points, monthly_change_percent)
        - Optional yoy_inflation_rate (CPI only)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    try:
        indicators = {
            "gdp_growth": (
                "A191RL1Q225SBEA",
                True,
                "quarterly_change_pct_points",
            ),
            "inflation_cpi": (
                "CPIAUCSL",
                False,
                "monthly_change_percent",
            ),
            "consumer_sentiment": (
                "UMCSENT",
                False,
                "monthly_change_percent",
            ),
        }

        results = {}

        for name, (code, is_rate, change_label) in indicators.items():
            try:
                # Fetch data from FRED
                data = pdr.DataReader(code, "fred", start_date, end_date)

                if not data.empty:
                    # Get the most recent non-null value
                    latest_value = data.iloc[-1, 0]
                    latest_date = data.index[-1]

                    # Calculate change from previous period
                    change = None
                    if len(data) > 1:
                        prev_value = data.iloc[-2, 0]
                        if pd.notna(prev_value) and pd.notna(latest_value):
                            if is_rate:
                                change = latest_value - prev_value
                            else:
                                change = (
                                    (latest_value - prev_value) / prev_value
                                ) * 100

                    # Prepare historical data
                    historical_data = []
                    for date, value in data.iterrows():
                        if pd.notna(value.iloc[0]):
                            historical_data.append(
                                HistoricalDataPoint(
                                    date=date.strftime("%Y-%m-%d"),
                                    value=float(value.iloc[0]),
                                )
                            )

                    # Build result dictionary with optional change field
                    result_kwargs = {
                        "latest_value": float(latest_value),
                        "latest_date": latest_date.strftime("%Y-%m-%d"),
                        "historical_data": historical_data,
                    }
                    if change is not None:
                        result_kwargs[change_label] = round(change, 2)

                    results[name] = IndicatorData(**result_kwargs)
                else:
                    results[name] = IndicatorData(
                        latest_value=0.0,
                        latest_date="",
                        historical_data=[],
                        error="No data available",
                    )

            except Exception as e:
                results[name] = IndicatorData(
                    latest_value=0.0,
                    latest_date="",
                    historical_data=[],
                    error=str(e),
                )

        # Calculate year-over-year inflation for CPI
        if "inflation_cpi" in results and results["inflation_cpi"].error is None:
            try:
                year_ago = end_date - timedelta(days=365)
                cpi_year_ago_data = pdr.DataReader(
                    "CPIAUCSL",
                    "fred",
                    year_ago - timedelta(days=30),
                    year_ago + timedelta(days=30),
                )
                if not cpi_year_ago_data.empty:
                    cpi_year_ago = cpi_year_ago_data.iloc[-1, 0]
                    current_cpi = results["inflation_cpi"].latest_value
                    yoy_inflation = ((current_cpi - cpi_year_ago) / cpi_year_ago) * 100
                    # Update the CPI result with YoY inflation
                    results["inflation_cpi"] = results["inflation_cpi"].model_copy(
                        update={"yoy_inflation_rate": round(yoy_inflation, 2)}
                    )
            except:
                pass

        return MacroDataResponse(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data=results,
        )

    except Exception as e:
        return MacroDataResponse(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data={},
            error=f"Failed to retrieve macro data: {str(e)}",
        )


if __name__ == "__main__":
    print("Fetching macroeconomic data from FRED...\n")

    # Get macro data
    response = get_macro_data(period_days=365)

    # Pretty print the results
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Data retrieved at: {response.timestamp}\n")
        print("=" * 80)

        for indicator, info in response.data.items():
            print(f"\n{indicator.replace('_', ' ').upper()}")
            print("-" * 40)

            if info.error:
                print(f"  Error: {info.error}")
            else:
                print(f"  Latest Value: {info.latest_value:.2f}")
                print(f"  Latest Date: {info.latest_date}")

                # Display change metrics based on indicator type
                if info.quarterly_change_pct_points is not None:
                    change = info.quarterly_change_pct_points
                    change_symbol = "↑" if change > 0 else "↓"
                    print(f"  Quarterly Change: {change_symbol} {abs(change):.2f} ppt")
                elif info.monthly_change_percent is not None:
                    change = info.monthly_change_percent
                    change_symbol = "↑" if change > 0 else "↓"
                    print(f"  Monthly Change: {change_symbol} {abs(change):.2f}%")

                if info.yoy_inflation_rate is not None:
                    print(f"  Year-over-Year Rate: {info.yoy_inflation_rate:.2f}%")

        print("\n" + "=" * 80)
        print("\nFull data saved to 'macro_data.json'")

        # Save to JSON file
        with open("macro_data.json", "w") as f:
            json.dump(response.model_dump(), f, indent=2)
