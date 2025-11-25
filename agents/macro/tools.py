from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import pandas as pd
import pandas_datareader as pdr
from langchain_core.tools import Tool

from models.tools import (
    HistoricalDataPoint,
    IndicatorData,
    MacroDataResponse,
    MacroDataInput,
)


# Configuration for macroeconomic indicators
# key: (FRED code, is_rate, change_label)
INDICATORS_CONFIG: Dict[str, Tuple[str, bool, str]] = {
    "gdp_growth": ("A191RL1Q225SBEA", True, "quarterly_change_pct_points"),
    "inflation_cpi": ("CPIAUCSL", False, "monthly_change_percent"),
    "consumer_sentiment": ("UMCSENT", False, "monthly_change_percent"),
}


def _fetch_indicator_data(
    code: str,
    is_rate: bool,
    change_label: str,
    start_date: datetime,
    end_date: datetime,
) -> IndicatorData:
    """
    Fetch and process data for a single macroeconomic indicator from FRED.
    """
    try:
        # Fetch data from FRED
        data = pdr.DataReader(code, "fred", start_date, end_date)

        if data.empty:
            return IndicatorData(
                latest_value=0.0,
                latest_date="",
                historical_data=[],
                error="No data available",
            )

        # Get the most recent non-null value
        latest_value = float(data.iloc[-1, 0])
        latest_date = data.index[-1].strftime("%Y-%m-%d")

        # Calculate change from previous period
        change = None
        if len(data) > 1:
            prev_value = data.iloc[-2, 0]
            if pd.notna(prev_value) and pd.notna(latest_value):
                if is_rate:
                    change = latest_value - prev_value
                else:
                    change = ((latest_value - prev_value) / prev_value) * 100

        # Prepare historical data
        historical_data = [
            HistoricalDataPoint(
                date=index.strftime("%Y-%m-%d"), value=float(row.iloc[0])
            )
            for index, row in data.iterrows()
            if pd.notna(row.iloc[0])
        ]

        # Build result dictionary with optional change field
        result_kwargs = {
            "latest_value": latest_value,
            "latest_date": latest_date,
            "historical_data": historical_data,
        }
        if change is not None:
            result_kwargs[change_label] = round(change, 2)

        return IndicatorData(**result_kwargs)

    except Exception as e:
        return IndicatorData(
            latest_value=0.0,
            latest_date="",
            historical_data=[],
            error=str(e),
        )


def _calculate_yoy_inflation(current_cpi: float, end_date: datetime) -> Optional[float]:
    """Calculate Year-over-Year inflation rate for CPI."""
    try:
        year_ago = end_date - timedelta(days=365)
        # Fetch a 60-day window around the target date to ensure we catch the monthly release
        cpi_year_ago_data = pdr.DataReader(
            "CPIAUCSL",
            "fred",
            year_ago - timedelta(days=30),
            year_ago + timedelta(days=30),
        )
        if not cpi_year_ago_data.empty:
            cpi_year_ago = cpi_year_ago_data.iloc[-1, 0]
            return ((current_cpi - cpi_year_ago) / cpi_year_ago) * 100
    except Exception:
        return None
    return None


def get_macro_data() -> MacroDataResponse:
    """
    Retrieve the most recent macroeconomic data from FRED (Federal Reserve Economic Data).

    Returns:
        MacroDataResponse containing GDP Growth, CPI, and Consumer Sentiment.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    results = {}

    try:
        for name, (code, is_rate, change_label) in INDICATORS_CONFIG.items():
            results[name] = _fetch_indicator_data(
                code, is_rate, change_label, start_date, end_date
            )

        # Calculate year-over-year inflation for CPI if available
        if "inflation_cpi" in results and results["inflation_cpi"].error is None:
            yoy_inflation = _calculate_yoy_inflation(
                results["inflation_cpi"].latest_value, end_date
            )
            if yoy_inflation is not None:
                results["inflation_cpi"] = results["inflation_cpi"].model_copy(
                    update={"yoy_inflation_rate": round(yoy_inflation, 2)}
                )

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


get_macro_data_tool = Tool(
    name="get_macro_data_tool",
    description="Use this tool to fetch macroeconomic data including GDP Growth Rate, Consumer Price Index (CPI/inflation), and Consumer Sentiment from FRED. Returns historical data and latest values with change calculations.",
    func=get_macro_data,
    args_schema=MacroDataInput,
)
