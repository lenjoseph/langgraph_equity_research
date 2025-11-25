import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import Tool

from models.tools import FundamentalsData, FundamentalsInput


# Financial metrics configuration
INCOME_METRICS = [
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "Net Income",
    "EBITDA",
    "Basic EPS",
    "Diluted EPS",
    "Interest Expense",
    "Tax Provision",
]

BALANCE_SHEET_METRICS = [
    "Total Assets",
    "Current Assets",
    "Cash And Cash Equivalents",
    "Inventory",
    "Total Liabilities",
    "Current Liabilities",
    "Total Debt",
    "Net Debt",
    "Stockholders Equity",
    "Working Capital",
]

CASH_FLOW_METRICS = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Free Cash Flow",
    "Capital Expenditure",
    "Repayment Of Debt",
    "Issuance Of Debt",
    "Repurchase Of Capital Stock",
    "Cash Dividends Paid",
]


def _clean_dict_for_json(obj: Any) -> Any:
    """Recursively clean dictionary to ensure JSON serializability."""
    if isinstance(obj, dict):
        return {str(k): _clean_dict_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_dict_for_json(item) for item in obj]
    if isinstance(obj, (float, int, np.number)):
        # Check for NaN or inf
        if pd.isna(obj) or np.isinf(obj):
            return None
        return float(obj) if isinstance(obj, (float, np.floating)) else int(obj)
    if pd.isna(obj):
        return None
    return obj


def _convert_df_to_dict(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Convert a pandas DataFrame to a JSON-serializable dictionary."""
    if df is None or df.empty:
        return None

    try:
        df_clean = df.copy()
        # Convert index/columns to strings if they are timestamps
        if isinstance(df_clean.columns, pd.DatetimeIndex):
            df_clean.columns = df_clean.columns.strftime("%Y-%m-%d")
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = df_clean.index.strftime("%Y-%m-%d")

        return _clean_dict_for_json(df_clean.to_dict())
    except Exception as e:
        print(f"Warning: Failed to convert DataFrame to JSON: {e}")
        return None


def _process_dataframe(
    df: Optional[pd.DataFrame], keep_rows: List[str], limit_columns: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Process DataFrame: limit columns, filter rows, and convert to dict.
    """
    if df is None or df.empty:
        return None

    try:
        # Limit columns (assume sorted by date descending)
        if len(df.columns) > limit_columns:
            df_subset = df.iloc[:, :limit_columns]
        else:
            df_subset = df

        # Filter rows
        valid_indices = [idx for idx in df_subset.index if idx in keep_rows]
        if valid_indices:
            df_subset = df_subset.loc[valid_indices]

        return _convert_df_to_dict(df_subset)
    except Exception as e:
        print(f"Warning: Error processing DataFrame: {e}")
        return _convert_df_to_dict(df)


def _get_earnings_data(
    annual_stmt: Optional[pd.DataFrame], quarterly_stmt: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """Extract Net Income (Earnings) from income statements."""
    earnings = {"annual": None, "quarterly": None}

    for key, stmt in [("annual", annual_stmt), ("quarterly", quarterly_stmt)]:
        if stmt is not None and "Net Income" in stmt.index:
            earnings[key] = _convert_df_to_dict(stmt.loc[["Net Income"]])

    return earnings


def get_earnings_and_financial_health(ticker: str) -> FundamentalsData:
    """
    Get comprehensive earnings and financial health data for a given ticker.

    Returns:
        FundamentalsData: Structured output with earnings, financial statements,
        ratios, and valuation metrics for fundamental analysis.
    """
    try:
        stock = yf.Ticker(ticker)

        # 1. Fetch Financial Statements (these trigger API calls)
        balance_sheet_annual = stock.balance_sheet
        balance_sheet_quarterly = stock.quarterly_balance_sheet
        income_annual = stock.income_stmt
        income_quarterly = stock.quarterly_income_stmt
        cash_flow_annual = stock.cashflow
        cash_flow_quarterly = stock.quarterly_cashflow

        # 2. Extract Earnings
        earnings = _get_earnings_data(income_annual, income_quarterly)

        # 3. Process Statements
        balance_sheets = {
            "annual": _process_dataframe(
                balance_sheet_annual, BALANCE_SHEET_METRICS, 3
            ),
            "quarterly": _process_dataframe(
                balance_sheet_quarterly, BALANCE_SHEET_METRICS, 4
            ),
        }
        income_statements = {
            "annual": _process_dataframe(income_annual, INCOME_METRICS, 3),
            "quarterly": _process_dataframe(income_quarterly, INCOME_METRICS, 4),
        }
        cash_flows = {
            "annual": _process_dataframe(cash_flow_annual, CASH_FLOW_METRICS, 3),
            "quarterly": _process_dataframe(cash_flow_quarterly, CASH_FLOW_METRICS, 4),
        }

        # 4. Get Company Info & Ratios
        info = stock.info

        ratios = {
            "P/E_ratio": info.get("trailingPE"),
            "forward_P/E": info.get("forwardPE"),
            "PEG_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
        }

        valuation_metrics = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "beta": info.get("beta"),
            "total_revenue": info.get("totalRevenue"),
            "revenue_per_share": info.get("revenuePerShare"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            "ebitda": info.get("ebitda"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
        }

        return FundamentalsData(
            ticker=ticker,
            company_name=info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            earnings=earnings,
            balance_sheet=balance_sheets,
            income_statement=income_statements,
            cash_flow=cash_flows,
            ratios=_clean_dict_for_json(ratios),
            valuation_metrics=_clean_dict_for_json(valuation_metrics),
            current_price=info.get("currentPrice"),
            target_price=_clean_dict_for_json(
                {
                    "mean": info.get("targetMeanPrice"),
                    "high": info.get("targetHighPrice"),
                    "low": info.get("targetLowPrice"),
                }
            ),
        )

    except Exception as e:
        traceback.print_exc()
        return FundamentalsData(
            ticker=ticker,
            error=str(e),
            message=f"Failed to retrieve data for {ticker}",
        )


get_fundamentals_tool = Tool(
    name="get_fundamentals_tool",
    description="Use this tool to get comprehensive fundamental analysis data for a stock. Returns earnings, financial statements (balance sheet, income statement, cash flow), key financial ratios (P/E, ROE, margins, debt ratios), and valuation metrics (market cap, EV, beta, FCF) useful for DCF and comparable company analysis.",
    func=get_earnings_and_financial_health,
    args_schema=FundamentalsInput,
)
