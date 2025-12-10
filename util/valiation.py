import yfinance as yf

from util.logger import get_logger
from models.state import EquityResearchState

logger = get_logger(__name__)


def validate_ticker(ticker: str, state: EquityResearchState) -> dict:
    try:
        yf_ticker = yf.Ticker(state.ticker)
        # Check if ticker has valid info by attempting to access basic info
        info = yf_ticker.info
        # A valid ticker should have at least some basic info like symbol or regularMarketPrice
        is_ticker = bool("longName" in info and info["longName"] is not None)

        if is_ticker:
            industry = info.get("industry")
            business = info.get("longName")
            # Cache the full info dict to avoid duplicate yfinance API calls
            return {
                "is_ticker_valid": True,
                "industry": industry,
                "business": business,
                "ticker_info": info,
            }
        else:
            return {"is_ticker_valid": False}
    except Exception as e:
        logger.warning(f"Ticker validation failed for {ticker}: {e}")
        return {"is_ticker_valid": False}
