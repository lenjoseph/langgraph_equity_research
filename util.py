from langgraph.types import CachePolicy
import yfinance as yf

from models.state import EquityResearchState


def create_cache_policy(ttl: int, static_key: str | None = None) -> CachePolicy:
    """Util to create cache policies for research agents.

    Args:
        ttl: Time to live in seconds
        static_key: If provided, uses a static key. Otherwise, uses ticker.

    Returns:
        CachePolicy instance
    """
    if static_key:
        return CachePolicy(key_func=lambda x: static_key.encode(), ttl=ttl)

    # Handle both dict and object state representations
    # graph drawing in langsmith requires dict representation
    # graph execution requires pydantic object representation
    def key_func(x):
        if isinstance(x, dict):
            # Handle missing keys (e.g., when drawing graph without state)
            ticker = x.get("ticker", "default")
            return f"{ticker}".encode()
        return f"{x.ticker}".encode()

    return CachePolicy(key_func=key_func, ttl=ttl)


def validate_ticker(ticker: str, state: EquityResearchState) -> dict:
    try:
        ticker = yf.Ticker(state.ticker)
        # Check if ticker has valid info by attempting to access basic info
        info = ticker.info
        # A valid ticker should have at least some basic info like symbol or regularMarketPrice
        is_ticker = bool("longName" in info and info["longName"] is not None)

        if is_ticker:
            industry = info.get("industry")
            business = info.get("longName")
            return {"is_ticker_valid": True, "industry": industry, "business": business}
        else:
            return {"is_ticker_valid": False}
    except:
        return {"is_ticker_valid": False}


def draw_architecture(graph_workflow):
    try:
        png_data = graph_workflow.get_graph().draw_mermaid_png()
        with open("architecture.png", "wb") as f:
            f.write(png_data)
    except Exception as e:
        print(f"Error generating architecture.png: {e}")
        # Fallback to writing mermaid text
        with open("architecture.mmd", "w") as f:
            f.write(graph_workflow.get_graph().draw_mermaid())
