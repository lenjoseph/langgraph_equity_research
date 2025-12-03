from typing import Any, Dict, Optional

import dotenv
from langchain_openai import ChatOpenAI

from agents.fundamentals.prompt import fundamentals_research_prompt
from agents.fundamentals.tools import (
    get_fundamentals_tool,
    get_earnings_and_financial_health,
)
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS

dotenv.load_dotenv()


def get_fundamental_sentiment(
    ticker: str, cached_info: Optional[Dict[str, Any]] = None
):
    """
    Generate fundamental sentiment analysis for a ticker.

    Args:
        ticker: Stock ticker symbol
        cached_info: Optional pre-fetched yfinance ticker.info to avoid duplicate API calls
    """
    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model, temperature=0.0)

    if cached_info is not None:
        # Use cached info - call function directly instead of via tool
        fundamentals_data = get_earnings_and_financial_health(
            ticker=ticker, cached_info=cached_info
        )
        # Inject the data directly into the prompt for analysis
        prompt = f"{fundamentals_research_prompt}\n\n"
        prompt += f"Analyze the business fundamentals for ticker: {ticker}\n\n"
        prompt += f"Here is the fundamental data:\n{fundamentals_data.model_dump_json(indent=2)}"
        result = llm.invoke(prompt)
        return result.content
    else:
        # No cached info - use tool-calling approach
        prompt = f"{fundamentals_research_prompt}\n\nAnalyze the business fundamentals for ticker: {ticker}"
        tools = [get_fundamentals_tool]
        result = run_agent_with_tools(llm, prompt, tools)
        return result
