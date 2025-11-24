import dotenv
from langchain_openai import ChatOpenAI

from models.api import TradeDuration
from models.trade_duration_utils import trade_duration_to_label
from prompts.fundamentals_prompt import fundamentals_research_prompt
from constants.llm_models import LLM_MODELS
from tools.get_fundamentals import get_fundamentals_tool
from agents.agent_utils import run_agent_with_tools

dotenv.load_dotenv()


def get_fundamental_sentiment(ticker: str, trade_duration: TradeDuration):
    trade_duration_label = trade_duration_to_label(trade_duration)
    prompt = f"{fundamentals_research_prompt}\n\nAnalyze the business fundamentals for ticker: {ticker}\nTrade Duration: {trade_duration_label}"
    tools = [get_fundamentals_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
