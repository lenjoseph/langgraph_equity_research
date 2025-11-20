import dotenv
from langchain_openai import ChatOpenAI

from models.agent_output import FundamentalAnalysisOutput
from prompts.fundamentals_prompt import fundamentals_research_prompt
from constants.llm_models import LLM_MODELS
from tools.get_fundamentals import get_fundamentals_tool
from agents.agent_utils import run_agent_with_tools, format_analysis_output

dotenv.load_dotenv()


def get_fundamental_sentiment(ticker: str, trade_duration_days: int):
    prompt = f"{fundamentals_research_prompt}\n\nAnalyze the business fundamentals for ticker: {ticker}\nTrade Duration: {trade_duration_days} days"
    tools = [get_fundamentals_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(
        llm, prompt, tools, structured_output=FundamentalAnalysisOutput
    )
    return format_analysis_output(result)
