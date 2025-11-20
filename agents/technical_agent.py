import dotenv
from langchain_openai import ChatOpenAI

from models.agent_output import TechnicalAnalysisOutput
from tools.get_technicals import get_technical_analysis_tool
from prompts.technicals_prompt import technical_research_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools, format_analysis_output

dotenv.load_dotenv()


def get_technical_sentiment(ticker: str, trade_duration_days: int):
    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}\nTrade Duration: {trade_duration_days} days"
    tools = [get_technical_analysis_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(
        llm, prompt, tools, structured_output=TechnicalAnalysisOutput
    )
    return format_analysis_output(result)
