import dotenv
from langchain_openai import ChatOpenAI

from tools.get_technicals import get_technical_analysis_tool
from prompts.technicals_prompt import technical_research_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools

dotenv.load_dotenv()


def get_technical_sentiment(ticker: str):
    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}"
    tools = [get_technical_analysis_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    return run_agent_with_tools(llm, prompt, tools)
