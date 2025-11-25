import dotenv
from langchain_openai import ChatOpenAI

from agents.fundamentals.prompt import fundamentals_research_prompt
from agents.fundamentals.tools import get_fundamentals_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS

dotenv.load_dotenv()


def get_fundamental_sentiment(ticker: str):
    prompt = f"{fundamentals_research_prompt}\n\nAnalyze the business fundamentals for ticker: {ticker}"
    tools = [get_fundamentals_tool]
    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model, temperature=0.0)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
