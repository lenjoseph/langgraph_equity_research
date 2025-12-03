import dotenv

from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.technical.prompt import technical_research_prompt
from agents.technical.tools import get_technical_analysis_tool

dotenv.load_dotenv()


def get_technical_sentiment(ticker: str):
    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}"
    tools = [get_technical_analysis_tool]
    llm = get_openai_llm(model=LLM_MODELS["open_ai_smart"], temperature=0.0)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
