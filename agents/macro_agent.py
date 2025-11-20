import dotenv
from langchain_openai import ChatOpenAI

from models.agent_output import MacroAnalysisOutput
from tools.get_macro import get_macro_data_tool
from prompts.macro_prompt import macro_research_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools, format_analysis_output

dotenv.load_dotenv()


def get_macro_sentiment(trade_duration_days: int):
    prompt = f"{macro_research_prompt}\n\nTrade Duration: {trade_duration_days} days"
    tools = [get_macro_data_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(
        llm, prompt, tools, structured_output=MacroAnalysisOutput
    )
    return format_analysis_output(result)
