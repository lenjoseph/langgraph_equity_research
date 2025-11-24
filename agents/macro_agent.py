import dotenv
from langchain_openai import ChatOpenAI

from tools.get_macro import get_macro_data_tool
from prompts.macro_prompt import macro_research_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools

dotenv.load_dotenv()


def get_macro_sentiment():
    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
