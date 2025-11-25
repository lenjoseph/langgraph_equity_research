import dotenv
from langchain_openai import ChatOpenAI

from agents.macro.prompt import macro_research_prompt
from agents.macro.tools import get_macro_data_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS

dotenv.load_dotenv()


def get_macro_sentiment():
    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model, temperature=0.0)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
