import dotenv

from agents.macro.prompt import macro_research_prompt
from agents.macro.tools import get_macro_data_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm

dotenv.load_dotenv()


def get_macro_sentiment():
    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    llm = get_openai_llm(model=LLM_MODELS["open_ai_smart"], temperature=0.0)
    result = run_agent_with_tools(llm, prompt, tools)
    return result
