import dotenv
from langchain_openai import ChatOpenAI

from agents.aggregation.prompt import research_aggregation_prompt
from models.state import EquityResearchState
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS


dotenv.load_dotenv()


def get_aggregated_sentiment(state: EquityResearchState):

    prompt = (
        f"{research_aggregation_prompt}\n\nAggregate the following equity research:\n\n"
    )
    prompt += f"Ticker: {state.ticker}\n"
    prompt += f"Fundamental Analysis:\n{state.fundamental_sentiment}\n\n"
    prompt += f"Technical Analysis:\n{state.technical_sentiment}\n\n"
    prompt += f"Macro Analysis:\n{state.macro_sentiment}\n\n"
    prompt += f"Industry Analysis:\n{state.industry_sentiment}\n\n"
    prompt += f"Headline Analysis:\n{state.headline_sentiment}\n\n"

    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model, temperature=0.2)
    result = run_agent_with_tools(llm, prompt)

    return result
