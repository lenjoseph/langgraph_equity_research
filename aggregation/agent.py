import dotenv
from langchain_openai import ChatOpenAI

from aggregation.prompt import research_aggregation_prompt
from models.state import EquityResearchState
from models.trade_duration_utils import trade_duration_to_label

from shared.agent_utils import run_agent_with_tools
from shared.llm_models import LLM_MODELS


dotenv.load_dotenv()


def get_aggregated_sentiment(state: EquityResearchState):
    trade_duration_label = trade_duration_to_label(state.trade_duration)

    prompt = (
        f"{research_aggregation_prompt}\n\nAggregate the following equity research:\n\n"
    )
    prompt += f"Ticker: {state.ticker}\n"
    prompt += f"Trade Duration: {trade_duration_label}\n\n"
    prompt += f"Fundamental Analysis:\n{state.fundamental_sentiment}\n\n"
    prompt += f"Technical Analysis:\n{state.technical_sentiment}\n\n"
    prompt += f"Macro Analysis:\n{state.macro_sentiment}\n\n"
    prompt += f"Industry Analysis:\n{state.industry_sentiment}\n\n"
    prompt += f"Headline Analysis:\n{state.headline_sentiment}\n\n"

    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(llm, prompt)

    return result
