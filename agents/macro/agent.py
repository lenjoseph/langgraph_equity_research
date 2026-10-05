import time
from typing import Optional

import dotenv

from agents.macro.prompt import macro_research_prompt
from agents.macro.tools import get_macro_data_tool
from agents.shared.agent_utils import run_agent_with_tools
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from agents.shared.run_result import finalize_llm_call
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import MacroSentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "macro"


def get_macro_sentiment(
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Generate macro sentiment analysis."""
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.macro
    model = LLM_MODELS["open_ai_smart"]

    llm = get_openai_llm(
        model=model,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    call = run_agent_with_tools(
        llm,
        macro_research_prompt,
        [get_macro_data_tool],
        MacroSentimentOutput,
        track_tokens=True,
        token_budget=config.token_budget,
    )
    return finalize_llm_call(AGENT_NAME, model, start_time, call, config.token_budget)
