from datetime import datetime, timedelta
from typing import Optional

import dotenv

from agents.headline.prompt import headline_research_prompt
from agents.shared.llm_models import LLM_MODELS
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from agents.shared.web_agent import run_search_grounded_agent
from models.agent import HeadlineSentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "headline"


def get_headline_sentiment(
    business: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Get headline sentiment using Google search grounding."""
    config = token_config or DEFAULT_TOKEN_CONFIG.headline
    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    prompt = headline_research_prompt.format(
        business=business,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )
    return run_search_grounded_agent(
        agent_name=AGENT_NAME,
        model=LLM_MODELS["google_fast"],
        prompt=prompt,
        output_schema=HeadlineSentimentOutput,
        token_config=config,
    )
