from datetime import datetime, timedelta
from typing import Optional

import dotenv

from agents.industry.prompt import industry_research_prompt
from agents.shared.llm_models import LLM_MODELS
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from agents.shared.web_agent import run_search_grounded_agent
from models.agent import IndustrySentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "industry"


def get_industry_sentiment(
    ticker: str,
    industry: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Get industry sentiment using Google search grounding."""
    config = token_config or DEFAULT_TOKEN_CONFIG.industry
    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    prompt = industry_research_prompt.format(
        ticker=ticker,
        industry=industry,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )
    return run_search_grounded_agent(
        agent_name=AGENT_NAME,
        model=LLM_MODELS["google_fast"],
        prompt=prompt,
        output_schema=IndustrySentimentOutput,
        token_config=config,
    )
