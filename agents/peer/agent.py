from datetime import datetime, timedelta
from typing import Optional

import dotenv

from agents.peer.prompt import peer_research_prompt
from agents.shared.llm_models import LLM_MODELS
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from agents.shared.web_agent import run_search_grounded_agent
from models.agent import PeerSentimentOutput
from models.result import AgentRunResult

dotenv.load_dotenv()

AGENT_NAME = "peer"


def get_peer_sentiment(
    business: str,
    token_config: Optional[AgentTokenConfig] = None,
) -> AgentRunResult:
    """Get peer sentiment using Google search grounding."""
    config = token_config or DEFAULT_TOKEN_CONFIG.peer
    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    prompt = peer_research_prompt.format(
        business=business,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )
    return run_search_grounded_agent(
        agent_name=AGENT_NAME,
        model=LLM_MODELS["google_fast"],
        prompt=prompt,
        output_schema=PeerSentimentOutput,
        token_config=config,
    )
