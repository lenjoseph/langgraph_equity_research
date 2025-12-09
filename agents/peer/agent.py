import time
from datetime import datetime, timedelta
from typing import Tuple

import dotenv

from agents.peer.prompt import peer_research_prompt
from agents.shared.llm_models import LLM_MODELS, get_google_llm
from agents.shared.agent_utils import invoke_llm_with_metrics
from models.agent import PeerSentimentOutput
from models.metrics import AgentMetrics


dotenv.load_dotenv()

AGENT_NAME = "peer"


def get_peer_sentiment(business: str) -> Tuple[PeerSentimentOutput, AgentMetrics]:
    """
    Get peer sentiment using Google's built-in search grounding.

    Returns:
        Tuple of (PeerSentimentOutput, AgentMetrics)
    """
    start_time = time.perf_counter()
    model = LLM_MODELS["google_fast"]

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    prompt = peer_research_prompt.format(
        business=business,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )

    llm = get_google_llm(
        model=model,
        temperature=0.0,
        with_search_grounding=True,
    )

    result, token_usage = invoke_llm_with_metrics(llm, prompt, PeerSentimentOutput)

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics = AgentMetrics(
        agent_name=AGENT_NAME,
        latency_ms=latency_ms,
        token_usage=token_usage,
        model=model,
    )
    return result, metrics
