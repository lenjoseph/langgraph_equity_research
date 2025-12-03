import dotenv
from datetime import datetime, timedelta

from agents.peer.prompt import peer_research_prompt
from agents.shared.llm_models import LLM_MODELS, get_google_llm


dotenv.load_dotenv()


def get_peer_sentiment(business: str):
    """
    Get peer sentiment using Google's built-in search grounding.
    """

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    prompt = peer_research_prompt.format(
        business=business,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )

    llm = get_google_llm(
        model=LLM_MODELS["google"],
        temperature=0.0,
        with_search_grounding=True,
    )

    result = llm.invoke(prompt)
    return result.content
