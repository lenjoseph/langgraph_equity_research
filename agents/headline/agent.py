import dotenv
from datetime import datetime, timedelta

from agents.headline.prompt import headline_research_prompt
from agents.shared.llm_models import LLM_MODELS, get_google_llm


dotenv.load_dotenv()


def get_headline_sentiment(business: str):
    """
    Get headline sentiment using Google's built-in search grounding.
    Google Search is configured via model_kwargs as it's a native Gemini feature.
    """

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    prompt = headline_research_prompt.format(
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
