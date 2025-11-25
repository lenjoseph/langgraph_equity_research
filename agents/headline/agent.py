import dotenv
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.headline.prompt import headline_research_prompt
from agents.shared.llm_models import LLM_MODELS


dotenv.load_dotenv()


def get_headline_sentiment(ticker: str):
    """
    Get headline sentiment using Google's built-in search grounding.
    Google Search is configured via model_kwargs as it's a native Gemini feature.
    """

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    prompt = headline_research_prompt.format(
        ticker=ticker,
        current_date=current_date,
        cutoff_date=cutoff_date,
    )
    model = LLM_MODELS["google"]

    # Configure Google Search grounding via model_kwargs
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        model_kwargs={"tools": [{"google_search_retrieval": {}}]},
    )

    result = llm.invoke(prompt)
    return result.content
