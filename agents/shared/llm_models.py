from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


LLM_MODELS = {
    "open_ai_fast": "gpt-4o-mini",
    "open_ai_smart": "gpt-5.1",
    "google": "gemini-2.5-flash",
}


@lru_cache(maxsize=8)
def get_openai_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Get a cached ChatOpenAI instance.

    Args:
        model: The OpenAI model name
        temperature: The temperature setting (default 0.0)

    Returns:
        Cached ChatOpenAI instance
    """
    return ChatOpenAI(model=model, temperature=temperature)


@lru_cache(maxsize=4)
def get_google_llm(
    model: str,
    temperature: float = 0.0,
    with_search_grounding: bool = False,
) -> ChatGoogleGenerativeAI:
    """
    Get a cached ChatGoogleGenerativeAI instance.

    Args:
        model: The Google model name
        temperature: The temperature setting (default 0.0)
        with_search_grounding: Whether to enable Google Search grounding

    Returns:
        Cached ChatGoogleGenerativeAI instance
    """
    model_kwargs = {}
    if with_search_grounding:
        model_kwargs["tools"] = [{"google_search_retrieval": {}}]

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        model_kwargs=model_kwargs if model_kwargs else None,
    )
