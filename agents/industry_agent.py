import dotenv
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from models.agent_output import IndustryAnalysisOutput
from agents.agent_utils import run_agent_with_tools, format_analysis_output
from constants.llm_models import LLM_MODELS
from prompts.industry_prompt import industry_research_prompt


dotenv.load_dotenv()


def get_industry_sentiment(ticker: str, trade_duration_days: int):
    """
    Get industry sentiment using Google's built-in search grounding.
    Google Search is configured via model_kwargs as it's a native Gemini feature.
    """

    current_date = datetime.now().strftime("%Y-%m-%d")
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    prompt = industry_research_prompt.format(
        ticker=ticker,
        current_date=current_date,
        cutoff_date=cutoff_date,
        trade_duration_days=trade_duration_days,
    )
    model = LLM_MODELS["google"]

    # Configure Google Search grounding via model_kwargs
    llm = ChatGoogleGenerativeAI(
        model=model, model_kwargs={"tools": [{"google_search_retrieval": {}}]}
    )

    result = run_agent_with_tools(llm, prompt, structured_output=IndustryAnalysisOutput)
    return format_analysis_output(result)
