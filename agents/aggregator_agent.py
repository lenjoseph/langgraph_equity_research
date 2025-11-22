import dotenv
from langchain_openai import ChatOpenAI

from models.agent_output import AggregatedSentimentOutput
from models.state import EquityResearchState
from prompts.aggregator_prompt import research_aggregation_prompt
from constants.llm_models import LLM_MODELS
from agents.agent_utils import run_agent_with_tools

dotenv.load_dotenv()


def get_aggregated_sentiment(state: EquityResearchState):
    prompt = f"{research_aggregation_prompt}\n\nAggregate the following equity research: {state}"
    model = LLM_MODELS["open_ai"]
    llm = ChatOpenAI(model=model)
    result = run_agent_with_tools(
        llm, prompt, structured_output=AggregatedSentimentOutput
    )

    # Convert structured output to string format
    if isinstance(result, AggregatedSentimentOutput):
        from models.trade_duration_utils import trade_duration_to_label

        trade_duration_label = trade_duration_to_label(state.trade_duration)
        formatted_output = f"**Trade Duration:** {trade_duration_label} - {result.trade_duration_category}\n\n"
        formatted_output += "**Summary of Research Findings:**\n"
        formatted_output += f"- Fundamental: {result.research_summaries.fundamental}\n"
        formatted_output += f"- Technical: {result.research_summaries.technical}\n"
        formatted_output += f"- Macro: {result.research_summaries.macro}\n"
        formatted_output += f"- Industry: {result.research_summaries.industry}\n"
        formatted_output += f"- Headline: {result.research_summaries.headline}\n"
        formatted_output += f"\n**Overall Sentiment:** {result.overall_sentiment}\n\n"
        formatted_output += f"**Conclusion:** {result.conclusion}"
        return formatted_output

    return result
