from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

from agents.headline_agent import get_headline_sentiment
from agents.industry_agent import get_industry_sentiment
from models.state import EquityResearchState
from agents.aggregator_agent import get_aggregated_sentiment
from agents.fundamentals_agent import get_fundamental_sentiment
from agents.macro_agent import get_macro_sentiment
from agents.technical_agent import get_technical_sentiment

load_dotenv()


# graph nodes
def fundamental_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate fundamental research sentiment"""
    fundamental_sentiment = get_fundamental_sentiment(ticker=state.ticker)
    return {"fundamental_sentiment": fundamental_sentiment}


def technical_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    technical_sentiment = get_technical_sentiment(ticker=state.ticker)
    return {"technical_sentiment": technical_sentiment}


def macro_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate macro research sentiment"""
    macro_sentiment = get_macro_sentiment()
    return {"macro_sentiment": macro_sentiment}


def industry_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    industry_sentiment = get_industry_sentiment(ticker=state.ticker)
    return {"industry_sentiment": industry_sentiment}


def headline_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    headline_sentiment = get_headline_sentiment(ticker=state.ticker)
    return {"headline_sentiment": headline_sentiment}


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment"""
    combined_sentiment = get_aggregated_sentiment(state)
    return {"combined_sentiment": combined_sentiment}


# build workflow
parallel_builder = StateGraph(EquityResearchState)

# add agent nodes
parallel_builder.add_node("fundamental_research_agent", fundamental_research_agent)
parallel_builder.add_node("technical_research_agent", technical_research_agent)
parallel_builder.add_node("macro_research_agent", macro_research_agent)
parallel_builder.add_node("industry_research_agent", industry_research_agent)
parallel_builder.add_node("headline_research_agent", headline_research_agent)
parallel_builder.add_node("aggregator", sentiment_aggregator)

# call research agents in parallel
parallel_builder.add_edge(START, "fundamental_research_agent")
parallel_builder.add_edge(START, "technical_research_agent")
parallel_builder.add_edge(START, "macro_research_agent")
parallel_builder.add_edge(START, "industry_research_agent")
parallel_builder.add_edge(START, "headline_research_agent")

# synthesize sentiment
parallel_builder.add_edge("fundamental_research_agent", "aggregator")
parallel_builder.add_edge("technical_research_agent", "aggregator")
parallel_builder.add_edge("macro_research_agent", "aggregator")
parallel_builder.add_edge("industry_research_agent", "aggregator")
parallel_builder.add_edge("headline_research_agent", "aggregator")

# terminate graph
parallel_builder.add_edge("aggregator", END)

# compile the graph workflow
parallel_workflow = parallel_builder.compile()

ticker = "CCJ"
state = EquityResearchState(
    ticker=ticker,
    fundamental_sentiment="",
    technical_sentiment="",
    macro_sentiment="",
    industry_sentiment="",
    headline_sentiment="",
    combined_sentiment="",
)


result = parallel_workflow.invoke(state)

print("=" * 80)
print("MACRO SENTIMENT:")
print("=" * 80)
print(result.get("macro_sentiment", "NOT FOUND"))
print("\n")
print("=" * 80)
print("INDUSTRY SENTIMENT:")
print("=" * 80)
print(result.get("industry_sentiment", "NOT FOUND"))
print("\n")
print(result["combined_sentiment"])


def input(ticker: str) -> EquityResearchState:
    state = EquityResearchState(
        ticker=ticker,
        fundamental_sentiment="",
        technical_sentiment="",
        macro_sentiment="",
        industry_sentiment="",
        headline_sentiment="",
        combined_sentiment="",
    )
    return state


def output(state: EquityResearchState) -> EquityResearchState:
    return state


research_chain = RunnableLambda(input) | parallel_workflow | RunnableLambda(output)
