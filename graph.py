from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph, START
from langgraph.cache.memory import InMemoryCache
from dotenv import load_dotenv

from agents.headline_agent import get_headline_sentiment
from agents.industry_agent import get_industry_sentiment
from aggregation.agent import get_aggregated_sentiment
from models.state import EquityResearchState
from agents.fundamentals_agent import get_fundamental_sentiment
from macro.agent import get_macro_sentiment
from agents.technical_agent import get_technical_sentiment
from util import create_cache_policy

load_dotenv()


# graph nodes
def fundamental_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate fundamental research sentiment"""
    fundamental_sentiment = get_fundamental_sentiment(
        ticker=state.ticker, trade_duration=state.trade_duration
    )
    return {"fundamental_sentiment": fundamental_sentiment}


def technical_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    technical_sentiment = get_technical_sentiment(
        ticker=state.ticker, trade_duration=state.trade_duration
    )
    return {"technical_sentiment": technical_sentiment}


def macro_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate macro research sentiment"""
    macro_sentiment = get_macro_sentiment()
    return {"macro_sentiment": macro_sentiment}


def industry_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    industry_sentiment = get_industry_sentiment(
        ticker=state.ticker, trade_duration=state.trade_duration
    )
    return {"industry_sentiment": industry_sentiment}


def headline_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    headline_sentiment = get_headline_sentiment(
        ticker=state.ticker, trade_duration=state.trade_duration
    )
    return {"headline_sentiment": headline_sentiment}


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment"""
    combined_sentiment = get_aggregated_sentiment(state)
    return {"combined_sentiment": combined_sentiment}


# configure graph cache
cache = InMemoryCache()


# build workflow
parallel_builder = StateGraph(EquityResearchState)

# add agent nodes
parallel_builder.add_node(
    "fundamental_research_agent",
    fundamental_research_agent,
    # evict fundamentals cache after one houe
    # todo: get smart about dynamic cache eviction; set ttl based on last earnings release for ticker
    cache_policy=create_cache_policy(ttl=3600),
)
parallel_builder.add_node(
    "technical_research_agent",
    technical_research_agent,
    # evict technical research cache after 5 minutes
    cache_policy=create_cache_policy(ttl=300),
)
parallel_builder.add_node(
    "macro_research_agent",
    macro_research_agent,
    # evict macro research cache after one hour
    # todo: get smart about dynamic cache eviction; set ttl based on last fed report issuance
    cache_policy=create_cache_policy(ttl=3600, static_key="macro_research"),
)
parallel_builder.add_node(
    "industry_research_agent",
    industry_research_agent,
    # evict industry research cache after one hour
    cache_policy=create_cache_policy(ttl=3600),
)

parallel_builder.add_node(
    "headline_research_agent",
    headline_research_agent,
    # evict headline research cache after one hour
    cache_policy=create_cache_policy(ttl=3600),
)
parallel_builder.add_node(
    "aggregator",
    sentiment_aggregator,
)

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
parallel_workflow = parallel_builder.compile(cache=cache)

# uncomment to regenerate architectural diagram

# png_data = parallel_workflow.get_graph().draw_mermaid_png()

# with open("architecture.png", "wb") as f:
#     f.write(png_data)


def input(input_dict: dict) -> EquityResearchState:
    state = EquityResearchState(
        ticker=input_dict["ticker"],
        trade_duration=input_dict["trade_duration"],
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


# pipeline to interface with the API
research_chain = RunnableLambda(input) | parallel_workflow | RunnableLambda(output)
