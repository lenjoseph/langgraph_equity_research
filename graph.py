from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph, START
from langgraph.cache.memory import InMemoryCache
from dotenv import load_dotenv
from fastapi import HTTPException


from agents.evaluation.agent import evaluate_aggregated_sentement
from agents.headline.agent import get_headline_sentiment
from agents.industry.agent import get_industry_sentiment
from agents.peer.agent import get_peer_sentiment
from agents.aggregation.agent import get_aggregated_sentiment
from logger import get_logger
from models.state import EquityResearchState
from agents.fundamentals.agent import get_fundamental_sentiment
from agents.macro.agent import get_macro_sentiment
from agents.technical.agent import get_technical_sentiment
from util import create_cache_policy, draw_architecture, validate_ticker

load_dotenv()
logger = get_logger(__name__)


# graph nodes
def ticker_validation(state: EquityResearchState) -> dict:
    """Validation node to ensure we have a real ticker"""
    logger.info(f"Validating ticker: {state.ticker}")
    result = validate_ticker(ticker=state.ticker, state=state)
    logger.info(f"Ticker validation complete for {state.ticker}")
    return result


def ticker_router(state: EquityResearchState):
    if state.is_ticker_valid:
        return [
            "fundamental_research_agent",
            "technical_research_agent",
            "macro_research_agent",
            "industry_research_agent",
            "peer_research_agent",
            "headline_research_agent",
        ]
    else:
        return END


def fundamental_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate fundamental research sentiment"""
    logger.info(f"Starting fundamental research for {state.ticker}")
    fundamental_sentiment = get_fundamental_sentiment(
        ticker=state.ticker,
        cached_info=state.ticker_info,  # Pass cached yfinance info to avoid duplicate API call
    )
    logger.info(f"Completed fundamental research for {state.ticker}")
    return {"fundamental_sentiment": fundamental_sentiment}


def technical_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    logger.info(f"Starting technical research for {state.ticker}")
    technical_sentiment = get_technical_sentiment(
        ticker=state.ticker,
    )
    logger.info(f"Completed technical research for {state.ticker}")
    return {"technical_sentiment": technical_sentiment}


def macro_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate macro research sentiment"""
    logger.info("Starting macro research")
    macro_sentiment = get_macro_sentiment()
    logger.info("Completed macro research")
    return {"macro_sentiment": macro_sentiment}


def industry_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    logger.info(f"Starting industry research for {state.ticker}")
    industry_sentiment = get_industry_sentiment(
        ticker=state.ticker, industry=state.industry
    )
    logger.info(f"Completed industry research for {state.ticker}")
    return {"industry_sentiment": industry_sentiment}


def peer_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate peer research sentiment"""
    logger.info(f"Starting peer research for {state.business}")
    peer_sentiment = get_peer_sentiment(business=state.business)
    logger.info(f"Completed peer research for {state.business}")
    return {"peer_sentiment": peer_sentiment}


def headline_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    logger.info(f"Starting headline research for {state.business}")
    headline_sentiment = get_headline_sentiment(
        business=state.business,
    )
    logger.info(f"Completed headline research for {state.business}")
    return {"headline_sentiment": headline_sentiment}


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment"""
    logger.info(f"Starting sentiment aggregation for {state.ticker}")
    combined_sentiment = get_aggregated_sentiment(state)
    logger.info(f"Completed sentiment aggregation for {state.ticker}")
    return {"combined_sentiment": combined_sentiment}


def sentiment_evaluator(state: EquityResearchState) -> dict:
    """LLM call to evaluate sentiment aggregator output"""
    logger.info("starting sentiment evaluation")
    sentiment_evaluation = evaluate_aggregated_sentement(
        sentiment=state.combined_sentiment
    )
    logger.info("Completed sentiment evaluation")
    sentiment_evaluation["revision_iteration_count"] = (
        state.revision_iteration_count + 1
    )
    return sentiment_evaluation


def sentiment_router(state: EquityResearchState):
    "Route back to aggregator or terminate based on evaluator feedback"
    if state.compliant == True:
        return "Compliant"
    elif state.revision_iteration_count > 2:
        return "Compliant"
    else:
        return "Noncompliant"


# build workflow
graph_builder = StateGraph(EquityResearchState)

# add agent nodes
graph_builder.add_node("ticker_validation", ticker_validation)

graph_builder.add_node(
    "fundamental_research_agent",
    fundamental_research_agent,
    # evict fundamentals cache after one houe
    # todo: get smart about dynamic cache eviction; set ttl based on last earnings release for ticker
    cache_policy=create_cache_policy(ttl=3600),
)

graph_builder.add_node(
    "technical_research_agent",
    technical_research_agent,
    # evict technical research cache after 5 minutes
    cache_policy=create_cache_policy(ttl=300),
)

graph_builder.add_node(
    "macro_research_agent",
    macro_research_agent,
    # evict macro research cache after one hour
    # todo: get smart about dynamic cache eviction; set ttl based on last fed report issuance
    cache_policy=create_cache_policy(ttl=3600, static_key="macro_research"),
)

graph_builder.add_node(
    "industry_research_agent",
    industry_research_agent,
    # evict industry research cache after one hour
    cache_policy=create_cache_policy(ttl=3600),
)

graph_builder.add_node(
    "peer_research_agent",
    peer_research_agent,
    # evict peer research cache after one hour
    cache_policy=create_cache_policy(ttl=3600),
)

graph_builder.add_node(
    "headline_research_agent",
    headline_research_agent,
    # evict headline research cache after one hour
    cache_policy=create_cache_policy(ttl=3600),
)

graph_builder.add_node(
    "aggregator",
    sentiment_aggregator,
)

graph_builder.add_node("evaluator", sentiment_evaluator)

# call research agents in parallel when ticker validation passes, otherwise end
graph_builder.add_edge(START, "ticker_validation")
graph_builder.add_conditional_edges(
    "ticker_validation",
    ticker_router,
    [
        "fundamental_research_agent",
        "technical_research_agent",
        "macro_research_agent",
        "industry_research_agent",
        "peer_research_agent",
        "headline_research_agent",
        END,
    ],
)
# synthesize sentiment
graph_builder.add_edge("fundamental_research_agent", "aggregator")
graph_builder.add_edge("technical_research_agent", "aggregator")
graph_builder.add_edge("macro_research_agent", "aggregator")
graph_builder.add_edge("industry_research_agent", "aggregator")
graph_builder.add_edge("peer_research_agent", "aggregator")
graph_builder.add_edge("headline_research_agent", "aggregator")
graph_builder.add_edge("aggregator", "evaluator")
# evaluation-optimization feedback loop with configured iteraation count
graph_builder.add_conditional_edges(
    "evaluator", sentiment_router, {"Compliant": END, "Noncompliant": "aggregator"}
)

# compile the graph workflow with node caching
cache = InMemoryCache()

graph_workflow = graph_builder.compile(cache=cache)

# uncomment to regenerate architectural diagram

# draw_architecture(graph_workflow)


def input(input_dict: dict) -> EquityResearchState:
    state = EquityResearchState(
        ticker=input_dict["ticker"],
        trade_duration=input_dict["trade_duration"],
        trade_direction=input_dict["trade_direction"],
        industry="",
        business="",
        fundamental_sentiment="",
        technical_sentiment="",
        macro_sentiment="",
        industry_sentiment="",
        peer_sentiment="",
        headline_sentiment="",
        combined_sentiment="",
        compliant=False,
        feedback=None,
        is_ticker_valid=False,
        revision_iteration_count=0,
        ticker_info=None,  # Will be populated by ticker_validation node
    )
    return state


def output(state: dict | EquityResearchState) -> EquityResearchState:
    if isinstance(state, dict):
        state = EquityResearchState(**state)

    if not state.is_ticker_valid:
        raise HTTPException(status_code=400, detail=f"Ticker {state.ticker} is invalid")
    return state


# pipeline to interface with the API
research_chain = RunnableLambda(input) | graph_workflow | RunnableLambda(output)
