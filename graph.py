from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph, START
from langgraph.cache.memory import InMemoryCache
from dotenv import load_dotenv
from fastapi import HTTPException


from agents.evaluation.agent import evaluate_aggregated_sentement
from agents.filings.agent import get_filings_sentiment
from agents.headline.agent import get_headline_sentiment
from agents.industry.agent import get_industry_sentiment
from agents.peer.agent import get_peer_sentiment
from agents.aggregation.agent import get_aggregated_sentiment
from data.util.ingest_sec_filings import ensure_filings_ingested
from logger import get_logger
from models.metrics import RequestMetrics
from models.state import EquityResearchState
from agents.fundamentals.agent import get_fundamental_sentiment
from agents.macro.agent import get_macro_sentiment
from agents.technical.agent import get_technical_sentiment
from util import (
    create_cache_policy,
    create_fundamentals_cache_policy,
    create_technical_cache_policy,
    create_macro_cache_policy,
    draw_architecture,
    format_sentiment_output,
    validate_ticker,
)

load_dotenv()
logger = get_logger(__name__)


# graph nodes
def ticker_validation(state: EquityResearchState) -> dict:
    """Validation node to ensure we have a real ticker"""
    logger.info(f"Validating ticker: {state.ticker}")
    result = validate_ticker(ticker=state.ticker, state=state)
    logger.info(f"Ticker validation complete for {state.ticker}")
    return result


def filings_ingestion(state: EquityResearchState) -> dict:
    """Ingest SEC filings into vector store before research agents run"""
    logger.info(f"Starting SEC filings ingestion for {state.ticker}")
    try:
        was_ingested = ensure_filings_ingested(ticker=state.ticker)
        if was_ingested:
            logger.info(f"SEC filings ingested for {state.ticker}")
        else:
            logger.info(f"SEC filings already available for {state.ticker}")
        return {"filings_ingested": True}
    except Exception as e:
        logger.error(
            f"SEC filings ingestion failed for {state.ticker}: {e}", exc_info=True
        )
        return {"filings_ingested": False}


def ticker_router(state: EquityResearchState):
    """Route to filings ingestion and research agents if ticker is valid, otherwise end"""
    if state.is_ticker_valid:
        return [
            "filings_ingestion",
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
    try:
        fundamental_sentiment, agent_metrics = get_fundamental_sentiment(
            ticker=state.ticker,
            cached_info=state.ticker_info,  # Pass cached yfinance info to avoid duplicate API call
        )
        logger.info(f"Completed fundamental research for {state.ticker}")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "fundamental_sentiment": format_sentiment_output(fundamental_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(
            f"Fundamental research failed for {state.ticker}: {e}", exc_info=True
        )
        return {
            "fundamental_sentiment": "Analysis unavailable due to data retrieval error."
        }


def technical_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    logger.info(f"Starting technical research for {state.ticker}")
    try:
        technical_sentiment, agent_metrics = get_technical_sentiment(
            ticker=state.ticker,
        )
        logger.info(f"Completed technical research for {state.ticker}")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "technical_sentiment": format_sentiment_output(technical_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(
            f"Technical research failed for {state.ticker}: {e}", exc_info=True
        )
        return {
            "technical_sentiment": "Analysis unavailable due to data retrieval error."
        }


def macro_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate macro research sentiment"""
    logger.info("Starting macro research")
    try:
        macro_sentiment, agent_metrics = get_macro_sentiment()
        logger.info("Completed macro research")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "macro_sentiment": format_sentiment_output(macro_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Macro research failed: {e}", exc_info=True)
        return {"macro_sentiment": "Analysis unavailable due to data retrieval error."}


def industry_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate industry research sentiment"""
    logger.info(f"Starting industry research for {state.ticker}")
    try:
        industry_sentiment, agent_metrics = get_industry_sentiment(
            ticker=state.ticker, industry=state.industry
        )
        logger.info(f"Completed industry research for {state.ticker}")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "industry_sentiment": format_sentiment_output(industry_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Industry research failed for {state.ticker}: {e}", exc_info=True)
        return {
            "industry_sentiment": "Analysis unavailable due to data retrieval error."
        }


def peer_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate peer research sentiment"""
    logger.info(f"Starting peer research for {state.business}")
    try:
        peer_sentiment, agent_metrics = get_peer_sentiment(business=state.business)
        logger.info(f"Completed peer research for {state.business}")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "peer_sentiment": format_sentiment_output(peer_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Peer research failed for {state.business}: {e}", exc_info=True)
        return {"peer_sentiment": "Analysis unavailable due to data retrieval error."}


def headline_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate headline research sentiment"""
    logger.info(f"Starting headline research for {state.business}")
    try:
        headline_sentiment, agent_metrics = get_headline_sentiment(
            business=state.business,
        )
        logger.info(f"Completed headline research for {state.business}")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "headline_sentiment": format_sentiment_output(headline_sentiment),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(
            f"Headline research failed for {state.business}: {e}", exc_info=True
        )
        return {
            "headline_sentiment": "Analysis unavailable due to data retrieval error."
        }


def filings_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate SEC filings research sentiment"""
    logger.info(f"Starting filings research for {state.ticker}")
    try:
        filings_sentiment, agent_metrics = get_filings_sentiment(ticker=state.ticker)
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        if filings_sentiment:
            logger.info(f"Completed filings research for {state.ticker}")
            return {
                "filings_sentiment": format_sentiment_output(filings_sentiment),
                "metrics": metrics,
            }
        else:
            return {
                "filings_sentiment": "No SEC filings available for analysis.",
                "metrics": metrics,
            }
    except Exception as e:
        logger.error(f"Filings research failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_sentiment": "Analysis unavailable due to data retrieval error."
        }


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment"""
    iteration = state.revision_iteration_count + 1
    logger.info(
        f"Starting sentiment aggregation for {state.ticker} (iteration {iteration})"
    )
    try:
        combined_sentiment, agent_metrics = get_aggregated_sentiment(state, iteration)
        logger.info(
            f"Completed sentiment aggregation for {state.ticker} (iteration {iteration})"
        )
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        return {
            "combined_sentiment": combined_sentiment,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(
            f"Sentiment aggregation failed for {state.ticker}: {e}", exc_info=True
        )
        return {
            "combined_sentiment": (
                f"**Conclusion:** Unable to fully synthesize sentiment due to an internal error. "
                f"Please review individual research components for partial analysis."
            )
        }


def sentiment_evaluator(state: EquityResearchState) -> dict:
    """LLM call to evaluate sentiment aggregator output"""
    iteration = state.revision_iteration_count + 1
    logger.info(f"Starting sentiment evaluation (iteration {iteration})")
    try:
        sentiment_evaluation, agent_metrics = evaluate_aggregated_sentement(
            sentiment=state.combined_sentiment,
            iteration=iteration,
        )
        logger.info(f"Completed sentiment evaluation (iteration {iteration})")
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        sentiment_evaluation["revision_iteration_count"] = iteration
        sentiment_evaluation["metrics"] = metrics
        return sentiment_evaluation
    except Exception as e:
        logger.error(f"Sentiment evaluation failed: {e}", exc_info=True)
        # Mark as compliant to avoid infinite retry loops, but log the failure
        return {
            "compliant": True,
            "feedback": "Evaluation skipped due to internal error.",
            "revision_iteration_count": state.revision_iteration_count + 1,
        }


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
graph_builder.add_node("filings_ingestion", filings_ingestion)

graph_builder.add_node(
    "fundamental_research_agent",
    fundamental_research_agent,
    # Dynamic cache: shorter TTL when earnings are imminent, key changes on earnings status
    cache_policy=create_fundamentals_cache_policy(),
)

graph_builder.add_node(
    "technical_research_agent",
    technical_research_agent,
    # Dynamic cache: key includes hour bucket for time-sensitive price data
    cache_policy=create_technical_cache_policy(),
)

graph_builder.add_node(
    "macro_research_agent",
    macro_research_agent,
    # Dynamic cache: key includes date bucket for daily invalidation
    cache_policy=create_macro_cache_policy(),
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
    "filings_research_agent",
    filings_research_agent,
    # evict filings research cache after one day
    cache_policy=create_cache_policy(ttl=86400),
)

graph_builder.add_node(
    "aggregator",
    sentiment_aggregator,
)

graph_builder.add_node("evaluator", sentiment_evaluator)

# validate ticker, ingest filings, then call research agents in parallel
graph_builder.add_edge(START, "ticker_validation")
graph_builder.add_conditional_edges(
    "ticker_validation",
    ticker_router,
    [
        "filings_ingestion",
        "fundamental_research_agent",
        "technical_research_agent",
        "macro_research_agent",
        "industry_research_agent",
        "peer_research_agent",
        "headline_research_agent",
        END,
    ],
)
# ingest SEC filings before filings research agent runs
graph_builder.add_edge("filings_ingestion", "filings_research_agent")
# synthesize sentiment
graph_builder.add_edge("fundamental_research_agent", "aggregator")
graph_builder.add_edge("technical_research_agent", "aggregator")
graph_builder.add_edge("macro_research_agent", "aggregator")
graph_builder.add_edge("industry_research_agent", "aggregator")
graph_builder.add_edge("peer_research_agent", "aggregator")
graph_builder.add_edge("headline_research_agent", "aggregator")
graph_builder.add_edge("filings_research_agent", "aggregator")
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
    # Initialize metrics with request start time stored in state
    metrics = RequestMetrics()
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
        filings_sentiment="",
        combined_sentiment="",
        compliant=False,
        feedback=None,
        is_ticker_valid=False,
        revision_iteration_count=0,
        ticker_info=None,  # Will be populated by ticker_validation node
        filings_ingested=False,  # Will be populated by filings_ingestion node
        metrics=metrics,
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
