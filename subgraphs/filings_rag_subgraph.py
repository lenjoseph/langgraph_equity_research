from langgraph.graph import END, StateGraph, START
from agents.filings.agents.query_builder import generate_search_queries
from agents.filings.agents.retriever import get_filings_context
from agents.filings.agents.synthesis import generate_filings_sentiment
from agents.shared.token_config import get_token_config
from data.util.ingest_sec_filings import ensure_filings_ingested
from models.state import EquityResearchState
from models.metrics import RequestMetrics
from util.cache import create_cache_policy
from util.formating import format_sentiment_output
from util.logger import get_logger

logger = get_logger(__name__)


def filings_rag_ingestion(state: EquityResearchState) -> dict:
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


def filings_rag_query_builder(state: EquityResearchState) -> dict:
    """Generate contextual search queries based on trade context"""
    logger.info(
        f"Building search queries for {state.ticker} "
        f"({state.trade_direction.value}, {state.trade_duration.value})"
    )
    try:
        config = get_token_config(state.token_preset)
        search_queries, agent_metrics = generate_search_queries(
            ticker=state.ticker,
            trade_direction=state.trade_direction,
            trade_duration=state.trade_duration,
            token_config=config.filings_query_builder,
        )
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        logger.info(f"Generated search queries for {state.ticker}: {search_queries}")
        return {
            "filings_search_queries": search_queries,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Query building failed for {state.ticker}: {e}", exc_info=True)
        return {"filings_search_queries": None}


def filings_rag_retriever(state: EquityResearchState) -> dict:
    """Retrieve SEC filings context using dynamically generated queries"""
    logger.info(f"Starting filings retrieval for {state.ticker}")
    try:
        filings_context, agent_metrics = get_filings_context(
            ticker=state.ticker,
            search_queries=state.filings_search_queries,
        )
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)
        logger.info(f"Completed filings retrieval for {state.ticker}")
        return {
            "filings_context": filings_context,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Filings retrieval failed for {state.ticker}: {e}", exc_info=True)
        # We don't return filings_sentiment error here, we let the synthesis agent handle missing context
        return {
            "filings_context": None,
        }


def filings_rag_synthesis_agent(state: EquityResearchState) -> dict:
    """LLM call to generate SEC filings research sentiment"""
    logger.info(f"Starting filings synthesis for {state.ticker}")
    try:
        config = get_token_config(state.token_preset)
        filings_sentiment, agent_metrics = generate_filings_sentiment(
            ticker=state.ticker,
            context=state.filings_context,
            token_config=config.filings_synthesis,
        )
        metrics = RequestMetrics()
        metrics.add_agent_metrics(agent_metrics)

        if filings_sentiment:
            logger.info(f"Completed filings synthesis for {state.ticker}")
            return {
                "filings_sentiment": format_sentiment_output(filings_sentiment),
                "metrics": metrics,
            }
        else:
            msg = "No SEC filings available for analysis."
            if state.filings_context is None:
                msg = "No SEC filings context retrieved."

            return {
                "filings_sentiment": msg,
                "metrics": metrics,
            }
    except Exception as e:
        logger.error(f"Filings synthesis failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_sentiment": "Analysis unavailable due to data retrieval error."
        }


# build filings subgraph
filings_rag_builder = StateGraph(EquityResearchState)
filings_rag_builder.add_node("filings_ingestion", filings_rag_ingestion)

# Add query builder node to generate contextual search queries
filings_rag_builder.add_node(
    "filings_query_builder",
    filings_rag_query_builder,
    cache_policy=create_cache_policy(ttl=86400),
)

filings_rag_builder.add_node(
    "filings_retriever",
    filings_rag_retriever,
    cache_policy=create_cache_policy(ttl=86400),
)

filings_rag_builder.add_node(
    "filings_synthesis_agent",
    filings_rag_synthesis_agent,
    cache_policy=create_cache_policy(ttl=86400),
)

filings_rag_builder.add_edge(START, "filings_ingestion")
filings_rag_builder.add_edge("filings_ingestion", "filings_query_builder")
filings_rag_builder.add_edge("filings_query_builder", "filings_retriever")
filings_rag_builder.add_edge("filings_retriever", "filings_synthesis_agent")
filings_rag_builder.add_edge("filings_synthesis_agent", END)

filings_rag_subgraph = filings_rag_builder.compile()
