"""SEC filings RAG subgraph."""

from langgraph.graph import END, StateGraph, START

from agents.filings.agents.query_builder import generate_search_queries
from agents.filings.agents.retriever import get_filings_context
from agents.filings.agents.synthesis import generate_filings_sentiment
from agents.shared.token_config import get_token_config
from data.util.fetch_sec_filings import MissingSECIdentity
from data.util.ingest_sec_filings import ensure_filings_ingested
from models.metrics import RequestMetrics
from models.result import AgentStatus
from models.state import EquityResearchState
from util.formating import format_sentiment_output
from util.logger import get_logger

logger = get_logger(__name__)

SEC_UNAVAILABLE = (
    "SEC filings unavailable because SEC_EDGAR_AGENT_KEY is not configured."
)


def filings_rag_ingestion(state: EquityResearchState) -> dict:
    """Ingest SEC filings into the vector store, refreshing new accessions."""
    logger.info(f"Starting SEC filings ingestion for {state.ticker}")
    try:
        available = ensure_filings_ingested(ticker=state.ticker)
        if available:
            logger.info(f"SEC filings available for {state.ticker}")
        else:
            logger.info(f"No SEC filings available for {state.ticker}")
        return {"filings_ingested": available}
    except MissingSECIdentity:
        logger.error("SEC filings ingestion skipped: SEC_EDGAR_AGENT_KEY is not set")
        return {
            "filings_ingested": False,
            "filings_sentiment": SEC_UNAVAILABLE,
            "agent_status": {"filings_ingestion": AgentStatus.FAILED.value},
        }
    except Exception as e:
        logger.error(
            f"SEC filings ingestion failed for {state.ticker}: {e}", exc_info=True
        )
        return {
            "filings_ingested": False,
            "agent_status": {"filings_ingestion": AgentStatus.FAILED.value},
        }


def filings_after_ingestion(state: EquityResearchState) -> str:
    """Stop the filings branch when ingestion already produced a final message."""
    if state.filings_sentiment and not state.filings_ingested:
        return END
    return "filings_query_builder"


def filings_rag_query_builder(state: EquityResearchState) -> dict:
    """Generate contextual search queries based on trade context."""
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
        metrics = RequestMetrics(token_budget=config.request_budget)
        metrics.add_agent_metrics(agent_metrics)
        logger.info(f"Generated search queries for {state.ticker}: {search_queries}")
        return {
            "filings_search_queries": search_queries,
            "metrics": metrics,
            "agent_status": {"filings_query_builder": AgentStatus.OK.value},
        }
    except Exception as e:
        logger.error(f"Query building failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_search_queries": None,
            "agent_status": {"filings_query_builder": AgentStatus.FAILED.value},
        }


def filings_rag_retriever(state: EquityResearchState) -> dict:
    """Retrieve SEC filings context using dynamically generated queries."""
    logger.info(f"Starting filings retrieval for {state.ticker}")
    try:
        config = get_token_config(state.token_preset)
        filings_context, agent_metrics = get_filings_context(
            ticker=state.ticker,
            search_queries=state.filings_search_queries,
        )
        metrics = RequestMetrics(token_budget=config.request_budget)
        metrics.add_agent_metrics(agent_metrics)
        logger.info(f"Completed filings retrieval for {state.ticker}")
        status = AgentStatus.OK if filings_context else AgentStatus.DEGRADED
        return {
            "filings_context": filings_context,
            "metrics": metrics,
            "agent_status": {"filings_retrieval": status.value},
        }
    except Exception as e:
        logger.error(f"Filings retrieval failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_context": None,
            "agent_status": {"filings_retrieval": AgentStatus.FAILED.value},
        }


def filings_rag_synthesis_agent(state: EquityResearchState) -> dict:
    """LLM call to generate SEC filings research sentiment."""
    logger.info(f"Starting filings synthesis for {state.ticker}")
    try:
        config = get_token_config(state.token_preset)
        run = generate_filings_sentiment(
            ticker=state.ticker,
            context=state.filings_context,
            token_config=config.filings_synthesis,
        )
        metrics = RequestMetrics(token_budget=config.request_budget)
        metrics.add_agent_metrics(run.metrics)
        if run.output is not None:
            logger.info(f"Completed filings synthesis for {state.ticker}")
            text = format_sentiment_output(run.output)
        else:
            text = run.error or "No SEC filings available for analysis."
        return {
            "filings_sentiment": text,
            "metrics": metrics,
            "agent_status": {"filings_synthesis": run.status.value},
        }
    except Exception as e:
        logger.error(f"Filings synthesis failed for {state.ticker}: {e}", exc_info=True)
        return {
            "filings_sentiment": "Analysis unavailable due to an internal error.",
            "agent_status": {"filings_synthesis": AgentStatus.FAILED.value},
        }


filings_rag_builder = StateGraph(EquityResearchState)
filings_rag_builder.add_node("filings_ingestion", filings_rag_ingestion)
filings_rag_builder.add_node("filings_query_builder", filings_rag_query_builder)
filings_rag_builder.add_node("filings_retriever", filings_rag_retriever)
filings_rag_builder.add_node("filings_synthesis_agent", filings_rag_synthesis_agent)

filings_rag_builder.add_edge(START, "filings_ingestion")
filings_rag_builder.add_conditional_edges(
    "filings_ingestion",
    filings_after_ingestion,
    ["filings_query_builder", END],
)
filings_rag_builder.add_edge("filings_query_builder", "filings_retriever")
filings_rag_builder.add_edge("filings_retriever", "filings_synthesis_agent")
filings_rag_builder.add_edge("filings_synthesis_agent", END)

filings_rag_subgraph = filings_rag_builder.compile()
