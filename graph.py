import asyncio
import os
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents.aggregation.agent import get_aggregated_sentiment
from agents.evaluation.agent import evaluate_state
from agents.evaluation.policy import MAX_REVISION_COUNT, evaluation_failure_update
from agents.fundamentals.agent import get_fundamental_sentiment
from agents.headline.agent import get_headline_sentiment
from agents.industry.agent import get_industry_sentiment
from agents.macro.agent import get_macro_sentiment
from agents.peer.agent import get_peer_sentiment
from agents.shared.request_budget import (
    begin_request,
    bind_request_budget,
    end_request,
    request_budget_allows,
    reset_request_budget,
)
from agents.shared.token_config import TokenBudgetConfig, get_token_config
from agents.technical.agent import get_technical_sentiment
from models.metrics import AgentMetrics, RequestMetrics
from models.result import AgentRunResult, AgentStatus
from models.state import EquityResearchState
from subgraphs.filings_rag_subgraph import filings_rag_subgraph
from util.cache import (
    create_cache_policy,
    create_filings_cache_policy,
    create_fundamentals_cache_policy,
    create_macro_cache_policy,
    create_technical_cache_policy,
)
from util.checkpointer import SqliteSnapshotSaver
from util.formating import format_sentiment_output
from util.logger import get_logger
from util.valiation import validate_ticker

load_dotenv()
logger = get_logger(__name__)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "data/checkpoints.sqlite")

NODE_AGENT_NAMES = {
    "fundamental_research_agent": ("fundamental",),
    "technical_research_agent": ("technical",),
    "macro_research_agent": ("macro",),
    "industry_research_agent": ("industry",),
    "peer_research_agent": ("peer",),
    "headline_research_agent": ("headline",),
    "filings_workflow": (
        "filings_query_builder",
        "filings_retrieval",
        "filings_synthesis",
    ),
}


def render_agent_output(run: AgentRunResult) -> str:
    """Render a structured payload, or the error text when parsing did not succeed."""
    if isinstance(run.output, BaseModel):
        return format_sentiment_output(run.output)
    if isinstance(run.output, str) and run.output.strip():
        return run.output
    return run.error or "Analysis unavailable due to an internal error."


def _with_request_budget(state: EquityResearchState):
    config = get_token_config(state.token_preset)
    tokens = bind_request_budget(state.request_id, config.request_budget)
    return config, tokens


def _budget_skip(agent_name: str, state_field: str, config: TokenBudgetConfig) -> dict:
    metrics = RequestMetrics(token_budget=config.request_budget)
    metrics.add_agent_metrics(
        AgentMetrics(
            agent_name=agent_name,
            latency_ms=0,
            model=None,
            budget_exceeded=True,
        )
    )
    return {
        state_field: "Analysis skipped because the request token budget was exceeded.",
        "metrics": metrics,
        "agent_status": {agent_name: AgentStatus.FAILED.value},
    }


@dataclass(frozen=True)
class Specialist:
    """One parallel research node: callable, cache policy, and the state field it writes."""

    node_name: str
    state_field: str
    agent_name: str
    cache_policy: object
    invoke: Callable[[EquityResearchState, TokenBudgetConfig], AgentRunResult]


def make_specialist_node(spec: Specialist):
    """Build a graph node that records status instead of dropping failures into an exception string."""

    def node(state: EquityResearchState) -> dict:
        logger.info(f"Starting {spec.agent_name} research for {state.ticker}")
        config, tokens = _with_request_budget(state)
        try:
            if not request_budget_allows():
                return _budget_skip(spec.agent_name, spec.state_field, config)
            run = spec.invoke(state, config)
            metrics = RequestMetrics(token_budget=config.request_budget)
            metrics.add_agent_metrics(run.metrics)
            logger.info(f"Completed {spec.agent_name} research for {state.ticker}")
            return {
                spec.state_field: render_agent_output(run),
                "metrics": metrics,
                "agent_status": {spec.agent_name: run.status.value},
            }
        except Exception as exc:
            logger.error(
                f"{spec.agent_name} research failed for {state.ticker}: {exc}",
                exc_info=True,
            )
            return {
                spec.state_field: "Analysis unavailable due to an internal error.",
                "agent_status": {spec.agent_name: AgentStatus.FAILED.value},
            }
        finally:
            reset_request_budget(tokens)

    node.__name__ = spec.node_name
    return node


def ticker_validation(state: EquityResearchState) -> dict:
    """Validation node to ensure we have a real ticker."""
    logger.info(f"Validating ticker: {state.ticker}")
    result = validate_ticker(ticker=state.ticker, state=state)
    logger.info(f"Ticker validation complete for {state.ticker}")
    return result


def ticker_router(state: EquityResearchState):
    """Route to filings workflow and research agents if ticker is valid, otherwise end."""
    if state.is_ticker_valid:
        return [spec.node_name for spec in SPECIALISTS] + ["filings_workflow"]
    return END


SPECIALISTS = (
    Specialist(
        "fundamental_research_agent",
        "fundamental_sentiment",
        "fundamental",
        create_fundamentals_cache_policy(),
        lambda state, config: get_fundamental_sentiment(
            ticker=state.ticker,
            cached_info=state.ticker_info,
            token_config=config.fundamental,
        ),
    ),
    Specialist(
        "technical_research_agent",
        "technical_sentiment",
        "technical",
        create_technical_cache_policy(),
        lambda state, config: get_technical_sentiment(
            ticker=state.ticker,
            token_config=config.technical,
        ),
    ),
    Specialist(
        "macro_research_agent",
        "macro_sentiment",
        "macro",
        create_macro_cache_policy(),
        lambda state, config: get_macro_sentiment(token_config=config.macro),
    ),
    Specialist(
        "industry_research_agent",
        "industry_sentiment",
        "industry",
        create_cache_policy(ttl=3600),
        lambda state, config: get_industry_sentiment(
            ticker=state.ticker,
            industry=state.industry,
            token_config=config.industry,
        ),
    ),
    Specialist(
        "peer_research_agent",
        "peer_sentiment",
        "peer",
        create_cache_policy(ttl=3600),
        lambda state, config: get_peer_sentiment(
            business=state.business,
            token_config=config.peer,
        ),
    ),
    Specialist(
        "headline_research_agent",
        "headline_sentiment",
        "headline",
        create_cache_policy(ttl=3600),
        lambda state, config: get_headline_sentiment(
            business=state.business,
            token_config=config.headline,
        ),
    ),
)


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment."""
    iteration = state.revision_iteration_count + 1
    logger.info(
        f"Starting sentiment aggregation for {state.ticker} (iteration {iteration})"
    )
    config, tokens = _with_request_budget(state)
    try:
        if not request_budget_allows():
            return _budget_skip("aggregation", "combined_sentiment", config)
        run = get_aggregated_sentiment(
            state,
            iteration,
            token_config=config.aggregation,
        )
        metrics = RequestMetrics(token_budget=config.request_budget)
        metrics.add_agent_metrics(run.metrics)
        logger.info(
            f"Completed sentiment aggregation for {state.ticker} (iteration {iteration})"
        )
        return {
            "combined_sentiment": render_agent_output(run),
            "metrics": metrics,
            "agent_status": {run.metrics.agent_name: run.status.value},
        }
    except Exception as exc:
        logger.error(
            f"Sentiment aggregation failed for {state.ticker}: {exc}", exc_info=True
        )
        return {
            "combined_sentiment": (
                "**Conclusion:** Unable to fully synthesize sentiment due to an internal error. "
                "Please review individual research components for partial analysis."
            ),
            "agent_status": {"aggregation": AgentStatus.FAILED.value},
        }
    finally:
        reset_request_budget(tokens)


def sentiment_evaluator(state: EquityResearchState) -> dict:
    """LLM call to evaluate sentiment aggregator output."""
    iteration = state.revision_iteration_count + 1
    logger.info(f"Starting sentiment evaluation (iteration {iteration})")
    config, tokens = _with_request_budget(state)
    try:
        if not request_budget_allows():
            update = evaluation_failure_update(
                iteration,
                "Evaluation skipped because the request token budget was exceeded.",
            )
            update["agent_status"] = {"evaluation": AgentStatus.FAILED.value}
            return update
        update, agent_metrics = evaluate_state(
            state,
            iteration,
            token_config=config.evaluation,
        )
        metrics = RequestMetrics(token_budget=config.request_budget)
        metrics.add_agent_metrics(agent_metrics)
        update["metrics"] = metrics
        status = (
            AgentStatus.FAILED.value
            if update.get("evaluation_status") == "skipped"
            else AgentStatus.OK.value
        )
        update["agent_status"] = {agent_metrics.agent_name: status}
        logger.info(f"Completed sentiment evaluation (iteration {iteration})")
        return update
    except Exception as exc:
        logger.error(f"Sentiment evaluation failed: {exc}", exc_info=True)
        update = evaluation_failure_update(
            iteration, "Evaluation skipped due to an internal error."
        )
        update["agent_status"] = {"evaluation": AgentStatus.FAILED.value}
        return update
    finally:
        reset_request_budget(tokens)


def sentiment_router(state: EquityResearchState):
    """Route back to the aggregator, or stop when the draft passed or evaluation is skipped."""
    if state.evaluation_status == "skipped":
        return "Compliant"
    if state.compliant is True:
        return "Compliant"
    if state.revision_iteration_count > MAX_REVISION_COUNT:
        return "Compliant"
    return "Noncompliant"


def _read_state_value(result, key):
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def run_filings_subgraph(state: EquityResearchState) -> dict:
    """Run the filings subgraph and copy only the fields the parent graph should merge."""
    _config, tokens = _with_request_budget(state)
    try:
        result = filings_rag_subgraph.invoke(state)
        metrics = _read_state_value(result, "metrics")
        if isinstance(metrics, dict):
            metrics = RequestMetrics.model_validate(metrics)
        agent_status = _read_state_value(result, "agent_status") or {}
        update = {
            "filings_sentiment": _read_state_value(result, "filings_sentiment"),
            "filings_ingested": _read_state_value(result, "filings_ingested"),
            "agent_status": agent_status,
        }
        if metrics is not None:
            update["metrics"] = metrics
        return update
    except Exception as exc:
        logger.error(f"Filings workflow failed for {state.ticker}: {exc}", exc_info=True)
        return {
            "filings_sentiment": "Analysis unavailable due to an internal error.",
            "filings_ingested": False,
            "agent_status": {"filings": AgentStatus.FAILED.value},
        }
    finally:
        reset_request_budget(tokens)


graph_builder = StateGraph(EquityResearchState)
graph_builder.add_node("ticker_validation", ticker_validation)
graph_builder.add_node(
    "filings_workflow",
    run_filings_subgraph,
    cache_policy=create_filings_cache_policy(ttl=3600),
)
for spec in SPECIALISTS:
    graph_builder.add_node(
        spec.node_name,
        make_specialist_node(spec),
        cache_policy=spec.cache_policy,
    )
graph_builder.add_node("aggregator", sentiment_aggregator)
graph_builder.add_node("evaluator", sentiment_evaluator)

graph_builder.add_edge(START, "ticker_validation")
graph_builder.add_conditional_edges(
    "ticker_validation",
    ticker_router,
    [spec.node_name for spec in SPECIALISTS] + ["filings_workflow", END],
)
for spec in SPECIALISTS:
    graph_builder.add_edge(spec.node_name, "aggregator")
graph_builder.add_edge("filings_workflow", "aggregator")
graph_builder.add_edge("aggregator", "evaluator")
graph_builder.add_conditional_edges(
    "evaluator",
    sentiment_router,
    {"Compliant": END, "Noncompliant": "aggregator"},
)

cache = InMemoryCache()
checkpointer = SqliteSnapshotSaver(CHECKPOINT_PATH)
graph_workflow = graph_builder.compile(cache=cache, checkpointer=checkpointer)


def build_input_state(input_dict: dict) -> EquityResearchState:
    """Create the initial graph state and start request-level token accounting."""
    preset = input_dict.get("token_preset", "standard")
    config = get_token_config(preset)
    request_id = input_dict.get("request_id") or str(uuid4())
    begin_request(request_id)
    return EquityResearchState(
        ticker=input_dict["ticker"],
        trade_duration=input_dict["trade_duration"],
        trade_direction=input_dict["trade_direction"],
        token_preset=preset,
        request_id=request_id,
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
        evaluation_status=None,
        faithfulness_score=None,
        is_ticker_valid=False,
        revision_iteration_count=0,
        ticker_info=None,
        filings_ingested=False,
        metrics=RequestMetrics(token_budget=config.request_budget),
    )


def apply_cache_hits(state: EquityResearchState, cached_nodes: set[str]) -> None:
    """Mark metrics served from the LangGraph node cache and stop counting those tokens twice."""
    for node_name in cached_nodes:
        for agent_name in NODE_AGENT_NAMES.get(node_name, ()):
            metrics = state.metrics.agent_metrics.get(agent_name)
            if metrics is None or metrics.cached:
                continue
            state.metrics.total_input_tokens -= metrics.token_usage.input_tokens
            state.metrics.total_output_tokens -= metrics.token_usage.output_tokens
            state.metrics.total_tokens -= metrics.token_usage.total_tokens
            metrics.cached = True
    state.metrics.total_input_tokens = max(0, state.metrics.total_input_tokens)
    state.metrics.total_output_tokens = max(0, state.metrics.total_output_tokens)
    state.metrics.total_tokens = max(0, state.metrics.total_tokens)
    agent_budget_exceeded = any(
        item.budget_exceeded for item in state.metrics.agent_metrics.values()
    )
    over_request_budget = (
        state.metrics.token_budget is not None
        and state.metrics.total_tokens > state.metrics.token_budget
    )
    state.metrics.budget_exceeded = agent_budget_exceeded or over_request_budget


def _coerce_state(value) -> EquityResearchState:
    if isinstance(value, EquityResearchState):
        return value
    return EquityResearchState.model_validate(value)


async def execute_research(input_dict: dict) -> EquityResearchState:
    """Run the research graph and record which nodes were served from cache."""
    state = build_input_state(input_dict)
    config = {"configurable": {"thread_id": state.request_id}}
    cached_nodes: set[str] = set()
    final = None
    try:
        async for mode, chunk in graph_workflow.astream(
            state,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if (
                mode == "updates"
                and isinstance(chunk, dict)
                and chunk.get("__metadata__", {}).get("cached")
            ):
                cached_nodes.update(key for key in chunk if key != "__metadata__")
            elif mode == "values":
                final = chunk
    finally:
        end_request(state.request_id)

    final_state = _coerce_state(final if final is not None else state)
    apply_cache_hits(final_state, cached_nodes)
    return final_state


class ResearchChain:
    """Sync and async entry points used by the API and the eval script."""

    def invoke(self, input_dict: dict, config=None) -> EquityResearchState:
        return asyncio.run(execute_research(input_dict))

    async def ainvoke(self, input_dict: dict, config=None) -> EquityResearchState:
        return await execute_research(input_dict)


research_chain = ResearchChain()
