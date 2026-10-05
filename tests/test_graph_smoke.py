"""Graph routing with mocked agents, plus a durable sqlite checkpoint."""

from typing import TypedDict

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from agents.shared.agent_utils import _execute_tool_calls
from graph import build_input_state, graph_workflow
from main import app
from models.metrics import AgentMetrics
from models.result import AgentRunResult, AgentStatus
from models.state import EquityResearchState, TradeDirection, TradeDuration
from util.checkpointer import SqliteSnapshotSaver
from util.formating import format_sentiment_output


def _ok(name: str) -> AgentRunResult:
    return AgentRunResult(
        status=AgentStatus.OK,
        output=f"{name} analysis",
        metrics=AgentMetrics(agent_name=name, latency_ms=1, model="test"),
    )


def _patch_specialists(monkeypatch):
    monkeypatch.setattr("graph.get_fundamental_sentiment", lambda **kwargs: _ok("fundamental"))
    monkeypatch.setattr("graph.get_technical_sentiment", lambda **kwargs: _ok("technical"))
    monkeypatch.setattr("graph.get_macro_sentiment", lambda **kwargs: _ok("macro"))
    monkeypatch.setattr("graph.get_industry_sentiment", lambda **kwargs: _ok("industry"))
    monkeypatch.setattr("graph.get_peer_sentiment", lambda **kwargs: _ok("peer"))
    monkeypatch.setattr("graph.get_headline_sentiment", lambda **kwargs: _ok("headline"))
    monkeypatch.setattr(
        "graph.filings_rag_subgraph.invoke",
        lambda state: {
            "filings_sentiment": "filings analysis",
            "filings_ingested": True,
            "metrics": None,
            "agent_status": {"filings_synthesis": "ok"},
        },
    )


def _valid_ticker(ticker, state):
    return {
        "is_ticker_valid": True,
        "industry": "Software",
        "business": "Test Co",
        "ticker_info": {},
    }


def _invoke(ticker: str):
    state = build_input_state(
        {
            "ticker": ticker,
            "trade_duration": TradeDuration.POSITION_TRADE,
            "trade_direction": TradeDirection.LONG,
            "request_id": f"thread-{ticker}",
        }
    )
    result = graph_workflow.invoke(
        state, config={"configurable": {"thread_id": state.request_id}}
    )
    if isinstance(result, dict):
        return EquityResearchState.model_validate(result)
    return result


def test_invalid_ticker_stops_without_raising(monkeypatch):
    monkeypatch.setattr(
        "graph.validate_ticker", lambda ticker, state: {"is_ticker_valid": False}
    )
    result = _invoke("BADTICK")
    assert result.is_ticker_valid is False


def test_revision_cap_stops_after_three_failed_evaluations(monkeypatch):
    _patch_specialists(monkeypatch)
    monkeypatch.setattr("graph.validate_ticker", _valid_ticker)
    calls = {"n": 0}

    def evaluate(state, iteration, token_config=None):
        calls["n"] += 1
        return {
            "compliant": False,
            "feedback": "Add peer weighting.",
            "evaluation_status": "failed",
            "revision_iteration_count": iteration,
            "faithfulness_score": 0,
        }, AgentMetrics(agent_name="evaluation", latency_ms=1, model="test")

    monkeypatch.setattr("graph.evaluate_state", evaluate)
    monkeypatch.setattr(
        "graph.get_aggregated_sentiment",
        lambda state, iteration=1, token_config=None: _ok("aggregation"),
    )
    result = _invoke("REVTEST")
    assert calls["n"] == 3
    assert result.revision_iteration_count == 3
    assert result.compliant is False


def test_skipped_evaluation_stops_the_loop(monkeypatch):
    _patch_specialists(monkeypatch)
    monkeypatch.setattr("graph.validate_ticker", _valid_ticker)
    calls = {"n": 0}

    def evaluate(state, iteration, token_config=None):
        calls["n"] += 1
        return {
            "compliant": False,
            "feedback": "Evaluation skipped due to an internal error.",
            "evaluation_status": "skipped",
            "revision_iteration_count": iteration,
            "faithfulness_score": None,
        }, AgentMetrics(agent_name="evaluation", latency_ms=1, model="test")

    monkeypatch.setattr("graph.evaluate_state", evaluate)
    monkeypatch.setattr(
        "graph.get_aggregated_sentiment",
        lambda state, iteration=1, token_config=None: _ok("aggregation"),
    )
    result = _invoke("SKPTEST")
    assert calls["n"] == 1
    assert result.evaluation_status == "skipped"
    assert result.compliant is False


def test_invalid_ticker_is_http_400(monkeypatch):
    async def fake_execute(payload):
        return EquityResearchState(
            ticker=payload["ticker"],
            trade_duration=payload["trade_duration"],
            trade_direction=payload["trade_direction"],
            is_ticker_valid=False,
        )

    monkeypatch.delenv("PRELOAD_TICKERS", raising=False)
    monkeypatch.setattr("main.execute_research", fake_execute)
    client = TestClient(app)
    response = client.post(
        "/research-equity",
        json={
            "ticker": "BAD",
            "trade_duration": "position_trade",
            "trade_direction": "long",
        },
    )
    assert response.status_code == 400


def test_successful_response_includes_disclaimer(monkeypatch):
    async def fake_execute(payload):
        return EquityResearchState(
            ticker=payload["ticker"],
            trade_duration=payload["trade_duration"],
            trade_direction=payload["trade_direction"],
            is_ticker_valid=True,
            combined_sentiment="**Overall Sentiment:** NEUTRAL",
            evaluation_status="passed",
        )

    monkeypatch.delenv("PRELOAD_TICKERS", raising=False)
    monkeypatch.setattr("main.execute_research", fake_execute)
    client = TestClient(app)
    response = client.post(
        "/research-equity",
        json={
            "ticker": "AAPL",
            "trade_duration": "day_trade",
            "trade_direction": "short",
            "token_preset": "economy",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert "not investment advice" in body["disclaimer"]
    assert body["evaluation_status"] == "passed"


def test_unknown_tool_returns_an_error_message_instead_of_raising():
    class _Tool:
        name = "known"
        func = staticmethod(lambda **kwargs: "ok")

    messages, unknown = _execute_tool_calls(
        type("Resp", (), {"tool_calls": [{"name": "missing", "id": "1", "args": {}}]})(),
        {"known": _Tool()},
    )
    assert unknown is True
    assert "Unknown tool" in messages[0]["content"]


def test_format_helper_accepts_plain_text():
    assert format_sentiment_output("already text") == "already text"
    assert format_sentiment_output(None) == ""


def test_sqlite_checkpointer_reloads_after_a_new_process_saver(tmp_path):
    class State(TypedDict):
        n: int

    path = tmp_path / "checkpoints.sqlite"

    def add_one(state: State):
        return {"n": state["n"] + 1}

    def build(saver):
        builder = StateGraph(State)
        builder.add_node("add_one", add_one)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)
        return builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "thread-1"}}
    first = build(SqliteSnapshotSaver(str(path)))
    assert first.invoke({"n": 1}, config=config)["n"] == 2

    second = build(SqliteSnapshotSaver(str(path)))
    snapshot = second.get_state(config)
    assert snapshot.values["n"] == 2
