"""Request budget accounting and cache-key behavior."""

from datetime import datetime, timedelta

from agents.shared.request_budget import (
    begin_request,
    bind_request_budget,
    end_request,
    record_request_usage,
    request_budget_allows,
    request_tokens_used,
    reset_request_budget,
)
from graph import apply_cache_hits
from models.metrics import AgentMetrics, RequestMetrics, TokenUsage
from models.state import EquityResearchState, TradeDirection, TradeDuration
from util.cache import create_fundamentals_cache_policy, is_earnings_imminent
from util.retry import is_retryable_error
from agents.shared.agent_utils import RequestBudgetExceeded, TokenBudgetExceeded


class _State:
    def __init__(self, ticker, ticker_info):
        self.ticker = ticker
        self.ticker_info = ticker_info


def test_request_budget_blocks_after_the_running_sum_is_spent():
    begin_request("req-1")
    tokens = bind_request_budget("req-1", 100)
    try:
        assert request_budget_allows(40) is True
        record_request_usage(80)
        assert request_tokens_used("req-1") == 80
        assert request_budget_allows(30) is False
        assert request_budget_allows(20) is True
    finally:
        reset_request_budget(tokens)
        end_request("req-1")


def test_budget_errors_are_not_retried():
    assert is_retryable_error(TokenBudgetExceeded(10, 11)) is False
    assert is_retryable_error(RequestBudgetExceeded()) is False
    assert is_retryable_error(TimeoutError()) is True


def test_imminent_earnings_rotate_the_cache_key():
    soon = int((datetime.now() + timedelta(days=2)).timestamp())
    info = {"earningsTimestamp": soon}
    assert is_earnings_imminent(info) is True
    policy = create_fundamentals_cache_policy()
    imminent = policy.key_func(_State("AAPL", info))
    normal = policy.key_func(_State("AAPL", {}))
    assert b"earnings_imminent" in imminent
    assert imminent != normal
    assert policy.ttl == 3600


def test_cache_hit_marks_metrics_and_does_not_keep_the_tokens():
    state = EquityResearchState(
        ticker="AAPL",
        trade_duration=TradeDuration.DAY_TRADE,
        trade_direction=TradeDirection.LONG,
        metrics=RequestMetrics(
            token_budget=1000,
            total_input_tokens=40,
            total_output_tokens=10,
            total_tokens=50,
            agent_metrics={
                "fundamental": AgentMetrics(
                    agent_name="fundamental",
                    latency_ms=5,
                    token_usage=TokenUsage(input_tokens=40, output_tokens=10, total_tokens=50),
                    model="test",
                )
            },
        ),
    )
    apply_cache_hits(state, {"fundamental_research_agent"})
    assert state.metrics.agent_metrics["fundamental"].cached is True
    assert state.metrics.total_tokens == 0
