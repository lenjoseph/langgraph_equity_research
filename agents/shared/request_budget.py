"""Process-wide request token budget shared across parallel agent threads."""

import threading
from contextvars import ContextVar, Token
from typing import Optional

_lock = threading.Lock()
_usage: dict[str, int] = {}

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_request_budget: ContextVar[Optional[int]] = ContextVar("request_budget", default=None)


def begin_request(request_id: str) -> None:
    """Start usage tracking for a research request."""
    with _lock:
        _usage[request_id] = 0


def end_request(request_id: str) -> None:
    """Drop usage tracking after a request finishes."""
    with _lock:
        _usage.pop(request_id, None)


def bind_request_budget(
    request_id: str, request_budget: Optional[int]
) -> tuple[Token, Token]:
    """Bind the active request on this thread so LLM calls can see the budget."""
    return (
        _request_id.set(request_id or None),
        _request_budget.set(request_budget),
    )


def reset_request_budget(tokens: tuple[Token, Token]) -> None:
    """Restore the previous request-budget context."""
    request_token, budget_token = tokens
    _request_id.reset(request_token)
    _request_budget.reset(budget_token)


def request_budget_allows(additional_tokens: int = 0) -> bool:
    """Return False when this request has already used its token budget."""
    budget = _request_budget.get()
    request_id = _request_id.get()
    if budget is None or not request_id:
        return True
    with _lock:
        used = _usage.get(request_id, 0)
        return used + additional_tokens <= budget


def record_request_usage(tokens: int) -> None:
    """Add tokens from a completed LLM call to the active request."""
    request_id = _request_id.get()
    if not request_id or tokens <= 0:
        return
    with _lock:
        _usage[request_id] = _usage.get(request_id, 0) + tokens


def request_tokens_used(request_id: str) -> int:
    """Return tokens recorded for a request."""
    with _lock:
        return _usage.get(request_id, 0)
