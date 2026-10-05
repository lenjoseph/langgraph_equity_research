"""Retries for transient LLM and HTTP failures."""

import random
import time
from typing import Callable, TypeVar

import requests

T = TypeVar("T")

MAX_ATTEMPTS = 3


def is_retryable_error(exc: BaseException) -> bool:
    """True for timeouts, connection failures, and rate limits. Budget errors are not retried."""
    name = type(exc).__name__
    if name in {
        "TokenBudgetExceeded",
        "RequestBudgetExceeded",
        "OutputParserException",
        "ValidationError",
    }:
        return False
    if name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectError",
        "RemoteProtocolError",
    }:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


def call_with_retry(func: Callable[[], T], attempts: int = MAX_ATTEMPTS) -> T:
    """Call func up to `attempts` times with jitter. The last error is raised."""
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1 or not is_retryable_error(exc):
                raise
            delay = min(4.0, 0.4 * (2**attempt)) + random.random() * 0.25
            time.sleep(delay)
    assert last_error is not None
    raise last_error
