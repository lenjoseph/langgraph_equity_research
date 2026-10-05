import os
import re
import time
import asyncio
import threading
from contextlib import asynccontextmanager

# Suppress gRPC/absl logging before importing anything that uses it
# Ensure huggingface tokenizer thread pool doesn't fork when graph executes parallel agents
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from graph import execute_research
from models.api import RESEARCH_DISCLAIMER, EquityResearchRequest
from agents.shared.token_config import BUDGET_PRESETS
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from data.util.ingest_sec_filings import ingest_ticker_filings
from util.logger import get_logger

logger = get_logger(__name__)

TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")
REQUEST_TIMEOUT_SECONDS = 240


def sanitize_ticker(ticker: str) -> str:
    """Sanitize and validate stock ticker input."""
    sanitized = ticker.strip().upper()

    if not sanitized:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    if not TICKER_PATTERN.match(sanitized):
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker format. Ticker must contain only letters, numbers, dots, or hyphens (max 10 characters)",
        )

    return sanitized


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    tickers_env = os.environ.get("PRELOAD_TICKERS")
    if tickers_env:
        tickers = tickers_env.replace(",", " ").split()
        if tickers:
            logger.info(
                f"Found PRELOAD_TICKERS env var. Preloading filings for: {tickers}"
            )

            def ingest_all():
                for ticker in tickers:
                    try:
                        logger.info(f"Starting background ingestion for {ticker}")
                        ingest_ticker_filings(ticker=ticker, years=2)
                        logger.info(f"Completed background ingestion for {ticker}")
                    except Exception as e:
                        logger.error(f"Error ingesting {ticker}: {e}", exc_info=True)

            thread = threading.Thread(target=ingest_all)
            thread.daemon = True  # Daemon thread ensures it doesn't block shutdown
            thread.start()

    yield
    # Shutdown logic (if any)


app = FastAPI(lifespan=lifespan)
_default_origins = "http://localhost:8501,http://127.0.0.1:8501"
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/")
def ping():
    return {"message": "Running"}


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/research-equity")
@limiter.limit("10/minute")
async def research_equity(request: Request, req: EquityResearchRequest):
    start_time = time.perf_counter()
    sanitized_ticker = sanitize_ticker(req.ticker)
    if req.token_preset not in BUDGET_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown token preset: {req.token_preset}. "
                f"Available presets: {sorted(BUDGET_PRESETS)}"
            ),
        )

    try:
        res = await asyncio.wait_for(
            execute_research(
                {
                    "ticker": sanitized_ticker,
                    "trade_duration": req.trade_duration,
                    "trade_direction": req.trade_direction,
                    "token_preset": req.token_preset,
                }
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Research request exceeded the 240 second deadline.",
        )

    if not res.is_ticker_valid:
        raise HTTPException(status_code=400, detail=f"Ticker {res.ticker} is invalid")

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    res.metrics.total_latency_ms = total_latency_ms

    return {
        "ticker": res.ticker,
        "request_id": res.request_id,
        "sentiment_analysis": {
            "fundamental": res.fundamental_sentiment,
            "technical": res.technical_sentiment,
            "macro": res.macro_sentiment,
            "peer": res.peer_sentiment,
            "industry": res.industry_sentiment,
            "headline": res.headline_sentiment,
            "filings": res.filings_sentiment,
        },
        "combined_sentiment": res.combined_sentiment,
        "evaluation_status": res.evaluation_status,
        "faithfulness_score": res.faithfulness_score,
        "agent_status": res.agent_status,
        "disclaimer": RESEARCH_DISCLAIMER,
        "metrics": res.metrics.to_response_dict(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
