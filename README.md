# Langgraph Equity Research with RAG

Agentic equity research implemented on LangGraph, ChromaDB, yfinance, and FastAPI

# Running in Docker

To create a local docker container to run the application, follow these steps:

1. Hydrate env variables in a new .env file (variables are specified in the env.dev file, langsmith is unnecessary for docker)
2. Ensure the local Docker daemon is running
3. Run `docker compose up` from root of the project (note: built image is ~1GB)
4. In a separate terminal, execute the following curl command:
   curl -X POST "http://localhost:8000/research-equity" -H "Content-Type: application/json" -d '{"ticker": "TICKER", "trade_duration": "position_trade", "trade_direction": "short"}'

# Running Locally

To run the program from the terminal, follow these steps:

1. Initialize a python environment on python version 3.12
2. Install deps with pip or other python package manager
3. Hydrate env variables in a new .env file (variables are specified in the env.dev file, langsmith is unnecessary for terminal use)
4. Run `python main.py` from the root of the project
5. From a separate terminal window, execute the following curl command:
   curl -X POST "http://localhost:8000/research-equity" -H "Content-Type: application/json" -d '{"ticker": "TICKER", "trade_duration": "position_trade", "trade_direction": "short"}'

# Running in LangSmith for Observability

To run the program in Langsmith, which offers observability for graph execution and tracing, follow these steps:

1. Initialize a python environment on python version 3.12
2. Install deps with pip or other python package manager
3. Hydrate env variables in a new .env file (variables are specified in the env.dev file, in this case langsmith is needed)
4. Run `langsmith dev` from the root of the project
5. A browser window should open showing the built graph and execution controls

# Running the Streamlit Demo

To run the interactive demo UI, follow these steps:

1. Start the API server with `python main.py`
2. In a separate terminal, run `streamlit run demo.py`
3. Open http://localhost:8501 in your browser

# Architecture

This app implements an agentic ai architecture to compile equity research on a single stock across different analytical lenses.

![Architecture Diagram](/architecture.png)

The app structures ai agents as research domain specialists that perform data gathering and analysis scoped to their respective domain. The filing agent implements a RAG architecture to analyze relevant SEC filings.
The app manages state through langgraph's graph state model.
The graph implements a node-based cache that configures cache keys and ttl at the agent level.
The entrypoint of the graph vaidates that the ticker is valid using yfinance.
Once the ticker is vaidated, research agents are executed in parallel.
When all agents have executed, a aggregator agent synthesizes overall sentiment for the stock.
An evaluator agent reviews the aggregator's output and determines whether it complies with criteria. If not, it provides feedback for revision.

# Architecture Components

- HTTP API: FastAPI
- Agentic Architecture: Langgraph
- Agent node caching: Langgraph Node Cache
- Agent Observability: LangSmith
- Type Package: Pydantic
- LLM Models: OpenAI, Google Gemini
- Financial Data API: YFinance
- Economic Data API: Pandas Datareader
- SEC Filings RAG: ChromaDB, SentenceTransformer

# Agent Observability

The application tracks detailed metrics for each agent execution and aggregates them at the request level. This enables cost monitoring, performance debugging, and cache effectiveness analysis.

![Observability Metrics](/metrics.png)

**Per-Agent Metrics:**

- `latency_ms` - Execution time in milliseconds
- `token_usage` - Input, output, and total tokens consumed
- `model` - The LLM model used (e.g., gpt-4o-mini)
- `cached` - Whether the result was served from cache

**Request-Level Metrics:**

- `total_latency_ms` - End-to-end request latency
- `total_tokens` - Aggregate token usage across all agents

Metrics are returned in the API response under the `metrics` key and displayed in the Streamlit demo's "Performance Metrics" panel. For parallel agent execution, metrics are merged using a LangGraph reducer to accurately aggregate totals.

For deeper tracing and visualization, connect to LangSmith (see "Running in LangSmith for Observability" above).

# API Definition

The API exposes one POST enpoint at `/research-equity`
This endpoint requires three params:

- ticker: The stock ticker being researched. It must be a publically traded company to pass validation (e.g. "AAPL")
- trade_direction: The trade bias ("long" or "short")
- trade_duration: The intended duration for the trade ("day_trade", "swing_trade", or "position_trade")

# AI Agent Descriptions

1. Technical Researcher - Focuses on technical price data over fixed timeframes
2. Fundamentals Researcher - Focuses on business fundamentals via most recent earnings results
3. Filings Researcher - Analyzes SEC filings (10-K, 10-Q) using RAG over embedded filing excerpts
4. Macro Economic Researcher - Focuses on the macro economic components provided by federal reserve data
5. Industry Researcher - Focuses on forecasted headwinds / tailwinds relative to the industry of the stock
6. Headline Researcher - Focuses on recent (within the last month) headlines about the stock
7. Sentiment Aggregator - Compiles overall stock sentiment based on aggregate findings of research specialist agents
8. Sentiment Evaluator - Evaluates the aggregated sentiment for compliance with target criteria
