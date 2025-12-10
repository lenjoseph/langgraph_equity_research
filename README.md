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

![Architecture Diagram](/assets/architecture.png)

The app structures AI agents as research domain specialists that perform data gathering and analysis scoped to their respective domain. Agents utilize different architectural patterns:

- **Vector RAG**: For analyzing dense SEC filings (Filings Agent)
- **Web RAG**: For real-time internet research (Headline, Industry, Peer Agents)
- **Tool-Use**: For fetching structured API data (Fundamental, Technical, Macro Agents)
  The app manages state through langgraph's graph state model.
  The graph implements node-level caching with configurable TTLs and dynamic cache policies per agent.
  The entrypoint of the graph validates that the ticker is valid using yfinance.
  Once validated, the graph fans out to start SEC filings ingestion and parallel execution of most research agents (fundamental, technical, macro, industry, peer, headline).
  The filings research agent waits for SEC filings ingestion to complete before execution.
  When all agents have completed, an aggregator agent synthesizes overall sentiment for the stock.
  An evaluator agent reviews the aggregator's output for compliance. If non-compliant, it provides feedback and the aggregator revises (up to 3 iterations).

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

![Observability Metrics](/assets/metrics.png)

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

# Agent Details

This system employs a multi-agent architecture where specialized agents use different methods to gather and analyze data.

### 1. Vector RAG Agent (Retrieval-Augmented Generation)

Uses a local ChromaDB vector store to index and semantically search dense documents.

- **Filings Agent**: Ingests SEC filings (10-K, 10-Q) into a vector database. It embeds user queries to retrieve relevant text chunks for analysis.

### 2. Web RAG Agents (Search Grounding)

Leverages Google Gemini's search grounding capability to perform real-time web research.

- **Headline Agent**: Searches for recent news and headlines (last 30 days) to gauge market sentiment.
- **Industry Agent**: Researches industry-specific trends, headwinds, and tailwinds.
- **Peer Agent**: Analyzes competitor performance and market positioning.

### 3. Tool-Use Agents

Uses Python tools to fetch structured data from external APIs (yfinance, FRED) before analysis.

- **Fundamentals Agent**: Calls `yfinance` to retrieve balance sheets, income statements, and cash flow data.
- **Technical Agent**: Fetches `yfinance` price history to calculate indicators like RSI, MACD, and Bollinger Bands.
- **Macro Agent**: Connects to the FRED (Federal Reserve Economic Data) API to fetch GDP, inflation, and consumer sentiment data.

### 4. Core Workflow Agents

Synthesizes information and manages the research process.

- **Sentiment Aggregator**: Combines outputs from all specialist agents into a cohesive research report.
- **Sentiment Evaluator**: Reviews the aggregated report for quality and compliance, triggering revisions if necessary.
