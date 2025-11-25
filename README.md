# lg_equity_research

Agentic equity research implemented on LangGraph

# Running Locally

To run the program from the terminal, follow these steps:

1. Initialize a python environment on python version 3.12
2. Install deps with pip or other python package manager
3. Hydrate env variables in a new .env file (variables are specified in the env.dev file, langsmith is unnecessary for terminal use)
4. Run `python main.py` from the root of the project
5. From a separate terminal window, execute the following curl command:
   curl -X POST "http://localhost:8000/research-equity" -H "Content-Type: application/json" -d '{"ticker": "TICKER"}'

# Running in LangSmith for Observability

To run the program in Langsmith, which offers observability for graph execution and tracing, follow these steps:

1. Initialize a python environment on python version 3.12
2. Install deps with pip or other python package manager
3. Hydrate env variables in a new .env file (variables are specified in the env.dev file, in this case langsmith is needed)
4. Run `langsmith dev` from the root of the project
5. A browser window should open showing the built graph and execution controls

# Architecture

This app implements an agentic ai architecture to compile equity research on a single stock across different analytical lenses.

![Architecture Diagram](/architecture.png)

The app structures ai agents as research domain specialists that perform data gathering and analysis scoped to their respective domain.
The app manages state through langgraph's graph state model.
The graph implements a node-based cache that configures cache keys and ttl at the agent level.
The entrypoint of the graph vaidates that the ticker is valid using yfinance.
Once the ticker is vaidated, research agents are executed in parallel.
When all agents have executed, a aggregator agent synthesizes overall sentiment for the stock.

# Architecture Components

- HTTP API: FastAPI
- Agentic Architecture: Langgraph
- Agent node caching: Langgraph Node Cache
- Agent Observability: LangSmith
- Type Package: Pydantic
- LLM Models: OpenAI, Google Gemini
- Financial Data API: YFinance
- Economic Data API: Pandas Datareader

# AI Agent Descriptions

1. Technical Researcher - Focuses on technical price data over fixed timeframes
2. Fundamentals Researcher - Focuses on business fundamentals via most recent earnings results
3. Macro Economic Researcher - Focuses on the macro economic components provided by federal reserve data
4. Industry Researcher - Focuses on forecasted headwinds / tailwinds relative to the industry of the stock
5. Headline Researcher - Focuses on recent (within the last month) headlines about the stock
6. Sentiment Aggregator - Compiles overall stock sentiment based on aggregate findings of research specialist agents
