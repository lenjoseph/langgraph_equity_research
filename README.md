# lg_equity_research

Agentic equity research implemented on LangGraph

# Architecture Description

This app implements an agentic ai architecture to compile equity research on a single stock across different analytical lenses.
The app exposes a POST endpoint that receives a single stock ticker which serves as the focal point of the research.
The app structures ai agents as research domain specialists that perform data gathering and analysis scoped to their respective domain.
The app manages state through langgraph's graph state model.
Research agents are executed in parallel at the start of the workflow.
When all agents have executed, a aggregator agent synthesizes overall sentiment for the stock.

# Architecture Components

HTTP API: FastAPI
Agentic Architecture: Langgraph
Agent Observability: LangSmith
Data Typing: Pydantic
LLM Models: OpenAI
In-Memory Vector Store: ChromaDB
Vector Embedding Model: HuggingFace
Financial Data API: YFinance
Economic Data API: Pandas Datareader

# AI Agent Descriptions

1. Technical Researcher - Focuses on technical price data over fixed timeframes
2. Fundamentals Researcher - Focuses on business fundamentals via most recent earnings results
3. Macro Economic Researcher - Focuses on the macro economic components provided by federal reserve data
4. Industry Researcher - Focuses on forecasted headwinds / tailwinds relative to the industry of the stock
5. Headline Researcher - Focuses on recent (within the last month) headlines about the stock
6. Sentiment Aggregator - Compiles overall stock sentiment based on aggregate findings of research specialist agents
