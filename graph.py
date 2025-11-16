from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from IPython.display import Image, display
from dotenv import load_dotenv
from pydantic import BaseModel

import llm_models

load_dotenv()

llm = init_chat_model(llm_models.LLM_MODELS["OAI"])


# graph state
class EquityResearchState(BaseModel):
    ticker: str
    fundamental_sentiment: str
    technical_sentiment: str
    combined_sentiment: str


# graph nodes
def receive_ticker(state: EquityResearchState) -> EquityResearchState:
    """LLM call to add ticker to state"""
    ticker = "TSLA"
    return state.model_copy(update={"ticker": ticker})


def call_fundamentals_research_agent(state: EquityResearchState) -> EquityResearchState:
    """LLM call to generate fundamental research sentiment"""


def call_technicals_research_agent(state: EquityResearchState) -> EquityResearchState:
    """LLM call to generate technical research sentiment"""


def synthesize_sentiment_agent(state: EquityResearchState) -> EquityResearchState:
    """LLM call to combine research findings and synthesize sentiment"""


# build workflow
parallel_builder = StateGraph(EquityResearchState)

# add agent nodes
parallel_builder.add_node(
    "call_fundamentals_research_agent", call_fundamentals_research_agent
)
parallel_builder.add_node(
    "call_technicals_research_agent", call_technicals_research_agent
)
parallel_builder.add_node("receive_ticker", receive_ticker)

# add edges to connect agent nodes
parallel_builder.set_entry_point("receive_ticker")

parallel_builder.add_edge("receive_ticker", "call_fundamentals_research_agent")
parallel_builder.add_edge("receive_ticker", "call_technicals_research_agent")
parallel_builder.add_edge(
    "call_fundamentals_research_agent", "synthesize_sentiment_agent"
)
parallel_builder.add_edge(
    "call_technicals_research_agent", "synthesize_sentiment_agent"
)
parallel_builder.add_edge("synthesize_sentiment_agent", END)

# compile the graph workflow
parallel_workflow = parallel_builder.compile()


def input(ticker: str) -> dict:
    return {"ticker": list({ticker})}


def output(state: EquityResearchState) -> str:
    result = state["combined_sentiment"]
    return result


research_chain = RunnableLambda(input) | parallel_workflow | RunnableLambda(output)
