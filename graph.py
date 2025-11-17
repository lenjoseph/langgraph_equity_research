from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
from pydantic import BaseModel

from agents.macro_agent import get_macro_sentiment
from agents.technical_agent import get_technical_sentiment

load_dotenv()


# graph state
class EquityResearchState(BaseModel):
    ticker: str
    fundamental_sentiment: str
    technical_sentiment: str
    macro_sentiment: str
    industry_sentiment: str
    headline_sentiment: str
    combined_sentiment: str


# graph nodes
def fundamental_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate fundamental research sentiment"""
    fundamental_sentiment = "sdfadsfa"
    return {"fundamental_sentiment": fundamental_sentiment}


def technical_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    technical_sentiment = get_technical_sentiment(ticker=state.ticker)
    return {"technical_sentiment": technical_sentiment}


def macro_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate fundamental research sentiment"""
    macro_sentiment = get_macro_sentiment()
    return {"macro_sentiment": macro_sentiment}


def industry_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    industry_sentiment = "gewrgw"
    return {"industry_sentiment": industry_sentiment}


def headline_research_agent(state: EquityResearchState) -> dict:
    """LLM call to generate technical research sentiment"""
    headline_sentiment = "sodfiowd"
    return {"headline_sentiment": headline_sentiment}


def sentiment_aggregator(state: EquityResearchState) -> dict:
    """LLM call to aggregate research findings and synthesize sentiment"""
    combined_sentiment = "oahfioahg"
    return {"combined_sentiment": combined_sentiment}


# build workflow
parallel_builder = StateGraph(EquityResearchState)

# add agent nodes
parallel_builder.add_node("fundamental_research_agent", fundamental_research_agent)
parallel_builder.add_node("technical_research_agent", technical_research_agent)
parallel_builder.add_node("macro_research_agent", macro_research_agent)
parallel_builder.add_node("industry_research_agent", industry_research_agent)
parallel_builder.add_node("headline_research_agent", headline_research_agent)
parallel_builder.add_node("aggregator", sentiment_aggregator)

# call research agents in parallel
parallel_builder.add_edge(START, "fundamental_research_agent")
parallel_builder.add_edge(START, "technical_research_agent")
parallel_builder.add_edge(START, "macro_research_agent")
parallel_builder.add_edge(START, "industry_research_agent")
parallel_builder.add_edge(START, "headline_research_agent")

# synthesize sentiment
parallel_builder.add_edge("fundamental_research_agent", "aggregator")
parallel_builder.add_edge("technical_research_agent", "aggregator")
parallel_builder.add_edge("macro_research_agent", "aggregator")
parallel_builder.add_edge("industry_research_agent", "aggregator")
parallel_builder.add_edge("headline_research_agent", "aggregator")

# terminate graph
parallel_builder.add_edge("aggregator", END)

# compile the graph workflow
parallel_workflow = parallel_builder.compile()

ticker = "TSLA"
state = EquityResearchState(
    ticker=ticker,
    fundamental_sentiment="",
    technical_sentiment="",
    macro_sentiment="",
    industry_sentiment="",
    headline_sentiment="",
    combined_sentiment="",
)
result = parallel_workflow.invoke(state)

print(result)


def input(ticker: str) -> EquityResearchState:
    state = EquityResearchState(
        ticker=ticker,
        fundamental_sentiment="",
        technical_sentiment="",
        macro_sentiment="",
        industry_sentiment="",
        headline_sentiment="",
        combined_sentiment="",
    )
    return state


def output(state: EquityResearchState) -> EquityResearchState:
    return state


research_chain = RunnableLambda(input) | parallel_workflow | RunnableLambda(output)
