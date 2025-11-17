from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "", temperature=0
)

@tool
def get_fundamentals(ticker: str):
    return ticker

tools = [get_fundamentals]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)