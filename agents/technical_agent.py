from langchain_openai import ChatOpenAI
import dotenv

from tools.get_technicals import get_technical_analysis_tool
from prompts.technicals_prompt import technical_research_prompt
from constants.llm_models import LLM_MODELS

dotenv.load_dotenv()


def get_technical_sentiment(ticker: str):
    model = LLM_MODELS["open_ai"]
    prompt = f"{technical_research_prompt}\n\nAnalyze the technical indicators for ticker: {ticker}"
    tools = [get_technical_analysis_tool]
    tools_map = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(model=model).bind_tools(tools)

    response = llm.invoke(prompt)

    if response.tool_calls:
        tool_call = response.tool_calls[0]

        requested_tool_name = tool_call["name"]
        requested_tool = tools_map[requested_tool_name]

        tool_args = tool_call["args"]
        tool_result = requested_tool.func(**tool_args)

        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [response.tool_calls[0]],
            },
            {
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"],
            },
        ]

        final_response = llm.invoke(messages)
        return final_response.content
    else:
        return response.content
