import os
from langchain_openai import ChatOpenAI
import dotenv

from tools.get_macro import get_macro_data_tool
from prompts.macro_prompt import macro_research_prompt
from constants.llm_models import LLM_MODELS

dotenv.load_dotenv()


def get_macro_sentiment():
    model = LLM_MODELS["open_ai"]
    prompt = macro_research_prompt
    tools = [get_macro_data_tool]
    tools_map = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(model=model).bind_tools(tools)

    # First LLM call - it will request to call the tool
    response = llm.invoke(prompt)

    # Check if there are tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]

        # DYNAMIC: Look up which tool the LLM requested
        requested_tool_name = tool_call["name"]
        requested_tool = tools_map[requested_tool_name]

        # Call the actual tool function
        tool_args = tool_call["args"]
        tool_result = requested_tool.func(**tool_args)

        # Create messages for the second LLM call with tool results
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

        # Second LLM call with tool results to get the analysis
        final_response = llm.invoke(messages)
        return final_response.content
    else:
        # No tool call needed, return the response directly
        return response.content
