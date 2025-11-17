from typing import Any, Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def run_agent_with_tools(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI], prompt: str, tools: list = []
):
    """
    Generic agent executor that handles tool calling flow.

    Args:
        llm: The llm model to use for the agent
        prompt: The prompt to send to the LLM
        tools: List of tools to bind to the LLM

    Returns:
        The final LLM response content after executing any tool calls
    """
    try:
        tools_map = {tool.name: tool for tool in tools}

        llm_with_tools = llm.bind_tools(tools)

        # initial invocation
        response = llm_with_tools.invoke(prompt)

        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]

            # Look up which tool the LLM requested
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
            final_response = llm_with_tools.invoke(messages)
            return final_response.content
        else:
            # No tool call, return the response
            return response.content
    except Exception as e:
        print(f"ERROR in run_agent_with_tools: {e}")
        import traceback

        traceback.print_exc()
        return f"Error executing agent: {str(e)}"
