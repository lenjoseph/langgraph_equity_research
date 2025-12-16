from typing import Optional, Tuple, Type, Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from util.logger import get_logger
from models.metrics import TokenUsage

logger = get_logger(__name__)


class TokenBudgetExceeded(Exception):
    """Raised when a token budget has been exceeded."""

    def __init__(self, budget: int, used: int, message: str = None):
        self.budget = budget
        self.used = used
        self.message = message or f"Token budget exceeded: {used}/{budget} tokens used"
        super().__init__(self.message)


def check_token_budget(
    used_tokens: int,
    budget: Optional[int],
    raise_on_exceed: bool = False,
) -> bool:
    """
    Check if token usage is within budget.

    Args:
        used_tokens: Number of tokens already used
        budget: Maximum allowed tokens (None = unlimited)
        raise_on_exceed: If True, raise TokenBudgetExceeded instead of returning False

    Returns:
        True if within budget, False if exceeded

    Raises:
        TokenBudgetExceeded: If raise_on_exceed=True and budget is exceeded
    """
    if budget is None:
        return True

    if used_tokens >= budget:
        if raise_on_exceed:
            raise TokenBudgetExceeded(budget=budget, used=used_tokens)
        return False
    return True


def _extract_token_usage(response) -> TokenUsage:
    """Extract token usage from LangChain response."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return TokenUsage(
            input_tokens=response.usage_metadata.get("input_tokens", 0),
            output_tokens=response.usage_metadata.get("output_tokens", 0),
            total_tokens=response.usage_metadata.get("total_tokens", 0),
        )
    # For structured output, try response_metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        if token_usage:
            return TokenUsage(
                input_tokens=token_usage.get("prompt_tokens", 0),
                output_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
            )
    return TokenUsage()


def _aggregate_token_usage(*usages: TokenUsage) -> TokenUsage:
    """Aggregate multiple token usages into one."""
    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
    )


def run_agent_with_tools(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI],
    prompt: str,
    tools: list = None,
    output_schema: Optional[Type[BaseModel]] = None,
    track_tokens: bool = False,
    token_budget: Optional[int] = None,
) -> Union[any, Tuple[any, TokenUsage]]:
    """
    Generic agent executor that handles tool calling flow.

    Args:
        llm: The llm model to use for the agent
        prompt: The prompt to send to the LLM
        tools: List of tools to bind to the LLM
        output_schema: Optional Pydantic model for structured output
        track_tokens: If True, return tuple of (result, TokenUsage)
        token_budget: Optional maximum total tokens allowed for this agent execution.
                      If exceeded, stops further LLM calls and returns partial result.

    Returns:
        The final LLM response (structured if output_schema provided, else content string).
        If track_tokens=True, returns (result, TokenUsage).

    Raises:
        TokenBudgetExceeded: If token_budget is exceeded and no partial result available
    """
    total_usage = TokenUsage()

    try:
        tools = tools or []

        tools_map = {tool.name: tool for tool in tools}

        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # Check token budget before initial call
        if token_budget:
            try:
                # Estimate input tokens
                input_tokens = llm.get_num_tokens(prompt)
                if not check_token_budget(input_tokens, token_budget):
                    logger.warning(
                        f"Token budget would be exceeded by input: {input_tokens}/{token_budget}"
                    )
                    error_msg = f"Token budget exceeded by input: {input_tokens}/{token_budget} tokens"
                    if track_tokens:
                        # Return empty usage since we didn't run
                        return error_msg, TokenUsage(
                            input_tokens=input_tokens, total_tokens=input_tokens
                        )
                    return error_msg
            except Exception as e:
                logger.warning(f"Could not estimate tokens before call: {e}")

        # initial invocation
        response = llm_with_tools.invoke(prompt)
        total_usage = _aggregate_token_usage(
            total_usage, _extract_token_usage(response)
        )

        # Check token budget after initial call
        if not check_token_budget(total_usage.total_tokens, token_budget):
            logger.warning(
                f"Token budget exceeded after initial call: {total_usage.total_tokens}/{token_budget}"
            )
            # Return what we have so far
            if track_tokens:
                return (
                    response.content if hasattr(response, "content") else str(response)
                ), total_usage
            return response.content if hasattr(response, "content") else str(response)

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
            if output_schema:
                structured_llm = llm.with_structured_output(
                    output_schema, include_raw=True
                )
                raw_result = structured_llm.invoke(messages)
                final_response = raw_result["parsed"]
                if "raw" in raw_result:
                    total_usage = _aggregate_token_usage(
                        total_usage, _extract_token_usage(raw_result["raw"])
                    )
                if track_tokens:
                    return final_response, total_usage
                return final_response
            else:
                final_response = llm_with_tools.invoke(messages)
                total_usage = _aggregate_token_usage(
                    total_usage, _extract_token_usage(final_response)
                )
                if track_tokens:
                    return final_response.content, total_usage
                return final_response.content
        else:
            # No tool call, return the response
            if output_schema:
                structured_llm = llm.with_structured_output(
                    output_schema, include_raw=True
                )
                raw_result = structured_llm.invoke(prompt)
                final_response = raw_result["parsed"]
                if "raw" in raw_result:
                    total_usage = _aggregate_token_usage(
                        total_usage, _extract_token_usage(raw_result["raw"])
                    )
                if track_tokens:
                    return final_response, total_usage
                return final_response
            else:
                if track_tokens:
                    return response.content, total_usage
                return response.content
    except TokenBudgetExceeded:
        raise
    except Exception as e:
        logger.error(f"Error in run_agent_with_tools: {e}", exc_info=True)
        error_msg = f"Error executing agent: {str(e)}"
        if track_tokens:
            return error_msg, total_usage
        return error_msg


def invoke_llm_with_metrics(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI],
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    token_budget: Optional[int] = None,
    current_usage: int = 0,
) -> Tuple[any, TokenUsage]:
    """
    Invoke LLM and return result with token usage.

    For direct LLM invocations without tool calling.

    Args:
        llm: The LLM to invoke
        prompt: The prompt to send
        output_schema: Optional Pydantic model for structured output
        token_budget: Optional maximum total tokens allowed (None = unlimited)
        current_usage: Current token usage count (for budget tracking)

    Returns:
        Tuple of (result, TokenUsage)

    Raises:
        TokenBudgetExceeded: If token_budget is specified and would be exceeded
    """
    # Check if we're already over budget before making the call
    if not check_token_budget(current_usage, token_budget):
        logger.warning(f"Token budget already exceeded: {current_usage}/{token_budget}")
        raise TokenBudgetExceeded(budget=token_budget, used=current_usage)

    # Check if input tokens would exceed budget
    if token_budget:
        try:
            input_tokens = llm.get_num_tokens(prompt)
            if not check_token_budget(current_usage + input_tokens, token_budget):
                logger.warning(
                    f"Token budget would be exceeded by input: {current_usage + input_tokens}/{token_budget}"
                )
                raise TokenBudgetExceeded(
                    budget=token_budget, used=current_usage + input_tokens
                )
        except AttributeError:
            # LLM might not support get_num_tokens
            pass
        except Exception as e:
            logger.warning(f"Could not estimate tokens before call: {e}")

    try:
        if output_schema:
            structured_llm = llm.with_structured_output(output_schema, include_raw=True)
            raw_result = structured_llm.invoke(prompt)
            result = raw_result["parsed"]
            usage = _extract_token_usage(raw_result.get("raw"))
        else:
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else response
            usage = _extract_token_usage(response)

        # Log if we exceeded budget after the call
        new_total = current_usage + usage.total_tokens
        if not check_token_budget(new_total, token_budget):
            logger.warning(
                f"Token budget exceeded after call: {new_total}/{token_budget}"
            )

        return result, usage
    except TokenBudgetExceeded:
        raise
    except Exception as e:
        logger.error(f"Error in invoke_llm_with_metrics: {e}", exc_info=True)
        return None, TokenUsage()
