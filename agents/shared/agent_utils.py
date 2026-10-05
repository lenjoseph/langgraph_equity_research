from typing import Any, NamedTuple, Optional, Type, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.shared.grounding import extract_grounding_sources
from agents.shared.request_budget import record_request_usage, request_budget_allows
from models.metrics import TokenUsage
from models.result import AgentStatus, GroundingSource
from util.logger import get_logger
from util.retry import call_with_retry

logger = get_logger(__name__)


class TokenBudgetExceeded(Exception):
    """Raised when an agent token budget has been exceeded before a call."""

    def __init__(self, budget: int, used: int, message: str = None):
        self.budget = budget
        self.used = used
        self.message = message or f"Token budget exceeded: {used}/{budget} tokens used"
        super().__init__(self.message)


class RequestBudgetExceeded(Exception):
    """Raised when the request-level token budget is already exhausted."""

    def __init__(self, message: str = "Request token budget exceeded."):
        super().__init__(message)


class LLMCall(NamedTuple):
    """One LLM invocation outcome. `result` is the parsed payload when parsing succeeded."""

    result: Any
    usage: TokenUsage
    grounding_sources: list
    status: AgentStatus
    error: Optional[str]


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
    """Extract token usage from a LangChain response."""
    if response is None:
        return TokenUsage()
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return TokenUsage(
            input_tokens=response.usage_metadata.get("input_tokens", 0),
            output_tokens=response.usage_metadata.get("output_tokens", 0),
            total_tokens=response.usage_metadata.get("total_tokens", 0),
        )
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


def _estimate_tokens(llm, payload) -> int:
    try:
        if isinstance(payload, str):
            return llm.get_num_tokens(payload)
        if isinstance(payload, list):
            text = "\n".join(
                str(message.get("content", "")) if isinstance(message, dict) else str(message)
                for message in payload
            )
            return llm.get_num_tokens(text)
    except Exception as exc:
        logger.warning(f"Could not estimate tokens before call: {exc}")
    return 0


def _invoke_with_retry(runnable, payload):
    return call_with_retry(lambda: runnable.invoke(payload))


def _budget_block(
    llm,
    payload,
    token_budget: Optional[int],
    current_usage: int,
) -> Optional[LLMCall]:
    """Return a failed call when an agent or request budget blocks the invocation."""
    if not request_budget_allows(0):
        return LLMCall(
            result=None,
            usage=TokenUsage(),
            grounding_sources=[],
            status=AgentStatus.FAILED,
            error="Request token budget exceeded.",
        )
    if token_budget is None:
        return None
    if not check_token_budget(current_usage, token_budget):
        return LLMCall(
            result=None,
            usage=TokenUsage(total_tokens=current_usage, input_tokens=current_usage),
            grounding_sources=[],
            status=AgentStatus.DEGRADED,
            error=f"Token budget exceeded: {current_usage}/{token_budget} tokens used",
        )
    estimated = _estimate_tokens(llm, payload)
    if estimated and not check_token_budget(current_usage + estimated, token_budget):
        return LLMCall(
            result=None,
            usage=TokenUsage(input_tokens=estimated, total_tokens=estimated),
            grounding_sources=[],
            status=AgentStatus.DEGRADED,
            error=(
                f"Token budget exceeded by input: {current_usage + estimated}/{token_budget} tokens"
            ),
        )
    if estimated and not request_budget_allows(estimated):
        return LLMCall(
            result=None,
            usage=TokenUsage(),
            grounding_sources=[],
            status=AgentStatus.FAILED,
            error="Request token budget exceeded.",
        )
    return None


def _execute_tool_calls(response, tools_map: dict) -> tuple[list[dict], bool]:
    """Execute every tool call. Unknown names return an error message and set unknown=True."""
    tool_messages = []
    unknown = False
    for tool_call in response.tool_calls:
        name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        if name not in tools_map:
            unknown = True
            available = ", ".join(sorted(tools_map)) or "(none)"
            tool_messages.append(
                {
                    "role": "tool",
                    "content": f"Unknown tool '{name}'. Available tools: {available}.",
                    "tool_call_id": tool_call_id,
                }
            )
            continue
        try:
            tool_result = tools_map[name].func(**(tool_call.get("args") or {}))
            content = str(tool_result)
        except Exception as exc:
            content = f"Tool execution failed: {exc}"
        tool_messages.append(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            }
        )
    return tool_messages, unknown


def _assistant_tool_message(response) -> dict:
    return {
        "role": "assistant",
        "content": response.content if getattr(response, "content", None) else "",
        "tool_calls": list(response.tool_calls),
    }


def run_agent_with_tools(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI],
    prompt: str,
    tools: list = None,
    output_schema: Optional[Type[BaseModel]] = None,
    track_tokens: bool = False,
    token_budget: Optional[int] = None,
) -> Union[Any, LLMCall]:
    """
    Run a tool-calling agent.

    Every tool call in a response is executed. If the model names an unknown tool,
    it gets one retry with the error, then a final answer is requested.
    `track_tokens=True` returns an LLMCall. Otherwise the parsed payload is returned.
    """
    total_usage = TokenUsage()
    try:
        tools = tools or []
        tools_map = {tool.name: tool for tool in tools}
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        blocked = _budget_block(llm, prompt, token_budget, 0)
        if blocked:
            return blocked if track_tokens else blocked.error

        response = _invoke_with_retry(llm_with_tools, prompt)
        total_usage = _aggregate_token_usage(total_usage, _extract_token_usage(response))
        record_request_usage(total_usage.total_tokens)

        if not check_token_budget(total_usage.total_tokens, token_budget):
            logger.warning(
                f"Token budget exceeded after initial call: {total_usage.total_tokens}/{token_budget}"
            )
            call = LLMCall(
                result=response.content if hasattr(response, "content") else str(response),
                usage=total_usage,
                grounding_sources=extract_grounding_sources(response),
                status=AgentStatus.DEGRADED,
                error=(
                    f"Token budget exceeded after initial call: "
                    f"{total_usage.total_tokens}/{token_budget}"
                ),
            )
            return call if track_tokens else call.result

        messages = [{"role": "user", "content": prompt}]
        if getattr(response, "tool_calls", None):
            messages.append(_assistant_tool_message(response))
            tool_messages, unknown = _execute_tool_calls(response, tools_map)
            messages.extend(tool_messages)

            if unknown and request_budget_allows(0) and check_token_budget(
                total_usage.total_tokens, token_budget
            ):
                retry_response = _invoke_with_retry(llm_with_tools, messages)
                retry_usage = _extract_token_usage(retry_response)
                total_usage = _aggregate_token_usage(total_usage, retry_usage)
                record_request_usage(retry_usage.total_tokens)
                if getattr(retry_response, "tool_calls", None):
                    messages.append(_assistant_tool_message(retry_response))
                    extra_messages, _unknown_again = _execute_tool_calls(
                        retry_response, tools_map
                    )
                    messages.extend(extra_messages)
                else:
                    response = retry_response

        if getattr(response, "tool_calls", None) or (
            messages and any(message.get("role") == "tool" for message in messages)
        ):
            if output_schema:
                structured_llm = llm.with_structured_output(output_schema, include_raw=True)
                blocked = _budget_block(llm, messages, token_budget, total_usage.total_tokens)
                if blocked:
                    blocked = blocked._replace(
                        usage=_aggregate_token_usage(total_usage, blocked.usage)
                    )
                    return blocked if track_tokens else blocked.error
                raw_result = _invoke_with_retry(structured_llm, messages)
                parsed = raw_result["parsed"]
                raw = raw_result.get("raw")
                usage = _extract_token_usage(raw)
                total_usage = _aggregate_token_usage(total_usage, usage)
                record_request_usage(usage.total_tokens)
                status = AgentStatus.OK if parsed is not None else AgentStatus.FAILED
                error = None if parsed is not None else "Structured output parsing returned no result."
                if not check_token_budget(total_usage.total_tokens, token_budget):
                    status = AgentStatus.DEGRADED if parsed is not None else status
                    error = (
                        f"Token budget exceeded after tool call: "
                        f"{total_usage.total_tokens}/{token_budget}"
                    )
                call = LLMCall(
                    result=parsed,
                    usage=total_usage,
                    grounding_sources=extract_grounding_sources(raw),
                    status=status,
                    error=error,
                )
                return call if track_tokens else parsed
            final_response = _invoke_with_retry(llm_with_tools, messages)
            usage = _extract_token_usage(final_response)
            total_usage = _aggregate_token_usage(total_usage, usage)
            record_request_usage(usage.total_tokens)
            content = final_response.content
            status = AgentStatus.OK
            error = None
            if not check_token_budget(total_usage.total_tokens, token_budget):
                status = AgentStatus.DEGRADED
                error = (
                    f"Token budget exceeded after tool call: "
                    f"{total_usage.total_tokens}/{token_budget}"
                )
            call = LLMCall(
                result=content,
                usage=total_usage,
                grounding_sources=extract_grounding_sources(final_response),
                status=status,
                error=error,
            )
            return call if track_tokens else content

        if output_schema:
            structured_llm = llm.with_structured_output(output_schema, include_raw=True)
            raw_result = _invoke_with_retry(structured_llm, prompt)
            parsed = raw_result["parsed"]
            raw = raw_result.get("raw")
            usage = _extract_token_usage(raw)
            total_usage = _aggregate_token_usage(total_usage, usage)
            record_request_usage(usage.total_tokens)
            status = AgentStatus.OK if parsed is not None else AgentStatus.FAILED
            error = None if parsed is not None else "Structured output parsing returned no result."
            if not check_token_budget(total_usage.total_tokens, token_budget):
                status = AgentStatus.DEGRADED if parsed is not None else status
                error = (
                    f"Token budget exceeded: {total_usage.total_tokens}/{token_budget}"
                )
            call = LLMCall(
                result=parsed,
                usage=total_usage,
                grounding_sources=extract_grounding_sources(raw),
                status=status,
                error=error,
            )
            return call if track_tokens else parsed

        content = response.content
        call = LLMCall(
            result=content,
            usage=total_usage,
            grounding_sources=extract_grounding_sources(response),
            status=AgentStatus.OK,
            error=None,
        )
        return call if track_tokens else content
    except (TokenBudgetExceeded, RequestBudgetExceeded) as exc:
        call = LLMCall(
            result=None,
            usage=total_usage,
            grounding_sources=[],
            status=AgentStatus.DEGRADED
            if isinstance(exc, TokenBudgetExceeded)
            else AgentStatus.FAILED,
            error=str(exc),
        )
        if track_tokens:
            return call
        return call.error
    except Exception as exc:
        logger.error(f"Error in run_agent_with_tools: {exc}", exc_info=True)
        call = LLMCall(
            result=None,
            usage=total_usage,
            grounding_sources=[],
            status=AgentStatus.FAILED,
            error=f"Error executing agent: {exc}",
        )
        if track_tokens:
            return call
        return call.error


def invoke_llm_with_metrics(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI],
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    token_budget: Optional[int] = None,
    current_usage: int = 0,
) -> LLMCall:
    """Invoke an LLM and return the parsed result, usage, and grounding sources."""
    blocked = _budget_block(llm, prompt, token_budget, current_usage)
    if blocked:
        return blocked

    try:
        if output_schema:
            structured_llm = llm.with_structured_output(output_schema, include_raw=True)
            raw_result = _invoke_with_retry(structured_llm, prompt)
            result = raw_result["parsed"]
            raw = raw_result.get("raw")
            usage = _extract_token_usage(raw)
            grounding: list[GroundingSource] = extract_grounding_sources(raw)
        else:
            response = _invoke_with_retry(llm, prompt)
            result = response.content if hasattr(response, "content") else response
            usage = _extract_token_usage(response)
            grounding = extract_grounding_sources(response)

        record_request_usage(usage.total_tokens)
        new_total = current_usage + usage.total_tokens
        status = AgentStatus.OK if result is not None else AgentStatus.FAILED
        error = None if result is not None else "Structured output parsing returned no result."
        if not check_token_budget(new_total, token_budget):
            logger.warning(f"Token budget exceeded after call: {new_total}/{token_budget}")
            if result is not None:
                status = AgentStatus.DEGRADED
            error = f"Token budget exceeded after call: {new_total}/{token_budget}"

        return LLMCall(
            result=result,
            usage=usage,
            grounding_sources=grounding,
            status=status,
            error=error,
        )
    except Exception as exc:
        logger.error(f"Error in invoke_llm_with_metrics: {exc}", exc_info=True)
        return LLMCall(
            result=None,
            usage=TokenUsage(),
            grounding_sources=[],
            status=AgentStatus.FAILED,
            error=f"Error executing agent: {exc}",
        )
