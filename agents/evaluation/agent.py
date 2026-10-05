import time
from typing import Any, Dict, Optional, Tuple

import dotenv

from agents.aggregation.agent import specialist_research_block
from agents.evaluation.format_check import check_aggregation_format
from agents.evaluation.policy import evaluation_failure_update
from agents.evaluation.prompt import sentiment_evaluator_prompt
from agents.shared.agent_utils import invoke_llm_with_metrics
from agents.shared.llm_models import LLM_MODELS, get_google_llm
from agents.shared.token_config import DEFAULT_TOKEN_CONFIG, AgentTokenConfig
from models.agent import AggregatorFeedback
from models.metrics import AgentMetrics

dotenv.load_dotenv()

AGENT_NAME = "evaluation"
# Judge is a different model family from the OpenAI writer.
JUDGE_MODEL = LLM_MODELS["google_smart"]


def evaluate_aggregated_sentiment(
    sentiment: str,
    iteration: int = 1,
    token_config: Optional[AgentTokenConfig] = None,
    specialist_research: str = "",
    trade_duration: str = "",
    trade_direction: str = "",
) -> Tuple[Dict[str, Any], AgentMetrics]:
    """
    Evaluate aggregated sentiment for template compliance and faithfulness.

    The judge sees the specialist analyses. Format checks run in code and can
    fail a draft even when the model marks it compliant. Judge failures do not
    mark the draft compliant.
    """
    start_time = time.perf_counter()
    config = token_config or DEFAULT_TOKEN_CONFIG.evaluation
    format_issues = check_aggregation_format(sentiment)

    prompt = (
        f"Trade duration: {trade_duration}\n"
        f"Trade direction: {trade_direction}\n\n"
        f"Specialist research:\n{specialist_research}\n\n"
        f"Synthesis to evaluate:\n{sentiment}\n\n"
        f"Evaluation criteria:\n{sentiment_evaluator_prompt}"
    )

    base_llm = get_google_llm(
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    call = invoke_llm_with_metrics(
        base_llm, prompt, AggregatorFeedback, token_budget=config.token_budget
    )

    budget_exceeded = bool(
        (config.token_budget and call.usage.total_tokens > config.token_budget)
        or (call.error and "budget" in call.error.lower())
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    agent_name = f"{AGENT_NAME}_{iteration}" if iteration > 1 else AGENT_NAME
    metrics = AgentMetrics(
        agent_name=agent_name,
        latency_ms=latency_ms,
        token_usage=call.usage,
        model=JUDGE_MODEL,
        budget_exceeded=budget_exceeded,
    )

    if call.result is None:
        reason = call.error or "Evaluation unavailable due to an internal error."
        if format_issues:
            reason = "Format issues: " + "; ".join(format_issues) + " " + reason
        return evaluation_failure_update(iteration, reason), metrics

    feedback_parts = []
    if format_issues:
        feedback_parts.append("Format issues: " + "; ".join(format_issues))
    if call.result.feedback:
        feedback_parts.append(call.result.feedback)
    compliant = bool(call.result.compliant) and not format_issues
    if not compliant and not feedback_parts:
        feedback_parts.append("Revise the synthesis so it matches the template and the specialist research.")

    return {
        "compliant": compliant,
        "feedback": " ".join(feedback_parts),
        "evaluation_status": "passed" if compliant else "failed",
        "revision_iteration_count": iteration,
        "faithfulness_score": call.result.faithfulness,
    }, metrics


def evaluate_state(state, iteration: int, token_config: Optional[AgentTokenConfig] = None):
    """Evaluate using specialist text stored on the graph state."""
    duration = state.trade_duration.value if hasattr(state.trade_duration, "value") else state.trade_duration
    direction = (
        state.trade_direction.value
        if hasattr(state.trade_direction, "value")
        else state.trade_direction
    )
    return evaluate_aggregated_sentiment(
        sentiment=state.combined_sentiment or "",
        iteration=iteration,
        token_config=token_config,
        specialist_research=specialist_research_block(state),
        trade_duration=str(duration),
        trade_direction=str(direction),
    )


def evaluate_aggregated_sentement(
    sentiment: str,
    iteration: int = 1,
    token_config: Optional[AgentTokenConfig] = None,
) -> Tuple[Dict[str, Any], AgentMetrics]:
    """Backward-compatible alias for callers that still use the original name."""
    return evaluate_aggregated_sentiment(
        sentiment=sentiment,
        iteration=iteration,
        token_config=token_config,
    )
