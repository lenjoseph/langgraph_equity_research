"""Shared decision for what to do when the evaluator itself fails."""

MAX_REVISION_COUNT = 2


def evaluation_failure_update(iteration: int, reason: str) -> dict:
    """Build the evaluator state update after the judge cannot score a draft.

    While revision iterations remain, the draft is not treated as compliant.
    Once the iteration cap is reached, evaluation_status is skipped so the
    graph stops without labeling the draft as passed.
    """
    if iteration > MAX_REVISION_COUNT:
        return {
            "compliant": False,
            "evaluation_status": "skipped",
            "feedback": reason,
            "revision_iteration_count": iteration,
            "faithfulness_score": None,
        }
    return {
        "compliant": False,
        "evaluation_status": "failed",
        "feedback": reason,
        "revision_iteration_count": iteration,
        "faithfulness_score": None,
    }
