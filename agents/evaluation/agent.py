import dotenv

from agents.evaluation.prompt import sentiment_evaluator_prompt
from agents.shared.llm_models import LLM_MODELS, get_openai_llm
from models.agent import AggregatorFeedback

dotenv.load_dotenv()


def evaluate_aggregated_sentement(sentiment: str):

    prompt = f"Evaluate this sentiment for criteria compliance: {sentiment}"

    prompt += (
        f"Use these criteria as the evaluation target: {sentiment_evaluator_prompt}"
    )

    # Get cached base LLM, then wrap with structured output
    base_llm = get_openai_llm(model=LLM_MODELS["open_ai_smart"], temperature=0.0)
    llm = base_llm.with_structured_output(schema=AggregatorFeedback)
    result = llm.invoke(prompt)

    return {"compliant": result.compliant, "feedback": result.feedback}
