import dotenv
from langchain_openai import ChatOpenAI

from agents.evaluation.prompt import sentiment_evaluator_prompt
from agents.shared.llm_models import LLM_MODELS
from models.agent import AggregatorFeedback

dotenv.load_dotenv()


def evaluate_aggregated_sentement(sentiment: str):

    prompt = f"Evaluate this sentiment for criteria compliance: {sentiment}"

    prompt += (
        f"Use these criteria as the evaluation target: {sentiment_evaluator_prompt}"
    )

    model = LLM_MODELS["open_ai_smart"]
    llm = ChatOpenAI(model=model, temperature=0.0).with_structured_output(
        schema=AggregatorFeedback
    )
    result = llm.invoke(prompt)

    return {"compliant": result.compliant, "feedback": result.feedback}
