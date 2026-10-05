"""LangSmith eval entrypoint."""

from langsmith import Client, evaluate

from agents.aggregation.agent import specialist_research_block
from agents.evaluation.agent import evaluate_aggregated_sentiment
from graph import research_chain
from models.state import TradeDirection, TradeDuration

client = Client()
dataset_name = "Equity Research Golden Dataset"

FIXTURE_INPUTS = [
    {
        "ticker": "AAPL",
        "trade_duration": TradeDuration.POSITION_TRADE.value,
        "trade_direction": TradeDirection.LONG.value,
    },
    {
        "ticker": "TSLA",
        "trade_duration": TradeDuration.SWING_TRADE.value,
        "trade_direction": TradeDirection.SHORT.value,
    },
]

# Human-labeled expectations used when the dataset is first created.
FIXTURE_OUTPUTS = [
    {
        "overall_sentiment": "BULLISH",
        "must_mention": ["Fundamental", "SEC Filings", "Position Implication"],
    },
    {
        "overall_sentiment": "BEARISH",
        "must_mention": ["Fundamental", "SEC Filings", "Position Implication"],
    },
]


def _state_from_outputs(outputs):
    if hasattr(outputs, "combined_sentiment"):
        return outputs
    return outputs or {}


def relevance_evaluator(run, example):
    outputs = _state_from_outputs(run.outputs)
    if hasattr(outputs, "combined_sentiment"):
        sentiment = outputs.combined_sentiment
        specialists = specialist_research_block(outputs)
        duration = outputs.trade_duration
        direction = outputs.trade_direction
    else:
        sentiment = outputs.get("combined_sentiment")
        specialists = "\n".join(
            f"{name}: {outputs.get(name, '')}"
            for name in (
                "fundamental_sentiment",
                "technical_sentiment",
                "macro_sentiment",
                "industry_sentiment",
                "peer_sentiment",
                "headline_sentiment",
                "filings_sentiment",
            )
        )
        duration = outputs.get("trade_duration", "")
        direction = outputs.get("trade_direction", "")

    update, _metrics = evaluate_aggregated_sentiment(
        sentiment=sentiment or "",
        specialist_research=specialists,
        trade_duration=str(getattr(duration, "value", duration)),
        trade_direction=str(getattr(direction, "value", direction)),
    )
    faithfulness = update.get("faithfulness_score") or 0
    return {
        "key": "faithfulness",
        "score": faithfulness / 2,
        "comment": update.get("feedback") or update.get("evaluation_status"),
    }


def main():
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=FIXTURE_INPUTS,
            outputs=FIXTURE_OUTPUTS,
            dataset_id=dataset.id,
        )

    return evaluate(
        research_chain.invoke,
        data=dataset_name,
        evaluators=[relevance_evaluator],
        experiment_prefix="equity-research-v1",
    )


if __name__ == "__main__":
    main()
