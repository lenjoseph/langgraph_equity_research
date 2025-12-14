from langsmith import Client


def create_eval_dataset():

    client = Client()

    dataset_name = "Equity Research Golden Dataset"
    description = "Standard test cases for validating equity research agent performance across different sectors and market caps."

    # check for data set, create if it doesn't exist
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' already exists. Appending new examples...")
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        print(f"Creating new dataset '{dataset_name}'...")
        dataset = client.create_dataset(
            dataset_name=dataset_name, description=description
        )

    # define test cases
    examples = [
        {
            "inputs": {
                "ticker": "AAPL",
                "trade_duration": "1 year",
                "trade_direction": "long",
            },
            "outputs": {"compliant": True},
        },
        {
            "inputs": {
                "ticker": "TSLA",
                "trade_duration": "1 week",
                "trade_direction": "short",
            }
        },
        {
            "inputs": {
                "ticker": "NVDA",
                "trade_duration": "3 months",
                "trade_direction": "neutral",
            }
        },
        {
            "inputs": {
                "ticker": "GME",
                "trade_duration": "1 day",
                "trade_direction": "volatility",
            }
        },
    ]

    # upload examples to langsmith via batch creation
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e.get("outputs") for e in examples],
        dataset_id=dataset.id,
    )

    print(f"Successfully added '{len(examples)}' examples to dataset '{dataset_name}")

    if __name__ == "__main__":
        create_eval_dataset()
