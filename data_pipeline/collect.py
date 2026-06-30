from datasets import load_dataset
import json


def load_code_alpaca():
    """Download and explore the CodeAlpaca dataset."""
    print("Downloading CodeAlpaca dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    print(f"\n✅ Total examples: {len(dataset['train'])}")
    print(f"\n--- Sample Example ---")
    sample = dataset['train'][0]
    for key, value in sample.items():
        print(f"{key}: {value}")

    print(f"\n--- Another Sample ---")
    sample2 = dataset['train'][100]
    for key, value in sample2.items():
        print(f"{key}: {value}")

    return dataset


if __name__ == "__main__":
    dataset = load_code_alpaca()
