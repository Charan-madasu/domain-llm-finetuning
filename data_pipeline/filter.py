from datasets import load_dataset
import json
import re


def format_instruction(example: dict) -> str:
    """Convert example to instruction format."""
    if example.get("input", "").strip():
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def is_quality_example(example: dict) -> bool:
    """Check if example passes quality filters."""
    output = example.get("output", "")
    instruction = example.get("instruction", "")

    # Filter 1: too short outputs are usually low quality
    if len(output.strip()) < 10:
        return False

    # Filter 2: too long outputs may be noisy or off-topic
    if len(output) > 2000:
        return False

    # Filter 3: instruction must be meaningful
    if len(instruction.strip()) < 10:
        return False

    # Filter 4: output should contain actual code-like content
    # (has common code symbols, not just plain English)
    code_indicators = [
        "(", ")", "{", "}", "=", "def ", "function", "return", "print", "import"]
    if not any(indicator in output for indicator in code_indicators):
        return False

    return True


def remove_duplicates(examples: list) -> list:
    """Remove exact duplicate instructions."""
    seen = set()
    unique = []
    for ex in examples:
        key = ex["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def build_clean_dataset(max_samples: int = 5000):
    """Full pipeline: load, filter, deduplicate, format."""
    print("Loading raw dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k")
    raw_examples = dataset["train"]

    print(f"Raw examples: {len(raw_examples)}")

    # Apply quality filters
    filtered = [ex for ex in raw_examples if is_quality_example(ex)]
    print(f"After quality filter: {len(filtered)}")

    # Remove duplicates
    unique = remove_duplicates(filtered)
    print(f"After deduplication: {len(unique)}")

    # Limit to max_samples
    final = unique[:max_samples]
    print(f"Final dataset size: {len(final)}")

    # Format and save
    formatted_data = []
    for ex in final:
        formatted_data.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"],
            "text": format_instruction(ex)
        })

    with open("data_pipeline/clean_dataset.json", "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2)

    print(
        f"\n✅ Saved {len(formatted_data)} clean examples to data_pipeline/clean_dataset.json")

    # Show a sample
    print("\n--- Sample Formatted Example ---")
    print(formatted_data[0]["text"])

    return formatted_data


if __name__ == "__main__":
    build_clean_dataset(max_samples=5000)
