from datasets import load_dataset
import json, pathlib

ds_train = load_dataset("ChilleD/StrategyQA", split="train")

def to_orz_test(example):
    """Convert StrategyQA format to ORZ evaluation format."""

    answer = "yes" if example["answer"] else "no"
    question = example["question"].strip()

    # Convert to target format
    return {
        "prompt": [
            {
                "from": "user",
                "value": question
            }
        ],
        "final_answer": answer
    }

orz_test  = [to_orz_test(ex) for ex in ds_train]

with open("data/eval_data/strategyqa_train.json", "w") as f:
    json.dump(orz_test, f, ensure_ascii=False)