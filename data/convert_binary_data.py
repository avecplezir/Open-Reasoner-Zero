from datasets import load_dataset
import json, pathlib

# ds_train = load_dataset("voidful/StrategyQA", split="train")
# ds_test  = load_dataset("voidful/StrategyQA", split="test")

ds_train = load_dataset("ChilleD/StrategyQA", split="train")
ds_test  = load_dataset("ChilleD/StrategyQA", split="test")

def to_orz_train(example):
    ans = "yes" if example["answer"] else "no"
    return [
        {"from": "human",     "value": example["question"].strip()},
        {"from": "assistant", "ground_truth": {"value": ans}}
    ]

orz_train = [to_orz_train(ex) for ex in ds_train]

pathlib.Path("data").mkdir(exist_ok=True)
with open("data/strategyqa.json", "w") as f:
    json.dump(orz_train, f, ensure_ascii=False)

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

orz_test  = [to_orz_test(ex) for ex in ds_test]

with open("data/eval_data/strategyqa_test.json", "w") as f:
    json.dump(orz_test, f, ensure_ascii=False)