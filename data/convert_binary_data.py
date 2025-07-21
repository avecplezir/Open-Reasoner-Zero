from datasets import load_dataset


# ds_train = load_dataset("voidful/StrategyQA", split="train")
# ds_test  = load_dataset("voidful/StrategyQA", split="test")

ds_train = load_dataset("ChilleD/StrategyQA", split="train")
ds_test  = load_dataset("ChilleD/StrategyQA", split="test")

def to_orz(example):
    ans = "yes" if example["answer"] else "no"
    return [
        {"from": "human",     "value": example["question"].strip()},
        {"from": "assistant", "ground_truth": {"value": ans}}
    ]

orz_train = [to_orz(ex) for ex in ds_train]
orz_test  = [to_orz(ex) for ex in ds_test]

import json, pathlib
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/strategyqa.json", "w") as f:
    json.dump(orz_train, f, ensure_ascii=False)

with open("data/eval_data/strategyqa_test.json", "w") as f:
    json.dump(orz_test, f, ensure_ascii=False)