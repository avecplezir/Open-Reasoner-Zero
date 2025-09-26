"""
Convert BoolQ/BoolIQ JSONL files into ORZ formats used for StrategyQA.

Inputs (default paths):
- data/BoolQ_train.jsonl  -> outputs train data like data/strategyqa.json
- data/BoolIQ_dev.jsonl   -> outputs eval data like data/eval_data/strategyqa_test.json

Usage:
    python data/convert_boolq.py \
        --train-in data/BoolQ_train.jsonl \
        --train-out data/boolq.json \
        --dev-in data/BoolIQ_dev.jsonl \
        --dev-out data/eval_data/booliq_dev.json

If arguments are omitted, the defaults above are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def to_orz_train(record: Dict) -> List[Dict]:
    # StrategyQA train format: list of two messages with ground_truth
    ans = "yes" if record.get("answer") else "no"
    question = (record.get("question") or "").strip()
    if question and not question.endswith("?"):
        question = question + "?"
    return [
        {"from": "human", "value": question},
        {"from": "assistant", "ground_truth": {"value": ans}},
    ]


def to_orz_eval(record: Dict) -> Dict:
    # StrategyQA eval format: {prompt: [{from:user, value:...}], final_answer: yes/no}
    ans = "yes" if record.get("answer") else "no"
    question = (record.get("question") or "").strip()
    if question and not question.endswith("?"):
        question = question + "?"
    return {
        "prompt": [{"from": "user", "value": question}],
        "final_answer": ans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BoolQ/BoolIQ to ORZ formats")
    parser.add_argument(
        "--train-in",
        type=Path,
        default=Path("data/BoolQ_train.jsonl"),
        help="Input BoolQ train JSONL path",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/boolq.json"),
        help="Output train JSON path (StrategyQA-like)",
    )
    parser.add_argument(
        "--dev-in",
        type=Path,
        default=Path("data/BoolIQ_dev.jsonl"),
        help="Input BoolIQ dev JSONL path",
    )
    parser.add_argument(
        "--dev-out",
        type=Path,
        default=Path("data/eval_data/booliq_dev.json"),
        help="Output eval JSON path (StrategyQA-test-like)",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.dev_out.parent.mkdir(parents=True, exist_ok=True)

    # Convert train
    train_records = load_jsonl(args.train_in)
    train_out = [to_orz_train(r) for r in train_records]
    with args.train_out.open("w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False)

    # Convert dev -> eval format
    dev_records = load_jsonl(args.dev_in)
    dev_out = [to_orz_eval(r) for r in dev_records]
    with args.dev_out.open("w", encoding="utf-8") as f:
        json.dump(dev_out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
