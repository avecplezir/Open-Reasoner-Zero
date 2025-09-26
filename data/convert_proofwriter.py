"""
Convert the Hugging Face ProofWriter dataset to ORZ formats similar to StrategyQA.
Adds context before the question and ensures a trailing '?'.

Outputs (default paths):
- data/proofwriter_train.json          -> ORZ train list[list[dict]]
- data/eval_data/proofwriter_dev.json  -> ORZ eval list[dict]

Usage examples:
    # Default config and splits
    python data/convert_proofwriter.py

    # Specify dataset/config/splits explicitly
    python data/convert_proofwriter.py \
        --dataset tasksource/proofwriter \
        --config default \
        --train-split train \
        --dev-split dev \
        --train-out data/proofwriter_train.json \
        --dev-out data/eval_data/proofwriter_dev.json

Prompt format:
    Context:
    <context text>
    Question: <question?>
If no context exists, the prompt is just '<question?>'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None


def ensure_qmark(text: str) -> str:
    s = (text or "").strip()
    if s and not s.endswith("?"):
        s = s + "?"
    return s


def coerce_yes_no(value: Any) -> str:
    v = value
    if isinstance(v, dict):
        for k in ("answer", "label", "labels", "target"):
            if k in v:
                v = v[k]
                break
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return "yes" if v != 0 else "no"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"yes", "y", "true", "t", "1"}:
            return "yes"
        if s in {"no", "n", "false", "f", "0"}:
            return "no"
    return "yes" if bool(v) else "no"


def collect_context(record: Dict[str, Any], context_fields: Optional[List[str]]) -> str:
    keys = context_fields or [
        "context",
        "facts",
        "premises",
        "supports",
        "evidence",
        "story",
        "passage",
        "theory",
        "kb",
        "rules",
        "proof",
    ]
    parts: List[str] = []
    for k in keys:
        if k not in record:
            continue
        v = record[k]
        if v is None:
            continue
        if isinstance(v, str):
            t = v.strip()
            if t:
                parts.append(t)
        elif isinstance(v, (list, tuple)):
            sub: List[str] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        sub.append(t)
                elif isinstance(item, dict):
                    for dk in ("text", "sentence", "fact", "value"):
                        if dk in item and isinstance(item[dk], str):
                            t = item[dk].strip()
                            if t:
                                sub.append(t)
                                break
                else:
                    t = str(item).strip()
                    if t:
                        sub.append(t)
            if sub:
                parts.append("\n".join(sub))
        elif isinstance(v, dict):
            sub = [str(x).strip() for x in v.values() if isinstance(x, str) and x.strip()]
            if sub:
                parts.append("\n".join(sub))
        else:
            t = str(v).strip()
            if t:
                parts.append(t)
    return "\n".join(p for p in parts if p)


def format_prompt(record: Dict[str, Any], question_field: Optional[str], context_fields: Optional[List[str]]) -> str:
    q: Optional[str] = None
    if question_field and question_field in record:
        q = str(record[question_field])
    else:
        for key in ("question", "query", "q", "hypothesis"):
            if key in record:
                q = str(record[key])
                break
    if q is None:
        q = str(record)
    q = ensure_qmark(q)

    ctx = collect_context(record, context_fields)
    if ctx:
        return f"Context:\n{ctx}\nQuestion: {q}"
    return q


def extract_answer(record: Dict[str, Any], label_field: Optional[str]) -> str:
    if label_field and label_field in record:
        return coerce_yes_no(record[label_field])
    for key in ("answer", "label", "labels", "target"):
        if key in record:
            return coerce_yes_no(record[key])
    return "no"


def to_orz_train(
    record: Dict[str, Any],
    question_field: Optional[str],
    label_field: Optional[str],
    context_fields: Optional[List[str]],
) -> List[Dict[str, Any]]:
    question = format_prompt(record, question_field, context_fields)
    ans = extract_answer(record, label_field)
    return [
        {"from": "human", "value": question},
        {"from": "assistant", "ground_truth": {"value": ans}},
    ]


def to_orz_eval(
    record: Dict[str, Any],
    question_field: Optional[str],
    label_field: Optional[str],
    context_fields: Optional[List[str]],
) -> Dict[str, Any]:
    question = format_prompt(record, question_field, context_fields)
    ans = extract_answer(record, label_field)
    return {
        "prompt": [{"from": "user", "value": question}],
        "final_answer": ans,
    }


def normalize_split_name(name: str) -> str:
    n = name.strip().lower()
    if n in {"validation", "val", "valid"}:
        return "dev"
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ProofWriter to ORZ formats")
    parser.add_argument("--dataset", type=str, default="tasksource/proofwriter", help="HF dataset path")
    parser.add_argument("--config", type=str, default=None, help="HF dataset config name (if any)")
    parser.add_argument("--train-split", type=str, default="train", help="Train split name")
    parser.add_argument("--dev-split", type=str, default="dev", help="Dev/validation split name")
    parser.add_argument("--question-field", type=str, default=None, help="Override question field name")
    parser.add_argument("--label-field", type=str, default=None, help="Override label/answer field name")
    parser.add_argument(
        "--context-fields",
        type=str,
        default=None,
        help="Comma-separated context field names to include before the question",
    )
    parser.add_argument(
        "--train-out", type=Path, default=Path("data/proofwriter_train.json"), help="Output train JSON path"
    )
    parser.add_argument(
        "--dev-out",
        type=Path,
        default=Path("data/eval_data/proofwriter_dev.json"),
        help="Output eval JSON path",
    )
    args = parser.parse_args()

    if load_dataset is None:
        raise RuntimeError("datasets is not installed. Please `pip install datasets`.")

    ds_kwargs = {}
    if args.config:
        ds_kwargs["name"] = args.config

    train_split = normalize_split_name(args.train_split)
    dev_split = normalize_split_name(args.dev_split)

    ds_train = load_dataset(args.dataset, **ds_kwargs, split=train_split)
    ds_dev = load_dataset(args.dataset, **ds_kwargs, split=dev_split)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.dev_out.parent.mkdir(parents=True, exist_ok=True)

    context_fields: Optional[List[str]] = (
        [s.strip() for s in args.context_fields.split(",") if s.strip()] if args.context_fields else None
    )

    train_out: List[List[Dict[str, Any]]] = [
        to_orz_train(rec, args.question_field, args.label_field, context_fields) for rec in ds_train
    ]
    with args.train_out.open("w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False)

    dev_out: List[Dict[str, Any]] = [
        to_orz_eval(rec, args.question_field, args.label_field, context_fields) for rec in ds_dev
    ]
    with args.dev_out.open("w", encoding="utf-8") as f:
        json.dump(dev_out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

