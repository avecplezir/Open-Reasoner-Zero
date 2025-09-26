"""
Convert the Hugging Face `tasksource/ruletaker` dataset to ORZ formats
consistent with StrategyQA converters. Ensures each question ends with '?',
includes the example's context before the question, and handles Unknown labels
via a configurable policy (drop or map to yes/no).

Outputs (default paths):
- data/ruletaker_train.json            -> ORZ train list[list[dict]]
- data/eval_data/ruletaker_dev.json    -> ORZ eval list[dict]

Usage examples:
    # Default config and splits
    python data/convert_ruletaker.py

    # Specify a dataset config (if needed) and custom splits
    python data/convert_ruletaker.py \
        --dataset tasksource/ruletaker \
        --config default \
        --train-split train \
        --dev-split dev \
        --train-out data/ruletaker_train.json \
        --dev-out data/eval_data/ruletaker_dev.json \
        --unknown-policy drop

Notes:
- The prompt is formatted as:
    Context:
    <context text>
    Question: <question?>
  If no context is found, the prompt is just <question?>.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception as e:  # pragma: no cover - import-time dependency check
    load_dataset = None  # allow module import without datasets installed


def ensure_qmark(text: str) -> str:
    s = (text or "").strip()
    if s and not s.endswith("?"):
        s = s + "?"
    return s


def try_coerce_yes_no(value: Any) -> Optional[str]:
    """Map labels to 'yes'/'no' where possible; return None for Unknown.

    Recognizes: bool; int 0/1; strings in yes={yes,y,true,t,1} and no={no,n,false,f,0}.
    Treats strings like {unknown, both, neither, uncertain, idk, maybe} as Unknown.
    For dicts, looks into common keys.
    """
    v = value
    if isinstance(v, dict):
        for k in ("answer", "label", "labels", "target"):
            if k in v:
                v = v[k]
                break
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        if v in (0, 1):
            return "yes" if v == 1 else "no"
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        yes_set = {"yes", "y", "true", "t", "1"}
        no_set = {"no", "n", "false", "f", "0"}
        unknown_set = {"unknown", "both", "neither", "uncertain", "idk", "maybe"}
        if s in yes_set:
            return "yes"
        if s in no_set:
            return "no"
        if s in unknown_set:
            return None
        if s.isdigit():
            return "yes" if int(s) == 1 else ("no" if int(s) == 0 else None)
        return None
    return None


def collect_context(record: Dict[str, Any], context_fields: Optional[List[str]]) -> str:
    """Collect context text from commonly used fields.

    Attempts the provided `context_fields` first; falls back to typical keys
    observed in boolean QA datasets.
    """
    keys = context_fields or [
        "context",
        "premises",
        "facts",
        "story",
        "passage",
        "supports",
        "evidence",
    ]

    parts: List[str] = []
    for k in keys:
        if k not in record:
            continue
        v = record[k]
        if v is None:
            continue
        # Normalize to strings
        if isinstance(v, str):
            txt = v.strip()
            if txt:
                parts.append(txt)
        elif isinstance(v, (list, tuple)):
            subparts: List[str] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        subparts.append(t)
                elif isinstance(item, dict):
                    # Try common text keys
                    for dk in ("text", "sentence", "fact", "value"):
                        if dk in item and isinstance(item[dk], str):
                            t = item[dk].strip()
                            if t:
                                subparts.append(t)
                                break
                else:
                    t = str(item).strip()
                    if t:
                        subparts.append(t)
            if subparts:
                parts.append("\n".join(subparts))
        elif isinstance(v, dict):
            # Join all string values
            sub = [str(x).strip() for x in v.values() if isinstance(x, str) and x.strip()]
            if sub:
                parts.append("\n".join(sub))
        else:
            t = str(v).strip()
            if t:
                parts.append(t)
    return "\n".join(p for p in parts if p)


def format_prompt(record: Dict[str, Any], question_field: Optional[str], context_fields: Optional[List[str]]) -> str:
    # Question
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

    # Context
    ctx = collect_context(record, context_fields)
    if ctx:
        return f"Context:\n{ctx}\nQuestion: {q}"
    return q


def extract_answer(record: Dict[str, Any], label_field: Optional[str]) -> Optional[str]:
    if label_field and label_field in record:
        return try_coerce_yes_no(record[label_field])
    for key in ("answer", "label", "labels", "target"):
        if key in record:
            return try_coerce_yes_no(record[key])
    return None


def to_orz_train(
    record: Dict[str, Any],
    question_field: Optional[str],
    label_field: Optional[str],
    context_fields: Optional[List[str]],
) -> Optional[List[Dict[str, Any]]]:
    question = format_prompt(record, question_field, context_fields)
    ans = extract_answer(record, label_field)
    if ans is None:
        return None
    return [
        {"from": "human", "value": question},
        {"from": "assistant", "ground_truth": {"value": ans}},
    ]


def to_orz_eval(
    record: Dict[str, Any],
    question_field: Optional[str],
    label_field: Optional[str],
    context_fields: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    question = format_prompt(record, question_field, context_fields)
    ans = extract_answer(record, label_field)
    if ans is None:
        return None
    return {
        "prompt": [{"from": "user", "value": question}],
        "final_answer": ans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RuleTaker (tasksource/ruletaker) to ORZ formats")
    parser.add_argument("--dataset", type=str, default="tasksource/ruletaker", help="HF dataset path")
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
        "--unknown-policy",
        type=str,
        default="drop",
        choices=["drop", "map-yes", "map-no", "keep"],
        help="How to handle Unknown labels: drop (default), map-yes, map-no, or keep (sets answer to 'unknown').",
    )
    parser.add_argument(
        "--train-out", type=Path, default=Path("data/ruletaker_train.json"), help="Output train JSON path"
    )
    parser.add_argument(
        "--dev-out",
        type=Path,
        default=Path("data/eval_data/ruletaker_dev.json"),
        help="Output eval JSON path",
    )
    args = parser.parse_args()

    if load_dataset is None:
        raise RuntimeError("datasets is not installed. Please `pip install datasets`." )

    # Load dataset splits
    ds_kwargs = {}
    if args.config:
        ds_kwargs["name"] = args.config
    ds_train = load_dataset(args.dataset, **ds_kwargs, split=args.train_split)
    ds_dev = load_dataset(args.dataset, **ds_kwargs, split=args.dev_split)

    # Ensure output directories exist
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.dev_out.parent.mkdir(parents=True, exist_ok=True)

    # Convert and save train
    context_fields: Optional[List[str]] = (
        [s.strip() for s in args.context_fields.split(",") if s.strip()] if args.context_fields else None
    )
    train_out: List[List[Dict[str, Any]]] = []
    for rec in ds_train:
        ans = extract_answer(rec, args.label_field)
        if ans is None:
            if args.unknown_policy == "map-yes":
                ans = "yes"
            elif args.unknown_policy == "map-no":
                ans = "no"
            elif args.unknown_policy == "keep":
                ans = "unknown"
            else:
                continue
        q = format_prompt(rec, args.question_field, context_fields)
        train_out.append([
            {"from": "human", "value": q},
            {"from": "assistant", "ground_truth": {"value": ans}},
        ])
    with args.train_out.open("w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False)

    # Convert and save dev
    dev_out: List[Dict[str, Any]] = []
    for rec in ds_dev:
        ans = extract_answer(rec, args.label_field)
        if ans is None:
            if args.unknown_policy == "map-yes":
                ans = "yes"
            elif args.unknown_policy == "map-no":
                ans = "no"
            elif args.unknown_policy == "keep":
                ans = "unknown"
            else:
                continue
        q = format_prompt(rec, args.question_field, context_fields)
        dev_out.append({
            "prompt": [{"from": "user", "value": q}],
            "final_answer": ans,
        })
    with args.dev_out.open("w", encoding="utf-8") as f:
        json.dump(dev_out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
