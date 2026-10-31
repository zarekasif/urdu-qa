"""Turn the translated Urdu SQuAD CSV into SQuAD-style JSONL.

Each output line is one question, compatible with Hugging Face ``datasets``:

    {"id", "paragraph_id", "context", "question",
     "answers": {"text": [...], "answer_start": [...]}}

Rows whose answer no longer appears verbatim in the translated context are dropped, since
extractive QA needs a span. This is the same filter the 2022 project applied, now with the
reason for every dropped row counted.

Usage:
    python -m urduqa.prepare data/raw/final_translated_dataset.csv data/processed/all.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .normalize import clean_text

# Column names used across the 2022 CSV exports, first match wins.
_COLUMNS = {
    "context": ["Paragraphs", "context", "Context"],
    "question": ["Questions", "question", "Question"],
    "answer": ["Answers", "answer_text", "text", "Answer"],
}


def _pick(df: pd.DataFrame, key: str) -> str:
    for name in _COLUMNS[key]:
        if name in df.columns:
            return name
    raise KeyError(f"No {key} column found; expected one of {_COLUMNS[key]}, got {list(df.columns)}")


def paragraph_id(context: str) -> str:
    return hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]


def prepare(df: pd.DataFrame) -> tuple[list[dict], Counter]:
    c_col, q_col, a_col = _pick(df, "context"), _pick(df, "question"), _pick(df, "answer")
    records: list[dict] = []
    dropped: Counter = Counter()
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        if any(pd.isna(row[c]) for c in (c_col, q_col, a_col)):
            dropped["missing_field"] += 1
            continue
        context, question, answer = (clean_text(row[c]) for c in (c_col, q_col, a_col))
        if not (context and question and answer):
            dropped["empty_field"] += 1
            continue
        start = context.find(answer)
        if start < 0:
            dropped["answer_not_in_context"] += 1
            continue
        key = (context, question)
        if key in seen:
            dropped["duplicate_question"] += 1
            continue
        seen.add(key)

        pid = paragraph_id(context)
        records.append(
            {
                "id": f"{pid}-{len(records)}",
                "paragraph_id": pid,
                "context": context,
                "question": question,
                "answers": {"text": [answer], "answer_start": [start]},
            }
        )
    return records, dropped


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    records, dropped = prepare(df)
    write_jsonl(records, args.out)
    paragraphs = len({r["paragraph_id"] for r in records})
    print(f"rows in: {len(df)}  kept: {len(records)}  paragraphs: {paragraphs}")
    for reason, n in dropped.most_common():
        print(f"  dropped {reason}: {n}")


if __name__ == "__main__":
    main()
