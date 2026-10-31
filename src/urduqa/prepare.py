"""Turn the translated Urdu SQuAD CSV into SQuAD-style JSONL.

Each output line is one question, compatible with Hugging Face ``datasets``:

    {"id", "paragraph_id", "context", "question",
     "answers": {"text": [...], "answer_start": [...]}}

Rows whose answer no longer appears verbatim in the translated context are dropped, since
extractive QA needs a span. This is the same filter the 2022 project applied, now with the
reason for every dropped row counted.

Inputs can be the CSV export (columns ``Paragraphs``, ``Questions``, ``Answers``) and/or
JSON files in SQuAD format (``{"data": [...]}``) or the simpletransformers list format
(``[{"context", "qas": [...]}]``). Several files are merged and de-duplicated, so old
pre-split train/validation/test files can be pooled and re-split by paragraph.

Usage:
    python -m urduqa.prepare "data/raw/URDU TRANSLATED DATASET.csv" data/processed/all.jsonl
    python -m urduqa.prepare TRAIN90.json VALIDATION5.json TEST5.json data/processed/all.jsonl
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
    "start": ["answer_start", "Start Answer"],
}


def _pick(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for name in _COLUMNS[key]:
        if name in df.columns:
            return name
    if not required:
        return None
    raise KeyError(f"No {key} column found; expected one of {_COLUMNS[key]}, got {list(df.columns)}")


def _rows_from_paragraphs(paragraphs) -> list[tuple]:
    rows = []
    for para in paragraphs:
        for qa in para.get("qas", []):
            answers = qa.get("answers") or []
            if answers:
                rows.append((para["context"], qa["question"], answers[0]["text"], answers[0].get("answer_start")))
    return rows


def load_any(path: Path) -> pd.DataFrame:
    """Load a CSV or a SQuAD-style / simpletransformers-style JSON file as one DataFrame."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return pd.read_csv(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "data" in obj:  # official SQuAD layout
        paragraphs = [p for article in obj["data"] for p in article["paragraphs"]]
    elif isinstance(obj, list):  # simpletransformers layout
        paragraphs = obj
    else:
        raise ValueError(f"Unrecognised JSON layout in {path}")
    return pd.DataFrame(_rows_from_paragraphs(paragraphs), columns=["Paragraphs", "Questions", "Answers", "answer_start"])


def _given_start(raw_context: str, raw_answer: str, value) -> int | None:
    """Return the dataset's own answer offset, re-mapped onto the cleaned context, if usable."""
    try:
        start = int(value)
    except (TypeError, ValueError):
        return None
    if start < 0 or str(raw_context)[start : start + len(str(raw_answer))] != str(raw_answer):
        return None
    # Offsets refer to the raw text; map them onto the cleaned text via the cleaned prefix.
    return len(clean_text(str(raw_context)[:start] + "x")) - 1


def paragraph_id(context: str) -> str:
    return hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]


def prepare(df: pd.DataFrame) -> tuple[list[dict], Counter]:
    c_col, q_col, a_col = _pick(df, "context"), _pick(df, "question"), _pick(df, "answer")
    s_col = _pick(df, "start", required=False)
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
        start = None
        if s_col is not None:
            start = _given_start(row[c_col], row[a_col], row[s_col])
            if start is not None and context[start : start + len(answer)] != answer:
                start = None
        if start is None:
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
    ap.add_argument("inputs", type=Path, nargs="+", help="one or more CSV/JSON files, then the output path")
    args = ap.parse_args()
    *inputs, out = args.inputs
    if not inputs:
        ap.error("give at least one input file and an output path")

    frames = []
    for path in inputs:
        part = load_any(path)
        cols = [_pick(part, "context"), _pick(part, "question"), _pick(part, "answer")]
        names = ["Paragraphs", "Questions", "Answers"]
        start_col = _pick(part, "start", required=False)
        if start_col is not None:
            cols.append(start_col)
            names.append("answer_start")
        part = part[cols].set_axis(names, axis=1)
        print(f"{path.name}: {len(part)} rows")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    records, dropped = prepare(df)
    write_jsonl(records, out)
    paragraphs = len({r["paragraph_id"] for r in records})
    print(f"rows in: {len(df)}  kept: {len(records)}  paragraphs: {paragraphs}")
    for reason, n in dropped.most_common():
        print(f"  dropped {reason}: {n}")


if __name__ == "__main__":
    main()
