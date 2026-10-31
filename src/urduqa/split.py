"""Paragraph-level train/dev/test split, plus a leakage check.

All questions about the same paragraph go to the same split, so a model is never evaluated
on a passage it was trained on. The 2022 notebooks split by row instead; `leakage` measures
how much that overlap was.

Usage:
    python -m urduqa.split data/processed/all.jsonl data/processed --seed 13
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .prepare import read_jsonl, write_jsonl


def split_by_paragraph(
    records: list[dict], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 13
) -> dict[str, list[dict]]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    pids = sorted({r["paragraph_id"] for r in records})
    random.Random(seed).shuffle(pids)

    n = len(pids)
    n_train = round(n * ratios[0])
    n_dev = round(n * ratios[1])
    assignment = {pid: "train" for pid in pids[:n_train]}
    assignment.update({pid: "dev" for pid in pids[n_train : n_train + n_dev]})
    assignment.update({pid: "test" for pid in pids[n_train + n_dev :]})

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in records:
        splits[assignment[r["paragraph_id"]]].append(r)
    return splits


def leakage(train: list[dict], evaluation: list[dict]) -> float:
    """Share of evaluation questions whose paragraph also appears in training."""
    train_pids = {r["paragraph_id"] for r in train}
    if not evaluation:
        return 0.0
    return sum(r["paragraph_id"] in train_pids for r in evaluation) / len(evaluation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    splits = split_by_paragraph(read_jsonl(args.jsonl), seed=args.seed)
    for name, rows in splits.items():
        write_jsonl(rows, args.out_dir / f"{name}.jsonl")
        n_par = len({r["paragraph_id"] for r in rows})
        print(f"{name:5s} questions: {len(rows):6d}  paragraphs: {n_par:6d}")
    for name in ("dev", "test"):
        print(f"leakage train->{name}: {leakage(splits['train'], splits[name]):.1%}")


if __name__ == "__main__":
    main()
