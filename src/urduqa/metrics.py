"""SQuAD-style Exact Match and token F1, with Urdu-aware normalisation.

Follows the official SQuAD v1.1 evaluation: each prediction is scored against every gold
answer and the best score is kept; corpus scores are the mean over questions, in percent.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from .normalize import normalize_answer


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric, prediction: str, golds: Iterable[str]) -> float:
    return max((metric(prediction, g) for g in golds), default=0.0)


def evaluate(golds: Mapping[str, list[str]], predictions: Mapping[str, str]) -> dict:
    """Score predictions ``{id: answer}`` against golds ``{id: [answers]}``.

    Questions with no prediction count as wrong, so a partial prediction file cannot
    inflate the score.
    """
    em_total = f1_total = 0.0
    missing = 0
    for qid, answers in golds.items():
        if qid not in predictions:
            missing += 1
            continue
        pred = predictions[qid]
        em_total += best_over_golds(exact_match, pred, answers)
        f1_total += best_over_golds(f1, pred, answers)
    n = len(golds)
    return {
        "exact_match": 100.0 * em_total / n if n else 0.0,
        "f1": 100.0 * f1_total / n if n else 0.0,
        "total": n,
        "missing": missing,
    }
