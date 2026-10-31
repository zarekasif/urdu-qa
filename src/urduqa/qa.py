"""Feature preparation and answer extraction for extractive QA models.

Model-agnostic: works with any Hugging Face *fast* tokenizer (it needs offset mappings).
Long contexts are split into overlapping windows; at prediction time the best-scoring span
across all windows of a question is returned. Follows the standard SQuAD recipe used by
the Hugging Face question-answering examples, with two fixes over the 2022 code: windows
cover the whole context instead of truncating it, and the answer is chosen by model score
(start + end logits), not by string order.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _tokenize(tokenizer, examples, max_length: int, stride: int):
    pad_right = tokenizer.padding_side == "right"
    questions = [q.strip() for q in examples["question"]]
    return tokenizer(
        questions if pad_right else examples["context"],
        examples["context"] if pad_right else questions,
        truncation="only_second" if pad_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def prepare_train_features(examples, tokenizer, max_length: int = 384, stride: int = 128):
    """Tokenise a batch and label each window with the answer's start/end token.

    Windows that do not contain the answer are labelled with the CLS position (0), the usual
    SQuAD convention, so the model learns that such a window holds no answer.
    """
    enc = _tokenize(tokenizer, examples, max_length, stride)
    context_index = 1 if tokenizer.padding_side == "right" else 0
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    starts, ends = [], []

    for i, offsets in enumerate(offsets_all):
        cls_index = 0
        answers = examples["answers"][sample_map[i]]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        seq_ids = enc.sequence_ids(i)

        ctx_start = next(k for k, s in enumerate(seq_ids) if s == context_index)
        ctx_end = len(seq_ids) - 1 - next(k for k, s in enumerate(reversed(seq_ids)) if s == context_index)

        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            starts.append(cls_index)
            ends.append(cls_index)
            continue

        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        starts.append(tok_start - 1)
        ends.append(tok_end + 1)

    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def prepare_eval_features(examples, tokenizer, max_length: int = 384, stride: int = 128):
    """Tokenise a batch for prediction, keeping what is needed to map tokens back to text."""
    enc = _tokenize(tokenizer, examples, max_length, stride)
    context_index = 1 if tokenizer.padding_side == "right" else 0
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["example_id"] = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["example_id"].append(examples["id"][sample_map[i]])
        # Keep offsets only for context tokens, so question tokens can never be picked.
        enc["offset_mapping"][i] = [
            o if seq_ids[k] == context_index else None for k, o in enumerate(enc["offset_mapping"][i])
        ]
    return enc


def postprocess_predictions(
    examples,
    features,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    n_best: int = 20,
    max_answer_tokens: int = 30,
) -> dict[str, str]:
    """Pick the highest-scoring valid span per question across all of its windows.

    ``examples`` needs ``id`` and ``context``; ``features`` needs ``example_id`` and
    ``offset_mapping`` as produced by :func:`prepare_eval_features`.
    """
    by_example = defaultdict(list)
    for f_idx, ex_id in enumerate(features["example_id"]):
        by_example[ex_id].append(f_idx)

    predictions: dict[str, str] = {}
    for ex_id, context in zip(examples["id"], examples["context"]):
        best_text, best_score = "", -np.inf
        for f_idx in by_example.get(ex_id, []):
            s_log, e_log = start_logits[f_idx], end_logits[f_idx]
            offsets = features["offset_mapping"][f_idx]
            s_cand = np.argsort(s_log)[-1 : -n_best - 1 : -1]
            e_cand = np.argsort(e_log)[-1 : -n_best - 1 : -1]
            for s in s_cand:
                for e in e_cand:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = s_log[s] + e_log[e]
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0] : offsets[e][1]]
        predictions[ex_id] = best_text
    return predictions
