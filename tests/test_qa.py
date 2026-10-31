"""Tests for QA feature preparation and answer extraction, using a small whitespace
tokenizer that mimics the Hugging Face fast-tokenizer interface (offsets, overflowing
windows with stride, sequence_ids). The real-model path runs in the Colab notebook."""

import re

import numpy as np

from urduqa.qa import postprocess_predictions, prepare_eval_features, prepare_train_features


class _Encoding(dict):
    def __init__(self, data, seq_ids):
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, i):
        return self._seq_ids[i]


class WhitespaceTokenizer:
    padding_side = "right"

    @staticmethod
    def _words(text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def __call__(self, questions, contexts, truncation, max_length, stride,
                 return_overflowing_tokens, return_offsets_mapping, padding):
        data = {"input_ids": [], "attention_mask": [], "offset_mapping": [], "overflow_to_sample_mapping": []}
        seq_ids_all = []
        for sample, (q, c) in enumerate(zip(questions, contexts)):
            q_words = self._words(q)
            c_words = self._words(c)
            budget = max_length - len(q_words) - 3
            step = budget - stride
            assert step > 0
            begin = 0
            while True:
                window = c_words[begin : begin + budget]
                offsets = [(0, 0)] + q_words + [(0, 0)] + window + [(0, 0)]
                seq = [None] + [0] * len(q_words) + [None] + [1] * len(window) + [None]
                pad = max_length - len(offsets)
                data["input_ids"].append(list(range(len(offsets))) + [0] * pad)
                data["attention_mask"].append([1] * len(offsets) + [0] * pad)
                data["offset_mapping"].append(offsets + [(0, 0)] * pad)
                data["overflow_to_sample_mapping"].append(sample)
                seq_ids_all.append(seq + [None] * pad)
                if begin + budget >= len(c_words):
                    break
                begin += step
        return _Encoding(data, seq_ids_all)


CONTEXT = " ".join(f"لفظ{i}" for i in range(40)) + " جواب یہاں ہے " + " ".join(f"اور{i}" for i in range(40))
ANSWER = "جواب یہاں"
EXAMPLES = {
    "id": ["q1"],
    "question": ["جواب کہاں ہے؟"],
    "context": [CONTEXT],
    "answers": [{"text": [ANSWER], "answer_start": [CONTEXT.index(ANSWER)]}],
}


def test_train_labels_point_at_the_answer_in_every_window_that_contains_it():
    tok = WhitespaceTokenizer()
    enc = prepare_train_features(EXAMPLES, tok, max_length=32, stride=8)
    assert len(enc["input_ids"]) > 1  # long context was split into windows
    hits = 0
    for i, (s, e) in enumerate(zip(enc["start_positions"], enc["end_positions"])):
        if s == 0 and e == 0:
            continue  # window without the answer
        enc_offsets = tok(EXAMPLES["question"], EXAMPLES["context"], "only_second", 32, 8, True, True, "max_length")["offset_mapping"][i]
        assert CONTEXT[enc_offsets[s][0] : enc_offsets[e][1]] == ANSWER
        hits += 1
    assert hits >= 1


def test_postprocess_returns_the_best_scoring_span_and_ignores_question_tokens():
    tok = WhitespaceTokenizer()
    train = prepare_train_features(EXAMPLES, tok, max_length=32, stride=8)
    feats = prepare_eval_features(EXAMPLES, tok, max_length=32, stride=8)
    n, length = len(feats["input_ids"]), 32
    start = np.zeros((n, length))
    end = np.zeros((n, length))
    for i, (s, e) in enumerate(zip(train["start_positions"], train["end_positions"])):
        if s:
            start[i, s] = 5.0
            end[i, e] = 5.0
    # A stronger score on a question token must be ignored.
    start[0, 1] = end[0, 1] = 50.0
    preds = postprocess_predictions(EXAMPLES, feats, start, end)
    assert preds == {"q1": ANSWER}


def test_question_offsets_are_masked_in_eval_features():
    tok = WhitespaceTokenizer()
    feats = prepare_eval_features(EXAMPLES, tok, max_length=32, stride=8)
    assert all(o is None for o in feats["offset_mapping"][0][:5])
    assert feats["example_id"] == ["q1"] * len(feats["input_ids"])
