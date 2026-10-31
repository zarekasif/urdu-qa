# Data

The data files are not stored in this repository.

**Input:** `URDU TRANSLATED DATASET.csv`, the machine-translated Urdu SQuAD from the 2022 project
(columns `Paragraphs`, `Questions`, `Answers`, `answer_start`; 40,329 rows over 14,387 paragraphs).
Its own answer offsets are kept when they match the text.

**Build the splits:**

Old pre-split JSON files (e.g. `TRAIN90.json`, `VALIDATION5.json`, `TEST5.json`) work too: pass
them all to `prepare` and they are pooled, de-duplicated and re-split by paragraph.

```bash
python -m urduqa.prepare "data/raw/URDU TRANSLATED DATASET.csv" data/processed/all.jsonl
python -m urduqa.split data/processed/all.jsonl data/processed --seed 13
```

This writes `train.jsonl`, `dev.jsonl` and `test.jsonl` (80/10/10 by paragraph) and prints the
number of rows dropped for each reason and the train-to-eval paragraph overlap (should be 0%).
On the 2022 data this gives 40,273 questions: train 32,290 / dev 3,959 / test 4,024, 0% overlap.

The translations come from Google Cloud Translation applied to
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0), so any redistributed
version of this dataset should carry the same licence.
