# Data

The data files are not stored in this repository.

**Input:** `data/raw/final_translated_dataset.csv`, the machine-translated Urdu SQuAD from the
2022 project (columns `Paragraphs`, `Questions`, `Answers`; about 40k rows over 14.4k paragraphs).

**Build the splits:**

Old pre-split JSON files (e.g. `TRAIN90.json`, `VALIDATION5.json`, `TEST5.json`) work too: pass
them all to `prepare` and they are pooled, de-duplicated and re-split by paragraph.

```bash
python -m urduqa.prepare data/raw/final_translated_dataset.csv data/processed/all.jsonl
python -m urduqa.split data/processed/all.jsonl data/processed --seed 13
```

This writes `train.jsonl`, `dev.jsonl` and `test.jsonl` (80/10/10 by paragraph) and prints the
number of rows dropped for each reason and the train-to-eval paragraph overlap (should be 0%).

The translations come from Google Cloud Translation applied to
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0), so any redistributed
version of this dataset should carry the same licence.
