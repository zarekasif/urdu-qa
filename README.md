# Urdu QA

Extractive question answering for Urdu, a low-resource language with over 100 million speakers.
This repository revisits my 2022 BSc final-year project, an Urdu version of SQuAD built by
machine translation, and re-evaluates it properly: paragraph-level splits with no train/test
overlap, standard SQuAD metrics, and correctly trained baselines.

**Status:** v0.1. First clean result: XLM-RoBERTa base, 61.7 EM / 73.2 F1 on the leakage-free dev set.

## Why a re-evaluation

Reviewing the original notebooks in 2026 turned up problems that inflated or invalidated the
reported scores (details in [`legacy/README.md`](legacy/README.md)):

- The mT5 run split train/validation by row: about 92% of validation questions came from paragraphs seen in training.
- The "F1" was a custom metric, not SQuAD token F1; the headline 73% was ROUGE-1.
- The XLM-RoBERTa run used a learning rate of 4e-10 and never learned; the only saved checkpoint is English RoBERTa, not XLM-R.

The only result the old code supports is **mT5-small: 63.9% exact match, 0.71 ROUGE-1** on a
2,017-question random-row split. That is an upper bound until it is re-run on clean splits.

## Results

Paragraph-level split, standard SQuAD Exact Match and token F1 with Urdu normalisation.

| Model | Split | Questions | Exact match | F1 |
| --- | --- | --- | --- | --- |
| XLM-RoBERTa base (2 epochs, lr 3e-5, 384 tokens, stride 128) | dev | 3,959 | 61.7 | 73.2 |

Test-set score and an mT5 re-run on the same split are next. For comparison, the 2022 mT5
number (63.9% exact match) was measured on a split where ~92% of evaluation paragraphs were
also in training.

## Dataset

- 40,273 Urdu questions over 14,387 paragraphs (after removing 55 duplicates), from SQuAD v1.1
  via Google Cloud Translation.
- Paragraph-level split (seed 13): train 32,290 / dev 3,959 / test 4,024 questions, 0% overlap.
- Rows whose answer did not survive translation verbatim in the context are dropped, since
  extractive QA needs an answer span.
- Data files are not in the repository; see [`data/README.md`](data/README.md).

## Quick start

```bash
pip install -e ".[dev]"

python -m urduqa.prepare "data/raw/URDU TRANSLATED DATASET.csv" data/processed/all.jsonl
python -m urduqa.split   data/processed/all.jsonl data/processed --seed 13
python -m urduqa.evaluate data/processed/test.jsonl outputs/<model>/predictions.json

pytest
```

**Train on a free GPU:** open [`notebooks/train_xlmr_colab.ipynb`](notebooks/train_xlmr_colab.ipynb)
in [Colab](https://colab.research.google.com/github/zarekasif/urdu-qa/blob/main/notebooks/train_xlmr_colab.ipynb).

| Module | Purpose |
| --- | --- |
| `urduqa.prepare` | CSV to SQuAD-style JSONL: HTML-entity repair, answer-span location, dedup, drop reasons counted |
| `urduqa.split` | 80/10/10 split by paragraph, with a leakage check |
| `urduqa.metrics` | SQuAD Exact Match and token F1 with Urdu normalisation (Arabic/Urdu letter variants, Urdu digits, diacritics, `۔ ، ؟`) |
| `urduqa.evaluate` | Score a `{id: answer}` predictions file; missing predictions count as wrong |
| `urduqa.qa` | Sliding-window features and best-span answer extraction for any HF fast tokenizer |
| `urduqa.train_extractive` | Fine-tune and evaluate an extractive model (default XLM-RoBERTa base) |
| `urduqa.predict` | Score a saved model on a split without retraining (for the one-off test run) |
| `legacy/` | The original 2022 notebooks, outputs stripped |

## Roadmap

1. ~~XLM-RoBERTa base on the paragraph-level split~~ (done, dev); report test once; re-run mT5 on the same split.
2. Zero-shot and few-shot baselines with current open multilingual LLMs.
3. Error analysis of translation artefacts: answer spans lost or shifted by translation.
4. Publish the cleaned splits on the Hugging Face Hub and a small demo.

## Credits

Original project (Forman Christian College, Lahore, 2022): **Salman Anwar, Zarek Asif, Eraj Khurshid**.
Advisors: Ali Faheem, Akheem Yousaf. Re-evaluation and maintenance: Zarek Asif.

Built on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016), CC BY-SA 4.0.

## Licence

Code: [MIT](LICENSE). Data derived from SQuAD v1.1 follows its CC BY-SA 4.0 licence.
