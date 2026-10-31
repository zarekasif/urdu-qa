# Urdu QA

Extractive question answering for Urdu, a low-resource language with over 100 million speakers.
This repository revisits my 2022 BSc final-year project, an Urdu version of SQuAD built by
machine translation, and re-evaluates it properly: paragraph-level splits with no train/test
overlap, standard SQuAD metrics, and correctly trained baselines.

**Status:** v0.1. Data pipeline, evaluation and the XLM-RoBERTa training script are in place and tested; first clean results are next.

## Why a re-evaluation

Reviewing the original notebooks in 2026 turned up problems that inflated or invalidated the
reported scores (details in [`legacy/README.md`](legacy/README.md)):

- Train and validation were split by row, so most validation paragraphs were also seen in training.
- The "F1" was a custom metric, not SQuAD token F1; the headline 73% was ROUGE-1.
- The XLM-RoBERTa run used a learning rate of 4e-10 and never learned.

The only result the old code supports is **mT5-small: 63.9% exact match, 0.71 ROUGE-1** on a
2,017-question random-row split. That is an upper bound until it is re-run on clean splits.

## Dataset

- About 40k Urdu questions over 14.4k paragraphs, from SQuAD v1.1 via Google Cloud Translation.
- Rows whose answer did not survive translation verbatim in the context are dropped, since
  extractive QA needs an answer span.
- Data files are not in the repository; see [`data/README.md`](data/README.md).

## Quick start

```bash
pip install -e ".[dev]"

python -m urduqa.prepare data/raw/final_translated_dataset.csv data/processed/all.jsonl
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
| `legacy/` | The original 2022 notebooks, outputs stripped |

## Roadmap

1. Re-train XLM-RoBERTa (extractive) and mT5 (generative) on the paragraph-level split; report EM/F1.
2. Zero-shot and few-shot baselines with current open multilingual LLMs.
3. Error analysis of translation artefacts: answer spans lost or shifted by translation.
4. Publish the cleaned splits on the Hugging Face Hub and a small demo.

## Credits

Original project (Forman Christian College, Lahore, 2022): **Salman Anwar, Zarek Asif, Eraj Khurshid**.
Advisors: Ali Faheem, Akheem Yousaf. Re-evaluation and maintenance: Zarek Asif.

Built on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016), CC BY-SA 4.0.

## Licence

Code: [MIT](LICENSE). Data derived from SQuAD v1.1 follows its CC BY-SA 4.0 licence.
