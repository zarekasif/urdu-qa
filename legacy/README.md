# Legacy: BSc final-year project notebooks (2022)

The original notebooks from the Urdu Machine Reading Comprehension final-year project
(Forman Christian College, Lahore, 2022), kept for reference. Outputs are stripped and
paths still point to the original Google Drive / local files, so they will not run as-is.
The maintained pipeline lives in [`../src`](../src).

| Notebook | What it does |
| --- | --- |
| `mt5_qa.ipynb` | Fine-tunes `google/mt5-small` as a generative QA model (PyTorch Lightning) |
| `xlm_roberta_qa.ipynb` | Fine-tunes XLM-RoBERTa for extractive QA (simpletransformers) |
| `sliding_window.ipynb` | Sliding-window + spaCy similarity baseline |
| `cosine_similarity_urdu.ipynb` | Sentence-level cosine-similarity baseline; also the answer-in-context filter |
| `cosine_similarity_english.ipynb` | Same baseline on English SQuAD, for comparison |
| `semantic_similarity_urdu.ipynb` | Sentence-embedding (MiniLM) retrieval baseline |

## Known issues found on review (2026)

These are why the project is being re-run in `src/`:

1. **Train/validation leakage.** `mt5_qa.ipynb` splits rows at random (95/5). The data has
   ~2.8 questions per paragraph, so most validation paragraphs also appear in training.
2. **Non-standard metrics.** The custom "F1" only scores a prediction when the gold answer is
   a substring of it, and is not SQuAD token-overlap F1. The 73% figure in the 2022 report
   matches ROUGE-1 from an earlier run, not F1.
3. **XLM-RoBERTa did not train.** The final run uses `learning_rate = 4e-10`; training loss
   stays flat (~4.95) and evaluation gets 5 of 6,049 answers right.
4. **Answer selection bug.** XLM-R predictions use `max(res)` over the n-best list, which picks
   the alphabetically last string rather than the highest-scoring answer.

## Results the notebooks actually support

| Model | Eval set | Exact match | ROUGE-1 F |
| --- | --- | --- | --- |
| mT5-small | 2,017 random-row held-out questions (leaky split) | 63.9% (1,288 / 2,017) | 0.71 |

Treat this as an upper bound until it is re-run on a paragraph-level split.
