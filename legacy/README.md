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

Checked against the full project folder (notebooks, data files and the saved checkpoint):

1. **mT5: train/validation leakage.** `mt5_qa.ipynb` splits the full CSV at random by row (95/5).
   With 2.8 questions per paragraph, about 92% of validation questions come from paragraphs
   that are also in training (measured by repeating the split). The other split files in the
   project (`TRAIN/VALIDATION/TEST.json`, 70/15/15, and `TRAIN90/VALIDATION5/TEST5.json`) are
   split by paragraph and are clean; the mT5 run did not use them.
2. **Non-standard metrics.** The custom "F1" only scores a prediction when the gold answer is a
   substring of it, and is not SQuAD token-overlap F1. The 73% "F1" in the 2022 report matches
   ROUGE-1 from an earlier mT5 run.
3. **No working XLM-RoBERTa model.** The final run in `xlm_roberta_qa.ipynb` uses
   `learning_rate = 4e-10`; loss stays flat and evaluation gets 5 of 6,049 answers right. The only
   saved checkpoint (`best_model`) is English `roberta-base` (BPE vocabulary, not XLM-R), trained
   with `max_seq_length = 128`, and its own `eval_results.txt` reports 0 correct of 6,049.
   No saved artefact supports the 36% figure in the report.
4. **Answer selection bug.** Predictions use `max(res)` over the n-best list, which picks the
   alphabetically last string rather than the highest-scoring answer.

## Results the notebooks actually support

| Model | Eval set | Exact match | ROUGE-1 F |
| --- | --- | --- | --- |
| mT5-small | 2,017 random-row held-out questions (leaky split) | 63.9% (1,288 / 2,017) | 0.71 |

Treat this as an upper bound until it is re-run on a paragraph-level split.
