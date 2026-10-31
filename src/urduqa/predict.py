"""Run a saved extractive QA model on a split and score it, without retraining.

Use this to report the held-out test score once, after choosing settings on dev.

Usage:
    python -m urduqa.predict --model-dir outputs/xlmr-base/model \
        --data-dir data/processed --split test --out-dir outputs/xlmr-base/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate
from .qa import postprocess_predictions, prepare_eval_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    from .train_extractive import _supported

    raw = load_dataset("json", data_files={args.split: str(args.data_dir / f"{args.split}.jsonl")})[args.split]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    feats = raw.map(
        prepare_eval_features,
        batched=True,
        remove_columns=raw.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "stride": args.stride},
    )
    model = AutoModelForQuestionAnswering.from_pretrained(str(args.model_dir))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            **_supported(
                TrainingArguments,
                output_dir=str(args.out_dir / "tmp"),
                per_device_eval_batch_size=args.batch_size,
                fp16=torch.cuda.is_available(),
                report_to="none",
            )
        ),
        data_collator=default_data_collator,
    )
    start_logits, end_logits = trainer.predict(feats.remove_columns(["example_id", "offset_mapping"])).predictions[:2]
    predictions = postprocess_predictions(
        {"id": raw["id"], "context": raw["context"]},
        {"example_id": feats["example_id"], "offset_mapping": feats["offset_mapping"]},
        start_logits,
        end_logits,
    )
    metrics = evaluate({ex["id"]: ex["answers"]["text"] for ex in raw}, predictions)
    metrics.update({"model_dir": str(args.model_dir), "split": args.split})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
