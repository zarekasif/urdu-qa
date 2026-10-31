"""Fine-tune an extractive QA model (default: XLM-RoBERTa base) and evaluate it.

Reads the paragraph-level splits written by ``urduqa.split``, trains on ``train.jsonl``,
predicts on the chosen evaluation split, and writes ``predictions.json`` and
``metrics.json`` (SQuAD EM/F1 from ``urduqa.metrics``) to the output directory.

Usage (GPU recommended, e.g. Colab):
    python -m urduqa.train_extractive --data-dir data/processed --out-dir outputs/xlmr

Quick pipeline check on a few hundred examples:
    python -m urduqa.train_extractive --data-dir data/processed --out-dir outputs/smoke \
        --max-train-samples 200 --max-eval-samples 100 --epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate
from .qa import postprocess_predictions, prepare_eval_features, prepare_train_features


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eval-split", default="dev", choices=["dev", "test"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-eval-samples", type=int)
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    raw = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            args.eval_split: str(args.data_dir / f"{args.eval_split}.jsonl"),
        },
    )
    train_raw, eval_raw = raw["train"], raw[args.eval_split]
    if args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_raw))))
    if args.max_eval_samples:
        eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    fn_kwargs = {"tokenizer": tokenizer, "max_length": args.max_length, "stride": args.stride}
    train_feats = train_raw.map(
        prepare_train_features, batched=True, remove_columns=train_raw.column_names, fn_kwargs=fn_kwargs
    )
    eval_feats = eval_raw.map(
        prepare_eval_features, batched=True, remove_columns=eval_raw.column_names, fn_kwargs=fn_kwargs
    )
    print(f"train: {len(train_raw)} questions -> {len(train_feats)} windows")
    print(f"{args.eval_split}: {len(eval_raw)} questions -> {len(eval_feats)} windows")

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_feats,
        data_collator=default_data_collator,
    )
    trainer.train()

    model_dir = args.out_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    pred_input = eval_feats.remove_columns(["example_id", "offset_mapping"])
    start_logits, end_logits = trainer.predict(pred_input).predictions[:2]
    predictions = postprocess_predictions(
        {"id": eval_raw["id"], "context": eval_raw["context"]},
        {"example_id": eval_feats["example_id"], "offset_mapping": eval_feats["offset_mapping"]},
        start_logits,
        end_logits,
    )

    golds = {ex["id"]: ex["answers"]["text"] for ex in eval_raw}
    metrics = evaluate(golds, predictions)
    metrics.update({"model": args.model, "split": args.eval_split, "epochs": args.epochs, "lr": args.lr})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
