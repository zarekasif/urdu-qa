"""Score a predictions file against a gold JSONL split.

Predictions are a JSON object mapping question id to answer string, the same format as
the official SQuAD evaluation script:

    {"3f2a9c1e5b7d-0": "1990 کی دہائی کے آخر میں", ...}

Usage:
    python -m urduqa.evaluate data/processed/test.jsonl outputs/mt5/predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate
from .prepare import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gold", type=Path)
    ap.add_argument("predictions", type=Path)
    args = ap.parse_args()

    golds = {r["id"]: r["answers"]["text"] for r in read_jsonl(args.gold)}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = evaluate(golds, predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
