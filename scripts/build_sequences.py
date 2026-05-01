"""
scripts/build_sequences.py
==========================
Phase 3 - Sequence Construction

Reads graph_data.json and builds sliding-window temporal sequences:

  [graph_t-W, ..., graph_t-1]  ->  target: graph_t
  label = change_label of graph_t
  bug   = bug_label_prev_buggy of graph_t

Window size W = 3 by default.

Output
------
data/sequences/train_sequences.json
data/sequences/val_sequences.json
data/sequences/test_sequences.json

Split: 70% train / 15% val / 15% test  (chronological, NOT random)

Format of each sequence record:
{
  "input": [
    {"commit_id": "...", "graph_before": {...}, "graph_after": {...}, ...},
    ...
  ],
  "target": {"commit_id": "...", "graph_before": {...}, "graph_after": {...}, ...},
  "label":  "MODIFY",
  "bug":    0
}

Usage:
  python scripts/build_sequences.py
  python scripts/build_sequences.py --window 3 --input data/processed/graph_data.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

# Project root - works regardless of CWD
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------
#  Config
# ---------------------------------------------

DEFAULT_INPUT  = ROOT / "data" / "processed" / "graph_data.json"
DEFAULT_OUTDIR = ROOT / "data" / "sequences"
WINDOW_SIZE    = 3
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
# TEST_FRAC    = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


# ---------------------------------------------
#  Sequence builder
# ---------------------------------------------

def build_sequences(records: list[dict], window: int) -> list[dict]:
    """
    Slide a window of size `window` over the sorted commit list.
    Each sequence has `window` input commits and one target commit.
    """
    sequences = []
    for i in range(window, len(records)):
        context = records[i - window: i]
        target  = records[i]

        sequences.append({
            "input":  context,
            "target": target,
            "label":  target.get("change_label", "MODIFY"),
            "bug":    target.get("bug_label_prev_buggy", 0),
        })
    return sequences


# ---------------------------------------------
#  Dataset split (chronological)
# ---------------------------------------------

def split(sequences: list[dict],
          train_frac: float = TRAIN_FRAC,
          val_frac: float   = VAL_FRAC
          ) -> tuple[list, list, list]:

    n     = len(sequences)
    n_tr  = int(n * train_frac)
    n_val = int(n * val_frac)

    train = sequences[:n_tr]
    val   = sequences[n_tr: n_tr + n_val]
    test  = sequences[n_tr + n_val:]
    return train, val, test


def save(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data):>5} sequences -> {path}")


# ---------------------------------------------
#  Label distribution helper
# ---------------------------------------------

def label_dist(sequences: list[dict]) -> dict:
    from collections import Counter
    return dict(Counter(s["label"] for s in sequences))


# ---------------------------------------------
#  Main
# ---------------------------------------------

def run(input_path: Path, output_dir: Path, window: int) -> None:
    print(f"\n{'='*55}")
    print(f"  Building sequences  (window={window})")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*55}\n")

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"  Loaded {len(records)} graph records")

    if len(records) < window + 1:
        raise ValueError(
            f"Need at least {window+1} records to form a sequence, got {len(records)}."
        )

    sequences = build_sequences(records, window)
    print(f"  Total sequences     : {len(sequences)}")
    print(f"  Label distribution  : {label_dist(sequences)}")

    train, val, test = split(sequences)

    save(train, output_dir / "train_sequences.json")
    save(val,   output_dir / "val_sequences.json")
    save(test,  output_dir / "test_sequences.json")

    print(f"\n  Split  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"\n  [OK]  Sequence construction complete.\n")


# ---------------------------------------------
#  CLI
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build temporal sequences from graph data.")
    p.add_argument("--input",  default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTDIR))
    p.add_argument("--window", type=int, default=WINDOW_SIZE,
                   help="Context window size (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input), Path(args.output), args.window)
