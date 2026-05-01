"""
utils/data_utils.py
===================
Dataset classes for PyTorch DataLoader.

CodeSequenceDataset
-------------------
Wraps train/val/test_sequences.json files.
Each item:
  - graph_seq_embeddings : placeholder (computed inside training loop)
  - change_label : int
  - bug_label    : int
  - raw_input    : list of graph dicts (for on-the-fly GNN encoding)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional, Iterable

try:
    import ijson
except ImportError:
    ijson = None

import torch
from torch.utils.data import Dataset, IterableDataset

from models.multitask_model import CHANGE_TO_IDX


class CodeSequenceDataset(IterableDataset):
    """
    Memory-efficient IterableDataset for temporal commit sequences.
    Uses ijson to stream records from a large JSON array.
    """

    def __init__(
        self,
        json_path: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.json_path = Path(json_path)
        self.transform = transform
        self._length = None

    def __iter__(self) -> Iterable[dict]:
        if not self.json_path.exists():
            return

        with open(self.json_path, "rb") as f:
            if ijson is None:
                # Fallback to standard json if ijson is missing (will likely fail for large files)
                data = json.load(f)
                items = data
            else:
                items = ijson.items(f, "item")

            for rec in items:
                change_label_str = rec.get("label", "MODIFY")
                change_label_int = CHANGE_TO_IDX.get(change_label_str, 2)
                bug_label        = int(rec.get("bug", 0))

                item = {
                    "raw_input":    rec.get("input", []),
                    "raw_target":   rec.get("target", {}),
                    "change_label": torch.tensor(change_label_int, dtype=torch.long),
                    "bug_label":    torch.tensor(bug_label,         dtype=torch.float),
                }

                if self.transform:
                    item = self.transform(item)

                yield item

    def __len__(self) -> int:
        """
        Returns the number of records. 
        Note: The first call will scan the file to count if length isn't cached.
        """
        if self._length is not None:
            return self._length
        
        if not self.json_path.exists():
            return 0

        print(f"  Counting records in {self.json_path.name}...")
        count = 0
        with open(self.json_path, "rb") as f:
            if ijson is None:
                self._length = len(json.load(f))
            else:
                for _ in ijson.items(f, "item"):
                    count += 1
                    if count % 10000 == 0:
                        sys.stdout.write(f"\r  Counted {count:,} items...")
                        sys.stdout.flush()
                self._length = count
                print(f"\r  Total: {self._length:,} items.        ")
        
        return self._length


def load_label_weights(train_json: str | Path) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for change labels using streaming.
    """
    from collections import Counter
    print(f"  Calculating label weights for {Path(train_json).name}...")
    
    counts = Counter()
    total = 0
    
    with open(train_json, "rb") as f:
        if ijson is None:
            data = json.load(f)
            counts = Counter(CHANGE_TO_IDX.get(r.get("label", "MODIFY"), 2) for r in data)
            total = len(data)
        else:
            for i, r in enumerate(ijson.items(f, "item")):
                label_idx = CHANGE_TO_IDX.get(r.get("label", "MODIFY"), 2)
                counts[label_idx] += 1
                total += 1
                if i > 0 and i % 10000 == 0:
                    sys.stdout.write(f"\r  Scanning for weights: {i:,} items...")
                    sys.stdout.flush()
            print(f"\r  Weight scan complete. {total:,} items.        ")

    weights = torch.zeros(3)
    for cls in range(3):
        cnt = counts.get(cls, 0)
        if cnt > 0:
            weights[cls] = total / (3 * cnt)
        else:
            weights[cls] = 1.0  # fallback
            
    return weights
