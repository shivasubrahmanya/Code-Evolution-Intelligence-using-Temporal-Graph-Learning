"""
scripts/binarize_data.py
========================
Converts JSON sequence files into binary .pt files for faster training.
Uses ijson for memory-efficient streaming.
"""

import os
import torch
from pathlib import Path
import sys
import ijson

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.graph_utils import dict_to_pyg
from models.multitask_model import CHANGE_TO_IDX

def binarize(json_path: Path, output_path: Path):
    print(f"Streaming {json_path}...")
    
    binarized = []
    bug_count = 0
    total = 0
    
    # Pre-count for progress tracking (optional but helpful)
    # Using a fast way to count items if possible
    
    with open(json_path, 'rb') as f:
        items = ijson.items(f, 'item')
        
        for i, rec in enumerate(items):
            if i % 100 == 0:
                sys.stdout.write(f"\r  Processed {i} items...")
                sys.stdout.flush()

            # 1. Convert sequence of graphs
            input_graphs = []
            for step in rec.get("input", []):
                # Use graph_after
                g = dict_to_pyg(step.get("graph_after", {"nodes": [], "edges": []}))
                input_graphs.append(g)
                
            # 2. Convert labels
            change_label_str = rec.get("label", "MODIFY")
            change_label_int = CHANGE_TO_IDX.get(change_label_str, 2)
            bug_label        = float(rec.get("bug", 0))
            
            if bug_label > 0:
                bug_count += 1

            binarized.append({
                "input": input_graphs,
                "change_label": torch.tensor(change_label_int, dtype=torch.long),
                "bug_label": torch.tensor(bug_label, dtype=torch.float)
            })
            total += 1

    print(f"\nSaving {total} items to {output_path}...")
    torch.save(binarized, output_path)
    
    pos_weight = (total - bug_count) / (bug_count + 1e-6)
    print(f"Done. Bug ratio: {bug_count}/{total} ({bug_count/(total+1e-6):.2%})")
    print(f"Suggested pos_weight: {pos_weight:.4f}")
    return pos_weight

if __name__ == "__main__":
    data_dir = ROOT / "data" / "sequences"
    
    # Process Val first (it's smaller, 4GB)
    val_json = data_dir / "val_sequences.json"
    val_pt   = data_dir / "val_sequences.pt"
    if val_json.exists():
        binarize(val_json, val_pt)

    # Process Train (25GB)
    train_json = data_dir / "train_sequences.json"
    train_pt   = data_dir / "train_sequences.pt"
    if train_json.exists():
        binarize(train_json, train_pt)
