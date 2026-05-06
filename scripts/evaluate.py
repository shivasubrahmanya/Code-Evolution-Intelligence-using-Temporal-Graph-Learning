"""
scripts/evaluate.py
===================
Final Evaluation Script
- Loads the final model
- Evaluates on the TEST set
- Computes Accuracy, F1, and Confusion Matrix
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import sys

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.multitask_model import CodeEvolutionModel, CHANGE_LABELS
from utils.data_utils        import BinaryCodeSequenceDataset

# Project root
ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate(batch: list[dict]) -> dict:
    return {
        "input":        [item["input"]      for item in batch],
        "change_label": torch.stack([item["change_label"] for item in batch]),
        "bug_label":    torch.stack([item["bug_label"]     for item in batch]),
    }

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    
    all_change_preds = []
    all_change_true  = []
    all_bug_preds    = []
    all_bug_true     = []
    
    print(f"  Evaluating on {len(loader)} batches...")
    
    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(f"    Batch {i}/{len(loader)}...")
            
        input_data    = batch["input"]
        change_labels = batch["change_label"].to(DEVICE)
        bug_labels    = batch["bug_label"].to(DEVICE)

        # Helper from train.py logic (re-implemented here for standalone use)
        from scripts.train import encode_sequence_batch
        seq_emb = encode_sequence_batch(input_data, model.gnn)
        
        change_logits, bug_logit = model(seq_emb)
        
        # Change predictions
        preds = change_logits.argmax(dim=-1)
        all_change_preds.extend(preds.cpu().numpy())
        all_change_true.extend(change_labels.cpu().numpy())
        
        # Bug predictions
        bug_probs = torch.sigmoid(bug_logit.squeeze(-1))
        all_bug_preds.extend((bug_probs > 0.5).float().cpu().numpy())
        all_bug_true.extend(bug_labels.cpu().numpy())
        
    return (
        np.array(all_change_true), np.array(all_change_preds),
        np.array(all_bug_true), np.array(all_bug_preds)
    )

def print_metrics(y_true, y_pred, labels, title):
    print(f"\n--- {title} ---")
    acc = (y_true == y_pred).mean()
    print(f"  Overall Accuracy: {acc:.4f}")
    
    # Simple manual Confusion Matrix
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
        
    print("\n  Confusion Matrix:")
    header = "True \\ Pred | " + " | ".join([f"{l:7}" for l in labels])
    print("  " + header)
    print("  " + "-" * len(header))
    for i, label in enumerate(labels):
        row = " | ".join([f"{cm[i, j]:7}" for j in range(n_classes)])
        print(f"  {label:10} | {row}")
    
    # Class-wise F1
    print("\n  Class-wise Performance:")
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"    {label:10}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

def run_eval():
    # 1. Load Model
    model_path = ROOT / "outputs" / "checkpoints" / "final_model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Need vocab size to init model
    vocab_path = ROOT / "data" / "processed" / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 10

    model = CodeEvolutionModel(vocab_size=vocab_size, hidden_dim=256, embed_dim=256, num_heads=8, num_tf_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"  Loaded model from {model_path}")

    # 2. Load Test Data
    test_pt = ROOT / "data" / "sequences" / "test_sequences.pt"
    if not test_pt.exists():
        print(f"Error: Test binary data not found at {test_pt}. Run binarization first.")
        return
        
    test_ds = BinaryCodeSequenceDataset(test_pt)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # 3. Evaluate
    y_change_true, y_change_pred, y_bug_true, y_bug_pred = evaluate(model, test_loader)

    # 4. Report
    print_metrics(y_change_true, y_change_pred, CHANGE_LABELS, "CHANGE DETECTION")
    print_metrics(y_bug_true, y_bug_pred, ["NO_BUG", "BUG"], "BUG DETECTION")

if __name__ == "__main__":
    run_eval()
