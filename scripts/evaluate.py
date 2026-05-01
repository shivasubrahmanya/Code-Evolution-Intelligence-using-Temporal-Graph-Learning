"""
scripts/evaluate.py
===================
Phase 7 - Evaluation: Accuracy, Top-3, F1, Confusion Matrix, Baselines.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.multitask_model import CodeEvolutionModel, CHANGE_LABELS
from utils.data_utils        import CodeSequenceDataset
from scripts.train           import encode_sequence_batch, custom_collate, DEVICE

# Project root - works regardless of CWD
ROOT = Path(__file__).resolve().parent.parent


def compute_change_metrics(logits, labels):
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    acc   = (preds == labels).float().mean().item()
    top3  = probs.topk(min(3, probs.size(-1)), dim=-1).indices
    top3_acc = sum(
        labels[i].item() in top3[i].tolist() for i in range(len(labels))
    ) / len(labels)
    n = probs.size(-1)
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t][p] += 1
    return {"accuracy": acc, "top3_accuracy": top3_acc, "confusion_matrix": cm.tolist()}


def compute_bug_metrics(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).float()
    labels = labels.float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    tn = ((1 - preds) * (1 - labels)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def baseline_random(labels, n_classes=3):
    import random
    preds = [random.randint(0, n_classes - 1) for _ in labels]
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def baseline_majority(labels):
    m = Counter(labels).most_common(1)[0][0]
    return sum(l == m for l in labels) / len(labels)


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    cl, la, bl, lb = [], [], [], []
    for batch in loader:
        seq = encode_sequence_batch(batch["raw_input"], model.gnn)
        c_l, b_l = model(seq)
        cl.append(c_l.cpu()); la.append(batch["change_label"])
        bl.append(b_l.squeeze(-1).cpu()); lb.append(batch["bug_label"])
    return torch.cat(cl), torch.cat(la), torch.cat(bl), torch.cat(lb)


def run(args):
    print(f"\n{'='*55}\n  Evaluation  -  {args.checkpoint}\n{'='*55}\n")
    vocab_path = ROOT / "data" / "processed" / "vocab.json"
    vocab_size = (len(json.load(open(vocab_path))) + 10) if vocab_path.exists() else 512

    model = CodeEvolutionModel(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))

    test_ds = CodeSequenceDataset(ROOT / "data" / "sequences" / "test_sequences.json")
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                         collate_fn=custom_collate, num_workers=0)

    c_log, c_lab, b_log, b_lab = run_inference(model, loader)

    cm = compute_change_metrics(c_log, c_lab)
    bm = compute_bug_metrics(b_log, b_lab)
    label_list = c_lab.tolist()

    print(f"  [Change]  acc={cm['accuracy']:.4f}  top3={cm['top3_accuracy']:.4f}")
    print(f"  [Bug]     P={bm['precision']:.3f}  R={bm['recall']:.3f}  F1={bm['f1']:.3f}")
    print(f"  [Random]  {baseline_random(label_list):.4f}")
    print(f"  [Majority]{baseline_majority(label_list):.4f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval_results.json", "w") as f:
        json.dump({"change": cm, "bug": bm}, f, indent=2)
    print(f"\n  [OK]  Results -> {out}/eval_results.json\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs" / "checkpoints" / "best_model.pt"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
