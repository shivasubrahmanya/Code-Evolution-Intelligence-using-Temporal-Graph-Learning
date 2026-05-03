"""
scripts/train.py
================
Phase 5 - Training Pipeline

Full flow:
  1. Load sequence datasets (train / val)
  2. For each batch:
       a. Convert raw graph dicts -> PyG Data
       b. Encode each graph with GNN -> embedding
       c. Stack into [B, W, D] sequence
       d. Pass through Transformer -> context
       e. Predict change label + bug flag
       f. Compute joint loss and backpropagate
  3. Validate every epoch
  4. Save best checkpoint

Hyperparameters (defaults):
  LR         : 1e-3
  Batch size : 16
  Epochs     : 20
  Window     : 3

Usage:
  python scripts/train.py
  python scripts/train.py --epochs 30 --batch-size 32 --lr 5e-4
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch

from models.multitask_model import CodeEvolutionModel, CHANGE_LABELS
from utils.data_utils        import CodeSequenceDataset, BinaryCodeSequenceDataset, load_label_weights
from utils.graph_utils       import dict_to_pyg

# Project root - works regardless of CWD
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------
#  Helpers
# ---------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_sequence_batch(
    batch_inputs: list[list[dict | Data]],    # [B, W] list of graph dicts OR Data objects
    gnn: nn.Module,
    use_after: bool = True,
) -> torch.Tensor:
    """
    Convert a batch of graph sequences -> tensor [B, W, D].
    Supports both raw dicts (slow) and pre-converted Data objects (fast).
    """
    B   = len(batch_inputs)
    W   = len(batch_inputs[0])

    all_embeddings = []

    for w in range(W):
        # Collect all graphs at position w across the batch
        graphs = []
        for b in range(B):
            item = batch_inputs[b][w]
            if isinstance(item, dict):
                key = "graph_after" if use_after else "graph_before"
                graphs.append(dict_to_pyg(item.get(key, {"nodes": [], "edges": []})))
            else:
                graphs.append(item)
        
        pyg_batch = Batch.from_data_list(graphs).to(DEVICE)

        with torch.no_grad() if not gnn.training else torch.enable_grad():
            emb = gnn(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)  # [B, D]

        all_embeddings.append(emb)   # W × [B, D]

    return torch.stack(all_embeddings, dim=1)   # [B, W, D]


def custom_collate(batch: list[dict]) -> dict:
    """Custom collate that handles both 'raw_input' and 'input' keys."""
    # Use 'input' if available (binary), else 'raw_input' (json)
    input_key = "input" if "input" in batch[0] else "raw_input"
    return {
        "input":        [item[input_key]      for item in batch],
        "change_label": torch.stack([item["change_label"] for item in batch]),
        "bug_label":    torch.stack([item["bug_label"]     for item in batch]),
    }


# ---------------------------------------------
#  Metrics helpers
# ---------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def f1_binary(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
    labels = labels.float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


# ---------------------------------------------
#  Training loop
# ---------------------------------------------

def train_epoch(
    model:      CodeEvolutionModel,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    loss_change: nn.Module,
    loss_bug:    nn.Module,
) -> dict:

    model.train()
    total_loss   = 0.0
    total_change = 0.0
    total_bug    = 0.0
    total_acc    = 0.0
    n_batches    = 0

    for batch in loader:
        input_data    = batch["input"]
        change_labels = batch["change_label"].to(DEVICE)
        bug_labels    = batch["bug_label"].to(DEVICE)

        seq_emb = encode_sequence_batch(input_data, model.gnn)  # [B, W, D]

        change_logits, bug_logit = model(seq_emb)

        l_change = loss_change(change_logits, change_labels)
        l_bug    = loss_bug(bug_logit.squeeze(-1), bug_labels)
        loss     = l_change + l_bug

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_change += l_change.item()
        total_bug    += l_bug.item()
        total_acc    += accuracy(change_logits, change_labels)
        n_batches    += 1

    return {
        "loss":         total_loss   / n_batches,
        "loss_change":  total_change / n_batches,
        "loss_bug":     total_bug    / n_batches,
        "change_acc":   total_acc    / n_batches,
    }


@torch.no_grad()
def eval_epoch(
    model:       CodeEvolutionModel,
    loader:      DataLoader,
    loss_change: nn.Module,
    loss_bug:    nn.Module,
) -> dict:

    model.eval()
    total_loss   = 0.0
    total_acc    = 0.0
    total_f1     = 0.0
    n_batches    = 0

    for batch in loader:
        input_data    = batch["input"]
        change_labels = batch["change_label"].to(DEVICE)
        bug_labels    = batch["bug_label"].to(DEVICE)

        seq_emb = encode_sequence_batch(input_data, model.gnn)

        change_logits, bug_logit = model(seq_emb)

        l_change = loss_change(change_logits, change_labels)
        l_bug    = loss_bug(bug_logit.squeeze(-1), bug_labels)

        total_loss += (l_change + l_bug).item()
        total_acc  += accuracy(change_logits, change_labels)
        total_f1   += f1_binary(bug_logit, bug_labels)
        n_batches  += 1

    return {
        "loss":       total_loss / n_batches,
        "change_acc": total_acc  / n_batches,
        "bug_f1":     total_f1   / n_batches,
    }


# ---------------------------------------------
#  Main
# ---------------------------------------------

def run(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Code Evolution Training")
    print(f"  Device      : {DEVICE}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"{'='*55}\n")

    # -- Load vocab ------------------------------------
    vocab_path = ROOT / "data" / "processed" / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab) + 10   # +10 buffer for unseen types
    else:
        vocab_size = 512               # safe default
    print(f"  Vocab size  : {vocab_size}")

    # -- Datasets --------------------------------------
    train_json = ROOT / "data" / "sequences" / "train_sequences.json"
    val_json   = ROOT / "data" / "sequences" / "val_sequences.json"
    train_pt   = ROOT / "data" / "sequences" / "train_sequences.pt"
    val_pt     = ROOT / "data" / "sequences" / "val_sequences.pt"

    if train_pt.exists() and not args.force_json:
        train_ds = BinaryCodeSequenceDataset(train_pt)
        val_ds   = BinaryCodeSequenceDataset(val_pt)
        use_binary = True
    else:
        print("  [WARN] Using slow JSON streaming. Run scripts/binarize_data.py for 10x speed.")
        train_ds = CodeSequenceDataset(train_json)
        val_ds   = CodeSequenceDataset(val_json)
        use_binary = False

    # -- Sampler for balancing -------------------------
    sampler = None
    if use_binary:
        print("  Creating WeightedRandomSampler for class balancing...")
        # Access the raw labels from the binary data
        labels = torch.stack([item["change_label"] for item in train_ds.data])
        class_counts = torch.bincount(labels)
        
        # Weight is inverse frequency
        class_weights = 1. / (class_counts.float() + 1e-6)
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        
        print(f"  Class counts: {class_counts.tolist()}")
        print(f"  Class weights: {[round(w, 4) for w in class_weights.tolist()]}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=(sampler is None and use_binary),
        collate_fn=custom_collate, 
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate, num_workers=0,
    )
    print(f"  Train batches: {len(train_loader)},  Val batches: {len(val_loader)}\n")

    # -- Model -----------------------------------------
    model = CodeEvolutionModel(
        vocab_size    = vocab_size,
        hidden_dim    = args.hidden_dim,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_tf_layers = args.num_layers,
        dropout       = args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    # -- Loss functions --------------------------------
    if use_binary:
        # Binary dataset allows fast inverse-frequency calculation
        bug_labels = torch.stack([item["bug_label"] for item in train_ds.data])
        num_pos = bug_labels.sum().item()
        num_neg = len(bug_labels) - num_pos
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)]).to(DEVICE)
        print(f"  Calculated bug pos_weight: {pos_weight.item():.4f}")
        
        import math
        prior_prob = num_pos / len(bug_labels)
        init_bias = math.log(prior_prob / (1 - prior_prob + 1e-6))
        model.bug_head[-1].bias.data.fill_(init_bias)
        print(f"  Initialized bug head bias to: {init_bias:.4f}")

        
        # Calculate change label weights from binary data
        change_labels = torch.stack([item["change_label"] for item in train_ds.data])
        counts = torch.bincount(change_labels, minlength=3)
        total = len(change_labels)
        class_weights = (total / (3.0 * counts + 1e-6)).to(DEVICE)
    else:
        pos_weight = torch.tensor([3.0]).to(DEVICE) # Fallback
        class_weights = load_label_weights(train_json).to(DEVICE)

    print(f"  Label weights: {class_weights.tolist()}")

    loss_change = nn.CrossEntropyLoss(weight=class_weights)
    loss_bug    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- Optimizer & Scheduler -------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop --------------------------------
    history    = []
    best_score = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, loss_change, loss_bug)
        va = eval_epoch(model,  val_loader,              loss_change, loss_bug)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={tr['loss']:.4f} acc={tr['change_acc']:.3f} | "
            f"val loss={va['loss']:.4f} acc={va['change_acc']:.3f} "
            f"bug_f1={va['bug_f1']:.3f} | {elapsed:.1f}s"
        )

        record = {"epoch": epoch, "train": tr, "val": va}
        history.append(record)

        # Save best model based on a balance of Change Acc and Bug F1
        score = va["change_acc"] + va["bug_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  [NEW BEST] Score: {score:.4f}")

    # -- Save checkpoint & history ---------------------
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best epoch  : {best_epoch}  (score={best_score:.4f})")
    print(f"  Saved to    : {out_dir}")
    print(f"\n  [OK]  Training complete.\n")


# ---------------------------------------------
#  CLI
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the Code Evolution model.")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--hidden-dim",  type=int,   default=256)
    p.add_argument("--embed-dim",   type=int,   default=256)
    p.add_argument("--num-heads",   type=int,   default=8)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--output-dir",  default=str(ROOT / "outputs" / "checkpoints"))
    p.add_argument("--force-json",  action="store_true", help="Force slow JSON loading")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
