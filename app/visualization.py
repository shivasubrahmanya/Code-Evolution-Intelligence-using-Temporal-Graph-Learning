"""
app/visualization.py
====================
Visualization helpers for the demo app and notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path


def plot_training_curves(history_path: str = "outputs/checkpoints/train_history.json"):
    """Plot train/val loss and accuracy curves using matplotlib."""
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss   = [h["val"]["loss"]   for h in history]
    train_acc  = [h["train"]["change_acc"] for h in history]
    val_acc    = [h["val"]["change_acc"]   for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_loss, label="Train", color="#667eea")
    ax1.plot(epochs, val_loss,   label="Val",   color="#ef473a", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train", color="#38ef7d")
    ax2.plot(epochs, val_acc,   label="Val",   color="#ffd200", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Change Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], title: str = "Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    plt.tight_layout()
    plt.show()


def graph_size_summary(graph_data_path: str = "data/processed/graph_data.json"):
    """Print and plot graph size statistics."""
    import matplotlib.pyplot as plt

    with open(graph_data_path) as f:
        data = json.load(f)

    before_n = [d["graph_before"]["num_nodes"] for d in data]
    after_n  = [d["graph_after"]["num_nodes"]  for d in data]
    deltas   = [a - b for a, b in zip(after_n, before_n)]

    print(f"  Avg nodes before : {sum(before_n)/len(before_n):.1f}")
    print(f"  Avg nodes after  : {sum(after_n)/len(after_n):.1f}")
    print(f"  Avg delta        : {sum(deltas)/len(deltas):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(before_n, bins=40, alpha=0.6, label="Before", color="#667eea")
    axes[0].hist(after_n,  bins=40, alpha=0.6, label="After",  color="#38ef7d")
    axes[0].set_title("Node Count Distribution"); axes[0].legend()
    axes[1].hist(deltas, bins=40, color="#ffd200")
    axes[1].set_title("Node Count Delta (after - before)")
    plt.tight_layout(); plt.show()
