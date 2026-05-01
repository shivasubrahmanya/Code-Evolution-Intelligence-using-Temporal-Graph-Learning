"""
models/multitask_model.py
=========================
Phase 4.3 - Multi-task Model

Combines:
  1. GraphEncoder (GNN) - encodes each graph in a sequence
  2. TemporalTransformer - encodes the sequence of graph embeddings
  3. Two prediction heads:
       - change_head : Linear -> Softmax   (3 classes: ADD/DELETE/MODIFY)
       - bug_head    : Linear -> Sigmoid   (binary: 0/1)

Full forward pass:
  sequence of graphs -> per-graph GNN embeddings
    -> stack into [B, W, D]
    -> Transformer -> context [B, D]
    -> change_head -> [B, 3]  (logits)
    -> bug_head    -> [B, 1]  (logit)

Usage:
  from models.multitask_model import CodeEvolutionModel
  model = CodeEvolutionModel(vocab_size=512)
  change_logits, bug_logit = model(batch_graph_sequences)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.gnn      import GraphEncoder
from models.temporal import TemporalTransformer


# Label maps - shared with training / evaluation
CHANGE_LABELS = ["ADD", "DELETE", "MODIFY"]
CHANGE_TO_IDX = {lbl: i for i, lbl in enumerate(CHANGE_LABELS)}
NUM_CHANGE_CLASSES = len(CHANGE_LABELS)


class CodeEvolutionModel(nn.Module):
    """
    End-to-end multi-task model for code evolution prediction.

    Parameters
    ----------
    vocab_size      : int   Size of AST node-type vocabulary
    hidden_dim      : int   Shared embedding dimension (GNN output & Transformer)
    embed_dim       : int   Initial node embedding size inside GNN
    num_heads       : int   Transformer attention heads
    num_tf_layers   : int   Number of Transformer encoder layers
    dropout         : float Shared dropout rate
    """

    def __init__(
        self,
        vocab_size:    int,
        hidden_dim:    int = 128,
        embed_dim:     int = 64,
        num_heads:     int = 4,
        num_tf_layers: int = 2,
        dropout:       float = 0.3,
    ):
        super().__init__()

        self.gnn = GraphEncoder(
            vocab_size  = vocab_size,
            hidden_dim  = hidden_dim,
            embed_dim   = embed_dim,
            dropout     = dropout,
        )

        self.transformer = TemporalTransformer(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            num_layers  = num_tf_layers,
            dropout     = dropout,
        )

        # -- Prediction heads --------------------------------
        self.change_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_CHANGE_CLASSES),
        )

        self.bug_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_graph(self, graph_batch: Batch) -> torch.Tensor:
        """Encode a PyG Batch of graphs -> [N_graphs, hidden_dim]."""
        return self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

    def forward(
        self,
        graph_seq_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        graph_seq_embeddings : [B, W, D]
            Pre-computed GNN embeddings for a window of W graphs per batch item.
            Compute with `encode_graph` and stack, or use the training loop helper.

        Returns
        -------
        change_logits : [B, 3]   (raw; apply softmax for probabilities)
        bug_logit     : [B, 1]   (raw; apply sigmoid for probability)
        """
        ctx = self.transformer(graph_seq_embeddings)  # [B, D]
        return self.change_head(ctx), self.bug_head(ctx)


# ---------------------------------------------
#  Quick self-test
# ---------------------------------------------

def _test():
    from torch_geometric.data import Data

    vocab_size = 200
    hidden_dim = 128
    batch_size = 4
    window     = 3

    model = CodeEvolutionModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.eval()

    # Simulate pre-computed graph embeddings for a batch
    fake_embeddings = torch.randn(batch_size, window, hidden_dim)

    change_logits, bug_logit = model(fake_embeddings)
    print(f"change_logits shape : {change_logits.shape}")   # [4, 3]
    print(f"bug_logit     shape : {bug_logit.shape}")        # [4, 1]

    assert change_logits.shape == (batch_size, NUM_CHANGE_CLASSES)
    assert bug_logit.shape     == (batch_size, 1)
    print("  [OK]  CodeEvolutionModel test passed.")


if __name__ == "__main__":
    _test()
