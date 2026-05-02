"""
models/gnn.py
=============
Phase 4.1 - Graph Neural Network

Architecture:
  Node Embedding (nn.Embedding)
    -> GCNConv
    -> ReLU + Dropout
    -> GCNConv
    -> ReLU + Dropout
    -> Global Mean Pool
    -> Linear projection
    -> Graph Embedding Vector  (hidden_dim)

Usage:
  from models.gnn import GraphEncoder
  model = GraphEncoder(vocab_size=512, hidden_dim=128)
  emb = model(x, edge_index, batch)   # -> [B, 128]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    TG_OK = True
except ImportError:
    TG_OK = False
    raise ImportError(
        "PyTorch Geometric is required for models/gnn.py\n"
        "Install: pip install torch-geometric"
    )


class GraphEncoder(nn.Module):
    """
    Encodes a single graph (set of node types + edges) into a
    fixed-size embedding vector using two GCN layers + mean pooling.

    Parameters
    ----------
    vocab_size  : int   Number of distinct AST node types (from vocab.json)
    hidden_dim  : int   Output embedding dimension  (default 128)
    embed_dim   : int   Initial node embedding size (default 64)
    dropout     : float Dropout rate between GCN layers (default 0.3)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        embed_dim:  int = 64,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1      = GCNConv(embed_dim,  hidden_dim)
        self.conv2      = GCNConv(hidden_dim, hidden_dim)
        self.conv3      = GCNConv(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.proj       = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x:          torch.Tensor,   # [N] long - node type IDs
        edge_index: torch.Tensor,   # [2, E] long - edges
        batch:      torch.Tensor,   # [N] long - batch assignment
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor  shape [B, hidden_dim]
        """
        h = self.embedding(x)               # [N, embed_dim]

        h = self.conv1(h, edge_index)       # [N, hidden_dim]
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)       # [N, hidden_dim]
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv3(h, edge_index)       # [N, hidden_dim]
        h = F.relu(h)
        h = self.dropout(h)

        h = global_mean_pool(h, batch)      # [B, hidden_dim]
        h = self.proj(h)                    # [B, hidden_dim]
        return h


# ---------------------------------------------
#  Quick self-test
# ---------------------------------------------

def _test():
    import torch
    from torch_geometric.data import Data, Batch

    model = GraphEncoder(vocab_size=100, hidden_dim=128)
    model.eval()

    # Fake graph 1: 5 nodes
    g1 = Data(
        x          = torch.randint(0, 100, (5,)),
        edge_index = torch.tensor([[0,1,1,2],[1,2,3,4]], dtype=torch.long),
    )
    # Fake graph 2: 7 nodes
    g2 = Data(
        x          = torch.randint(0, 100, (7,)),
        edge_index = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,6]], dtype=torch.long),
    )

    batch = Batch.from_data_list([g1, g2])
    emb   = model(batch.x, batch.edge_index, batch.batch)
    print(f"GraphEncoder output shape: {emb.shape}")   # -> [2, 128]
    assert emb.shape == (2, 128), f"Unexpected shape {emb.shape}"
    print("  [OK]  GraphEncoder test passed.")


if __name__ == "__main__":
    _test()
