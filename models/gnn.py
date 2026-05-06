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
from transformers import AutoModel

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
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
        # -- Semantic Backbone (CodeBERT) --------------------
        print("  Loading CodeBERT backbone...")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        for param in self.codebert.parameters():
            param.requires_grad = False  # Keep it frozen for speed/memory
            
        self.semantic_proj = nn.Linear(768, embed_dim)

        # -- Structural Encoder (GNN) ------------------------
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1      = GCNConv(embed_dim,  hidden_dim)
        self.conv2      = GCNConv(hidden_dim, hidden_dim)
        self.conv3      = GCNConv(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        
        self.proj       = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x:           torch.Tensor,   # [N] long - node type IDs
        edge_index:  torch.Tensor,   # [2, E] long - edges
        batch:       torch.Tensor,   # [N] long - batch assignment
        node_tokens: torch.Tensor = None, # [N, 32]
        node_mask:   torch.Tensor = None, # [N, 32]
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor  shape [B, hidden_dim]
        """
        # 1. Structural features
        h_struct = self.embedding(x)              # [N, embed_dim]
        
        # 2. Semantic features (CodeBERT)
        if node_tokens is not None:
            # SAFETY: If there are too many nodes, we must process in chunks to avoid 14GB OOM
            MAX_SEM_NODES = 1000 # Safety cap for CodeBERT
            num_nodes = node_tokens.size(0)
            
            if num_nodes > MAX_SEM_NODES:
                # Only take the first MAX_SEM_NODES to stay within 6GB VRAM
                node_tokens = node_tokens[:MAX_SEM_NODES]
                node_mask   = node_mask[:MAX_SEM_NODES]
                # We'll need to pad h_sem back to the original N later
            
            with torch.no_grad():
                # Process in small mini-chunks of 128 nodes to stay safe
                chunk_size = 128
                sem_chunks = []
                for i in range(0, node_tokens.size(0), chunk_size):
                    tok_chunk = node_tokens[i:i+chunk_size]
                    mask_chunk = node_mask[i:i+chunk_size]
                    outputs = self.codebert(input_ids=tok_chunk, attention_mask=mask_chunk)
                    sem_chunks.append(outputs.last_hidden_state[:, 0, :]) # [chunk, 768]
                
                h_sem_raw = torch.cat(sem_chunks, dim=0) # [min(N, 1000), 768]
            
            h_sem = self.semantic_proj(h_sem_raw)     # [min(N, 1000), embed_dim]
            
            # If we capped, pad with zeros for the remaining nodes
            if num_nodes > MAX_SEM_NODES:
                padding = torch.zeros((num_nodes - MAX_SEM_NODES, h_sem.size(1)), device=h_sem.device)
                h_sem = torch.cat([h_sem, padding], dim=0)
                
            h0 = h_struct + h_sem                  # Fuse
        else:
            h0 = h_struct

        h1 = self.conv1(h0, edge_index)     # [N, hidden_dim]
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.conv2(h1, edge_index)     # [N, hidden_dim]
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1                        # Residual connection

        h3 = self.conv3(h2, edge_index)     # [N, hidden_dim]
        h3 = F.relu(h3)
        h3 = self.dropout(h3)
        h3 = h3 + h2                        # Residual connection

        h_mean = global_mean_pool(h3, batch) # [B, hidden_dim]
        h_max  = global_max_pool(h3, batch)  # [B, hidden_dim]
        
        h_pool = torch.cat([h_mean, h_max], dim=-1) # [B, hidden_dim * 2]
        out    = self.proj(h_pool)                  # [B, hidden_dim]
        
        return out


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
