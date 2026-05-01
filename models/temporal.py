"""
models/temporal.py
==================
Phase 4.2 - Temporal Transformer

Encodes a sequence of graph embeddings using a Transformer encoder
(with learned positional encoding) and returns a context vector.

Architecture:
  Input: [B, W, D]   - batch of W graph embeddings each of dim D
    -> Positional Encoding
    -> Transformer Encoder (2-4 layers, multi-head attention)
    -> CLS token or mean pool over sequence
    -> Context Vector [B, D]

Usage:
  from models.temporal import TemporalTransformer
  model = TemporalTransformer(embed_dim=128, num_heads=4, num_layers=2)
  ctx   = model(seq)   # seq: [B, W, 128] -> [B, 128]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds position information to each token in a sequence.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over a sequence of graph embedding vectors.

    Parameters
    ----------
    embed_dim   : int   Dimension of each graph embedding (must match GNN output)
    num_heads   : int   Number of attention heads
    num_layers  : int   Number of Transformer encoder layers
    ff_dim      : int   Feed-forward hidden size (default 4x embed_dim)
    dropout     : float Dropout rate
    max_len     : int   Maximum sequence length supported
    """

    def __init__(
        self,
        embed_dim:  int = 128,
        num_heads:  int = 4,
        num_layers: int = 2,
        ff_dim:     int | None = None,
        dropout:    float = 0.1,
        max_len:    int = 512,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4

        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,   # [B, T, D] convention
            norm_first      = True,   # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm       = nn.LayerNorm(embed_dim)
        self.embed_dim  = embed_dim

    def forward(
        self,
        seq: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        seq                  : [B, W, D]  sequence of graph embeddings
        src_key_padding_mask : [B, W] bool  True = ignore (padding)

        Returns
        -------
        context : [B, D]  mean-pooled context vector
        """
        x = self.pos_enc(seq)                                   # [B, W, D]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Mean pool over the sequence dimension
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, W, 1]
            x    = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(dim=1)                                   # [B, D]

        return x


# ---------------------------------------------
#  Quick self-test
# ---------------------------------------------

def _test():
    model = TemporalTransformer(embed_dim=128, num_heads=4, num_layers=2)
    model.eval()

    batch_size = 4
    window     = 3
    seq        = torch.randn(batch_size, window, 128)
    ctx        = model(seq)
    print(f"TemporalTransformer output shape: {ctx.shape}")   # -> [4, 128]
    assert ctx.shape == (batch_size, 128), f"Unexpected shape {ctx.shape}"
    print("  [OK]  TemporalTransformer test passed.")


if __name__ == "__main__":
    _test()
