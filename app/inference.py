"""
app/inference.py
================
Standalone inference helper - load model and predict on raw code strings.

Usage:
  from app.inference import Predictor
  p = Predictor("outputs/checkpoints/best_model.pt")
  result = p.predict_sequence([code1, code2, code3])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multitask_model import CodeEvolutionModel, CHANGE_LABELS
from scripts.build_graphs   import ast_to_graph
from utils.graph_utils      import dict_to_pyg
from torch_geometric.data   import Batch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, checkpoint: str, vocab_size: int | None = None):
        if vocab_size is None:
            vp = ROOT / "data" / "processed" / "vocab.json"
            vocab_size = (len(json.load(open(vp))) + 10) if vp.exists() else 512

        self.model = CodeEvolutionModel(vocab_size=vocab_size).to(DEVICE)
        self.model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        self.model.eval()

    def predict_sequence(self, code_list: list[str]) -> dict:
        """
        Parameters
        ----------
        code_list : list of Python source strings (length = window)

        Returns
        -------
        dict with change_label, change_probs, bug_prob
        """
        graphs = [ast_to_graph(c) for c in code_list]
        if any(g is None for g in graphs):
            raise ValueError("One or more code strings could not be parsed.")

        embs = []
        for g in graphs:
            data  = dict_to_pyg(g).to(DEVICE)
            batch = Batch.from_data_list([data])
            with torch.no_grad():
                e = self.model.gnn(batch.x, batch.edge_index, batch.batch)
            embs.append(e)

        seq = torch.stack(embs, dim=1)   # [1, W, D]
        with torch.no_grad():
            c_log, b_log = self.model(seq)

        probs     = F.softmax(c_log, dim=-1).squeeze().tolist()
        bug_prob  = torch.sigmoid(b_log).item()
        top_idx   = int(torch.tensor(probs).argmax())

        return {
            "change_label":  CHANGE_LABELS[top_idx],
            "change_probs":  {l: round(p, 4) for l, p in zip(CHANGE_LABELS, probs)},
            "bug_prob":      round(bug_prob, 4),
        }
