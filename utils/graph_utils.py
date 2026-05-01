"""
utils/graph_utils.py
====================
Utility functions for converting raw graph dicts (from graph_data.json)
into PyTorch Geometric Data objects.

Used by: scripts/train.py, app/demo.py
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data, Batch


def dict_to_pyg(graph_dict: dict) -> Data:
    """
    Convert a graph dict from graph_data.json to a PyG Data object.

    Expected dict format:
      {
        "nodes": [int, ...],
        "edges": [[src, dst], ...],
        "num_nodes": int,
        "num_edges": int
      }
    """
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    x = torch.tensor(nodes, dtype=torch.long)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))


def sequence_to_pyg_list(seq_record: dict, use_after: bool = True) -> list[Data]:
    """
    Convert a sequence record (from train_sequences.json) to a list of PyG Data.

    Parameters
    ----------
    seq_record : dict    one sequence record with "input" list
    use_after  : bool    if True, use graph_after; else graph_before

    Returns
    -------
    list[Data]  length = window size
    """
    key    = "graph_after" if use_after else "graph_before"
    graphs = []
    for item in seq_record.get("input", []):
        g = dict_to_pyg(item.get(key, {"nodes": [], "edges": []}))
        graphs.append(g)
    return graphs


def graphs_to_batch(graphs: list[Data]) -> Batch:
    """Collate a list of PyG Data objects into a single Batch."""
    return Batch.from_data_list(graphs)


def pad_graph(graph: Data, target_nodes: int) -> Data:
    """
    Pad a graph to have exactly `target_nodes` nodes (with zeros).
    Used for fixed-size input if needed.
    """
    n = graph.num_nodes
    if n >= target_nodes:
        return graph
    pad = target_nodes - n
    x   = torch.cat([graph.x, torch.zeros(pad, dtype=torch.long)])
    return Data(x=x, edge_index=graph.edge_index, num_nodes=target_nodes)
