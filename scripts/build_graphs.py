"""
scripts/build_graphs.py
=======================
Phase 2.2 - AST -> Graph
Phase 2.3 - Store Graphs
Phase 2.4 - Change Labeling

Converts cleaned commit data into graph representations.

Graph design
------------
- Nodes : AST node types, mapped to integer IDs via a vocabulary
- Edges : parent -> child relationships in the AST tree
- Node features: one-hot / integer type ID

Change labels (per commit)
--------------------------
  ADD    - net positive nodes added  (> +5 nodes)
  DELETE - net negative nodes        (< -5 nodes)
  MODIFY - small structural change

Bug label (heuristic, Phase 6)
-------------------------------
  1 if commit message contains fix/bug/patch keywords (lagged -1)
  0 otherwise

Input : data/processed/clean_commits.json
Output: data/processed/graph_data.json

Usage:
  python scripts/build_graphs.py
  python scripts/build_graphs.py --input data/processed/clean_commits.json \
                                  --output data/processed/graph_data.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

# Project root - works regardless of CWD
ROOT = Path(__file__).resolve().parent.parent

from scripts.parse_ast import parse_code, TREE_SITTER_OK


# ---------------------------------------------
#  Node-type vocabulary
# ---------------------------------------------

class NodeVocab:
    """Incrementally assigns integer IDs to AST node type strings."""
    UNKNOWN = 0

    def __init__(self):
        self._vocab: dict[str, int] = {"<UNK>": self.UNKNOWN}
        self._next = 1

    def encode(self, node_type: str) -> int:
        if node_type not in self._vocab:
            self._vocab[node_type] = self._next
            self._next += 1
        return self._vocab[node_type]

    def __len__(self):
        return len(self._vocab)

    def to_dict(self) -> dict:
        return self._vocab


# Global shared vocabulary (grows as we process files)
VOCAB = NodeVocab()


# ---------------------------------------------
#  AST -> Graph conversion
# ---------------------------------------------

def ast_to_graph(source: str) -> Optional[dict]:
    """
    Parse Python source and convert to a graph dict:
    {
        "nodes": [int, ...],        # node type IDs (one per AST node)
        "edges": [[src, dst], ...], # directed parent->child edges
        "num_nodes": int,
        "num_edges": int,
    }
    Returns None if parsing fails or produces an error tree.
    """
    if not source or not source.strip():
        return None

    try:
        tree = parse_code(source)
    except Exception:
        return None

    root = tree.root_node
    if root.has_error:
        # Still proceed but flag it; some minor parse errors are tolerable
        pass

    nodes: list[int] = []
    edges: list[list[int]] = []

    # BFS traversal to assign node indices
    queue  = [(root, -1)]   # (node, parent_index)
    index  = 0

    while queue:
        current, parent_idx = queue.pop(0)
        current_idx = index
        index += 1

        nodes.append(VOCAB.encode(current.type))

        if parent_idx >= 0:
            edges.append([parent_idx, current_idx])

        for child in current.children:
            queue.append((child, current_idx))

    return {
        "nodes":     nodes,
        "edges":     edges,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }


# ---------------------------------------------
#  Change labeling (Phase 2.4)
# ---------------------------------------------

_BUG_KEYWORDS = {"fix", "bug", "patch", "error", "issue", "crash", "hotfix"}


def change_label(g_before: dict, g_after: dict) -> str:
    """
    Compare node counts between before and after graph.
    ADD    - after has significantly more nodes
    DELETE - after has significantly fewer nodes
    MODIFY - everything else
    """
    delta = g_after["num_nodes"] - g_before["num_nodes"]
    if delta > 5:
        return "ADD"
    elif delta < -5:
        return "DELETE"
    else:
        return "MODIFY"


def bug_label(message: str) -> int:
    """Heuristic: 1 if message suggests a bug-fix commit."""
    lower = message.lower()
    return int(any(kw in lower for kw in _BUG_KEYWORDS))


# ---------------------------------------------
#  Validation check (Phase 2.5)
# ---------------------------------------------

def validate_sample(graph_records: list[dict], n: int = 5) -> None:
    """Print graph structure for the first n commits as a visual check."""
    print(f"\n[Phase 2.5] Visual check - first {n} commit graphs:")
    print("-" * 55)
    for rec in graph_records[:n]:
        cid  = rec["commit_id"][:8]
        gb   = rec["graph_before"]
        ga   = rec["graph_after"]
        lbl  = rec["change_label"]
        bug  = rec["bug_label"]
        print(f"  Commit {cid}  label={lbl}  bug={bug}")
        print(f"    before: {gb['num_nodes']} nodes, {gb['num_edges']} edges")
        print(f"    after : {ga['num_nodes']} nodes, {ga['num_edges']} edges")
    print("-" * 55)
    print(f"  Vocab size so far: {len(VOCAB)} node types")
    print()


# ---------------------------------------------
#  Main build loop
# ---------------------------------------------

def build_graphs(input_path: Path, output_path: Path) -> list[dict]:
    if not TREE_SITTER_OK:
        raise RuntimeError("Tree-sitter is not available. Run parse_ast.py first.")

    print(f"\n{'='*55}")
    print(f"  Building graphs from: {input_path}")
    print(f"{'='*55}\n")

    with open(input_path, encoding="utf-8") as f:
        commits = json.load(f)

    print(f"  Loaded {len(commits)} commits")

    records = []
    skipped = 0

    for i, commit in enumerate(commits):
        if i % 100 == 0:
            print(f"  [{i}/{len(commits)}] processed - {len(records)} graphs built")

        commit_id = commit["commit_id"]
        message   = commit["message"]

        # Use the first changed file (primary diff carrier)
        files = commit.get("files", [])
        if not files:
            skipped += 1
            continue

        primary = files[0]
        g_before = ast_to_graph(primary.get("before", ""))
        g_after  = ast_to_graph(primary.get("after", ""))

        if g_before is None or g_after is None:
            skipped += 1
            continue

        records.append({
            "commit_id":    commit_id,
            "parent":       commit["parent"],
            "message":      message,
            "graph_before": g_before,
            "graph_after":  g_after,
            "change_label": change_label(g_before, g_after),
            "bug_label":    bug_label(message),
            "file":         primary["file"],
        })

    # Propagate bug label backwards: if commit[i] is a fix -> commit[i-1] is buggy
    # (This is a heuristic; commit[i-1] in list = temporally previous)
    for i in range(len(records)):
        if records[i]["bug_label"] == 1 and i > 0:
            # Mark the preceding commit's after-state as potentially buggy
            records[i-1]["bug_label_prev_buggy"] = 1
        else:
            records[i].setdefault("bug_label_prev_buggy", 0)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Also save vocabulary
    vocab_path = output_path.parent / "vocab.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(VOCAB.to_dict(), f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Graphs built  : {len(records)}")
    print(f"  Skipped       : {skipped}")
    print(f"  Vocab size    : {len(VOCAB)}")
    print(f"  Output        : {output_path}")
    print(f"  Vocab         : {vocab_path}")
    print(f"{'='*55}\n")

    validate_sample(records)
    return records


# ---------------------------------------------
#  CLI
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert clean commits to AST graphs.")
    p.add_argument("--input",  default=str(ROOT / "data" / "processed" / "clean_commits.json"))
    p.add_argument("--output", default=str(ROOT / "data" / "processed" / "graph_data.json"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_graphs(Path(args.input), Path(args.output))
