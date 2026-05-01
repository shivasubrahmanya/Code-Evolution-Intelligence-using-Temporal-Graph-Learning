"""
scripts/utils_debug.py
======================
Quick debug/inspection utilities.
Run at any stage to inspect intermediate outputs.

Usage:
  python scripts/utils_debug.py --stage graphs     # inspect graph_data.json
  python scripts/utils_debug.py --stage sequences  # inspect train_sequences.json
  python scripts/utils_debug.py --stage commits    # inspect clean_commits.json
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

try:
    import ijson
except ImportError:
    ijson = None


def inspect_commits(path="data/processed/clean_commits.json", n=5):
    print(f"\n[Commits] {path}")
    if not Path(path).exists():
        print(f"  Error: File not found: {path}")
        return

    total = 0
    samples = []
    
    if ijson is None:
        with open(path) as f:
            data = json.load(f)
        total = len(data)
        samples = data[:n]
    else:
        with open(path, "rb") as f:
            items = ijson.items(f, "item")
            for i, c in enumerate(items):
                total += 1
                if len(samples) < n:
                    samples.append(c)
                if i > 0 and i % 10000 == 0:
                    sys.stdout.write(f"\r  Processed {i:,} items...")
                    sys.stdout.flush()
        print(f"\r  Processed {total:,} items total.        ")

    print(f"  Total: {total}")
    for c in samples:
        nf = len(c.get("files", []))
        print(f"  {c.get('commit_id', 'unknown')[:8]}  files={nf}  msg={c.get('message', '')[:60]}")


def inspect_graphs(path="data/processed/graph_data.json", n=5):
    print(f"\n[Graphs] {path}")
    if not Path(path).exists():
        print(f"  Error: File not found: {path}")
        return

    labels = Counter()
    bugs = 0
    total = 0
    samples = []

    if ijson is None:
        with open(path) as f:
            data = json.load(f)
        total = len(data)
        labels = Counter(d["change_label"] for d in data)
        bugs = sum(d.get("bug_label", 0) for d in data)
        samples = data[:n]
    else:
        with open(path, "rb") as f:
            items = ijson.items(f, "item")
            for i, d in enumerate(items):
                total += 1
                labels[d.get("change_label", "unknown")] += 1
                bugs += d.get("bug_label", 0)
                if len(samples) < n:
                    samples.append(d)
                if i > 0 and i % 10000 == 0:
                    sys.stdout.write(f"\r  Processed {i:,} items...")
                    sys.stdout.flush()
        print(f"\r  Processed {total:,} items total.        ")

    print(f"  Total     : {total}")
    print(f"  Labels    : {dict(labels)}")
    print(f"  Bug fixes : {bugs}")
    for d in samples:
        gb = d.get("graph_before", {"num_nodes": 0, "num_edges": 0})
        ga = d.get("graph_after", {"num_nodes": 0, "num_edges": 0})
        print(f"  {d.get('commit_id', 'unknown')[:8]}  {d.get('change_label', 'unknown')}"
              f"  before={gb.get('num_nodes')}n/{gb.get('num_edges')}e"
              f"  after={ga.get('num_nodes')}n/{ga.get('num_edges')}e")


def inspect_sequences(path="data/sequences/train_sequences.json", n=3):
    print(f"\n[Sequences] {path}")
    if not Path(path).exists():
        print(f"  Error: File not found: {path}")
        return

    if ijson is None:
        print("  Warning: ijson not installed. Falling back to json.load (may cause MemoryError).")
        with open(path) as f:
            data = json.load(f)
        labels = Counter(d["label"] for d in data)
        bugs = sum(d.get("bug", 0) for d in data)
        samples = data[:n]
        total = len(data)
    else:
        print("  Streaming file (this may take a while for 25GB+ files)...")
        labels = Counter()
        bugs = 0
        samples = []
        total = 0
        
        with open(path, "rb") as f:
            # ijson.items streams the array elements one by one
            items = ijson.items(f, "item")
            for i, d in enumerate(items):
                total += 1
                labels[d.get("label", "unknown")] += 1
                bugs += d.get("bug", 0)
                if len(samples) < n:
                    samples.append(d)
                
                if i > 0 and i % 10000 == 0:
                    sys.stdout.write(f"\r  Processed {i:,} items...")
                    sys.stdout.flush()
        
        print(f"\r  Processed {total:,} items total.        ")

    print(f"  Total     : {total}")
    print(f"  Labels    : {dict(labels)}")
    print(f"  Bug seqs  : {bugs}")
    for d in samples:
        w = len(d.get("input", []))
        print(f"  window={w}  label={d.get('label')}  bug={d.get('bug')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["commits", "graphs", "sequences"],
                   default="graphs")
    p.add_argument("--n", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "commits":
        inspect_commits(n=args.n)
    elif args.stage == "graphs":
        inspect_graphs(n=args.n)
    elif args.stage == "sequences":
        inspect_sequences(n=args.n)
