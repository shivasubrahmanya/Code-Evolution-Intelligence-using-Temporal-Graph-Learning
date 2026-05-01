"""
scripts/parse_ast.py
====================
Phase 2.1 - AST Parsing with Tree-sitter

Provides:
  parse_code(source: str) -> tree_sitter.Tree
  tree_to_dict(node)      -> dict   (recursive, for inspection)

Also runnable as a standalone smoke-test:
  python scripts/parse_ast.py

Tree-sitter grammar note
------------------------
tree-sitter >= 0.21 ships prebuilt Python grammar wheels.
Install:  pip install tree-sitter tree-sitter-python
"""

from __future__ import annotations

import sys
from typing import Optional

# -- Tree-sitter import (graceful degradation) ----------------------------
TREE_SITTER_OK = False
_ts_err_msg    = ""
_PARSER        = None

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    _PARSER     = Parser(PY_LANGUAGE)
    TREE_SITTER_OK = True

except Exception as _ts_err:
    TREE_SITTER_OK = False
    _ts_err_msg    = str(_ts_err)
    # Don't print here to avoid noise during normal imports, 
    # but the info is available in _ts_err_msg



# ---------------------------------------------
#  Public API
# ---------------------------------------------

def get_parser():
    """Return the shared Tree-sitter Python parser."""
    if not TREE_SITTER_OK:
        raise RuntimeError(
            f"Tree-sitter is not available: {_ts_err_msg}\n"
            "Install with:  pip install tree-sitter tree-sitter-python"
        )
    return _PARSER


def parse_code(source: str):
    """
    Parse a Python source string and return a Tree-sitter Tree.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    tree_sitter.Tree
    """
    parser = get_parser()
    return parser.parse(source.encode("utf-8"))


def tree_to_dict(node, max_depth: int = 30, _depth: int = 0) -> dict:
    """
    Recursively convert a Tree-sitter node to a plain dict.
    Useful for inspection / debugging.

    Parameters
    ----------
    node      : tree_sitter.Node
    max_depth : int   (guard against infinite recursion)

    Returns
    -------
    dict with keys: type, start, end, children
    """
    if _depth > max_depth:
        return {"type": node.type, "truncated": True}

    return {
        "type":     node.type,
        "start":    node.start_point,
        "end":      node.end_point,
        "children": [
            tree_to_dict(ch, max_depth, _depth + 1)
            for ch in node.children
        ],
    }


def count_nodes(node) -> int:
    """Count total nodes in an AST tree (recursive)."""
    return 1 + sum(count_nodes(ch) for ch in node.children)


def collect_node_types(node, result: Optional[set] = None) -> set[str]:
    """Collect the set of unique node-type strings in the tree."""
    if result is None:
        result = set()
    result.add(node.type)
    for ch in node.children:
        collect_node_types(ch, result)
    return result


# ---------------------------------------------
#  Smoke-test (run directly)
# ---------------------------------------------

SAMPLE_CODE = """\
def add(a, b):
    \"\"\"Return a + b.\"\"\"
    return a + b

class Foo:
    def __init__(self):
        self.x = 0

    def increment(self, step=1):
        self.x += step
        return self.x
"""


def _smoke_test():
    print("=" * 55)
    print("  Tree-sitter Smoke Test")
    print("=" * 55)

    if not TREE_SITTER_OK:
        print(f"  [FAIL] Tree-sitter unavailable: {_ts_err_msg}")
        sys.exit(1)

    tree  = parse_code(SAMPLE_CODE)
    root  = tree.root_node
    d     = tree_to_dict(root, max_depth=10)
    types = collect_node_types(root)
    n     = count_nodes(root)

    print(f"  Root type   : {root.type}")
    print(f"  Total nodes : {n}")
    print(f"  Unique types: {sorted(types)[:10]} ...")
    print(f"  Error nodes : {root.has_error}")

    # Pretty-print first level
    print("\n  Top-level children:")
    for ch in root.children:
        print(f"    [{ch.type}]  lines {ch.start_point[0]+1}-{ch.end_point[0]+1}")

    print(f"\n  [OK] Tree-sitter is working correctly.\n")


if __name__ == "__main__":
    _smoke_test()
