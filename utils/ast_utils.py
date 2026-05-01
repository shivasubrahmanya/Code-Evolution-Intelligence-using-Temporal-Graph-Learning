"""
utils/ast_utils.py
==================
Additional helpers for working with Tree-sitter AST nodes.
"""

from __future__ import annotations
from typing import Generator


def walk(node) -> Generator:
    """Depth-first walk of a Tree-sitter node tree."""
    yield node
    for child in node.children:
        yield from walk(child)


def find_nodes_by_type(root, node_type: str) -> list:
    """Return all nodes of a given type in the tree."""
    return [n for n in walk(root) if n.type == node_type]


def node_text(node, source_bytes: bytes) -> str:
    """Extract source text for a node."""
    return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")


def tree_depth(node, _d=0) -> int:
    """Return maximum depth of the AST."""
    if not node.children:
        return _d
    return max(tree_depth(c, _d + 1) for c in node.children)


def count_functions(root) -> int:
    """Count function definitions in the tree."""
    return len(find_nodes_by_type(root, "function_definition"))


def count_classes(root) -> int:
    """Count class definitions in the tree."""
    return len(find_nodes_by_type(root, "class_definition"))
