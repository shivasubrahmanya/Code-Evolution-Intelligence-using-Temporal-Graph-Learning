"""
utils/label_utils.py
====================
Helpers for change and bug label encoding/decoding.
"""

from __future__ import annotations

CHANGE_LABELS = ["ADD", "DELETE", "MODIFY"]
CHANGE_TO_IDX = {l: i for i, l in enumerate(CHANGE_LABELS)}
IDX_TO_CHANGE = {i: l for i, l in enumerate(CHANGE_LABELS)}
BUG_KEYWORDS  = {"fix", "bug", "patch", "error", "issue", "crash", "hotfix", "broken"}


def encode_change(label: str) -> int:
    return CHANGE_TO_IDX.get(label.upper(), 2)


def decode_change(idx: int) -> str:
    return IDX_TO_CHANGE.get(idx, "MODIFY")


def is_bug_fix(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in BUG_KEYWORDS)


def label_sequences_with_bugs(records: list[dict]) -> list[dict]:
    """
    Apply lagged bug labeling:
    If record[i] is a bug fix -> record[i-1] has bug_label_prev_buggy = 1
    """
    for i, rec in enumerate(records):
        rec.setdefault("bug_label", int(is_bug_fix(rec.get("message", ""))))
        rec.setdefault("bug_label_prev_buggy", 0)

    for i in range(1, len(records)):
        if records[i]["bug_label"] == 1:
            records[i-1]["bug_label_prev_buggy"] = 1

    return records
