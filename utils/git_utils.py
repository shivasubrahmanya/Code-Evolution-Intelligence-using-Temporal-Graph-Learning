"""
utils/git_utils.py
==================
Lightweight git helper functions (subprocess-based).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_git(args: list[str], cwd: str | Path, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            ["git"] + args, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=timeout,
        )
        return r.stdout
    except Exception:
        return ""


def get_log(repo: Path, n: int = 100, fmt: str = "%H %s") -> list[str]:
    raw = run_git(["log", f"--format={fmt}", "--no-merges", f"-n{n}"], repo)
    return raw.strip().splitlines()


def get_file_at(repo: Path, commit: str, filepath: str) -> str:
    return run_git(["show", f"{commit}:{filepath}"], repo)


def get_changed_files(repo: Path, commit: str) -> list[str]:
    raw = run_git(["diff-tree", "--no-commit-id", "-r", "--name-only", commit], repo)
    return [f.strip() for f in raw.strip().splitlines() if f.strip()]


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()
