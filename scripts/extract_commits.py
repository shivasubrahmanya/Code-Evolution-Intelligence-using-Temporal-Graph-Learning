"""
scripts/extract_commits.py
==========================
Phase 1.3 - Commit Extraction
Extracts structured commit data from a cloned Git repo.

Output: data/raw/<repo_name>.json
Each record:
{
  "commit_id": "abc123",
  "parent":    "def456",
  "message":   "FIX: handle edge case in optimizer",
  "files": [
    {
      "file":   "optuna/samplers/_tpe.py",
      "before": "...",   # code before this commit
      "after":  "..."    # code after this commit
    }
  ]
}

Usage:
  python scripts/extract_commits.py \
      --repo   /path/to/cloned/optuna \
      --output data/raw/optuna.json \
      --limit  3000          # max commits to extract (0 = all)
      --after  2021-01-01    # optional: only commits after this date
      --before 2025-01-01    # optional: only commits before this date
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ---------------------------------------------
#  Helpers
# ---------------------------------------------

def run(cmd: list[str], cwd: str) -> str:
    """Run a shell command and return stdout. Returns '' on error."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",       # don't crash on non-UTF-8 bytes
            timeout=30
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {' '.join(cmd[:4])}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return ""


def get_commit_list(repo_path: str,
                    limit: int,
                    after: str | None,
                    before: str | None) -> list[dict]:
    """
    Returns list of {commit_id, parent, message} using git log.

    Format string:  commit_id<TAB>parent<TAB>message
    %H  = full commit hash
    %P  = parent hash (first parent only for merge commits)
    %s  = subject line (first line of message)
    """
    fmt = "%H\t%P\t%s"
    cmd = ["git", "log", f"--format={fmt}", "--no-merges"]

    if after:
        cmd += [f"--after={after}"]
    if before:
        cmd += [f"--before={before}"]
    if limit > 0:
        cmd += [f"-n", str(limit)]

    raw = run(cmd, repo_path)
    commits = []

    for line in raw.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        commit_id, parents, message = parts
        # For merge commits (multiple parents), take only the first
        parent = parents.split()[0] if parents.strip() else ""
        commits.append({
            "commit_id": commit_id.strip(),
            "parent":    parent.strip(),
            "message":   message.strip(),
        })

    return commits


def get_changed_python_files(repo_path: str, commit_id: str) -> list[str]:
    """
    Returns list of .py files changed in this commit.
    Uses git diff-tree to get the file list without fetching code yet.
    """
    cmd = [
        "git", "diff-tree",
        "--no-commit-id", "-r",
        "--name-only",
        "--diff-filter=M",      # M = Modified only (skip Added/Deleted/Renamed)
        commit_id
    ]
    raw = run(cmd, repo_path)
    files = [
        f.strip() for f in raw.strip().splitlines()
        if f.strip().endswith(".py")
    ]
    return files


def get_file_at_commit(repo_path: str, commit_id: str, filepath: str) -> str:
    """Returns file content at a specific commit. Returns '' if not found."""
    cmd = ["git", "show", f"{commit_id}:{filepath}"]
    return run(cmd, repo_path)


def extract_file_changes(repo_path: str,
                         commit_id: str,
                         parent_id: str,
                         py_files: list[str]) -> list[dict]:
    """
    For each changed .py file, fetch before (parent) and after (commit) content.
    Skips files larger than MAX_FILE_BYTES to avoid huge JSON blobs.
    """
    MAX_FILE_BYTES = 100_000   # ~100 KB per file
    MAX_FILES_PER_COMMIT = 10  # don't process commits touching 50 files

    if len(py_files) > MAX_FILES_PER_COMMIT:
        return []               # likely a bulk rename/reformat - skip

    file_records = []
    for filepath in py_files:
        before = ""
        after  = ""

        # "after"  = content at this commit
        after = get_file_at_commit(repo_path, commit_id, filepath)

        # "before" = content at parent commit
        if parent_id:
            before = get_file_at_commit(repo_path, parent_id, filepath)

        # Skip empty or oversized files
        if not after:
            continue
        if len(before.encode()) > MAX_FILE_BYTES or len(after.encode()) > MAX_FILE_BYTES:
            continue

        file_records.append({
            "file":   filepath,
            "before": before,
            "after":  after
        })

    return file_records


# ---------------------------------------------
#  Cleaning filters  (Phase 1.3 cleaning step)
# ---------------------------------------------

SKIP_KEYWORDS = [
    "merge branch", "merge pull request",   # merge commits (belt + suspenders)
    "bump version", "update changelog",      # release noise
    "update dependencies", "update setup",
]

def is_skip_message(message: str) -> bool:
    """Return True if the commit message signals noise."""
    lower = message.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


# ---------------------------------------------
#  Main extraction loop
# ---------------------------------------------

def extract(repo_path: str,
            output_path: str,
            limit: int = 0,
            after: str | None = None,
            before: str | None = None) -> None:

    repo_path = str(Path(repo_path).resolve())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Repo   : {repo_path}")
    print(f"  Output : {output_path}")
    print(f"  Limit  : {limit if limit > 0 else 'all'}")
    print(f"  After  : {after or '-'}")
    print(f"  Before : {before or '-'}")
    print(f"{'='*55}\n")

    # -- Step 1: get commit list --------------------------
    print("-> Fetching commit list ...")
    commits = get_commit_list(repo_path, limit, after, before)
    print(f"  Found {len(commits)} commits (after --no-merges filter)\n")

    results      = []
    skipped_msg  = 0
    skipped_nopy = 0
    errors       = 0

    # -- Step 2: iterate commits --------------------------
    for i, commit in enumerate(commits, 1):
        commit_id = commit["commit_id"]
        parent_id = commit["parent"]
        message   = commit["message"]

        # Progress every 50 commits
        if i % 50 == 0:
            print(f"  [{i}/{len(commits)}] processed - {len(results)} kept, "
                  f"{skipped_nopy} no-py, {skipped_msg} skipped-msg, {errors} errors")

        # Filter 1: skip noisy messages
        if is_skip_message(message):
            skipped_msg += 1
            continue

        # Filter 2: skip root commit (no parent to diff against)
        if not parent_id:
            skipped_nopy += 1
            continue

        # Get changed Python files
        py_files = get_changed_python_files(repo_path, commit_id)
        if not py_files:
            skipped_nopy += 1
            continue

        # Extract before/after for each file
        try:
            file_records = extract_file_changes(
                repo_path, commit_id, parent_id, py_files
            )
        except Exception as e:
            errors += 1
            print(f"  [WARN] commit {commit_id[:8]}: {e}", file=sys.stderr)
            continue

        if not file_records:
            skipped_nopy += 1
            continue

        results.append({
            "commit_id": commit_id,
            "parent":    parent_id,
            "message":   message,
            "files":     file_records
        })

    # -- Step 3: write JSON -------------------------------
    print(f"\n-> Writing {len(results)} records to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # -- Summary ------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Commits extracted : {len(results)}")
    print(f"  Skipped (no .py)  : {skipped_nopy}")
    print(f"  Skipped (message) : {skipped_msg}")
    print(f"  Errors            : {errors}")
    print(f"  Output            : {output_path}")
    print(f"{'='*55}\n")


# ---------------------------------------------
#  CLI
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract structured commit data from a Git repo."
    )
    p.add_argument(
        "--repo", required=True,
        help="Path to the cloned Git repository"
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSON path, e.g. data/raw/optuna.json"
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max number of commits to process (0 = all, default: 0)"
    )
    p.add_argument(
        "--after", default=None,
        help="Only include commits after this date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--before", default=None,
        help="Only include commits before this date (YYYY-MM-DD)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(
        repo_path   = args.repo,
        output_path = args.output,
        limit       = args.limit,
        after       = args.after,
        before      = args.before,
    )