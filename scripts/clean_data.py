"""
scripts/clean_data.py
=====================
Phase 1.4 - Data Cleaning

Reads raw commit JSON(s) produced by extract_commits.py and applies filters:
  - Remove merge commits (multiple parents already filtered, but double-check)
  - Remove commits with empty diffs
  - Remove large commits (>20 files)
  - Remove non-Python files (already filtered upstream, but validate)
  - Remove commits where before/after are identical
  - Remove commits where code is not valid Python (syntax check)

Phase 1.5 - Validation is embedded at the end.

Input : data/raw/*.json   (or a specific file via --input)
Output: data/processed/clean_commits.json

Usage:
  python scripts/clean_data.py
  python scripts/clean_data.py --input data/raw/optuna.json --output data/processed/clean_commits.json
  python scripts/clean_data.py --max-files 20 --min-lines 5
"""

import ast
import json
import argparse
import warnings
from pathlib import Path

# Project root = parent of this script file (works regardless of CWD)
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------
#  Config
# ---------------------------------------------

DEFAULT_INPUT_DIR  = ROOT / "data" / "raw"
DEFAULT_OUTPUT     = ROOT / "data" / "processed" / "clean_commits.json"
MAX_FILES_DEFAULT  = 20
MIN_LINES_DEFAULT  = 3    # minimum lines of code to keep a file pair


# ---------------------------------------------
#  Validators
# ---------------------------------------------

def is_valid_python(code: str) -> bool:
    """Return True if 'code' parses without SyntaxError.
    SyntaxWarnings (e.g. invalid escape sequences in raw code) are suppressed
    — they come from the code being analyzed, not our script.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False



def is_meaningful_diff(before: str, after: str, min_lines: int) -> bool:
    """Return True if before and after differ and both have enough lines."""
    if before.strip() == after.strip():
        return False                       # no actual change
    if len(after.splitlines()) < min_lines:
        return False                       # too short to be useful
    return True


def clean_file_list(files: list[dict], min_lines: int) -> list[dict]:
    """
    Filter a single commit's file list:
      - only .py files
      - must have both before and after
      - must be valid Python
      - must have a meaningful diff
    """
    cleaned = []
    for f in files:
        path   = f.get("file", "")
        before = f.get("before", "")
        after  = f.get("after", "")

        if not path.endswith(".py"):
            continue
        if not before or not after:
            continue
        if not is_meaningful_diff(before, after, min_lines):
            continue
        if not is_valid_python(before):
            continue
        if not is_valid_python(after):
            continue

        cleaned.append(f)
    return cleaned


# ---------------------------------------------
#  Main cleaning loop
# ---------------------------------------------

def clean(input_paths: list[Path],
          output_path: Path,
          max_files: int,
          min_lines: int) -> list[dict]:

    all_commits: list[dict] = []
    for p in input_paths:
        print(f"  Loading {p} ...")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        print(f"    -> {len(data)} raw commits")
        all_commits.extend(data)

    print(f"\n  Total raw commits : {len(all_commits)}")

    cleaned: list[dict] = []
    stats = {
        "no_parent":    0,
        "too_many_files": 0,
        "no_py_files":  0,
        "invalid_py":   0,
        "kept":         0,
    }

    for commit in all_commits:
        commit_id = commit.get("commit_id", "?")
        parent    = commit.get("parent", "")
        files     = commit.get("files", [])

        # 1. Must have a parent
        if not parent:
            stats["no_parent"] += 1
            continue

        # 2. Not too many files
        if len(files) > max_files:
            stats["too_many_files"] += 1
            continue

        # 3. Clean the file list
        clean_files = clean_file_list(files, min_lines)

        if not clean_files:
            stats["no_py_files"] += 1
            continue

        cleaned.append({
            "commit_id": commit_id,
            "parent":    parent,
            "message":   commit.get("message", ""),
            "files":     clean_files,
        })
        stats["kept"] += 1

    # -- Write output ------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # -- Summary -----------------------------------------
    print(f"\n{'='*55}")
    print(f"  Cleaning complete")
    print(f"  Kept              : {stats['kept']}")
    print(f"  Dropped no-parent : {stats['no_parent']}")
    print(f"  Dropped too-large : {stats['too_many_files']}")
    print(f"  Dropped no-py     : {stats['no_py_files']}")
    print(f"  Output            : {output_path}")
    print(f"{'='*55}\n")

    return cleaned


# ---------------------------------------------
#  Phase 1.5 - Validation (MANDATORY)
# ---------------------------------------------

def validate(commits: list[dict]) -> None:
    print("\n[Phase 1.5] Running validation ...")
    errors = []

    for i, c in enumerate(commits):
        cid = c.get("commit_id", f"index-{i}")

        if not c.get("parent"):
            errors.append(f"  FAIL [{cid}] - missing parent")

        for f in c.get("files", []):
            if not f.get("before"):
                errors.append(f"  FAIL [{cid}] - null before in {f.get('file')}")
            if not f.get("after"):
                errors.append(f"  FAIL [{cid}] - null after in {f.get('file')}")

    if errors:
        print("  VALIDATION FAILED - issues found:")
        for e in errors[:20]:          # show first 20
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
        print("\n  [FAIL]  Fix the data before proceeding to Phase 2.")
        raise SystemExit(1)
    else:
        print(f"  [OK]  All {len(commits)} commits passed validation.\n")


# ---------------------------------------------
#  CLI
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Clean raw commit JSON data.")
    p.add_argument("--input", default=None,
                   help="Specific raw JSON file (default: all files in data/raw/)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output JSON path")
    p.add_argument("--max-files", type=int, default=MAX_FILES_DEFAULT,
                   help="Max changed files per commit (default: 20)")
    p.add_argument("--min-lines", type=int, default=MIN_LINES_DEFAULT,
                   help="Min lines in after-code (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_paths = sorted(DEFAULT_INPUT_DIR.glob("*.json"))
        if not input_paths:
            print(f"No JSON files found in {DEFAULT_INPUT_DIR}. Run extract_commits.py first.")
            raise SystemExit(1)

    commits = clean(
        input_paths = input_paths,
        output_path = Path(args.output),
        max_files   = args.max_files,
        min_lines   = args.min_lines,
    )

    validate(commits)
