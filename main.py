"""
main.py
=======
Master runner - executes the full pipeline end-to-end.

Stages (run in order):
  1. validate_env     - check all imports
  2. clean_data       - filter raw commits
  3. build_graphs     - AST -> graph
  4. build_sequences  - sliding window sequences
  5. train            - train the model
  6. evaluate         - test metrics

Usage:
  python main.py                          # run all stages
  python main.py --from-stage graphs      # resume from stage
  python main.py --stage clean_data       # run only one stage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    ("validate_env",    None),
    ("clean_data",      ["python", "-m", "scripts.clean_data"]),
    ("build_graphs",    ["python", "-m", "scripts.build_graphs"]),
    ("build_sequences", ["python", "-m", "scripts.build_sequences"]),
    ("train",           ["python", "-m", "scripts.train"]),
    ("evaluate",        ["python", "-m", "scripts.evaluate"]),
]

STAGE_NAMES = [s[0] for s in STAGES]


def validate_env():
    print("  Checking environment ...")
    errors = []

    try:
        import torch
        print(f"    [OK]  torch {torch.__version__}")
    except ImportError:
        errors.append("torch")

    try:
        import torch_geometric
        print(f"    [OK]  torch_geometric {torch_geometric.__version__}")
    except ImportError:
        errors.append("torch_geometric")

    try:
        import tree_sitter  # noqa
        import importlib.metadata as _meta
        _ts_ver = _meta.version("tree-sitter")
        print(f"    [OK] tree_sitter {_ts_ver}")
    except ImportError:
        errors.append("tree_sitter")

    try:
        import tree_sitter_python  # noqa
        print(f"    [OK] tree_sitter_python")
    except ImportError:
        errors.append("tree_sitter_python")

    try:
        import git
        print(f"    [OK]  gitpython {git.__version__}")
    except ImportError:
        errors.append("gitpython")

    try:
        import tqdm
        print(f"    [OK]  tqdm {tqdm.__version__}")
    except ImportError:
        errors.append("tqdm")

    try:
        import networkx
        print(f"    [OK]  networkx {networkx.__version__}")
    except ImportError:
        errors.append("networkx")

    if errors:
        print(f"\n  [FAIL]  Missing: {errors}")
        print("  Install: pip install -r requirements.txt")
        raise SystemExit(1)
    else:
        print("  [OK]  All dependencies present.\n")


def run_stage(name: str, cmd: list[str] | None):
    print(f"\n{'='*55}")
    print(f"  STAGE: {name.upper()}")
    print(f"{'='*55}")

    if name == "validate_env":
        validate_env()
        return

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"\n  [FAIL]  Stage '{name}' failed (exit {result.returncode}).")
        raise SystemExit(result.returncode)
    print(f"\n  [OK]  Stage '{name}' complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Run the Code Evolution ML pipeline.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--stage",      choices=STAGE_NAMES, help="Run only this stage")
    group.add_argument("--from-stage", choices=STAGE_NAMES, help="Resume from this stage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.stage:
        # Run one stage only
        for name, cmd in STAGES:
            if name == args.stage:
                run_stage(name, cmd)
                break
    else:
        start_idx = 0
        if args.from_stage:
            start_idx = STAGE_NAMES.index(args.from_stage)

        for name, cmd in STAGES[start_idx:]:
            run_stage(name, cmd)

    print("\n  [DONE]  Pipeline complete!\n")
