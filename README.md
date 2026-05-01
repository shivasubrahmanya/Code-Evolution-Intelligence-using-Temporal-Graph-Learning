# Code Evolution Intelligence — Full Project Briefing

> **What this project does:**
> It watches how a Python codebase changes over time (commit by commit), converts each version of the code into a graph (via AST), and trains a deep learning model to:
> 1. **Predict the type of the next code change** (Add / Delete / Modify)
> 2. **Detect if the current code is likely buggy** (binary classification)

---

## Big Picture Architecture

```
Git Repo
   |
   v
[extract_commits.py]  ──>  Raw JSON  (commit id, before code, after code)
   |
   v
[clean_data.py]       ──>  Cleaned JSON  (filtered, validated Python-only commits)
   |
   v
[parse_ast.py]        ──>  Tree-sitter AST  (abstract syntax tree of Python code)
   |
   v
[build_graphs.py]     ──>  Graph JSON  (nodes = AST node types, edges = parent→child)
   |
   v
[build_sequences.py]  ──>  Sequence JSON  ([graph1, graph2, graph3] → predict graph4)
   |
   v
[GNN model]           ──>  Graph Embedding  (each graph → a 128-dim vector)
   |
   v
[Transformer model]   ──>  Context Vector  (sequence of embeddings → one vector)
   |
   v
[Two heads]           ──>  change label (ADD/DELETE/MODIFY) + bug probability
   |
   v
[evaluate.py]         ──>  Metrics, confusion matrix, comparison vs baselines
   |
   v
[demo.py]             ──>  Streamlit web app — live predictions
```

---

## Full File-by-File Breakdown

---

### ROOT LEVEL

#### `main.py`
**The master runner.** Chains all pipeline stages in the correct order.
- Run everything: `python main.py`
- Run one stage: `python main.py --stage clean_data`
- Resume from a stage: `python main.py --from-stage build_graphs`
- Internally calls each script as a subprocess in the right sequence.

#### `requirements.txt`
Lists every Python package needed. Install all at once:
```bash
pip install -r requirements.txt
```
Key packages: `torch`, `torch-geometric`, `tree-sitter`, `tree-sitter-python`, `gitpython`, `streamlit`, `tqdm`, `networkx`, `matplotlib`, `scikit-learn`

#### `README.md`
This file. Full documentation and briefing.

---

### `configs/`

#### `configs/config.yaml`
**Central control panel.** All hyperparameters in one place — window size, train split, model dimensions, learning rate, batch size, epochs, file paths.
- You change this file instead of touching source code when tuning.
- Key sections: `data`, `model`, `training`, `evaluation`, `app`

---

### `data/`

This folder holds everything that flows through the pipeline. It has three sub-levels:

```
data/
  raw/
    repos/          ← cloned git repositories live here
    <repo>.json     ← raw commit extraction output (per repo)
  processed/
    clean_commits.json   ← after cleaning/filtering
    graph_data.json      ← AST graphs for every commit
    vocab.json           ← mapping of AST node type → integer ID
  sequences/
    train_sequences.json ← 70% of data, sliding window format
    val_sequences.json   ← 15%
    test_sequences.json  ← 15%
```

**Nothing in `data/` is code** — it's all generated data produced by the scripts.

---

### `scripts/`

The core pipeline scripts. Run them in this exact order:

#### `scripts/extract_commits.py` ← **Phase 1.3** (you already had this)
Walks through a cloned git repo using `git log` and `git show`.
- Extracts: `commit_id`, `parent`, `message`, `before_code`, `after_code` for each changed `.py` file.
- Filters out merge commits, non-Python files, and oversized files automatically.
- **Usage:** `python scripts/extract_commits.py --repo data/raw/repos/optuna --output data/raw/optuna.json`
- **Output:** `data/raw/<repo>.json`

#### `scripts/clean_data.py` ← **Phase 1.4 + 1.5**
Reads all raw JSON files and applies strict filters:
- Removes commits with no parent (root commits)
- Removes commits touching more than 20 files
- Removes non-Python files
- Removes files where `before == after` (no real change)
- Removes files where `before` or `after` is not valid Python (uses `ast.parse`)
- At the end, runs **mandatory Phase 1.5 validation** — if any commit has null code or missing parent, it stops with exit code 1.
- **Usage:** `python scripts/clean_data.py`
- **Output:** `data/processed/clean_commits.json`

#### `scripts/parse_ast.py` ← **Phase 2.1**
Wraps Tree-sitter to parse Python source code into an AST.
- Exposes `parse_code(source)` → returns a Tree-sitter `Tree` object.
- Also exposes helpers: `tree_to_dict()`, `count_nodes()`, `collect_node_types()`.
- Running it directly (`python scripts/parse_ast.py`) does a smoke test to verify Tree-sitter works.
- **Used by:** `build_graphs.py`, `app/demo.py`

#### `scripts/build_graphs.py` ← **Phase 2.2, 2.3, 2.4**
The most important data processing script. Does three things:
1. **AST → Graph:** BFS-traverses the AST, assigns each node a type ID from a vocabulary, creates parent→child edge list.
2. **Stores graphs:** Saves `graph_before` and `graph_after` per commit as `{"nodes": [...], "edges": [[src, dst], ...]}`.
3. **Labels each commit:**
   - `change_label`: `ADD` if after has 5+ more nodes, `DELETE` if 5+ fewer, else `MODIFY`
   - `bug_label`: 1 if commit message has "fix/bug/patch/error/crash" keywords
   - `bug_label_prev_buggy`: 1 if the *next* commit was a bug fix (lagged heuristic)
4. Saves the vocabulary (node type → int ID) to `vocab.json`.
- **Usage:** `python scripts/build_graphs.py`
- **Output:** `data/processed/graph_data.json`, `data/processed/vocab.json`

#### `scripts/build_sequences.py` ← **Phase 3**
Converts the graph records into training sequences using a **sliding window**.
- Window of 3: takes commits `[C1, C2, C3]` as input, and `C4` as the target to predict.
- The split is **chronological** (not random) — 70% train, 15% val, 15% test.
- **Usage:** `python scripts/build_sequences.py --window 3`
- **Output:** `data/sequences/train_sequences.json`, `val_sequences.json`, `test_sequences.json`

#### `scripts/train.py` ← **Phase 5**
The full training loop.
- Loads sequence datasets, loads the model, runs epoch-by-epoch training.
- For each batch: converts raw graph dicts → PyG Data → encodes with GNN → stacks into sequence tensor → passes through Transformer → computes joint loss.
- **Loss = CrossEntropy (change type) + BCEWithLogitsLoss (bug detection)**
- Saves `best_model.pt` (lowest val loss) and `final_model.pt` at the end.
- Also saves `train_history.json` with per-epoch metrics.
- **Usage:** `python scripts/train.py --epochs 20 --batch-size 16 --lr 1e-3`
- **Output:** `outputs/checkpoints/best_model.pt`, `outputs/checkpoints/train_history.json`

#### `scripts/evaluate.py` ← **Phase 7**
Loads the best checkpoint and runs inference on the test set.
- **Change prediction metrics:** Accuracy, Top-3 Accuracy, Confusion Matrix
- **Bug detection metrics:** Precision, Recall, F1-score
- **Baselines:** Compares your model against Random predictor and Majority-class predictor
- **Usage:** `python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt`
- **Output:** `outputs/results/eval_results.json`

#### `scripts/utils_debug.py`
A quick inspection tool — not part of the training pipeline, just for debugging.
- Prints a summary of any data file at any stage.
- **Usage:**
  - `python scripts/utils_debug.py --stage commits`
  - `python scripts/utils_debug.py --stage graphs`
  - `python scripts/utils_debug.py --stage sequences`

---

### `models/`

Pure PyTorch model definitions. No data loading, no training logic here.

#### `models/gnn.py` ← **Phase 4.1**
**GraphEncoder** — encodes a single graph into a fixed-size vector.
```
Node IDs (integers)
  → nn.Embedding (64-dim)
  → GCNConv layer 1 (128-dim) + ReLU + Dropout
  → GCNConv layer 2 (128-dim) + ReLU + Dropout
  → Global Mean Pool (averages all node vectors)
  → Linear projection
  → Output: [128-dim graph embedding vector]
```
- Input: `x` (node type IDs), `edge_index` (edge list), `batch` (batch assignment)
- Output: one 128-dim vector **per graph**
- Uses PyTorch Geometric `GCNConv` and `global_mean_pool`

#### `models/temporal.py` ← **Phase 4.2**
**TemporalTransformer** — encodes a *sequence* of graph embeddings.
```
[graph_emb_1, graph_emb_2, graph_emb_3]  ← shape [B, W=3, D=128]
  → Sinusoidal Positional Encoding  (adds position awareness)
  → Transformer Encoder (2 layers, 4 attention heads, Pre-LayerNorm)
  → Mean pool over sequence positions
  → Output: [128-dim context vector]
```
- Uses `nn.TransformerEncoderLayer` with `batch_first=True` (standard PyTorch)
- No external dependencies beyond PyTorch

#### `models/multitask_model.py` ← **Phase 4.3**
**CodeEvolutionModel** — the full end-to-end model. Assembles GNN + Transformer + two prediction heads.
```
sequence of graphs
  → GraphEncoder (per graph)
  → stack → [B, W, 128]
  → TemporalTransformer
  → context [B, 128]
  → change_head: Linear(128→64) → ReLU → Linear(64→3)  → logits for ADD/DELETE/MODIFY
  → bug_head:    Linear(128→64) → ReLU → Linear(64→1)  → bug probability logit
```
- Also defines: `CHANGE_LABELS = ["ADD", "DELETE", "MODIFY"]`, `CHANGE_TO_IDX`, `NUM_CHANGE_CLASSES`

#### `models/__init__.py`
Makes `models/` a Python package. Exports `GraphEncoder`, `TemporalTransformer`, `CodeEvolutionModel`.

---

### `utils/`

Shared helper functions used across scripts and the app.

#### `utils/graph_utils.py`
Converts raw graph dicts (the JSON format from `graph_data.json`) into PyTorch Geometric `Data` objects.
- `dict_to_pyg(graph_dict)` → `Data` object with `.x` (node IDs) and `.edge_index`
- `sequence_to_pyg_list(seq_record)` → list of `Data` objects from one sequence record
- `graphs_to_batch(graphs)` → `Batch` object (feeds directly into GNN)
- `pad_graph(graph, target_nodes)` → pads a graph if fixed-size input is needed

#### `utils/data_utils.py`
PyTorch `Dataset` wrapper for the sequence JSON files.
- **`CodeSequenceDataset`** — wraps `train/val/test_sequences.json`. Each `__getitem__` returns `raw_input` (list of graph dicts), `change_label` (long tensor), `bug_label` (float tensor).
- **`load_label_weights()`** — computes inverse-frequency class weights to handle class imbalance in the CrossEntropy loss.

#### `utils/label_utils.py`
Clean functions for label encoding/decoding.
- `encode_change("ADD")` → `0`
- `decode_change(2)` → `"MODIFY"`
- `is_bug_fix(message)` → `True/False`
- `label_sequences_with_bugs(records)` → applies lagged bug labeling to a list

#### `utils/ast_utils.py`
Extra AST analysis tools (useful in notebooks or debugging).
- `walk(node)` — depth-first generator over all tree nodes
- `find_nodes_by_type(root, "function_definition")` → list of matching nodes
- `node_text(node, source_bytes)` → extract the raw source text for any node
- `tree_depth(node)` → maximum depth of the AST
- `count_functions(root)` / `count_classes(root)` → quick code metrics

#### `utils/git_utils.py`
Thin subprocess wrapper over git commands (alternative to GitPython for specific tasks).
- `run_git(args, cwd)` — runs any git command, returns stdout
- `get_log(repo, n)` — get last N commit hashes
- `get_file_at(repo, commit, filepath)` — get file content at a specific commit
- `get_changed_files(repo, commit)` — list files changed in a commit
- `is_git_repo(path)` — checks if a directory is a git repo

#### `utils/__init__.py`
Makes `utils/` a package, re-exports all key functions for convenient imports.

---

### `app/`

Everything related to the demo application.

#### `app/demo.py` ← **Phase 8** — THE DEMO
**Streamlit web app** — the user-facing interface.
- **Sidebar:** configure checkpoint path, window size, toggle demo mode
- **Demo mode:** uses 4 synthetic Python code snippets — no real repo or trained model required; shows the full UI flow
- **Live mode:** paste a GitHub repo URL → it clones it, extracts recent Python commits, parses their ASTs, runs them through the model, shows predictions
- **Displayed outputs:**
  - Commit timeline cards (commit hash, message, file, node/edge counts)
  - Next change type prediction (ADD / DELETE / MODIFY) with confidence bars
  - Bug probability gauge (green/yellow/red color coded)
  - Line chart of how graph size evolves across the commit sequence
- **Run:** `streamlit run app/demo.py`

#### `app/inference.py`
**Programmatic inference** — use this when you want to call the model from code (not the UI).
```python
from app.inference import Predictor
p = Predictor("outputs/checkpoints/best_model.pt")
result = p.predict_sequence([code1, code2, code3])
# → {"change_label": "MODIFY", "change_probs": {...}, "bug_prob": 0.23}
```

#### `app/visualization.py`
Matplotlib-based plotting functions for the notebook or post-training analysis.
- `plot_training_curves("outputs/checkpoints/train_history.json")` — loss + accuracy over epochs
- `plot_confusion_matrix(cm, labels)` — heatmap of any confusion matrix
- `graph_size_summary("data/processed/graph_data.json")` — histogram of before/after node counts and their delta

---

### `outputs/`

Generated at runtime — never manually edited.
```
outputs/
  checkpoints/
    best_model.pt         ← saved when val loss improves
    final_model.pt        ← saved at end of training
    train_history.json    ← per-epoch loss/acc/f1
  results/
    eval_results.json     ← test set metrics from evaluate.py
```

### `notebooks/`
Empty — intended for Jupyter notebooks for exploration, visualization, and ablation experiments.

### `tests/`
Empty — intended for unit tests (e.g., testing `dict_to_pyg`, `change_label`, `encode_change`).

### `env/`
Your virtual environment folder. Never commit this to git.

---

## Execution Order (strict)

```bash
# Step 0: Verify environment
python main.py --stage validate_env

# Step 1: Clone a repo
git clone https://github.com/optuna/optuna data/raw/repos/optuna

# Step 2: Extract commits
python scripts/extract_commits.py \
    --repo data/raw/repos/optuna \
    --output data/raw/optuna.json \
    --limit 2000

# Step 3: Clean + validate (MANDATORY check built in)
python scripts/clean_data.py

# Step 4: Build AST graphs + label them
python scripts/build_graphs.py

# Step 5: Build temporal sequences (70/15/15 split)
python scripts/build_sequences.py

# Step 6: Train
python scripts/train.py --epochs 20

# Step 7: Evaluate
python scripts/evaluate.py

# Step 8: Demo app
streamlit run app/demo.py

# Debug any stage:
python scripts/utils_debug.py --stage graphs
```

---

## Data Flow Summary

```
Git repo  →  extract_commits.py  →  data/raw/<repo>.json
                                          |
                                    clean_data.py
                                          |
                               data/processed/clean_commits.json
                                          |
                                    build_graphs.py
                                          |
                               data/processed/graph_data.json
                               data/processed/vocab.json
                                          |
                                  build_sequences.py
                                          |
                    data/sequences/train / val / test _sequences.json
                                          |
                                      train.py
                                          |
                           outputs/checkpoints/best_model.pt
                                          |
                                     evaluate.py
                                          |
                            outputs/results/eval_results.json
```

---

## What the Model Learns

| Input | What it sees |
|---|---|
| A sequence of 3 commits | 3 × (before-graph + after-graph encoded as one graph-after embedding) |
| Each graph | AST nodes of the Python code, connected by parent→child edges |
| Each node | An integer ID representing its AST type (e.g., `function_definition=5`) |

| Output | What it predicts |
|---|---|
| `change_label` | Will the next commit ADD, DELETE, or MODIFY code structure? |
| `bug_prob` | Is the current state of the code likely to contain a bug? |

---

## Common Questions

**Q: What is an AST node?**
Every part of Python code has a type: `function_definition`, `class_definition`, `assignment`, `return_statement`, `if_statement`, `for_statement`, etc. The AST (Abstract Syntax Tree) represents code as a tree of these typed nodes.

**Q: Why graphs instead of raw text?**
Graphs capture **structure** — which function calls which, how deep nesting goes, whether code was refactored into classes. Raw text models like LLMs don't naturally capture this hierarchical structure as compactly.

**Q: Why a Transformer on top of the GNN?**
The GNN only understands one snapshot of code. The Transformer understands **how code evolved over time** — it sees the sequence of snapshots and learns temporal patterns (e.g., "code usually gets refactored after 2-3 feature additions").

**Q: Is the bug label reliable?**
No — it's a noisy heuristic. If a commit message says "fix bug in parser", we label the *previous* commit as buggy. This is imprecise but good enough for a baseline model.
