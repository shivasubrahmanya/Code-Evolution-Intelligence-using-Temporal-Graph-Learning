"""
app/demo.py
===========
Phase 8 - Streamlit Demo App

Interface:
  - Input  : GitHub repo link (cloned locally)
  - Output : Predicted next change type + bug probability
  - Visuals: Commit sequence, graph sizes, confidence bar

Run:
  streamlit run app/demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import streamlit as st

# -- Page config ---------------------------------------------------------
st.set_page_config(
    page_title  = "Code Evolution Intelligence",
    page_icon   = "[AI]",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# -- Imports (after page config) ------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multitask_model import CodeEvolutionModel, CHANGE_LABELS, CHANGE_TO_IDX
from scripts.parse_ast      import parse_code, TREE_SITTER_OK
from scripts.build_graphs   import ast_to_graph, VOCAB
from utils.graph_utils      import dict_to_pyg
from torch_geometric.data   import Batch


# ------------------------------------------------------------------------
#  Styling
# ------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white; text-align: center;
  }
  .metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 1.5rem; border-radius: 12px; color: white;
    text-align: center; margin: 0.5rem 0;
  }
  .bug-safe  { background: linear-gradient(135deg, #11998e, #38ef7d); }
  .bug-warn  { background: linear-gradient(135deg, #f7971e, #ffd200); }
  .bug-risky { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
  .commit-card {
    background: #1e1e2e; border-radius: 10px; padding: 1rem;
    border-left: 4px solid #667eea; margin: 0.4rem 0; color: #cdd6f4;
  }
  .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANGE_EMOJI = {"ADD": "➕", "DELETE": "➖", "MODIFY": "✏️"}
CHANGE_COLOR = {"ADD": "#38ef7d", "DELETE": "#ef473a", "MODIFY": "#ffd200"}


@st.cache_resource
def load_model(checkpoint: str, vocab_size: int) -> CodeEvolutionModel:
    m = CodeEvolutionModel(vocab_size=vocab_size).to(DEVICE)
    m.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    m.eval()
    return m


def get_vocab_size() -> int:
    vp = ROOT / "data" / "processed" / "vocab.json"
    if vp.exists():
        return len(json.load(open(vp))) + 10
    return 512


def code_to_graph(code: str) -> dict | None:
    return ast_to_graph(code)


def graph_to_embedding(g: dict, gnn) -> torch.Tensor:
    data  = dict_to_pyg(g).to(DEVICE)
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        return gnn(batch.x, batch.edge_index, batch.batch)  # [1, D]


def predict(graphs: list[dict], model: CodeEvolutionModel):
    """Run a list of W graphs through the model and return predictions."""
    embs = []
    for g in graphs:
        embs.append(graph_to_embedding(g, model.gnn))   # [1, D]
    seq = torch.stack(embs, dim=1)                       # [1, W, D]
    with torch.no_grad():
        c_log, b_log = model(seq)
    import torch.nn.functional as F
    change_probs = F.softmax(c_log, dim=-1).squeeze().tolist()
    bug_prob     = torch.sigmoid(b_log).item()
    return change_probs, bug_prob


def clone_repo(url: str, target_dir: Path) -> bool:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "50", url, str(target_dir)],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except Exception as e:
        st.error(f"Clone failed: {e}")
        return False


def extract_recent_python_files(repo_path: Path, n: int = 10) -> list[dict]:
    """Get the last n modified Python files with before/after content."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--no-merges", "--format=%H", f"-n{n+5}"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=20,
        )
        hashes = result.stdout.strip().splitlines()
    except Exception:
        return []

    records = []
    for i in range(len(hashes) - 1):
        curr, prev = hashes[i], hashes[i+1]
        try:
            diff_r = subprocess.run(
                ["git", "diff", "--name-only", prev, curr],
                cwd=str(repo_path), capture_output=True, text=True, timeout=10,
            )
            py_files = [f for f in diff_r.stdout.strip().splitlines() if f.endswith(".py")]
            if not py_files:
                continue

            def get_file(h, fp):
                r = subprocess.run(["git", "show", f"{h}:{fp}"],
                                   cwd=str(repo_path), capture_output=True, text=True)
                return r.stdout

            before = get_file(prev, py_files[0])
            after  = get_file(curr, py_files[0])
            if before and after:
                msg_r = subprocess.run(
                    ["git", "log", "--format=%s", "-n1", curr],
                    cwd=str(repo_path), capture_output=True, text=True,
                )
                records.append({
                    "commit_id": curr[:8],
                    "message":   msg_r.stdout.strip(),
                    "before":    before,
                    "after":     after,
                    "file":      py_files[0],
                })
            if len(records) >= n:
                break
        except Exception:
            continue
    return records


# ------------------------------------------------------------------------
#  UI
# ------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
  <h1>[AI] Code Evolution Intelligence</h1>
  <p style="opacity:0.8">Temporal Graph Learning for Code Change Prediction & Bug Detection</p>
</div>
""", unsafe_allow_html=True)

# -- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    checkpoint = st.text_input(
        "Model checkpoint", value="outputs/checkpoints/best_model.pt"
    )
    window_size = st.slider("Context window (commits)", 2, 6, 3)
    use_demo    = st.checkbox("Use demo mode (no real repo needed)", value=True)

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("- GNN: 2× GCNConv + pool")
    st.markdown("- Transformer: 2 layers, 4 heads")
    st.markdown("- Labels: ADD / DELETE / MODIFY")
    st.markdown("- Bug: binary heuristic")

# -- Main content ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("[IN] Input")
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repo",
    )

    analyze = st.button("[RUN] Analyze Repository", type="primary", use_container_width=True)


# -- Demo mode data --------------------------------------------------------
DEMO_CODE_SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "def add(a, b, c=0):\n    \"\"\"Add numbers.\"\"\"\n    return a + b + c\n",
    "class Calculator:\n    def add(self, a, b):\n        return a + b\n",
    "class Calculator:\n    def add(self, *args):\n        return sum(args)\n    def mul(self, a, b):\n        return a * b\n",
]

if analyze or use_demo:
    if use_demo:
        st.info("🎭 **Demo mode** - using synthetic code snippets (no repo required)")
        file_records = [
            {"commit_id": f"demo{i:02d}", "message": msg, "file": "calc.py",
             "before": DEMO_CODE_SNIPPETS[i], "after": DEMO_CODE_SNIPPETS[i+1]}
            for i, msg in enumerate(["add default arg", "refactor to class", "add mul method"])
        ]
    elif repo_url:
        target = ROOT / "data" / "raw" / "repos" / repo_url.rstrip("/").split("/")[-1]
        if not target.exists():
            with st.spinner("Cloning repository..."):
                ok = clone_repo(repo_url, target)
            if not ok:
                st.stop()
        with st.spinner("Extracting commits..."):
            file_records = extract_recent_python_files(target, n=window_size + 2)
        if len(file_records) < window_size:
            st.warning(f"Only {len(file_records)} Python commits found (need {window_size}).")
            st.stop()
    else:
        st.warning("Enter a repo URL or enable demo mode.")
        st.stop()

    # -- Parse graphs ----------------------------------------------------
    graphs = []
    for rec in file_records:
        g = code_to_graph(rec["after"])
        if g:
            graphs.append((rec, g))

    if len(graphs) < window_size:
        st.error(f"Only {len(graphs)} parseable graphs - need at least {window_size}.")
        st.stop()

    window_graphs   = [g for _, g in graphs[-window_size:]]
    window_records  = [r for r, _ in graphs[-window_size:]]

    # -- Load model ------------------------------------------------------
    cp_path = ROOT / checkpoint
    model_loaded = cp_path.exists()
    if model_loaded:
        try:
            model        = load_model(str(cp_path), get_vocab_size())
            change_probs, bug_prob = predict(window_graphs, model)
            top_idx      = int(torch.tensor(change_probs).argmax())
            top_label    = CHANGE_LABELS[top_idx]
        except Exception as e:
            st.warning(f"Model inference failed: {e}. Showing graph stats only.")
            model_loaded = False

    # -- Display commit timeline ------------------------------------------
    with col_left:
        st.subheader("[LOG] Commit Sequence")
        for i, (rec, g) in enumerate(zip(window_records, window_graphs)):
            marker = "🎯 (latest)" if i == len(window_records)-1 else f"#{i+1}"
            st.markdown(f"""
            <div class="commit-card">
              <b>{marker}</b> &nbsp; <code>{rec['commit_id']}</code><br>
              [NOTE] {rec['message']}<br>
              📁 {rec['file']} &nbsp;|&nbsp; 🌳 {g['num_nodes']} nodes, {g['num_edges']} edges
            </div>""", unsafe_allow_html=True)

    # -- Predictions -----------------------------------------------------
    with col_right:
        st.subheader("[PREDICT] Predictions")

        if model_loaded:
            # Change type
            st.markdown("**Next Change Type**")
            for i, (lbl, prob) in enumerate(zip(CHANGE_LABELS, change_probs)):
                emoji = CHANGE_EMOJI[lbl]
                color = CHANGE_COLOR[lbl]
                badge = " ← **predicted**" if i == top_idx else ""
                st.markdown(f"{emoji} `{lbl}`{badge}")
                st.progress(prob)
                st.caption(f"{prob*100:.1f}%")

            # Bug probability
            st.markdown("---")
            st.markdown("**Bug Probability**")
            bug_class = "bug-safe" if bug_prob < 0.3 else ("bug-warn" if bug_prob < 0.6 else "bug-risky")
            bug_emoji = "[OK]" if bug_prob < 0.3 else ("[WARN]" if bug_prob < 0.6 else "🚨")
            st.markdown(f"""
            <div class="metric-card {bug_class}">
              <h2>{bug_emoji} {bug_prob*100:.1f}%</h2>
              <p>Bug Probability</p>
            </div>""", unsafe_allow_html=True)
            st.progress(bug_prob)
        else:
            st.info("[FIX] No trained model found at the checkpoint path.\nTrain first with `scripts/train.py`.")

        # Graph size visualization
        st.markdown("---")
        st.subheader("[CHART] Graph Size Evolution")
        import pandas as pd
        df = pd.DataFrame([
            {"commit": r["commit_id"], "nodes": g["num_nodes"], "edges": g["num_edges"]}
            for r, g in zip(window_records, window_graphs)
        ])
        st.line_chart(df.set_index("commit")[["nodes", "edges"]])
