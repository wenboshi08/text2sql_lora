#!/usr/bin/env python3
"""Generate the Colab-runnable notebook (text2sql_colab.ipynb, lightweight).

Design: source is NOT embedded in the notebook. Instead, run locally
    python tools/build_text2sql_zip.py   # -> text2sql.zip
then upload it in Colab via `google.colab.files.upload()`, extract to
/content/text2sql, and run the pipeline (M1 data -> M2 baseline -> M3 training
-> M4 eval) through `python -m` subprocess calls.

Note: after changing repo scripts, re-run build_text2sql_zip.py to repackage;
the notebook only needs regeneration when step text changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "text2sql_colab.ipynb"

sys.path.insert(0, str(ROOT))  # allow `from tools.build_text2sql_zip import ...`

# Must match tools/build_text2sql_zip.py
ZIP_RUNTIME_FILES = [
    "data/prep.py",
    "data/download_spider_dbs.py",
    "src/__init__.py",
    "src/prompt.py",
    "src/metrics.py",
    "src/judge.py",
    "src/harness.py",
    "src/baseline_eval.py",
    "src/train.py",
    "src/eval.py",
    "requirements.txt",
    "docs/TEXT2SQL_FINETUNE_DESIGN.md",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_cells() -> list:
    cells = []

    cells.append(md(f"""# Text-to-SQL Fine-Tuning Pipeline (Qwen2.5-Coder-7B + BIRD/Spider)

Runs M1→M4 on Colab: **data prep → baseline → QLoRA fine-tuning → evaluation**.

- Base model: `Qwen2.5-Coder-7B-Instruct` (locked); method: BIRD(evidence) + Spider
- Source is packaged separately: build `text2sql.zip` locally, then upload in this notebook
- Design doc: `docs/TEXT2SQL_FINETUNE_DESIGN.md` (included in the zip)

> Note: choose a **T4 GPU** runtime (free) or L4/A100 (Pro+). All files are
> session-scoped and disappear on reset; uncomment the Drive-mount cell to persist."""))

    cells.append(md("""## 0. Before you start (locally, once)

```bash
# Generate text2sql.zip from the repo root (contains data/, src/, requirements.txt, docs/)
python tools/build_text2sql_zip.py
```

Then prepare this notebook and `text2sql.zip` together (the zip is uploaded in step 3)."""))

    cells.append(md("""## 1. Run top-to-bottom

| Step | Command | Est. time |
|---|---|---|
| M1 data | `python data/prep.py` | ~5-10 min (downloads HF datasets) |
| M2 Spider DBs | `python data/download_spider_dbs.py` | ~1-3 min |
| M2 baseline | `python -m src.baseline_eval` | ~10 min for 50 samples; 1-2h for 500 (T4) |
| M3 training | `python -m src.train` | 1-2h full (A100) / several hours (T4) |
| M4 eval | `python -m src.eval` | same as baseline |

The **configuration** cell controls eval sample count, training smoke params, and the optional LLM semantic judge."""))

    cells.append(code("""# 2. Install dependencies (torch is preinstalled; training does not depend on TRL)
!pip install -q "datasets>=2.19" "sqlglot>=23" "transformers>=4.44" \\
               accelerate bitsandbytes gdown openai "peft>=0.12" unsloth
!python -c "import sqlglot, datasets; print('deps OK:', sqlglot.__version__, datasets.__version__)" """))

    cells.append(md("""## 3. Upload and extract the source package

Run `python tools/build_text2sql_zip.py` locally to produce `text2sql.zip`,
then run the cell below to upload it. After extraction the working directory is
switched to `/content/text2sql`."""))

    cells.append(code("""from google.colab import files
print('Upload text2sql.zip')
uploaded = files.upload()
import os, zipfile

os.makedirs('/content/text2sql', exist_ok=True)
for fn in uploaded.keys():
    with zipfile.ZipFile(fn) as z:
        z.extractall('/content/text2sql')

os.chdir('/content/text2sql')
print('cwd:', os.getcwd())
print('files:', sorted(os.listdir('.')))
# quick sanity check: required files present
missing = [f for f in ('data/prep.py', 'src/train.py', 'src/baseline_eval.py', 'requirements.txt')
           if not os.path.exists(f)]
assert not missing, f"zip missing required files: {missing}; repackage and re-upload" """))

    cells.append(code("""# 4. GPU check (must be a GPU runtime)
import torch
assert torch.cuda.is_available(), "Select a GPU runtime: Runtime -> Change runtime type -> T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))"""))

    cells.append(md("""## 5. Configuration

- `LIMIT_BASELINE`: baseline/eval sample count. **Use 50 for a smoke run, then 500 for real numbers**.
- `LIMIT_TRAIN` / `MAX_STEPS`: training smoke params (e.g. `LIMIT_TRAIN=200, MAX_STEPS=5`).
- `RUN_JUDGE`: whether to add the independent LLM semantic judge (needs a DeepSeek/Qwen API key)."""))

    cells.append(code("""# 5. Configuration
LIMIT_BASELINE = 50          # smoke: 50; real eval: 500 (~1-2h on T4)
LIMIT_TRAIN    = None        # training sample cap (None = full); smoke: 200
MAX_STEPS      = None        # early-stop steps (None = normal); smoke: 5
SKIP_TRAIN     = False       # True skips M3 training (data+baseline only)
RUN_JUDGE      = False       # whether to run the independent LLM semantic judge
JUDGE_API_KEY  = None        # DeepSeek key (or set OPENAI_API_KEY env var)
JUDGE_API_BASE = "https://api.deepseek.com"
JUDGE_MODEL    = "deepseek-chat"

if RUN_JUDGE and not JUDGE_API_KEY:
    raise ValueError("RUN_JUDGE=True but no JUDGE_API_KEY provided")"""))

    cells.append(md("## 6. M1 — Data prep (download/clean/dedupe/split/contamination check)\n\nProduces `data/processed/{train,val,test}.jsonl` + `meta.json`."))
    cells.append(code("""!python data/prep.py"""))

    cells.append(md("## 7. M2 — Download Spider databases (needed for execution eval)\n\nProduces `data/spider_database/`."))
    cells.append(code("""!python data/download_spider_dbs.py"""))

    cells.append(md(f"""## 8. M2 — Zero-shot baseline

Scores the **un-fine-tuned** model on the held-out set (Spider validation subset,
`LIMIT_BASELINE` samples). Results -> `results/baseline.json` (per-sample gold/pred
for M4 comparison)."""))
    cells.append(code("""import subprocess

cmd = ["python", "-m", "src.baseline_eval",
       "--db-root", "data/spider_database",
       "--limit", str(LIMIT_BASELINE)]
if RUN_JUDGE:
    cmd += ["--judge", "--judge-api-key", JUDGE_API_KEY,
            "--judge-api-base", JUDGE_API_BASE, "--judge-model", JUDGE_MODEL]
print("$", " ".join(cmd))
rc = subprocess.run(cmd).returncode
if rc != 0:
    print(f"command failed (exit {rc}) — scroll up for the full error")
    raise SystemExit(rc)"""))

    cells.append(md("## 9. M3 — QLoRA fine-tuning (Unsloth 4-bit + completion-only loss)\n\nProduces `outputs/lora/` (adapter + train_config.json + history.json)."))
    cells.append(code("""import subprocess

if not SKIP_TRAIN:
    cmd = ["python", "-m", "src.train"]
    if LIMIT_TRAIN: cmd += ["--limit", str(LIMIT_TRAIN)]
    if MAX_STEPS:   cmd += ["--max-steps", str(MAX_STEPS)]
    print("$", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"command failed (exit {rc}) — scroll up for the full error")
        raise SystemExit(rc)
else:
    print("SKIP_TRAIN=True; skipping training (M4 eval will not run)")"""))

    cells.append(md("## 10. M4 — Fine-tuned evaluation (same harness, same held-out as baseline)\n\nProduces `results/finetuned.json` + `results/comparison.json` (baseline/finetuned/delta)."))
    cells.append(code("""import subprocess

if not SKIP_TRAIN:
    cmd = ["python", "-m", "src.eval",
           "--adapter", "outputs/lora",
           "--db-root", "data/spider_database",
           "--limit", str(LIMIT_BASELINE)]
    if RUN_JUDGE:
        cmd += ["--judge", "--judge-api-key", JUDGE_API_KEY,
                "--judge-api-base", JUDGE_API_BASE, "--judge-model", JUDGE_MODEL]
    print("$", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"command failed (exit {rc}) — scroll up for the full error")
        raise SystemExit(rc)
else:
    print("SKIP_TRAIN=True; skipping eval")"""))

    cells.append(md("""## 11. Interpreting results

`results/comparison.json` structure:

```json
{
  "baseline":  {"exact_match": 0.04, "sql_validity": 0.99, "execution_accuracy": 0.12},
  "finetuned": {"exact_match": 0.60, "sql_validity": 0.99, "execution_accuracy": 0.48},
  "delta":     {"exact_match": 0.56, "sql_validity": 0.00, "execution_accuracy": 0.36}
}
```

- **exact_match**: lower bound (penalizes different-but-correct SQL);
- **execution_accuracy**: primary metric (identical result sets);
- delta > 0 means fine-tuning helped; focus on **execution_accuracy gains** (not EM).
- **semantic_equiv** appears only when RUN_JUDGE=True.

Next (M5): merge to 16-bit + GGUF quantization + push to HF Hub (`src/publish.py`, not yet implemented)."""))

    cells.append(code("""# 12. View results
import json, os

comp = "results/comparison.json"
if os.path.exists(comp):
    c = json.load(open(comp))
    print("=== comparison ===")
    for k, v in c["delta"].items():
        print(f"  {k:>18s}: {c['baseline'].get(k)} -> {c['finetuned'].get(k)}  (delta {v:+.4f})")
else:
    print("no comparison.json yet (M4 may not have run)")

base = "results/baseline.json"
if os.path.exists(base):
    b = json.load(open(base))
    print("\\n=== baseline sample examples ===")
    for s in b["samples"][:2]:
        print("- question:", s["question"][:80])
        print("  gold:", s["gold"][:100])
        print("  pred:", s["pred"][:100])"""))

    return cells


def main() -> None:
    # Ensure the zip manifest matches the packaging script.
    from tools.build_text2sql_zip import RUNTIME_FILES as ZIP_FILES
    assert set(ZIP_FILES) == set(ZIP_RUNTIME_FILES), "zip manifest and notebook are out of sync"

    nb = {
        "cells": build_cells(),
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"generated {OUT} ({len(nb['cells'])} cells, lightweight: source ships via zip)")


if __name__ == "__main__":
    main()
