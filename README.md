# Text-to-SQL Fine-Tuning Pipeline

Fine-tune **Qwen2.5-Coder-7B-Instruct** for text-to-SQL on Google Colab using BIRD + Spider datasets, QLoRA, and execution-based evaluation.

## Quick Start (Google Colab)

1. **Generate the runtime zip** (on your local machine):
   ```bash
   python tools/build_text2sql_zip.py
   ```

2. **Open `text2sql_colab.ipynb`** in [Google Colab](https://colab.research.google.com/), select **T4 GPU** runtime.

3. **Run all cells** top-to-bottom:
   - Upload `text2sql.zip` when prompted
   - M1: Prepare data (~5–10 min)
   - M2: Download Spider DBs + run baseline (~10 min for 50 samples)
   - M3: QLoRA fine-tuning (~1–2h on A100, several hours on T4)
   - M4: Evaluate fine-tuned model + compare with baseline

See the notebook for configurable parameters (`LIMIT_BASELINE`, `RUN_JUDGE`, etc.).

## Architecture

```
data/
├── prep.py                  # M1: download, clean, dedupe, split BIRD + Spider
└── download_spider_dbs.py   # M2: fetch SQLite databases for execution eval

src/
├── prompt.py                # Single source of truth for chat templates
├── metrics.py               # EM, SQL validity, execution accuracy
├── judge.py                 # LLM-based semantic equivalence judge
├── harness.py               # Shared eval harness (baseline & finetuned use same path)
├── baseline_eval.py         # M2: zero-shot baseline on held-out set
├── train.py                 # M3: Unsloth QLoRA SFT with completion-only loss
└── eval.py                  # M4: evaluate LoRA adapter + produce comparison

tools/
├── build_text2sql_zip.py    # Package runtime files into text2sql.zip
└── build_colab_notebook.py  # Generate text2sql_colab.ipynb from source
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Base model**: Qwen2.5-Coder-7B-Instruct | Fact standard for 7B text-to-SQL; mature Unsloth support |
| **Datasets**: BIRD (filtered) + Spider | BIRD provides external-knowledge reasoning (evidence field); Spider adds breadth |
| **Training**: Unsloth 4-bit QLoRA + custom completion-only collator | No TRL dependency (avoids API drift); stable across transformers versions |
| **Evaluation**: Execution accuracy > exact match | Execution is the gold standard; EM penalizes different-but-correct SQL |

For full details see [`docs/TEXT2SQL_FINETUNE_DESIGN.md`](docs/TEXT2SQL_FINETUNE_DESIGN.md).

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `datasets`, `sqlglot`, `transformers>=4.44`, `torch`, `accelerate`, `bitsandbytes`, `peft`, `unsloth`, `gdown`, `openai`.

## Project Structure

| Path | Description |
|---|---|
| `data/processed/` | Generated train/val/test JSONL files |
| `data/spider_database/` | Spider SQLite databases (for execution eval) |
| `outputs/lora/` | Trained LoRA adapter + config + training history |
| `results/` | Baseline and finetuned evaluation results + comparison |

## License

Apache-2.0 (Qwen models); CC-BY-SA-4.0 (Spider dataset); check individual dataset licenses.
