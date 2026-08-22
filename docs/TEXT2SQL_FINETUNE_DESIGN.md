# Fine-Tuning a 7B Coder for Text-to-SQL — Design Document

> Version v1.2 · Goal: SFT a 7B-class coder model into a text-to-SQL specialist on Google Colab.
> Focus: **is there a better, more challenging dataset? a better reference approach?**

---

## 0. TL;DR (Decision Summary)

| Decision | Conclusion |
|---|---|
| Base model | **Locked to `Qwen2.5-Coder-7B-Instruct`** (an official `Qwen3-Coder-7B` does not exist; at 7B this is the de-facto standard for text-to-SQL). 30B-A3B is a later upgrade only. |
| Dataset | **BIRD (`bird23-train-filtered`) primary + Spider supplementary + synthetic data (SynSQL family)**; `text-to-sql-mix-v2` is dropped. Core levers = **evidence injection + synthetic-data SFT**. |
| Training | Unsloth + transformers.Trainer, **QLoRA r=16 all-linear, completion-only loss** (custom collator; no TRL dependency to avoid version drift). |
| References | `junmingg/Unsloth-Qwen2.5-Coder-7b-Text-to-SQL-SFT` is an excellent engineering skeleton (adopt its eval discipline); `SamirMasato/text-to-sql-TinyLlama` is a Colab starter only; stronger references: BIRD official finetuning example + SEED paper + Defog sql-eval. |
| Evaluation | Execution accuracy as primary, exact-match as lower bound, independent LLM for semantic equivalence; baseline-first + train/test contamination check. |

---

## 1. Goals and Non-Goals

**Goals**
1. Complete QLoRA fine-tuning within Colab free/Pro compute, producing a loadable LoRA adapter (plus merged 16-bit + GGUF).
2. Teach the model "schema + natural-language question → a single executable SQL", emphasizing **cross-DB generalization, multi-table JOINs, and external-knowledge reasoning** — not memorization.
3. Build a trustworthy evaluation loop (baseline vs fine-tuned, execution verification, contamination check) to avoid fooling ourselves.

**Non-goals (out of scope this round)**
- Conversational SQL (CoSQL), follow-up / clarification.
- SQL-injection safety, access control.
- Production serving (vLLM/TGI deployment, latency optimization).
- Full-parameter fine-tuning (infeasible and unnecessary for 7B on Colab).

---

## 2. Key Upfront Decision: Base Model (`Qwen3-Coder-7B` does not exist)

**Fact check**: the official `Qwen3-Coder` family has only two MoE sizes — **there is no 7B**:

| Model | Params | Active | Context | Notes |
|---|---|---|---|---|
| `Qwen3-Coder-480B-A35B-Instruct` | 480B | 35B | 256K | Flagship; infeasible on Colab |
| `Qwen3-Coder-30B-A3B-Instruct` | 30.5B | 3.3B | 256K | MoE; QLoRA feasible on A100 40GB |

> As of 2025-12 Qwen3-Coder remains these two sizes ([model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)). Community 7B derivatives exist but are not recommended as a base.

**Three candidate routes**:

| Route | Model | Colab HW | Verdict |
|---|---|---|---|
| **A (recommended)** | `Qwen2.5-Coder-7B-Instruct` | T4(free)/L4/A100 | Real 7B dense coder; mature Unsloth support; same as the junmingg reference — lowest risk |
| B (upgrade) | `Qwen3-Coder-30B-A3B-Instruct` | A100 40GB (Pro+) | Higher ceiling, but ~15–16GB at 4-bit; MoE QLoRA on Unsloth is limited; higher engineering risk |
| C (avoid) | `Qwen3-8B` (non-coder) | T4/L4 | Loses the coder base's SQL/code prior |

**Decision (locked)**: **Route A `Qwen2.5-Coder-7B-Instruct`**, reproducible and identical to the junmingg reference. Rationale: at 7B it is the de-facto text-to-SQL standard (SQL-R1 66.60 / OMNI-SQL 66.10 on BIRD dev EX), and [SLM-SQL](https://arxiv.org/html/2507.22478v1) shows that at this size **method (evidence + synthetic data + eval) beats swapping in a same-size base**. Route B (30B-A3B) is only a later upgrade with A100 access.

---

## 3. Dataset Strategy (core)

### 3.1 Why `text-to-sql-mix-v2` is unsuitable as the primary set

Sampling the dataset ([HF card](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2)) shows it concatenates a dozen sources: `sql-create-context`, `kaxap-llama2`, `nstext2sql`(sede/wikisql/sql_create_context), `gretel-synthetic`, `motherduck-duckdb`, `pipable-spider-bird`, `clinton-text2sql`, etc. Problems:

1. **Label noise / case-bleed**: e.g. `SELECT COUNTry_iso_code FROM COUNTry WHERE COUNTry_name = 'Kyrgyz Republic'` (`COUNT` fused into `country`) — the model learns bad token patterns.
2. **Inconsistent string quoting**: some use double quotes for strings (`bush__percentage = "78.40%"`), some single quotes; in SQLite double quotes denote identifiers — harmful.
3. **Number/string type confusion**: `"Base Pairs" = '4,895,836'` (comma-formatted number as a string).
4. **Mixed dialects without a marker**: sqlite/duckdb/postgres/generic interleaved (even DuckDB `STRUCT`/`FILTER` syntax) but no dialect field fed to the model → contradictory learning signal.
5. **Malformed questions**: `"What average has 1 as rhe rank?"`, `"-- Which track has a Japanese title of?"`.
6. **Skewed difficulty**: mostly single-table/single-value queries (WikiSQL/sql-create-context derived); few hard multi-table JOIN/nested/aggregate samples → low ceiling.
7. **Cross-source duplication**: `sql-create-context` and `ns-sql_create_context` overlap heavily → oversampling.

**Conclusion**: usable only as "cleaned supplementary breadth", **not** as the SFT backbone — its ceiling is "single-table queries + learned bad formatting conventions".

### 3.2 Better, more challenging datasets compared

| Dataset | Size | Difficulty / traits | Has evidence | Suggested use |
|---|---|---|---|---|
| **BIRD** ([bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered), official filtered 6,601/9,428 ≈70%) | ~9.4k train / 1,534 dev / 1,789 test, 95+ real DBs | Cross-domain, real "dirty" DBs, multi-table JOIN, **external-knowledge reasoning** (evidence field), unit/time conversions | ✅ yes | **Primary training set (first choice)** |
| **Spider** ([Yale](https://yale-lily.github.io/spider)) | 10,181 questions / 200 DBs / 138 domains | Classic cross-domain benchmark, easy–extra hard | ❌ no | Breadth supplement; dev for early validation |
| **BIRD-Critic 1.0 / SWE-SQL** ([NeurIPS 2025](https://github.com/bird-bench/BIRD-CRITIC-1)) | 600 tasks ×4 dialects (open) | Diagnose/fix SQL defects; reasoning-heavy | ✅ yes | Advanced (hard for 7B; stretch goal) |
| **LiveSQLBench** ([livesqlbench.ai](https://livesqlbench.ai/)) | Base-Lite 270 / Base-Full 600 | Contamination-free, full SQL spectrum, HKB + test cases; o3-mini only 44.81 | ✅ yes | **Eval only** (too small to train) |
| **Spider robustness variants** (Dr.Spider / Spider-Syn / Spider-Realistic) | thousands | Perturbation, paraphrase, counterfactual | ❌ no | Optional robustness |
| **gretelai/synthetic_text_to_sql** (synthetic) | ~100k | Clean, uniform, broad schema coverage | ❌ no | Optional breadth (much cleaner than mix-v2) |
| **Defog sql-eval** ([defog-ai/sql-eval](https://github.com/defog-ai/sql-eval)) | ~175 hand-labeled | Real enterprise schemas, **execution-level** | ✅ partial hints | **Eval set (execution accuracy)** |

> In one line: **BIRD is the current gold standard for text-to-SQL fine-tuning** — the team even released `bird23-train-filtered` as a "drop-in finetuning replacement". Its `evidence` field (which spells out non-standard column names / units / semantics) is the key signal that lifts a 7B model from ~24% (Llama3-8b on BIRD Mini-Dev) toward 50%+.

### 3.3 Recommended data mix (train / dev / test)

| Split | Data | Size | Notes |
|---|---|---|---|
| **train** | `bird23-train-filtered` (with evidence) + Spider train + **synthetic (SynSQL family)** | ~6.6k + ~7k + controllable | BIRD teaches external-knowledge reasoning; Spider/synthetic add breadth + format consistency |
| **dev** (tuning/early-stop) | BIRD dev subset or Spider dev | 500–1,000 | val loss + early stopping |
| **held-out eval** | Defog sql-eval + remaining BIRD dev | 175 + ~500 | strict train/eval isolation |

**Optional third tier (breadth, cleaned)**: `gretelai/synthetic_text_to_sql` (clean synthetic) or a subset of `text-to-sql-mix-v2` after "sqlglot parse filter + dedup + dialect unification + quote normalization". **Do not mix in the raw mix-v2**.

**Cleaning pipeline (implemented in code)**:
1. Parse every gold SQL with `sqlglot`; **drop unparseable/obviously-invalid samples** (junmingg measured 0.32% noise already affects validity).
2. **Unify dialect**: train on a single dialect (SQLite); drop cross-dialect samples or add a dialect prefix.
3. **Dedupe** at two levels: `(schema, question)` and `(schema, question, sql)`.
4. **Length filter**: drop/truncate overly long schemas beyond `max_seq_length` (e.g. SEDE-style 20+ table schemas).
5. **Dedupe before splitting**: global dedup first, then split, to avoid train/test leakage.

### 3.4 Two primary levers: evidence injection + synthetic data (this round's focus)

Once we commit to 7B, most of the gain comes from method, not from swapping the base. Two levers:

**Lever 1 — evidence injection (the soul of BIRD)**
- BIRD's `evidence` field spells out non-standard column names / unit conversions / semantic ambiguity (e.g. `list with the most movies refers to MAX(list_movie_number)`).
- **Feed real evidence at training time** so the model learns "read external knowledge before writing SQL"; at inference, evidence comes from one of: retrieval/rule-based generation, a small-model generator (SEED idea), or omission.
- Reference: [SEED](https://arxiv.org/html/2506.07423v1) lifts CodeS-7B from 41.92 → 56.58 (BIRD dev) via automatic evidence generation.

**Lever 2 — synthetic-data SFT (breadth + CoT format)**
- Use **SynSQL-2.5M** (OmniSQL; includes Spider+BIRD with CoT annotations) or its cleaned **SynSQL-Think-916K** ([SLM-SQL](https://arxiv.org/html/2507.22478v1) heuristics: drop invalid SQL, drop duplicate CoT, ≤7k tokens, wrap in `<think>…</think>` + `<answer>…</answer>`).
- Purpose: real data (BIRD ~6.6k) is small; synthetic data supplies large-scale, format-uniform, difficulty-layered supplement; the SFT module contributed +21.93 (0.5B) in SLM-SQL.
- **Plan**: first BIRD+Spider real-data SFT; append synthetic data in a second phase (if introducing `<think>` CoT, update the template accordingly, see §4).

---

## 4. Prompt Template

**completion-only loss** (mask the prompt, compute loss only on the SQL answer). Template:

```
<|im_start|>system
You are a text-to-SQL assistant. Given a database schema and a question,
output a single valid SQLite query and nothing else.<|im_end|>
<|im_start|>user
### Database schema:
CREATE TABLE ... ; CREATE TABLE ... ;

### External knowledge (evidence):
<BIRD evidence field, or "None">

### Question:
<question>

### SQL:
<|im_end|>
<|im_start|>assistant
SELECT ...<|im_end|>
```

Key points:
- **Always feed `evidence`** — it is the core value of BIRD data; real evidence at training, retrieval/rule or small-model generation at inference (SEED idea, §7).
- **Response format**: require "SQL only, no explanation"; strip fences/extra text via `clean_sql()` at decode.
- **Single dialect**: the template says SQLite, matching the training data.
- The prompt builder is the single global source (`build_messages()`); baseline and fine-tuned use **exactly** the same template and decoding parameters.

---

## 5. Training Recipe

**Method**: QLoRA (4-bit NF4 + double quant) + Unsloth + transformers.Trainer, completion-only loss.
Custom collator: use the ChatML `<|im_start|>assistant` anchor; mask everything before it to -100 (loss only on the SQL answer).
No TRL — `DataCollatorForCompletionOnlyLM` was removed from recent transformers and TRL's SFTTrainer API drifts; the manual mask is version-stable.

```python
# Core config (aligned with junmingg's verified recipe, Qwen2.5-Coder-7B)
LoraConfig(
    task_type="CAUSAL_LM",
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",          # all linear layers; <1% trainable at 7B
    bias="none",
)
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16,
                   bnb_4bit_use_double_quant=True)
TrainingArguments(
    num_train_epochs=1,                   # 1 epoch usually saturates (~13k data)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch=16
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, optim="adamw_8bit",
    max_seq_length=2048,                  # separate truncation for long schemas
    logging_steps=10, eval_strategy="steps", eval_steps=100,
    save_steps=200, load_best_model_at_end=True,
)
```

**Colab hardware matrix (QLoRA, Qwen2.5-Coder-7B)**:

| GPU | VRAM | Workable config | Expectation |
|---|---|---|---|
| T4 (free) | 16GB | batch=1–2 + grad accum=8–16, max_seq=1024–1536 | Runs but slow, sequence-limited |
| L4 (Pro) | 22GB | batch=4, max_seq=2048 | Comfortable |
| A100 (Pro+) | 40GB | batch=4–8, or route B (30B-A3B) | Fastest / upgradeable |

**Must obey (fine-tuning-expert constraints)**: clean data before training; use PEFT (no full-param for 7B); monitor train/val loss; always use LR warmup; version data and checkpoints.

---

## 6. Evaluation (junmingg's "hard-to-fake" discipline)

1. **Baseline first**: score the un-fine-tuned base on the same 500-sample held-out set; otherwise "how much did fine-tuning help" is unfalsifiable.
2. **Three metrics**:
   - **Exact match** (lower bound): normalized/string match, penalizes different-but-correct SQL.
   - **Execution accuracy** (primary): execute on real SQLite via sqlglot/SQLite and compare result sets (Defog sql-eval `compare_df` idea). The most trustworthy text-to-SQL metric.
   - **Semantic equivalence** (fairness): an independent LLM (GLM/Claude, **not the model under test**) judges equivalence, avoiding self-preference.
   - Plus **SQL validity**: whether the prediction parses.
3. **Contamination check**: verify held-out `(schema, question)` and full triples do not appear in the train set.
4. **BIRD official scoring**: stretch goal — use the [official BIRD scoring script](https://github.com/bird-bench/mini_dev) (execution + exact match) on dev to compare against the leaderboard (7B baselines ~24–44).
5. **(Optional, inference-time) self-consistency voting**: sample N SQL per question, group by execution result, rewrite the minority (SLM-SQL corrective self-consistency, ~+3–5 points). Not a training item.

---

## 7. Reference Approaches (which is better)

| Reference | Role | Verdict |
|---|---|---|
| [junmingg/Unsloth-Qwen2.5-Coder-7b-Text-to-SQL-SFT](https://github.com/junmingg/Unsloth-Qwen2.5-Coder-7b-Text-to-SQL-SFT) | Complete engineering | **Strongly recommended as the primary template**. Value is the eval discipline (baseline-first, contamination check, independent judge, noise ablation): EM 3.8%→78.8%, semantic equiv 67%→86.2% (single-table data). Limitation: uses `sql-create-context` (single-table) — low ceiling; **swap in BIRD and it becomes this project** |
| [SamirMasato/text-to-sql-TinyLlama](https://github.com/SamirMasato/text-to-sql-TinyLlama) | Starter prototype | Colab/T4 onboarding only (TinyLlama 1.1B, LoRA+4bit+SFTTrainer). Too small, no eval — not a template |
| [BIRD official finetuning example](https://github.com/bird-bench/mini_dev/tree/main/finetuning) | Official data recipe | **Must-read**. The canonical BIRD training format; the official way to use bird23-train-filtered |
| [SEED](https://arxiv.org/html/2506.07423v1) (auto evidence generation) | Method | Advanced: auto-generate evidence then SFT; CodeS-7B 41.92→56.58 (BIRD dev). Maps to our evidence-guided design |
| [SLM-SQL](https://arxiv.org/html/2507.22478v1) (synthetic + SFT/RL + self-consistency) | Method | Evidence that **method > base swap**: 0.5B–1.5B + SynSQL SFT hits 56.9–67.1 EX (BIRD dev), 1.5B beats bare 7B/15B. Borrow its SynSQL-Think cleaning and `<think>/<answer>` template |
| [defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) | Eval harness | **Adopt**. Execution-level, real enterprise schemas |

**Recommended combo (locked)**: *junmingg's engineering skeleton + BIRD(evidence) with official finetuning format + SynSQL synthetic data (SLM-SQL cleaning) + Defog sql-eval execution eval + SEED's evidence-generation idea*. This is a tier above "copy a Colab notebook".

---

## 8. Colab Implementation Plan (milestones)

| Phase | Content | Deliverable | Acceptance |
|---|---|---|---|
| M1 data | Download bird23-train-filtered + Spider (+ optional SynSQL subset); clean (sqlglot filter/dedup/dialect); split per §3.3 | `data/prep.py`, train/dev/test jsonl | every gold parses; train/test zero leakage |
| M2 baseline | Zero-shot generation of un-fine-tuned Qwen2.5-Coder-7B on held-out | `results/baseline.json` | has EM/exec/semantic metrics |
| M3 training | Unsloth QLoRA SFT, completion-only | LoRA adapter + checkpoint | val loss decreases, no overfit; ~1–2h (A100) |
| M4 eval | Same harness + independent judge on baseline vs fine-tuned | `results/finetuned.json`, comparison | all three metrics beat baseline |
| M5 publish | merge 16-bit + GGUF + push HF Hub | merged model + GGUF + Model Card | loadable via `load_dataset`/ollama |

**Colab notebook skeleton**: follow junmingg's modular `src/` structure (`data.py / baseline_eval.py / train.py / eval.py / judge.py / push.py`), adapted to a single notebook or "notebook calls src modules". On first run, verify GPU and 4-bit loading.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| T4 out of memory | lower batch/seq, compensate with grad accum; upgrade to L4/A100 |
| BIRD data needs real DBs | `bird23-train-filtered` is already question+evidence+SQL text; training does **not** need DB files; only execution eval needs BIRD dev DBs or Defog sql-eval DBs |
| Overfitting BIRD naming/units | evidence field carries external knowledge; Spider adds breadth; held-out uses a different source (Defog) to test generalization |
| Learning format, not semantics | use execution accuracy + semantic equivalence (not EM) as primary, avoiding quote/case-style bias |
| Contamination inflates scores | global dedup before split + explicit contamination-check script |
| Route B (30B-A3B) Unsloth support | if choosing route B, validate 4-bit loading + LoRA trainability in small steps first; fall back to route A on failure |

---

## 10. Deliverables

1. `docs/TEXT2SQL_FINETUNE_DESIGN.md` (this document)
2. `data/prep.py` — download/clean/split script (sqlglot validation + dedup + contamination check)
3. `src/{prompt,metrics,judge,harness,baseline_eval,train,eval}.py` — modular pipeline (ported from junmingg)
4. `text2sql_colab.ipynb` — one-click Colab entry point (generated by `tools/build_colab_notebook.py`)
5. `results/` — baseline.json / finetuned.json / comparison
6. HF Hub: LoRA adapter + merged 16-bit + GGUF + Model Card

---

## Appendix: Key Links

- Base models: [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) · [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- Datasets: [bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered) · BIRD schema mirror [xu3kev/BIRD-SQL-data-train](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train) · [BIRD](https://bird-bench.github.io/) · [Spider](https://yale-lily.github.io/spider) · Spider schema mirror [SuperMax991/spider-text2sql](https://huggingface.co/datasets/SuperMax991/spider-text2sql) · SynSQL-2.5M (via [SLM-SQL code](https://github.com/CycloneBoy/slm_sql)) · [text-to-sql-mix-v2](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2)
- References: [junmingg SFT](https://github.com/junmingg/Unsloth-Qwen2.5-Coder-7b-Text-to-SQL-SFT) · [TinyLlama prototype](https://github.com/SamirMasato/text-to-sql-TinyLlama) · [BIRD finetuning example](https://github.com/bird-bench/mini_dev/tree/main/finetuning) · [SEED paper](https://arxiv.org/html/2506.07423v1) · [SLM-SQL paper](https://arxiv.org/html/2507.22478v1) · [Defog sql-eval](https://github.com/defog-ai/sql-eval)
