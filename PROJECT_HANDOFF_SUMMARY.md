# CoT Faithfulness Project — Phase 3 Handoff Prompt

Paste this into your new chat along with ROADMAP.md.

---

## CONTEXT

You are helping me implement a research project one phase at a time. ROADMAP.md is the single source of truth.

**Project**: Chain-of-thought faithfulness benchmark — 150 instances (50 math/GSM8K, 50 commonsense/CommonsenseQA, 50 causal/LogiQA), 5 variants each (original + 4 corruptions), evaluated across 7 models.

**Repo**: `~/Desktop/cot-faithfulness-personal-ai` — GitHub: `Adibhaibvn/cot-faithfulness-personal-ai`  
**Venv**: `.venv`, Python 3.11. Keys in `.env` (never print/commit).

### Models
| Key | Provider | ID |
|-----|----------|----|
| `llama3.2-3b` | Ollama | `llama3.2:3b` |
| `llama3.1-8b-q4` | Ollama | `llama3.1:8b` |
| `llama3.1-8b-q8` | Ollama | `llama3.1:8b-instruct-q8_0` |
| `gemma2-9b` | Ollama | `gemma2:9b` |
| `llama3.1-8b-cloud` | OpenRouter | `meta-llama/llama-3.1-8b-instruct` |
| `llama3.3-70b-cloud` | OpenRouter | `meta-llama/llama-3.3-70b-instruct` |
| `gemini-flash` | Google AI Studio | `GEMINI_MODEL` from `.env` |

### Completed Phases
- **Phase 0** (§5): Environment verified. All 7 models respond.
- **Phase 1** (§6): 150 benchmark instances curated and saved in `data/benchmark/`. Gold CoTs generated with `llama3.3-70b-cloud` for commonsense and causal tasks.
- **Phase 2** (§7): 600 corruptions generated (4 per instance: `logic_error`, `step_deletion`, `conclusion_swap`, `nonsequitur`). Human audit passed (κ=0.821, agreement=97.8%). Data frozen with SHA-256 hashes, tagged `benchmark-v1`. **Never edit `data/benchmark/` or `data/corrupted/`.**

### Approved Deviations
1. LogiQA loaded from upstream GitHub (`data/raw/logiqa_train.json`) instead of HF `lucasmccabe/logiqa` (legacy script incompatible with `datasets>=5.0`).
2. `check()` in `generate.py` strips `Step \d+:\s*` before comparing lines (GSM8K chains are unnumbered but LLM output adds numbering).
3. `step_deletion` minimum lines set to 2 for math (3-step chains → 2 after deletion), 3 for MC tasks.

### Existing Code (already committed)
- `src/config.py`, `src/common.py`, `src/models/clients.py`, `src/models/check_setup.py`
- `src/dataset/curate.py`, `src/dataset/generate_gold_cots.py`
- `src/corruption/prompts.py`, `src/corruption/generate.py`, `src/corruption/repair_failures.py`
- `src/corruption/make_review_sheets.py`, `src/corruption/agreement.py`, `src/corruption/freeze.py`
- `src/evaluation/extraction.py` — already exists per ROADMAP §8.1 (neutral EVAL_SYSTEM_PROMPT + 3-layer extractor)

## HARD RULES
1. Follow ROADMAP.md exactly. Stop and tell me if something seems wrong.
2. Never change EVAL_SYSTEM_PROMPT, answer-comparison logic, or metric definitions without my approval.
3. Never edit `data/benchmark/`, `data/corrupted/`, or delete from `results/raw/`.
4. Ask before any command that calls a paid/rate-limited API or makes >10 model calls. Use `--limit` for tests.
5. Temperature 0, fixed seeds per ROADMAP. Cache every response to disk. All scripts resumable.
6. Project venv only. No sudo, no global installs, no unapproved deps.
7. Never print/log/commit API keys or `.env`.
8. Record provider name for every OpenRouter response.
9. Small steps: plan → approval → implement → smallest test → summarize → commit only after I approve.

## CURRENT TASK: Phase 3 — Evaluation (ROADMAP §8)

Implement Phase 3 in order:
1. Verify `src/evaluation/extraction.py` matches ROADMAP §8.1 exactly (it should already exist).
2. Create `src/evaluation/runner.py` per §8.2 (model as outermost loop, immediate disk append, resume by `(id, model, variant)` key).
3. Create `src/evaluation/make_pilot.py` per §8.3 (5 instances per task from corrupted files).
4. Run pilot with `--group local --pilot` on ONE model first (`--limit` or `--models llama3.2-3b`). Show me extraction failure rate and sample responses before full run.
5. I own all models (local + cloud). No division between Aditya/Varun.

Before starting, confirm: Python version, venv active, ROADMAP.md present, frozen data hashes verify, and existing evaluation code matches roadmap.

