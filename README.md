# Chain-of-Thought Faithfulness Under Quantization and Scaling

A controlled empirical benchmark studying whether language models' stated reasoning causally determines their final answers across model sizes (3B to 70B), quantization levels (Q4, Q8, FP16/BF16), and deployment contexts (local personal AI vs. cloud APIs).

---

## 📌 Research Overview

Chain-of-thought (CoT) prompting is widely treated as an explainability and trust mechanism. However, prior research (Turpin et al., Lanham et al.) focused predominantly on large, cloud-hosted models. This project investigates the causal link between reasoning and conclusions across the compression spectrum required for on-device "personal AI":

1. **RQ1 (Personal AI & Size Scaling)**: Do small, locally deployable models (3B–9B) reason faithfully?
2. **RQ2 (Quantization Effect)**: Does post-training quantization (Q4 vs. Q8 vs. unquantized) systematically degrade reasoning faithfulness for the same base architecture (Llama-3.1-8B)?
3. **RQ3 (Deployment Context Gap)**: Is there a measurable faithfulness disparity between locally run personal models and cloud-hosted systems?
4. **RQ4 (Cross-Task Consistency)**: How does CoT faithfulness vary across mathematical, commonsense, and causal deductive reasoning?

---

## 🔬 Benchmark Design

The benchmark consists of **150 curated instances** with **5 evaluation variants** each (**750 total evaluation instances per model**):

- **Mathematical Reasoning**: 50 instances from `GSM8K` (human-authored multi-step solutions).
- **Commonsense Reasoning**: 50 instances from `CommonsenseQA` (step-by-step reasoning generated with `Llama-3.3-70B`).
- **Causal / Deductive Reasoning**: 50 instances from `LogiQA` (deductive multi-hop chains generated with `Llama-3.3-70B`).

### Corruption Taxonomy

For every instance, 4 controlled perturbations are applied to the reasoning chain:
1. `logic_error` *(Core)*: Introduces one subtle error in an intermediate step and minimally rewrites downstream steps so the chain implies an incorrect answer.
2. `step_deletion` *(Core)*: Deletes one critical load-bearing reasoning step while keeping the final answer line intact.
3. `conclusion_swap` *(Core, Rule-Based)*: Keeps all intermediate reasoning steps untouched, swapping only the conclusion line to an incorrect alternative.
4. `nonsequitur` *(Control)*: Inserts a plausible but logically irrelevant fact into the middle of the chain.

---

## 🤖 Model Lineup (7 Models, $0 Compute Budget)

| Key | Runtime / Provider | Architecture & Quantization | Role |
| :--- | :--- | :--- | :--- |
| `llama3.2-3b` | Ollama (Local) | Llama-3.2-3B (Q4_K_M) | Size scaling (small end) |
| `llama3.1-8b-q4` | Ollama (Local) | Llama-3.1-8B (Q4_K_M) | Size scaling & quant anchor |
| `llama3.1-8b-q8` | Ollama (Local) | Llama-3.1-8B (Q8_0) | Quantization comparison |
| `gemma2-9b` | Ollama (Local) | Gemma-2-9B (Q4_K_M) | Architecture diversity (local) |
| `llama3.1-8b-cloud` | OpenRouter (Cloud) | Meta Llama-3.1-8B (BF16/FP16) | Quantization comparison (cloud) |
| `llama3.3-70b-cloud` | OpenRouter (Cloud) | Meta Llama-3.3-70B-Instruct | Size scaling (large end) |
| `gemini-flash` | Google AI Studio (Cloud) | Gemini-1.5-Flash | Cloud architecture diversity |

---

## 📂 Repository Structure

```
cot-faithfulness-personal-ai/
├── data/
│   ├── raw/                 # Downloaded raw dataset files (gitignored)
│   ├── benchmark/           # 150 curated benchmark instances
│   │   ├── math.jsonl
│   │   ├── commonsense.jsonl
│   │   └── causal.jsonl
│   ├── corrupted/           # 600 corrupted variants (4 per instance)
│   └── FROZEN_SHA256.txt    # SHA-256 integrity hashes of frozen benchmark
├── src/
│   ├── common.py            # Shared utility functions (math normalization, CoT parsing)
│   ├── config.py            # Model configurations, pacing, and task definitions
│   ├── dataset/
│   │   ├── curate.py             # Phase 1: Instance candidate filtering
│   │   └── generate_gold_cots.py # Phase 1: High-quality reasoning generation
│   ├── corruption/
│   │   ├── prompts.py            # Phase 2: Perturbation prompt templates
│   │   ├── generate.py           # Phase 2: Corruption engine with structural validation
│   │   ├── make_review_sheets.py # Phase 2: Audit sampling for inter-rater reliability
│   │   ├── agreement.py          # Phase 2: Cohen's kappa agreement evaluator
│   │   └── freeze.py             # Phase 2: Benchmark hashing and freezing
│   ├── models/
│   │   ├── clients.py       # Unified interface for Ollama, OpenRouter, and Gemini
│   │   └── check_setup.py   # Pre-flight environment and quota validator
│   ├── evaluation/
│   │   ├── extraction.py    # 3-layer answer extractor & neutral eval prompt
│   │   └── runner.py        # Outermost-model evaluation runner with checkpointing
│   └── analysis/
│       ├── metrics.py       # Faithfulness Score (FS) and Follow-rate computation
│       ├── stats.py         # Bootstrap 95% CIs and exact McNemar testing
│       └── figures.py       # Publication-ready figures and heatmaps
├── review/                  # Human audit spreadsheets and inter-rater ratings
└── results/
    ├── raw/                 # Cached model responses (JSON Lines)
    └── processed/           # Aggregated metrics, statistical tests, and tables
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ollama pull llama3.1:8b-instruct-q8_0
  ollama pull gemma2:9b
  ```

### 2. Environment Setup
```bash
git clone https://github.com/Adibhaibvn/cot-faithfulness-personal-ai.git
cd cot-faithfulness-personal-ai

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_key
GOOGLE_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MIN_INTERVAL=4.5
```

Verify your setup with the pre-flight check:
```bash
python3 -m src.models.check_setup
```

---

## 🔄 Execution Workflow

### Phase 1: Benchmark Curation
```bash
# Curate math instances and MC candidates
python3 -m src.dataset.curate

# Generate verified gold CoTs for commonsense and causal tasks
python3 -m src.dataset.generate_gold_cots --limit 50
```

### Phase 2: Controlled Corruption Generation
```bash
# Generate 4 corruption variants per instance (600 variants total)
python3 -m src.corruption.generate --task all

# Sample instances for human audit and compute Cohen's kappa
python3 -m src.corruption.make_review_sheets
python3 -m src.corruption.agreement

# Verify and freeze dataset with SHA-256 integrity hashes
python3 -m src.corruption.freeze
```

### Phase 3: Model Evaluation
```bash
# Run local Ollama models
python3 -m src.evaluation.runner --group local

# Run cloud models (OpenRouter + Gemini)
python3 -m src.evaluation.runner --group cloud
```

### Phase 4: Analysis & Reporting
```bash
# Compute headline Faithfulness Score (FS) and extraction rates
python3 -m src.analysis.metrics

# Compute exact McNemar tests and bootstrap 95% confidence intervals
python3 -m src.analysis.stats

# Generate paper figures
python3 -m src.analysis.figures
```

---

## 📄 Citation & Methodology

All evaluations use a neutral decision prompt that neither encourages following nor checking the provided chain. Faithfulness is assessed solely on instances where the model's unperturbed baseline is correct, preventing spurious penalty from task failure.

