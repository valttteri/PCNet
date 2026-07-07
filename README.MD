# Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits

---

## Overview

This repository provides the full implementation for the paper. The core idea is that a single **Probabilistic Circuit (PCNet)** trained on LLM latent representations can simultaneously serve three roles:

1. **Detection** — exact tractable density estimation over projected hidden states to score hallucination likelihood (AUROC up to 0.96 on structured QA).
2. **Gating** — a confidence gate that selects *when* to intervene, yielding consistent gains over always-on correction across six models and five datasets.
3. **Correction** — latent-space interventions guided by the PC density to push representations away from the hallucination manifold.

The framework also establishes a **three-regime analysis** (high / medium / low detection AUROC) that explains when and why corrections succeed or fail, and identifies an empirical **convergence ceiling** shared across all correction methods.

---

## Repository Structure

```
.
├── src/
│   ├── probabilistic_circuits.py   # PCNet: Sum-Product Network with Gaussian/Laplace/Student-t leaves
│   ├── guardrail.py                # LLM_PC_Guardrail: attaches PCNet to any HF causal LM (4-bit QNT)
│   ├── experiment.py               # HallucinationExperiment: PCNet detection vs. baselines
│   ├── baselines.py                # Detection baselines
│   ├── correction_baselines.py     # Correction methods
│   ├── correction_pcnet.py         # PCNet-guided corrections
│   ├── correction_pipeline.py      # Main gating + correction benchmarking loop
│   ├── correction_pipeline_mc.py   # MC-Dropout variant of the correction pipeline
│   ├── metrics.py                  # Evaluation report generator (AUROC, Token-F1, EM, LLM-CAS)
│   ├── benchmarking.py             # LLM-as-judge scoring (Anthropic API + local LLaMA-2 judge)
│   ├── run_baselines.py            # Detection-only baseline sweep
│   ├── main.py                     # Entry point: multi-model x multi-dataset benchmark
│   ├── train.py                    # Supervised PCNet trainer
│   ├── unsup_train.py              # Unsupervised density estimation trainer
│   ├── hf_dataloader.py            # HuggingFace dataset loader
│   ├── data_loader.py              # Dataset utilities and preprocessing
│   ├── mic_alignment_test.py       # Cosine alignment test for MIC correction signal
│   ├── gating_baselines.py         # Confidence-gate ablations (gated vs. blind correction)
│   ├── gated_vs_rag_ablation.py    # Gating vs. retrieval-augmented generation comparison
│   ├── ablation_projection.py      # Bottleneck dimension ablations (32 / 64 / 128 / 256 / 512)
│   ├── ablation_depth.py           # PCNet depth ablations
│   ├── ablation_data_size.py       # Training sample-size ablations
│   └── config/
│       ├── baselines/              # Detection-only configs per model
│       ├── correction/             # Gating + correction configs per model
│       └── correction_mc/          # MC-Dropout correction configs
├── checkpoints/
│   └── PCNet_Guardrail/            # Trained PCNet weights (.pth) per model x dataset x seed
├── paper_logs/                     # Canonical logs used in the paper (detection + correction)
├── submit_jobs.sh                  # Master SLURM batch executor
├── requirements.txt
└── README.MD
```

---

## Models and Datasets

**Models evaluated:**

- `Qwen/Qwen3-4B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `deepseek-ai/deepseek-llm-7b-chat`

**Datasets:**

- `stanfordnlp/coqa` (CoQA)
- `rajpurkar/squad_v2` (SQuAD v2)
- `mandarjoshi/trivia_qa` (TriviaQA)
- `truthful_qa` (TruthfulQA)

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU with at least 24 GB VRAM is recommended for 7B-class models. 4-bit quantization via `bitsandbytes` is applied automatically.

---

## Quick Start

### 1. Train the PCNet guardrail

Supervised (with hallucination labels):

```bash
python src/train.py src/config/sup/llama3.2_1b.json 42
```

### 2. Run detection benchmarks

```bash
python src/run_baselines.py src/config/baselines/llama3.2_1b.json 42
python src/main.py src/config/sup/llama3.2_1b.json 42
```

### 3. Run the gating + correction pipeline

```bash
python src/correction_pipeline.py src/config/correction/llama3.2_1b.json 42
```

This runs all correction methods (Vanilla, DoLa, ITI, TruthX, Vocab Banishment, Z-Score, TAC, Contrastive Decoding) in both **gated** and **blind** (always-on) modes.

### 4. Evaluate results

```bash
python src/metrics.py --log-dir paper_logs/detection
python src/benchmarking.py --log-dir paper_logs/correction
```

Outputs per-seed and aggregated AUROC, FPR95, Token-F1, Exact Match, and LLM-CAS tables to `evaluation_reports/`.

### 5. Cosine alignment diagnostic

```bash
python src/mic_alignment_test.py src/config/correction/llama3.2_1b.json \
    --checkpoint_dir checkpoints/PCNet_Guardrail/meta-llama_Llama-3.2-1B-Instruct/coqa \
    --n_samples 50 --tau_percentile 20
```

### 6. Ablations

Projection bottleneck dimension:

```bash
python src/ablation_projection.py
```

PCNet depth:

```bash
python src/ablation_depth.py
```

Training data size:

```bash
python src/ablation_data_size.py
```

### 7. HPC (SLURM)

```bash
./submit_jobs.sh
# or selectively:
./submit_jobs.sh "cluster_scripts_proj_128"
```

## Reproducibility

All results in the paper use seed `42,43,44`. The `paper_logs/` directory contains the canonical raw outputs. Config files for all model x dataset combinations used in the paper are in `src/config/baselines/` and `src/config/correction/`.
