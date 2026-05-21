# Setup Guide

## Prerequisites

- Windows 11 with WSL2 + Ubuntu 22.04
- NVIDIA GPU (4090 for local dev)
- CUDA toolkit installed inside WSL (not just the driver)
- Python 3.11+

## 1. Verify GPU Access

```bash
# In WSL
nvidia-smi  # Should show the 4090
python -c "import torch; print(torch.cuda.is_available())"  # Should return True
```

## 2. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/DJeison-DVT/triton-refinement.git
cd triton-refinement
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## 3. Clone External Dependencies

These are gitignored and must be cloned manually:

```bash
# Professor's benchmark harness
git clone --depth 1 https://github.com/salvahin/TritonBench4Modal.git

# Upstream benchmark data (166 PyTorch ops + evaluation logic)
git clone --depth 1 https://github.com/thunlp/TritonBench.git
```

Pin the TritonBench4Modal commit for reproducibility:
```bash
cd TritonBench4Modal && git checkout 6103f69 && cd ..
```

## 4. Install Python Dependencies

```bash
pip install -e .
```

For development (includes pytest):
```bash
pip install -e ".[dev]"
```

For analysis (includes pandas, matplotlib, seaborn):
```bash
pip install -e ".[analysis]"
```

## 5. Modal Setup

```bash
pip install modal
modal setup  # Opens browser to authenticate
```

Set a $5 billing alert on [modal.com](https://modal.com).

### Configure LLM Secret

Only needed if using TritonBench4Modal's built-in generation (for baseline comparison):

```bash
# Anthropic
modal secret create tritonbench-llm ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
modal secret create tritonbench-llm OPENAI_API_KEY=sk-...
```

## 6. Local vLLM Setup (Development)

```bash
pip install vllm
```

Serve a model locally with XGrammar:
```bash
vllm serve Qwen/Qwen2.5-Coder-1.5B --guided-decoding-backend xgrammar
```

Start with the 1.5B model for fast iteration. Switch to 7B for real experiments.

### Test the endpoint

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-1.5B",
    "prompt": "import triton",
    "max_tokens": 100
  }'
```

## 7. Verify End-to-End

```bash
# Smoke test: 5 ops through the full pipeline
python scripts/run_experiment.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --max-iters 5 \
  --limit 5 \
  --base-url http://localhost:8000/v1

# Evaluate against TritonBench
modal run TritonBench4Modal/modal_app.py::evaluate_only -- --predictions ./predictions.jsonl
```

## Directory Layout After Setup

```
triton-refinement/
├── triton_lexeme/         # submodule (committed)
├── TritonBench4Modal/     # local clone (gitignored)
├── TritonBench/           # local clone (gitignored)
│   └── data/
│       ├── TritonBench_T_v1/              # 166 PyTorch .py files
│       ├── TritonBench_T_simp_alpac_v1.json
│       ├── TritonBench_T_comp_alpac_v1.json
│       └── TritonBench_T_v1.jsonl         # op metadata
├── prompts/
├── core/
├── adapters/
├── scripts/
├── results/               # gitignored, created during runs
└── ...
```
