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

For analysis (includes numpy, scipy, pandas, matplotlib, seaborn):
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

## 6. Ollama Setup (Quick Dev)

Ollama runs on Windows and serves models for local development. The refinement
loop must run from WSL2 (Triton requires Linux + CUDA), so Ollama must listen
on all interfaces.

```powershell
# Windows: set env var so WSL2 can reach Ollama
set OLLAMA_HOST=0.0.0.0
ollama serve
```

Pull the dev model:
```powershell
ollama pull qwen2.5-coder:7b
```

Find your Windows host IP from WSL2:
```bash
# In WSL2
ip route show default | awk '{print $3}'
# e.g. 172.28.48.1
```

Test from WSL2:
```bash
curl http://172.28.48.1:11434/v1/models
```

### Quick smoke test (no grammar, no compile verification):
```bash
# From Windows (fast iteration on prompts)
python scripts/run_local.py --limit 1 --model qwen2.5-coder:7b
```

### Smoke test with refinement (from WSL2, real compile):
```bash
# From WSL2 (Triton compile check works)
cd /mnt/c/Users/<you>/repos/school/triton-refinement
python3 scripts/run_local.py --limit 1 --model qwen2.5-coder:7b \
  --base-url http://172.28.48.1:11434/v1 --refine --max-iters 2
```

## 7. Local vLLM Setup (Grammar-Constrained Experiments)

vLLM is required for XGrammar constrained decoding. Must run inside WSL2.

```bash
# In WSL2
pip install vllm
```

Serve a model with XGrammar:
```bash
vllm serve Qwen/Qwen2.5-Coder-7B --guided-decoding-backend xgrammar
```

Start with 1.5B for fast iteration. Switch to 7B for real experiments.
The 4090 (24GB) handles 7B models comfortably.

### Test the endpoint

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B",
    "prompt": "import triton",
    "max_tokens": 100
  }'
```

## 8. Verify End-to-End

All experiment runs should happen from WSL2 (for Triton compile/test):

```bash
# Smoke test: 5 ops, single condition (from WSL2)
python3 scripts/run_experiment.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --condition refinement \
  --limit 5 \
  --base-url http://localhost:8000/v1

# Batch mode: all models x conditions x seeds (with resume)
python3 scripts/run_experiment.py --batch --limit 5

# Batch filtered to one model
python3 scripts/run_experiment.py --batch --models Qwen/Qwen2.5-Coder-7B --limit 5

# With TritonBench4Modal evaluation (updates op_results.json with real Phase 1/2/3)
python3 scripts/run_experiment.py --batch --evaluate --limit 5
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
├── analysis/              # statistical analysis package
├── scripts/
├── results/               # gitignored, created during runs
│   └── {model}_{condition}_seed{seed}/
│       ├── trajectories/{op}.jsonl
│       ├── op_results.json
│       ├── predictions.jsonl
│       └── summary.json
└── ...
```
