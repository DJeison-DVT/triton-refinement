# Triton Refinement

Reflexion-style iterative refinement for translating PyTorch operators to Triton GPU kernels using open-weights LLMs with XGrammar constrained decoding.

Benchmarked against [TritonBench-T](https://github.com/thunlp/TritonBench) (166 ops, 3-phase evaluation) via [TritonBench4Modal](https://github.com/salvahin/TritonBench4Modal).

## Pipeline

```
PyTorch source code
  │
  ├─→ Test Generation       (LLM, free-form)
  ├─→ Triton Translation    (LLM, XGrammar-constrained)
  └─→ Refinement Loop       (up to 5 iterations)
        compile → run tests → verify best practices → fix on failure
  │
  Final candidate → TritonBench4Modal evaluate_only
```

## Models

| Model | Size | Role |
|-------|------|------|
| Qwen2.5-Coder-7B | 7B | Primary |
| DeepSeek-Coder-6.7B | 6.7B | Comparison |
| CodeGemma-7B | 7B | Comparison |

## Quick Start

See [docs/setup.md](docs/setup.md) for full environment setup.

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/DJeison-DVT/triton-refinement.git
cd triton-refinement

# 2. Clone external dependencies (not committed)
git clone --depth 1 https://github.com/salvahin/TritonBench4Modal.git
git clone --depth 1 https://github.com/thunlp/TritonBench.git

# 3. Install
pip install -e .

# 4. Local dev — single op against local vLLM
python scripts/run_local.py --model Qwen/Qwen2.5-Coder-7B --op relu

# 5. Benchmark — full run via Modal
python scripts/run_experiment.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --max-iters 5 \
  --limit 5 \
  --evaluate
```

## Project Structure

```
triton-refinement/
├── prompts/           # LLM prompt templates (translator, reviewer, fixer, test_generator)
├── core/              # Refinement loop, LLM client, grammar loader, pattern memory
├── adapters/          # TritonBench4Modal bridge, dataset loader
├── scripts/           # CLI entry points (run_experiment, run_local, analyze)
├── modal_vllm.py      # vLLM deployment as Modal function
├── modal_app.py       # Orchestrator for benchmark runs
├── triton_lexeme/     # [submodule] EBNF grammar for constrained decoding
├── TritonBench4Modal/ # [local clone] Professor's benchmark harness
├── TritonBench/       # [local clone] Upstream benchmark data (166 PyTorch ops)
├── paper/             # LaTeX paper and figures
└── results/           # Experiment outputs (gitignored)
```

## Documentation

- [Setup Guide](docs/setup.md) — environment, dependencies, Modal configuration
- [Architecture](docs/architecture.md) — system design, data flow, component details
- [Decisions](docs/decisions.md) — key design choices and rationale
- [Scope](docs/scope.md) — research goals, deliverables, timeline

## External Dependencies

- [TritonBench](https://github.com/thunlp/TritonBench) — benchmark dataset and evaluation logic
- [TritonBench4Modal](https://github.com/salvahin/TritonBench4Modal) — Modal-based benchmark runner (pinned at commit `6103f69`)
- [triton_lexeme](https://github.com/DJeison-DVT/triton_lexeme) — EBNF grammar for Triton constrained decoding

## References

- Shinn et al., 2023 — [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- TritonBench paper — [https://arxiv.org/pdf/2502.14752](https://arxiv.org/pdf/2502.14752)
- XGrammar — constrained decoding for structured LLM outputs
