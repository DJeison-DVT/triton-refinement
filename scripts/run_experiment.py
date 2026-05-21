"""Full experiment CLI: run refinement across ops, models, and seeds.

Usage (single run):
    python scripts/run_experiment.py \
      --model Qwen/Qwen2.5-Coder-7B \
      --condition refinement \
      --limit 5 \
      --seed 42 \
      --base-url http://localhost:8000/v1

Usage (batch mode):
    python scripts/run_experiment.py --batch
    python scripts/run_experiment.py --batch --models Qwen2.5-Coder-7B --seeds 42 99
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.dataset import load_ops, build_stem_to_opname
from adapters.tritonbench import write_predictions, evaluate, update_op_results_with_eval
from core.grammar import load_grammar
from core.llm_client import LLMClient
from core.loop import generate_with_refinement, save_trajectory, build_op_result
from core.memory_inmemory import InMemoryPatternMemory
from prompts import extract_code
from prompts import test_generator


def set_seed(seed: int):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_run_id(model: str, condition: str, seed: int) -> str:
    """Build deterministic run_id: {model_short}_{condition}_seed{seed}."""
    model_short = model.split("/")[-1]
    return f"{model_short}_{condition}_seed{seed}"


def build_sweep_matrix(
    config_path: Path,
    models_filter: list[str] | None = None,
    conditions_filter: list[str] | None = None,
    seeds_filter: list[int] | None = None,
) -> list[dict]:
    """Read experiment_config.json and build the sweep matrix.

    Returns list of dicts with keys: model, condition, seed.
    Filters narrow the matrix when provided.
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    models = [m["hf_id"] for m in raw["models"]]
    conditions = list(raw["conditions"].keys())
    seeds = raw["reproducibility"]["seeds"]

    if models_filter:
        models = [m for m in models if m in models_filter or m.split("/")[-1] in models_filter]
    if conditions_filter:
        conditions = [c for c in conditions if c in conditions_filter]
    if seeds_filter:
        seeds = [s for s in seeds if s in seeds_filter]

    matrix = []
    for model in models:
        for condition in conditions:
            for seed in seeds:
                matrix.append({"model": model, "condition": condition, "seed": seed})
    return matrix


def resolve_condition_settings(config_path: Path, condition: str) -> dict:
    """Look up condition settings from experiment_config.json."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return raw["conditions"][condition]


def is_run_complete(run_dir: Path) -> bool:
    """Check if a run directory has a summary.json (complete)."""
    return (run_dir / "summary.json").exists()


def run_single(
    model: str,
    condition: str,
    seed: int,
    base_url: str,
    config_path: Path,
    output_dir: Path,
    limit: int | None,
    max_tokens: int,
) -> None:
    """Execute a single experiment run."""
    settings = resolve_condition_settings(config_path, condition)
    max_iters = settings["max_iterations"]
    use_grammar = settings["grammar_constrained"]
    mem_type = settings["pattern_memory"]

    set_seed(seed)
    client = LLMClient(base_url=base_url, model=model, api_key="EMPTY", max_tokens=max_tokens)
    ops = load_ops(limit=limit)
    grammar = load_grammar() if use_grammar else None
    mem = InMemoryPatternMemory() if mem_type == "inmemory" else None

    run_id = make_run_id(model, condition, seed)
    run_dir = output_dir / run_id
    traj_dir = run_dir / "trajectories"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    total = len(ops)
    predictions = []
    op_results = {}
    n_passed = 0

    for idx, op in enumerate(ops):
        print(f"[{idx + 1}/{total}] {op.op_name}", flush=True)

        test_msgs = test_generator.format_messages(op.pytorch_code)
        raw_tests = client.generate(test_msgs)
        test_code = extract_code(raw_tests)

        result = generate_with_refinement(
            op.op_name, op.pytorch_code, test_code, client,
            grammar=grammar, max_iters=max_iters, pattern_memory=mem,
        )

        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} ({result.total_iterations} iter(s))", flush=True)

        if result.passed:
            n_passed += 1

        save_trajectory(result, traj_dir)
        predictions.append({"instruction": op.instruction, "predict": result.final_code})
        op_results[op.op_name] = build_op_result(result)

    pred_path = run_dir / "predictions.jsonl"
    write_predictions(predictions, pred_path)

    (run_dir / "op_results.json").write_text(
        json.dumps(op_results, indent=2), encoding="utf-8",
    )

    pass_rate = n_passed / total if total else 0.0
    summary = {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "seed": seed,
        "max_iters": max_iters,
        "grammar": use_grammar,
        "pattern_memory": mem_type if mem_type else "none",
        "n_ops": total,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"Ops: {total} | Passed: {n_passed} | Pass rate: {pass_rate:.1%}")
    print(f"Results: {run_dir}")


def run_evaluation(run_dir: Path) -> None:
    """Run TritonBench4Modal evaluation on a completed run and update op_results.json."""
    pred_path = run_dir / "predictions.jsonl"
    op_results_path = run_dir / "op_results.json"

    if not pred_path.exists():
        print(f"  No predictions.jsonl in {run_dir}, skipping evaluation")
        return
    if not op_results_path.exists():
        print(f"  No op_results.json in {run_dir}, skipping evaluation")
        return

    run_id = run_dir.name
    print(f"  Evaluating {run_id} via TritonBench4Modal...", flush=True)

    eval_result = evaluate(pred_path, output_subdir=run_id)
    stem_map = build_stem_to_opname()
    update_op_results_with_eval(op_results_path, eval_result, stem_map)

    print(f"  Phase 1: {eval_result.phase1_passed}/{eval_result.total_predictions} ({eval_result.phase1_rate:.1f}%)")
    print(f"  Phase 2: {eval_result.phase2_passed}/{eval_result.total_predictions} ({eval_result.phase2_rate:.1f}%)")
    if eval_result.phase3_speedup is not None:
        print(f"  Phase 3: {eval_result.phase3_speedup:.2f}x vs PyTorch")
    print(f"  Updated {op_results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run refinement experiment across ops, models, and seeds."
    )
    # -- Single run flags --
    parser.add_argument("--model", help="Model identifier (e.g., Qwen/Qwen2.5-Coder-7B)")
    parser.add_argument("--condition", help="Condition name from experiment_config.json (e.g., single-shot, refinement)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM endpoint URL")
    parser.add_argument("--limit", type=int, default=None, help="Number of ops to process (default: all 166)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per LLM call")

    # -- Batch mode flags --
    parser.add_argument("--batch", action="store_true", help="Run all models x conditions x seeds from config")
    parser.add_argument("--models", nargs="+", default=None, help="Filter: only these models (batch mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Filter: only these seeds (batch mode)")
    parser.add_argument("--conditions", nargs="+", default=None, help="Filter: only these conditions (batch mode)")

    # -- Evaluation flag --
    parser.add_argument("--evaluate", action="store_true",
                        help="Run TritonBench4Modal evaluation after each run (requires Modal)")

    # -- Shared flags --
    parser.add_argument("--config", type=Path, default=Path("experiment_config.json"), help="Path to experiment_config.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory")

    args = parser.parse_args()

    if args.batch:
        matrix = build_sweep_matrix(
            args.config,
            models_filter=args.models,
            conditions_filter=args.conditions,
            seeds_filter=args.seeds,
        )
        total_runs = len(matrix)
        print(f"Batch mode: {total_runs} runs from {args.config}")

        for i, run_spec in enumerate(matrix):
            run_id = make_run_id(run_spec["model"], run_spec["condition"], run_spec["seed"])
            run_dir = args.output_dir / run_id

            if is_run_complete(run_dir):
                print(f"[{i + 1}/{total_runs}] {run_id} — skipped (already complete)")
                continue

            print(f"\n[{i + 1}/{total_runs}] {run_id}")
            run_single(
                model=run_spec["model"],
                condition=run_spec["condition"],
                seed=run_spec["seed"],
                base_url=args.base_url,
                config_path=args.config,
                output_dir=args.output_dir,
                limit=args.limit,
                max_tokens=args.max_tokens,
            )
            if args.evaluate:
                run_evaluation(run_dir)

        print(f"\nBatch complete: {total_runs} runs")
    else:
        if not args.model or not args.condition:
            parser.error("--model and --condition are required for single-run mode (or use --batch)")
        run_single(
            model=args.model,
            condition=args.condition,
            seed=args.seed,
            base_url=args.base_url,
            config_path=args.config,
            output_dir=args.output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
        )
        if args.evaluate:
            run_id = make_run_id(args.model, args.condition, args.seed)
            run_evaluation(args.output_dir / run_id)


if __name__ == "__main__":
    main()
