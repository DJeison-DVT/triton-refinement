"""Full experiment CLI: run refinement across ops, models, and seeds.

Usage:
    python scripts/run_experiment.py \
      --model Qwen/Qwen2.5-Coder-7B \
      --max-iters 5 \
      --limit 5 \
      --pattern-memory inmemory \
      --seed 42 \
      --base-url http://localhost:8000/v1

    python scripts/run_experiment.py \
      --model Qwen/Qwen2.5-Coder-7B \
      --max-iters 1 \
      --seed 42 \
      --evaluate
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.dataset import load_ops
from adapters.tritonbench import write_predictions, evaluate
from core.grammar import load_grammar
from core.llm_client import LLMClient
from core.loop import generate_with_refinement, save_trajectory
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


def main():
    parser = argparse.ArgumentParser(
        description="Run refinement experiment across ops, models, and seeds."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., Qwen/Qwen2.5-Coder-7B)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM endpoint URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the vLLM endpoint (default: EMPTY)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Max refinement iterations per op (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of ops to process (default: all 166)",
    )
    parser.add_argument(
        "--pattern-memory",
        choices=["inmemory", "none"],
        default="inmemory",
        help="Pattern memory backend (default: inmemory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--grammar",
        action="store_true",
        default=True,
        help="Enable XGrammar constraint (default: enabled)",
    )
    parser.add_argument(
        "--no-grammar",
        dest="grammar",
        action="store_false",
        help="Disable XGrammar constraint",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run TritonBench4Modal evaluation after experiment",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per LLM call (default: 4096)",
    )
    args = parser.parse_args()

    # 1. Set seed
    set_seed(args.seed)

    # 2. Create LLMClient
    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )

    # 3. Load ops
    ops = load_ops(limit=args.limit)
    total = len(ops)

    # 4. Load grammar if enabled
    grammar = load_grammar() if args.grammar else None

    # 5. Create pattern memory
    mem = InMemoryPatternMemory() if args.pattern_memory == "inmemory" else None

    # 6. Derive run_id
    model_short = args.model.split("/")[-1]
    run_id = f"{model_short}_seed{args.seed}_iters{args.max_iters}"

    # 7. Create output directories
    output_dir = Path(args.output_dir)
    run_dir = output_dir / run_id
    traj_dir = run_dir / "trajectories"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    n_passed = 0

    # 8. Process each op
    for idx, op in enumerate(ops):
        print(f"[{idx + 1}/{total}] {op.op_name}", flush=True)

        # a. Generate tests
        test_msgs = test_generator.format_messages(op.pytorch_code)
        raw_tests = client.generate(test_msgs)
        test_code = extract_code(raw_tests)

        # b. Run refinement
        result = generate_with_refinement(
            op.op_name,
            op.pytorch_code,
            test_code,
            client,
            grammar=grammar,
            max_iters=args.max_iters,
            pattern_memory=mem,
        )

        # c. Print result
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} ({result.total_iterations} iter(s))", flush=True)

        if result.passed:
            n_passed += 1

        # d. Save trajectory
        save_trajectory(result, traj_dir)

        # e. Append prediction
        predictions.append({"instruction": op.instruction, "predict": result.final_code})

    # 9. Write predictions.jsonl
    pred_path = run_dir / "predictions.jsonl"
    write_predictions(predictions, pred_path)

    # 10. Print summary
    pass_rate = n_passed / total if total else 0.0
    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"Ops: {total} | Passed: {n_passed} | Pass rate: {pass_rate:.1%}")
    print(f"Predictions: {pred_path}")

    # 12. Save summary.json
    summary = {
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "max_iters": args.max_iters,
        "grammar": args.grammar,
        "pattern_memory": args.pattern_memory,
        "n_ops": total,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")

    # 13. Evaluate if requested
    if args.evaluate:
        print("\nRunning TritonBench4Modal evaluation...", flush=True)
        eval_result = evaluate(pred_path, output_subdir=run_id)
        print(f"Phase 1 (call accuracy):  {eval_result.phase1_passed}/{eval_result.total_predictions} ({eval_result.phase1_rate:.1%})")
        print(f"Phase 2 (exec accuracy):  {eval_result.phase2_passed}/{eval_result.total_predictions} ({eval_result.phase2_rate:.1%})")
        if eval_result.phase3_speedup is not None:
            print(f"Phase 3 (speedup):        {eval_result.phase3_speedup:.2f}x vs PyTorch")
        else:
            print("Phase 3 (speedup):        N/A")


if __name__ == "__main__":
    main()
