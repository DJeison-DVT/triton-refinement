"""Local dev CLI: translate a single PyTorch op to Triton via LLM.

Supports optional refinement loop (--refine) for smoke testing the full
pipeline (dataset, LLM client, grammar, translate, compile, test, review).

Usage:
    python scripts/run_local.py                          # first op, Ollama
    python scripts/run_local.py --op abs                 # specific op
    python scripts/run_local.py --limit 3                # first 3 ops
    python scripts/run_local.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-7B
    python scripts/run_local.py --refine                 # enable refinement loop
    python scripts/run_local.py --refine --max-iters 3   # refinement with 3 iterations
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.dataset import load_ops
from core.grammar import load_grammar
from core.llm_client import LLMClient
from prompts import extract_code
from prompts import translator, test_generator


def main():
    parser = argparse.ArgumentParser(description="Translate PyTorch ops to Triton (local dev)")
    parser.add_argument("--op", help="Specific op name to translate")
    parser.add_argument("--limit", type=int, default=1, help="Number of ops to translate")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from base-url if omitted)")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--grammar", action="store_true", help="Enable XGrammar constraint (vLLM only)")
    parser.add_argument("--out", help="Write predictions.jsonl to this path")
    parser.add_argument("--refine", action="store_true", help="Enable refinement loop (compile > test > review > fix)")
    parser.add_argument("--max-iters", type=int, default=5, help="Maximum refinement iterations (default: 5)")
    args = parser.parse_args()

    # Auto-detect model name based on backend
    if args.model is None:
        if "11434" in args.base_url:
            args.model = "qwen2.5-coder:7b"
        else:
            args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )

    ops = load_ops()
    if args.op:
        ops = [op for op in ops if op.op_name == args.op]
        if not ops:
            print(f"Op '{args.op}' not found in dataset")
            sys.exit(1)
    else:
        ops = ops[: args.limit]

    grammar = load_grammar() if args.grammar else None
    predictions = []

    for op in ops:
        print(f"\n{'='*60}")
        print(f"Translating: {op.op_name}")
        print(f"{'='*60}")

        if args.refine:
            from core.loop import generate_with_refinement, save_trajectory

            # Generate tests first
            print("Generating tests...", flush=True)
            test_msgs = test_generator.format_messages(op.pytorch_code)
            raw_tests = client.generate(test_msgs)
            test_code = extract_code(raw_tests)

            # Run refinement loop
            print(f"Running refinement loop (max {args.max_iters} iters)...", flush=True)
            result = generate_with_refinement(
                op_name=op.op_name,
                pytorch_code=op.pytorch_code,
                test_code=test_code,
                client=client,
                grammar=grammar,
                max_iters=args.max_iters,
            )

            # Save trajectory
            traj_dir = Path(__file__).resolve().parent.parent / "results" / "local" / "trajectories"
            traj_path = save_trajectory(result, traj_dir)

            status = "PASS" if result.passed else "FAIL"
            print(f"\nStatus: {status} (iterations: {result.total_iterations})")
            print(f"Trajectory saved to: {traj_path}")

            code = result.final_code
        else:
            # Non-refine path: use translator prompt module
            if grammar:
                messages = translator.format_kernel_messages(op.pytorch_code)
            else:
                messages = translator.format_messages(op.pytorch_code)

            print("Calling LLM...", flush=True)
            raw = client.generate(messages, grammar=grammar)

            if grammar:
                # Grammar-constrained output is raw code — no extraction needed
                code = raw if raw.endswith("\n") else raw + "\n"
            else:
                code = extract_code(raw)

        print(f"\n--- Generated Triton code ({len(code.splitlines())} lines) ---")
        print(code[:2000])
        if len(code) > 2000:
            print(f"... ({len(code) - 2000} more chars)")

        predictions.append({"instruction": op.instruction, "predict": code})

    if args.out:
        from adapters.tritonbench import write_predictions

        out_path = write_predictions(predictions, Path(args.out))
        print(f"\nWrote {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
