"""Local dev CLI: translate a single PyTorch op to Triton via LLM.

No refinement loop — just test generation and translation for smoke testing
the plumbing (dataset, LLM client, grammar).

Usage:
    python scripts/run_local.py                          # first op, Ollama
    python scripts/run_local.py --op abs                 # specific op
    python scripts/run_local.py --limit 3                # first 3 ops
    python scripts/run_local.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-7B
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.dataset import load_ops
from core.grammar import load_grammar
from core.llm_client import LLMClient


TRANSLATE_SYSTEM = (
    "You are an expert in Triton GPU programming. Translate the given PyTorch "
    "operator into an equivalent Triton kernel and wrapper function.\n\n"
    "Output a single Python module containing:\n"
    "1. Necessary imports (torch, triton, triton.language as tl)\n"
    "2. The Triton kernel(s) decorated with @triton.jit\n"
    "3. A wrapper function matching the original PyTorch signature\n\n"
    "Wrap the entire module in one ```python ... ``` code block."
)


def extract_code(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


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

        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": f"Translate this PyTorch operator to Triton:\n\n```python\n{op.pytorch_code}\n```"},
        ]

        print("Calling LLM...", flush=True)
        raw = client.generate(messages, grammar=grammar)
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
