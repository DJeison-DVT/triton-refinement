"""Smoke-test TritonBench4Modal evaluate_only with a small predictions.jsonl.

Builds predictions for a handful of simple ops using hand-crafted Triton code,
then runs evaluate_only and prints exactly what comes back.

Usage:
    python scripts/test_evaluate.py              # build + evaluate (Modal)
    python scripts/test_evaluate.py --build-only # just build predictions.jsonl, skip Modal
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.dataset import load_ops
from adapters.tritonbench import write_predictions, evaluate


# Hand-crafted Triton kernels for a few simple ops.
# These are intentionally basic — we're testing the evaluator, not the generator.
MANUAL_TRITON = {
    # Wrapper signature must exactly match the golden test:
    #   add(input, other, alpha=1, out=None)
    # Tests pass scalars as `other` and use `alpha`, so we fall back to
    # torch.add for those cases and only use the kernel for simple tensor+tensor.
    "torch.add": '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y * alpha, mask=mask)

def add(input, other, alpha=1, out=None):
    if not isinstance(other, torch.Tensor):
        return torch.add(input, other, alpha=alpha, out=out)
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _add_kernel[grid](input, other, out, alpha, n, BLOCK=1024)
    return out
''',

    # Wrapper: relu(input: Tensor, inplace: bool = False) -> Tensor
    # inplace=True modifies the input tensor in-place.
    "torch.nn.functional.relu": '''
import torch
import triton
import triton.language as tl

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, tl.zeros_like(x)), mask=mask)

def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _relu_kernel[grid](input, out, n, BLOCK=1024)
    return out
''',

    # Wrapper: tanh(input_tensor, out_tensor=None)
    # Test passes empty tensors, so guard for n==0.
    "torch.tanh": '''
import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.math.tanh(x), mask=mask)

def tanh(input_tensor, out_tensor=None):
    if out_tensor is None:
        out_tensor = torch.empty_like(input_tensor)
    n = input_tensor.numel()
    if n == 0:
        return out_tensor
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _tanh_kernel[grid](input_tensor, out_tensor, n, BLOCK=1024)
    return out_tensor
''',
}


def build_predictions(op_names: list[str], output_path: Path) -> Path:
    """Build predictions.jsonl for the given ops."""
    all_ops = load_ops()
    op_map = {op.op_name: op for op in all_ops}

    predictions = []
    for name in op_names:
        if name not in op_map:
            print(f"  SKIP: '{name}' not in dataset")
            continue
        if name not in MANUAL_TRITON:
            print(f"  SKIP: '{name}' has no manual Triton code")
            continue

        op = op_map[name]
        code = MANUAL_TRITON[name].strip()
        predictions.append({
            "instruction": op.instruction,
            "predict": code,
        })
        print(f"  OK: {name} ({len(code)} chars)")

    path = write_predictions(predictions, output_path)
    print(f"\nWrote {len(predictions)} predictions to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Smoke-test evaluate_only")
    parser.add_argument("--build-only", action="store_true",
                        help="Build predictions.jsonl without running Modal evaluation")
    parser.add_argument("--output-subdir", default="eval-smoke-test",
                        help="Subdir name in Modal volume (default: eval-smoke-test)")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent.parent / "results" / "local" / "smoke_predictions.jsonl"
    op_names = list(MANUAL_TRITON.keys())

    print(f"Building predictions for {len(op_names)} ops...")
    pred_path = build_predictions(op_names, out_path)

    # Show what we built
    print("\n--- predictions.jsonl contents ---")
    with pred_path.open() as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            inst_preview = entry["instruction"][:80] + "..."
            code_preview = entry["predict"][:60].replace("\n", "\\n") + "..."
            print(f"  [{i}] instruction: {inst_preview}")
            print(f"      predict: {code_preview}")

    if args.build_only:
        print("\n--build-only: skipping Modal evaluation")
        return

    print(f"\n{'='*60}")
    print("Running evaluate_only on Modal...")
    print(f"{'='*60}")

    try:
        result = evaluate(pred_path, output_subdir=args.output_subdir)
        print(f"\n{'='*60}")
        print("EVALUATION RESULT")
        print(f"{'='*60}")
        print(f"  Total predictions: {result.total_predictions}")
        print(f"  Phase 1 (call acc):  {result.phase1_passed}/{result.total_predictions} = {result.phase1_rate:.1f}%")
        print(f"    Passed: {result.phase1_ops}")
        print(f"  Phase 2 (exec acc):  {result.phase2_passed}/{result.total_predictions} = {result.phase2_rate:.1f}%")
        print(f"    Passed: {result.phase2_ops}")
        print(f"  Phase 3 (speedup):   {result.phase3_speedup}")
    except Exception as e:
        print(f"\nEVALUATION FAILED: {e}")
        print("\nThis is expected if Modal isn't set up or the TritonBench4Modal image needs building.")
        print("The predictions.jsonl was still built successfully at:")
        print(f"  {pred_path}")


if __name__ == "__main__":
    main()
