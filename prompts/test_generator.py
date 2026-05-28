"""Prompt templates for test generation.

Completion-style prompts for base models. The model completes test code
for a given PyTorch operator.
"""

# ---------------------------------------------------------------------------
# Completion-style (base models)
# ---------------------------------------------------------------------------

_TEST_TEMPLATE = """\
# PyTorch operator to test:
{pytorch_code}

# Test code: create CUDA tensors, call the operator, assert correctness.
# Cover edge cases: small/large sizes, non-contiguous tensors.
import torch

def test_{func_name}():
"""


def format_prompt(pytorch_code: str) -> str:
    """Return completion prefix for test generation."""
    # Extract function name from first def line
    func_name = "operator"
    for line in pytorch_code.strip().split("\n"):
        if line.startswith("def "):
            func_name = line.split("(")[0].replace("def ", "").strip()
            break
    return _TEST_TEMPLATE.format(pytorch_code=pytorch_code, func_name=func_name)


# ---------------------------------------------------------------------------
# Chat-style (instruct models, backward compat)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert GPU kernel engineer. Generate test code for the given PyTorch \
operator. Create CUDA tensors, call the operator, compare against PyTorch reference. \
Cover edge cases. Output in a ```python``` fence.\
"""


def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for test generation (instruct models)."""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Generate tests for:\n\n{pytorch_code}"},
    ]
