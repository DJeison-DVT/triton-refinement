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
Generate a test function for a Triton kernel wrapper. The function is ALREADY \
defined in the same file — do NOT import it from anywhere.

RULES:
- Do NOT use import statements for the function being tested
- ALL tensors MUST use device='cuda': torch.randn(..., device='cuda')
- Only import torch (nothing else needed)
- Compare output against PyTorch reference using torch.allclose(result, expected, atol=1e-2)
- Use small tensor shapes (e.g., 128, 256)
- Define ALL variables before using them
- Call the test function at the end

Output in a ```python``` fence.\
"""


def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for test generation (instruct models)."""
    # Extract function name for the prompt
    func_name = "the_function"
    for line in pytorch_code.strip().split("\n"):
        if line.startswith("def "):
            func_name = line.split("(")[0].replace("def ", "").strip()
            break

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content":
         f"Write a test for `{func_name}`. It is already defined in the same file — just call it directly. "
         f"All tensors on CUDA.\n\nFunction signature:\n{pytorch_code.strip().split(chr(10))[0]}"},
    ]
