"""Prompt templates for PyTorch-to-Triton translation.

Completion-style prompts for base models. Each format_*_prompt() returns a
string prefix that the model completes. Grammar constraint handles structure
for the kernel step; free-form steps use stop sequences.
"""

# ---------------------------------------------------------------------------
# Free-form full-module translation (kernel + wrapper in one shot)
# ---------------------------------------------------------------------------

_FULL_TEMPLATE = """\
# PyTorch operator
{pytorch_code}

# Equivalent Triton implementation (kernel + wrapper, same interface as above)
import torch
import triton
import triton.language as tl

"""


def format_prompt(pytorch_code: str) -> str:
    """Return completion prompt for free-form full-module translation."""
    return _FULL_TEMPLATE.format(pytorch_code=pytorch_code)


# Keep chat-style for backward compat (instruct models)
def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for free-form full-module translation (instruct models)."""
    return [
        {"role": "system", "content": "You are an expert Triton GPU kernel engineer. "
         "Translate the PyTorch operator to a complete Triton module with @triton.jit kernel "
         "and wrapper function. Output the code in a ```python``` fence."},
        {"role": "user", "content": f"Translate to Triton:\n\n{pytorch_code}"},
    ]


# ---------------------------------------------------------------------------
# Grammar-constrained kernel-only generation
# ---------------------------------------------------------------------------

def format_kernel_prompt(pytorch_code: str) -> str:
    """Return completion prefix for grammar-constrained kernel generation.

    Ends with a blank line so the grammar starts fresh with imports.
    """
    lines = pytorch_code.strip().split("\n")
    commented = "\n".join(f"# {line}" for line in lines)
    return f"""\
# Translate this PyTorch operator to a Triton kernel.
{commented}

"""


def format_wrapper_messages(pytorch_code: str, kernel_code: str) -> list[dict[str, str]]:
    """Return chat messages for wrapper generation.

    Minimal prompt: give the LLM the kernel and PyTorch signature,
    ask for just the wrapper function. Nothing else.
    """
    return [
        {"role": "system", "content":
         "Write a Python wrapper function that launches the given Triton kernel. "
         "The wrapper must match the PyTorch operator's signature. "
         "Only output the function (no imports, no tests, no markdown). "
         "Keep it under 15 lines: allocate output, compute grid, launch kernel, return."},
        {"role": "user", "content":
         f"Kernel:\n{kernel_code}\n\nPyTorch signature:\n{pytorch_code.strip().split(chr(10))[0]}"},
    ]
