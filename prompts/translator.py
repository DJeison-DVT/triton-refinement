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

_KERNEL_TEMPLATE = """\
# PyTorch operator to translate:
# {pytorch_oneliner}
#
# Triton kernel (20-60 lines, use tl.load/tl.store with masks):
"""


def format_kernel_prompt(pytorch_code: str) -> str:
    """Return completion prefix for grammar-constrained kernel generation.

    The prefix ends right before the function definition. The grammar
    forces valid Triton from this point. The PyTorch code is compressed
    to a one-line comment to minimize prompt tokens.
    """
    # Compress PyTorch code to a single-line summary for the comment
    first_line = pytorch_code.strip().split("\n")[0]
    return _KERNEL_TEMPLATE.format(pytorch_oneliner=first_line)


def format_kernel_prompt_full(pytorch_code: str) -> str:
    """Return completion prefix with full PyTorch source for complex ops.

    The prefix ends with a comment. The grammar's root rule starts with
    imports and @triton.jit, so the model generates those as part of the
    grammar-constrained output — not as part of the prompt.
    """
    lines = pytorch_code.strip().split("\n")
    commented = "\n".join(f"# {line}" for line in lines)
    return f"""\
# PyTorch operator to translate into a Triton kernel:
{commented}
#
# Equivalent Triton kernel (20-60 lines, use tl.load/tl.store with masks):
"""


# Keep chat-style for backward compat
def format_kernel_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for grammar-constrained kernel generation (instruct models)."""
    return [
        {"role": "system", "content": "You are an expert Triton kernel engineer. "
         "Write ONLY the @triton.jit kernel function. No wrapper, no markdown fences. "
         "Keep it 20-60 lines. Output starts with imports and @triton.jit."},
        {"role": "user", "content": f"Write the kernel for:\n\n{pytorch_code}"},
    ]


# ---------------------------------------------------------------------------
# Free-form wrapper generation (after kernel is produced)
# ---------------------------------------------------------------------------

_WRAPPER_TEMPLATE = """\
# Triton kernel:
{kernel_code}

# Wrapper:
{func_signature}
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    {kernel_name}[grid]("""


def format_wrapper_prompt(pytorch_code: str, kernel_code: str) -> str:
    """Return completion prefix for wrapper generation.

    Extracts the function signature from PyTorch and kernel name from the
    generated kernel. Seeds the wrapper body with output allocation, grid
    computation, and kernel launch — the model only needs to fill in the
    kernel call arguments and return statement.
    """
    # Extract function signature from PyTorch code
    func_signature = "def wrapper():"
    for line in pytorch_code.strip().split("\n"):
        if line.startswith("def "):
            func_signature = line
            break

    # Extract kernel function name from generated kernel
    kernel_name = "_kernel"
    for line in kernel_code.strip().split("\n"):
        if line.startswith("def ") and "kernel" in line.lower():
            kernel_name = line.split("(")[0].replace("def ", "").strip()
            break

    return _WRAPPER_TEMPLATE.format(
        kernel_code=kernel_code,
        func_signature=func_signature,
        kernel_name=kernel_name,
    )


# Keep chat-style for backward compat
def format_wrapper_messages(pytorch_code: str, kernel_code: str) -> list[dict[str, str]]:
    """Return messages for wrapper generation (instruct models)."""
    return [
        {"role": "system", "content": "You are an expert Triton kernel engineer. "
         "Write the Python wrapper function that launches the given kernel. "
         "Match the PyTorch operator's interface. Output in a ```python``` fence."},
        {"role": "user", "content": f"PyTorch:\n{pytorch_code}\n\nKernel:\n{kernel_code}\n\nWrite the wrapper."},
    ]
