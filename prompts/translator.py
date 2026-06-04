"""Prompt templates for PyTorch-to-Triton translation.

Includes one-shot examples for few-shot prompting. Grammar constraint
handles kernel structure; wrapper uses chat API with example.
"""

# ---------------------------------------------------------------------------
# One-shot example (torch.add) used in both kernel and wrapper prompts
# ---------------------------------------------------------------------------

_EXAMPLE_PYTORCH = "def add(input, other, alpha=1, out=None):\n    return torch.add(input, other, alpha=alpha, out=out)"

_EXAMPLE_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y * alpha, mask=mask)"""

_EXAMPLE_WRAPPER = """\
def add(input, other, alpha=1, out=None):
    if not isinstance(other, torch.Tensor):
        return torch.add(input, other, alpha=alpha, out=out)
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _add_kernel[grid](input, other, out, alpha, n, BLOCK=1024)
    return out"""


# ---------------------------------------------------------------------------
# Free-form full-module translation (kernel + wrapper in one shot)
# ---------------------------------------------------------------------------

_FULL_TEMPLATE = """\
# Example: translate torch.add to Triton
# {example_pytorch}

import torch
import triton
import triton.language as tl

{example_kernel}

{example_wrapper}

# Now translate this operator to Triton (kernel + wrapper, same interface):
# {target_pytorch}

import torch
import triton
import triton.language as tl

"""


def format_prompt(pytorch_code: str) -> str:
    """Return completion prompt for free-form full-module translation."""
    first_line = pytorch_code.strip().split("\n")[0]
    return _FULL_TEMPLATE.format(
        example_pytorch=_EXAMPLE_PYTORCH.split("\n")[0],
        example_kernel=_EXAMPLE_KERNEL,
        example_wrapper=_EXAMPLE_WRAPPER,
        target_pytorch=first_line,
    )


def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for free-form full-module translation (instruct models)."""
    return [
        {"role": "system", "content": "You are an expert Triton GPU kernel engineer. "
         "Translate the PyTorch operator to a complete Triton module with @triton.jit kernel "
         "and wrapper function. Output the code in a ```python``` fence."},
        {"role": "user", "content": f"Translate to Triton:\n\n{pytorch_code}"},
    ]


# ---------------------------------------------------------------------------
# Grammar-constrained kernel-only generation (with one-shot example)
# ---------------------------------------------------------------------------

def format_kernel_prompt(pytorch_code: str) -> str:
    """Return completion prefix for grammar-constrained kernel generation.

    Includes a one-shot example of a correct kernel, then the target op.
    Ends with a blank line so the grammar starts fresh with imports.
    """
    lines = pytorch_code.strip().split("\n")
    commented = "\n".join(f"# {line}" for line in lines)
    return f"""\
# Example: translate torch.add to a Triton kernel
# {_EXAMPLE_PYTORCH.split(chr(10))[0]}
{_EXAMPLE_KERNEL}

# Now translate this operator to a Triton kernel:
{commented}

"""


# ---------------------------------------------------------------------------
# Wrapper generation (chat API with one-shot example)
# ---------------------------------------------------------------------------

def format_wrapper_messages(pytorch_code: str, kernel_code: str) -> list[dict[str, str]]:
    """Return chat messages for wrapper generation with one-shot example.

    Uses a multi-turn format: show example kernel → example wrapper,
    then the target kernel → model writes the target wrapper.
    """
    func_sig = pytorch_code.strip().split("\n")[0]

    return [
        {"role": "system", "content":
         "Write a Python wrapper function that launches a Triton kernel. "
         "The wrapper MUST use the EXACT same function name and parameters as the PyTorch operator. "
         "Pass tensors directly to the kernel — do NOT use .data_ptr(). "
         "Output ONLY the function. No imports, no tests, no markdown."},
        {"role": "user", "content":
         f"Kernel:\n{_EXAMPLE_KERNEL}\n\nPyTorch signature: {_EXAMPLE_PYTORCH.split(chr(10))[0]}"},
        {"role": "assistant", "content": _EXAMPLE_WRAPPER},
        {"role": "user", "content":
         f"Kernel:\n{kernel_code}\n\nPyTorch signature: {func_sig}"},
    ]
