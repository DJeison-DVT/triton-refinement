"""Prompt templates for PyTorch-to-Triton translation."""

# ---------------------------------------------------------------------------
# Free-form full-module translation (kernel + wrapper in one shot)
# ---------------------------------------------------------------------------

_SYSTEM_FULL = """\
You are an expert GPU kernel engineer specializing in Triton. \
Given a PyTorch operator, translate it into a complete Triton module that includes:
1. A @triton.jit kernel function
2. A Python wrapper function that launches the kernel with appropriate grid/block config

Follow Triton best practices:
- Use tl.load / tl.store with mask for memory safety
- Compute grid dimensions carefully to cover all elements
- Keep the wrapper interface compatible with the original PyTorch operator

Output the full Triton module wrapped in a ```python ... ``` markdown fence.\
"""

_USER_FULL_TEMPLATE = """\
Translate the following PyTorch operator to Triton:

```python
{pytorch_code}
```
"""


def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for free-form full-module translation (kernel + wrapper).

    Args:
        pytorch_code: The PyTorch operator source code.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": _SYSTEM_FULL},
        {"role": "user", "content": _USER_FULL_TEMPLATE.format(pytorch_code=pytorch_code)},
    ]


# ---------------------------------------------------------------------------
# Grammar-constrained kernel-only generation
# ---------------------------------------------------------------------------

_SYSTEM_KERNEL = """\
You are an expert GPU kernel engineer specializing in Triton. \
Given a PyTorch operator, write ONLY the @triton.jit kernel function.

STRICT OUTPUT RULES — you must follow these exactly:
- Do not include a wrapper function
- Do not use markdown fences (no ``` or ```python)
- Do not add any explanation or comments outside the kernel
- Output raw Python/Triton code only, starting directly with @triton.jit

The output will be parsed by an EBNF grammar; any deviation will cause a parse error.\
"""

_USER_KERNEL_TEMPLATE = """\
Write the @triton.jit kernel for the following PyTorch operator (raw code, no markdown):

{pytorch_code}
"""


def format_kernel_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return messages for grammar-constrained kernel-only generation.

    The system prompt explicitly forbids markdown fences and wrapper functions
    because the output is fed directly to XGrammar constrained decoding.

    Args:
        pytorch_code: The PyTorch operator source code.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": _SYSTEM_KERNEL},
        {"role": "user", "content": _USER_KERNEL_TEMPLATE.format(pytorch_code=pytorch_code)},
    ]


# ---------------------------------------------------------------------------
# Free-form wrapper generation (after kernel is produced)
# ---------------------------------------------------------------------------

_SYSTEM_WRAPPER = """\
You are an expert GPU kernel engineer specializing in Triton. \
Given a PyTorch operator and the corresponding @triton.jit kernel that has already been written, \
write the Python wrapper function that:
1. Prepares tensors (contiguous, CUDA, correct dtype)
2. Computes the launch grid
3. Calls the kernel with all required arguments

The wrapper should have the same interface as the original PyTorch operator.

Output only the wrapper function wrapped in a ```python ... ``` markdown fence.\
"""

_USER_WRAPPER_TEMPLATE = """\
PyTorch operator:
```python
{pytorch_code}
```

Triton kernel (already written):
```python
{kernel_code}
```

Write the Python wrapper function that launches the kernel above.
"""


def format_wrapper_messages(pytorch_code: str, kernel_code: str) -> list[dict[str, str]]:
    """Return messages for free-form wrapper generation after kernel is produced.

    Args:
        pytorch_code: The PyTorch operator source code.
        kernel_code: The already-generated @triton.jit kernel code.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": _SYSTEM_WRAPPER},
        {
            "role": "user",
            "content": _USER_WRAPPER_TEMPLATE.format(
                pytorch_code=pytorch_code,
                kernel_code=kernel_code,
            ),
        },
    ]
