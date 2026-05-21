"""Prompt templates for Triton code fixing."""

_SYSTEM = """\
You are an expert Triton GPU kernel engineer. You will be given:
1. A reference PyTorch operator
2. A Triton implementation that has a bug or error
3. The error message or test failure output

Your job is to fix the Triton implementation so it:
- Compiles without errors
- Produces numerically correct results matching the PyTorch reference
- Follows Triton best practices (masked loads/stores, correct grid, etc.)

Output the complete fixed Triton module (kernel + wrapper) wrapped in a \
```python ... ``` markdown fence. Do not include partial snippets — output the \
full corrected module.\
"""

_USER_TEMPLATE = """\
Reference PyTorch operator:
```python
{pytorch_code}
```

Triton implementation with issue:
```python
{triton_code}
```

Error / failure output:
```
{error}
```

Please fix the Triton implementation.
"""


def format_messages(
    pytorch_code: str,
    triton_code: str,
    error: str,
) -> list[dict[str, str]]:
    """Return OpenAI-style messages for Triton code fixing.

    Args:
        pytorch_code: The reference PyTorch operator source code.
        triton_code: The buggy Triton implementation (kernel + wrapper).
        error: The compile error, runtime error, or test failure text.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                pytorch_code=pytorch_code,
                triton_code=triton_code,
                error=error,
            ),
        },
    ]
