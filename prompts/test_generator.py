"""Prompt templates for test generation."""

_SYSTEM = """\
You are an expert GPU kernel engineer. Given a PyTorch operator implementation, \
generate comprehensive test code that exercises the operator on GPU.

Your tests should:
- Import torch and any other necessary libraries
- Create representative input tensors on CUDA
- Call the operator and compare results against a reference (e.g., torch built-ins)
- Cover edge cases: small sizes, large sizes, non-contiguous tensors, etc.
- Use assertions or pytest-style checks

Output the test code wrapped in a ```python ... ``` markdown fence.\
"""

_USER_TEMPLATE = """\
Generate GPU kernel tests for the following PyTorch operator:

```python
{pytorch_code}
```
"""


def format_messages(pytorch_code: str) -> list[dict[str, str]]:
    """Return OpenAI-style messages for test generation.

    Args:
        pytorch_code: The PyTorch operator source code.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TEMPLATE.format(pytorch_code=pytorch_code)},
    ]
