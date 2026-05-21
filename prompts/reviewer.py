"""Prompt templates for Triton code review."""

_SYSTEM_BASE = """\
You are an expert Triton GPU kernel reviewer. Your task is to evaluate a Triton \
kernel implementation for correctness and best practices.

Check for:
- Correct use of tl.load / tl.store with mask to avoid out-of-bounds memory access
- Correct grid and block dimension computation
- Proper handling of non-contiguous tensors (use .contiguous() in wrapper)
- Avoiding common Triton pitfalls (e.g., implicit broadcasting errors, missing masks)
- Numerical correctness relative to the reference PyTorch operator
- Clean, readable code style

{patterns_section}\
If the code looks correct and follows best practices, respond with exactly "APPROVED" on its own line. \
Otherwise, describe each issue concisely so the fixer can address them.\
"""

_PATTERNS_SECTION = """\
Known patterns from previous translations:
{patterns_list}

"""

_USER_TEMPLATE = """\
Reference PyTorch operator:
```python
{pytorch_code}
```

Triton implementation to review:
```python
{triton_code}
```

Please review the Triton implementation above.
"""


def format_messages(
    triton_code: str,
    pytorch_code: str,
    patterns: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Return OpenAI-style messages for Triton code review.

    Args:
        triton_code: The Triton kernel + wrapper code to review.
        pytorch_code: The reference PyTorch operator.
        patterns: Optional list of pattern-memory dicts from PatternMemory.retrieve(),
                  each with keys {op_name, pattern, outcome}. When None, the
                  patterns section is omitted.

    Returns:
        A list of two dicts with 'role' and 'content' keys.
    """
    if patterns:
        lines = []
        for p in patterns:
            lines.append(f"- [{p['outcome']}] {p['op_name']}: {p['pattern']}")
        patterns_list = "\n".join(lines)
        patterns_section = _PATTERNS_SECTION.format(patterns_list=patterns_list)
    else:
        patterns_section = ""

    system_content = _SYSTEM_BASE.format(patterns_section=patterns_section)

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                pytorch_code=pytorch_code,
                triton_code=triton_code,
            ),
        },
    ]
