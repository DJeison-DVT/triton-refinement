"""Prompt utilities for the Triton refinement pipeline."""

import re


def extract_code(text: str) -> str:
    """Strip markdown code fences and return the code inside.

    Handles:
    - ```python ... ``` fences
    - ``` ... ``` bare fences
    - Plain text with no fence (returned as-is)

    Always returns a string ending with a newline.
    """
    # Match ```python or ``` opening fence, capture everything until closing ```
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text.strip())
    if match:
        code = match.group(1)
    else:
        code = text

    # Ensure trailing newline
    if not code.endswith("\n"):
        code += "\n"
    return code
