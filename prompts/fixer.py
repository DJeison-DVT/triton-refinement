"""Prompt templates for Triton code fixing (chat mode).

Chat-style prompts for instruct models. The fixer receives the buggy code
and error in a multi-turn conversation, accumulating context across iterations.
"""

_SYSTEM = """\
You are an expert Triton GPU kernel engineer. Fix the buggy Triton implementation.

The fixed module MUST contain exactly:
1. Imports (torch, triton, triton.language as tl)
2. One or more @triton.jit kernel functions (20-60 lines each)
3. A Python wrapper function matching the PyTorch operator's signature

Do NOT include test code, example usage, function calls, or `if __name__` blocks.
Output in a ```python``` fence.\
"""

_INITIAL_TEMPLATE = """\
Reference PyTorch:
{pytorch_code}

Buggy Triton:
{triton_code}

Error:
{error}

Fix it."""

_FOLLOWUP_TEMPLATE = """\
The fix still has issues.

Current code:
{triton_code}

Error:
{error}

Fix it."""

# Max chars of error to include in messages
_ERROR_LIMIT = 1500


def _truncate_error(error: str) -> str:
    """Truncate error to the last _ERROR_LIMIT characters."""
    tail = error.strip()[-_ERROR_LIMIT:]
    return tail


def format_messages(
    pytorch_code: str,
    triton_code: str,
    error: str,
) -> list[dict[str, str]]:
    """Return initial messages for Triton code fixing.

    This starts a new fixer conversation thread. Use format_followup()
    for subsequent iterations in the same thread.
    """
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _INITIAL_TEMPLATE.format(
                pytorch_code=pytorch_code,
                triton_code=triton_code,
                error=_truncate_error(error),
            ),
        },
    ]


def format_followup(triton_code: str, error: str) -> dict[str, str]:
    """Return a follow-up user message for the next fix iteration.

    Appended to the existing fixer_messages list along with the
    assistant's prior response.
    """
    return {
        "role": "user",
        "content": _FOLLOWUP_TEMPLATE.format(
            triton_code=triton_code,
            error=_truncate_error(error),
        ),
    }
