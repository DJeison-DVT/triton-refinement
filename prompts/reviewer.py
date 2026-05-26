"""Prompt templates for Triton code review (chat mode).

Chat-style prompts for instruct models. The reviewer is called statelessly
each iteration — it does not accumulate conversation history.
"""

_SYSTEM_BASE = """\
You are an expert Triton GPU kernel reviewer. Check for correct masks, grid \
dimensions, contiguous tensors, and numerical correctness.

{patterns_section}\
If correct, respond with exactly "APPROVED". Otherwise, describe issues concisely.\
"""

_USER_TEMPLATE = """\
PyTorch:
{pytorch_code}

Triton:
{triton_code}

Review it."""


def format_messages(
    triton_code: str,
    pytorch_code: str,
    patterns: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Return messages for code review (instruct models)."""
    patterns_section = _format_patterns(patterns)
    return [
        {"role": "system", "content": _SYSTEM_BASE.format(patterns_section=patterns_section)},
        {"role": "user", "content": _USER_TEMPLATE.format(
            pytorch_code=pytorch_code, triton_code=triton_code)},
    ]


def _format_patterns(patterns: list[dict] | None) -> str:
    if not patterns:
        return ""
    lines = [f"- [{p['outcome']}] {p['op_name']}: {p['pattern']}" for p in patterns]
    return "Known patterns:\n" + "\n".join(lines) + "\n\n"
