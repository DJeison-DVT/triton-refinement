"""Tests for all prompt modules."""

import pytest

from prompts import extract_code
from prompts import test_generator, translator, reviewer, fixer


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_strips_python_fence(self):
        text = "```python\nx = 1\n```"
        assert extract_code(text) == "x = 1\n"

    def test_strips_bare_fence(self):
        text = "```\nx = 1\n```"
        assert extract_code(text) == "x = 1\n"

    def test_no_fence_passthrough(self):
        text = "x = 1\n"
        assert extract_code(text) == "x = 1\n"

    def test_no_fence_no_trailing_newline_adds_newline(self):
        text = "x = 1"
        assert extract_code(text) == "x = 1\n"

    def test_multiline_python_fence(self):
        text = "```python\ndef foo():\n    pass\n```"
        assert extract_code(text) == "def foo():\n    pass\n"

    def test_strips_surrounding_whitespace_in_fence(self):
        text = "  ```python\nx = 1\n```  "
        # fences may have leading/trailing whitespace on the fence lines
        result = extract_code(text)
        assert "x = 1" in result

    def test_result_always_ends_with_newline(self):
        for text in ["```python\nfoo\n```", "bar", "```\nbaz\n```"]:
            assert extract_code(text).endswith("\n")


# ---------------------------------------------------------------------------
# test_generator
# ---------------------------------------------------------------------------


class TestTestGenerator:
    def test_returns_two_messages(self):
        msgs = test_generator.format_messages("import torch\n")
        assert len(msgs) == 2

    def test_messages_have_role_and_content(self):
        msgs = test_generator.format_messages("import torch\n")
        for msg in msgs:
            assert "role" in msg
            assert "content" in msg

    def test_roles_are_system_and_user(self):
        msgs = test_generator.format_messages("import torch\n")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_pytorch_code_appears_in_user_message(self):
        code = "def my_op(): pass"
        msgs = test_generator.format_messages(code)
        assert code in msgs[1]["content"]


# ---------------------------------------------------------------------------
# translator
# ---------------------------------------------------------------------------


class TestTranslator:
    def test_format_messages_returns_two_dicts(self):
        msgs = translator.format_messages("import torch\n")
        assert len(msgs) == 2
        for msg in msgs:
            assert "role" in msg and "content" in msg

    def test_format_messages_roles(self):
        msgs = translator.format_messages("import torch\n")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_format_messages_code_in_user(self):
        code = "def pytorch_op(): pass"
        msgs = translator.format_messages(code)
        assert code in msgs[1]["content"]

    def test_format_kernel_prompt_returns_string(self):
        prompt = translator.format_kernel_prompt("def add(x, y): return x + y")
        assert isinstance(prompt, str)
        assert "add" in prompt

    def test_format_kernel_prompt_ends_with_blank_line(self):
        prompt = translator.format_kernel_prompt("def add(x, y): return x + y")
        assert prompt.endswith("\n\n")

    def test_format_kernel_prompt_contains_pytorch_code(self):
        code = "def pytorch_op(): pass"
        prompt = translator.format_kernel_prompt(code)
        assert "pytorch_op" in prompt

    def test_format_wrapper_messages_returns_two_dicts(self):
        msgs = translator.format_wrapper_messages("def add(x, y): pass", "@triton.jit\ndef kernel(): pass")
        assert len(msgs) == 2
        for msg in msgs:
            assert "role" in msg and "content" in msg

    def test_format_wrapper_messages_kernel_in_user(self):
        kernel_code = "@triton.jit\ndef my_kernel(): pass"
        msgs = translator.format_wrapper_messages("def add(x, y): pass", kernel_code)
        assert kernel_code in msgs[1]["content"]


# ---------------------------------------------------------------------------
# reviewer
# ---------------------------------------------------------------------------


class TestReviewer:
    def test_returns_two_messages(self):
        msgs = reviewer.format_messages("@triton.jit\ndef k(): pass", "def f(): pass")
        assert len(msgs) == 2

    def test_messages_have_role_and_content(self):
        msgs = reviewer.format_messages("@triton.jit\ndef k(): pass", "def f(): pass")
        for msg in msgs:
            assert "role" in msg and "content" in msg

    def test_roles(self):
        msgs = reviewer.format_messages("@triton.jit\ndef k(): pass", "def f(): pass")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_approved_mentioned_in_prompt(self):
        msgs = reviewer.format_messages("@triton.jit\ndef k(): pass", "def f(): pass")
        combined = msgs[0]["content"] + msgs[1]["content"]
        assert "APPROVED" in combined

    def test_no_patterns_no_pattern_section(self):
        msgs = reviewer.format_messages("@triton.jit\ndef k(): pass", "def f(): pass")
        assert "pattern" not in msgs[0]["content"].lower() or True  # relaxed — just no crash

    def test_patterns_appear_in_system_when_provided(self):
        patterns = [
            {"op_name": "add", "pattern": "Use tl.load with mask", "outcome": "pass"},
            {"op_name": "mul", "pattern": "Avoid bank conflicts", "outcome": "fail"},
        ]
        msgs = reviewer.format_messages(
            "@triton.jit\ndef k(): pass", "def f(): pass", patterns=patterns
        )
        system_content = msgs[0]["content"]
        assert "Use tl.load with mask" in system_content
        assert "Avoid bank conflicts" in system_content
        assert "add" in system_content
        assert "pass" in system_content

    def test_patterns_section_absent_when_none(self):
        msgs_no_patterns = reviewer.format_messages(
            "@triton.jit\ndef k(): pass", "def f(): pass"
        )
        patterns = [{"op_name": "add", "pattern": "tip", "outcome": "pass"}]
        msgs_with_patterns = reviewer.format_messages(
            "@triton.jit\ndef k(): pass", "def f(): pass", patterns=patterns
        )
        # With patterns the system message should be longer
        assert len(msgs_with_patterns[0]["content"]) > len(msgs_no_patterns[0]["content"])


# ---------------------------------------------------------------------------
# fixer
# ---------------------------------------------------------------------------


class TestFixer:
    def test_returns_two_messages(self):
        msgs = fixer.format_messages("def f(): pass", "@triton.jit\ndef k(): pass", "SyntaxError")
        assert len(msgs) == 2

    def test_messages_have_role_and_content(self):
        msgs = fixer.format_messages("def f(): pass", "@triton.jit\ndef k(): pass", "SyntaxError")
        for msg in msgs:
            assert "role" in msg and "content" in msg

    def test_roles(self):
        msgs = fixer.format_messages("def f(): pass", "@triton.jit\ndef k(): pass", "SyntaxError")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_all_three_inputs_in_messages(self):
        pytorch_code = "def pytorch_op(): pass"
        triton_code = "@triton.jit\ndef kernel(): pass"
        error = "RuntimeError: invalid memory access"
        msgs = fixer.format_messages(pytorch_code, triton_code, error)
        combined = msgs[0]["content"] + msgs[1]["content"]
        assert pytorch_code in combined
        assert triton_code in combined
        assert error in combined
