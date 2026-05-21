"""Load and cache the Triton EBNF grammar from triton_lexeme submodule."""

from pathlib import Path

_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "triton_lexeme" / "triton.ebnf"
_cache: str | None = None


def load_grammar() -> str:
    """Return the Triton EBNF grammar string. Cached after first read."""
    global _cache
    if _cache is None:
        _cache = _GRAMMAR_PATH.read_text(encoding="utf-8")
    return _cache
