"""Load PyTorch operators from TritonBench for the refinement pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "TritonBench" / "data"
_ALPACA_PATH = _DATA_DIR / "TritonBench_T_simp_alpac_v1.json"
_METADATA_PATH = _DATA_DIR / "TritonBench_T_v1.jsonl"
_PYTORCH_DIR = _DATA_DIR / "TritonBench_T_v1"


@dataclass
class Op:
    """A single PyTorch operator with its instruction and source code."""

    op_name: str
    instruction: str
    pytorch_code: str


def load_ops(limit: int | None = None) -> list[Op]:
    """Load operators from TritonBench dataset.

    Joins the Alpaca instructions with JSONL metadata and PyTorch source files.

    Args:
        limit: Only load the first N ops (for smoke tests).

    Returns:
        List of Op dataclasses with instruction, pytorch source, and op name.
    """
    # Load metadata (JSON array despite .jsonl extension)
    metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))

    # Load Alpaca instructions (same order as metadata)
    alpaca = json.loads(_ALPACA_PATH.read_text(encoding="utf-8"))

    # Build lookup: file stem -> alpaca instruction
    # Metadata and Alpaca are both 166 items but may be in different order.
    # Use the metadata's 'file' field to match with PyTorch source files,
    # and pair with alpaca by index (they share the same ordering).
    ops: list[Op] = []
    for meta, alp in zip(metadata, alpaca):
        if limit and len(ops) >= limit:
            break

        py_file = _PYTORCH_DIR / meta["file"]
        if not py_file.exists():
            continue

        # Read only the function definition (above the test separator)
        source = py_file.read_text(encoding="utf-8")
        # TritonBench files have a separator line of '#' before test code
        sep = "#" * 20
        if sep in source:
            source = source[: source.index(sep)].rstrip()

        ops.append(Op(
            op_name=meta["name"],
            instruction=alp["instruction"],
            pytorch_code=source,
        ))

    return ops


def build_stem_to_opname() -> dict[str, str]:
    """Build a mapping from TritonBench file stems to op names.

    TritonBench4Modal returns file stems (e.g., "add", "tanh") in evaluation
    results. Our pipeline uses op names from metadata (e.g., "torch.add",
    "torch.tanh"). This mapping bridges the two.
    """
    metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return {
        meta["file"].replace(".py", ""): meta["name"]
        for meta in metadata
    }
