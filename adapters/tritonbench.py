"""Bridge to TritonBench4Modal's evaluate_only entrypoint."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TRITONBENCH4MODAL_DIR = Path(__file__).resolve().parent.parent / "TritonBench4Modal"


@dataclass
class EvalResult:
    """Parsed output from TritonBench4Modal evaluation."""

    total_predictions: int
    phase1_passed: int
    phase1_rate: float
    phase2_passed: int
    phase2_rate: float
    phase3_speedup: float | None


def write_predictions(
    predictions: list[dict[str, str]],
    output_path: Path,
) -> Path:
    """Write predictions.jsonl in TritonBench4Modal's expected format.

    Each entry must have:
        - instruction: original Alpaca instruction (verbatim)
        - predict: generated Triton code

    Args:
        predictions: list of {instruction, predict} dicts
        output_path: where to write the file

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    return output_path


def evaluate(
    predictions_path: Path,
    output_subdir: str = "results",
) -> EvalResult:
    """Invoke TritonBench4Modal's evaluate_only via Modal CLI.

    Args:
        predictions_path: path to local predictions.jsonl
        output_subdir: subdirectory name for results in the Modal volume

    Returns:
        Parsed EvalResult with phase 1/2/3 scores.
    """
    cmd = [
        "modal", "run",
        str(_TRITONBENCH4MODAL_DIR / "modal_app.py") + "::evaluate_only",
        "--",
        "--predictions", str(predictions_path.resolve()),
        "--output-subdir", output_subdir,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60 * 60,  # 1 hour max
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"evaluate_only failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Parse the JSON summary from stdout
    summary = json.loads(result.stdout.strip())
    return EvalResult(
        total_predictions=summary["total_predictions"],
        phase1_passed=summary["phase1_call_acc"]["passed"],
        phase1_rate=summary["phase1_call_acc"]["rate"],
        phase2_passed=summary["phase2_exec_acc"]["passed"],
        phase2_rate=summary["phase2_exec_acc"]["rate"],
        phase3_speedup=summary["phase3_efficiency"].get("speedup_vs_pytorch"),
    )
