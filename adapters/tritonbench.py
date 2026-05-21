"""Bridge to TritonBench4Modal's evaluate_only entrypoint."""

import json
import os
import re
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
    phase1_ops: list[str]
    phase2_passed: int
    phase2_rate: float
    phase2_ops: list[str]
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
        "--predictions", str(predictions_path.resolve()),
        "--output-subdir", output_subdir,
    ]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60 * 60,  # 1 hour max
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"evaluate_only failed (exit {result.returncode}):\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )

    # Parse the JSON summary from stdout — Modal prints progress lines before
    # the final JSON, so find the last JSON object in the output.
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"evaluate_only returned empty stdout.\nSTDERR:\n{result.stderr}"
        )

    # The JSON object is printed mid-stream (Modal adds progress lines before
    # and after).  Scan for the first line starting with '{' and consume until
    # the matching '}'.
    m = re.search(r"^(\{.*?^\})", stdout, re.MULTILINE | re.DOTALL)
    if m:
        summary = json.loads(m.group(1))
    else:
        raise RuntimeError(
            f"Could not parse JSON from evaluate_only output.\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return EvalResult(
        total_predictions=summary["total_predictions"],
        phase1_passed=summary["phase1_call_acc"]["passed"],
        phase1_rate=summary["phase1_call_acc"]["rate"],
        phase1_ops=summary["phase1_call_acc"].get("ops", []),
        phase2_passed=summary["phase2_exec_acc"]["passed"],
        phase2_rate=summary["phase2_exec_acc"]["rate"],
        phase2_ops=summary["phase2_exec_acc"].get("ops", []),
        phase3_speedup=summary["phase3_efficiency"].get("speedup_vs_pytorch"),
    )


def update_op_results_with_eval(
    op_results_path: Path,
    eval_result: EvalResult,
    stem_to_opname: dict[str, str],
) -> None:
    """Update op_results.json with real TritonBench4Modal evaluation data.

    Adds eval_phase1, eval_phase2, and eval_speedup fields to each op entry.
    Phase 1/2 are per-op (from the ops lists). Speedup is aggregate-only
    (TritonBench doesn't provide per-op speedup).

    Args:
        op_results_path: Path to existing op_results.json
        eval_result: Parsed EvalResult from evaluate()
        stem_to_opname: Mapping from TritonBench file stems to our op_names
            (from adapters.dataset.build_stem_to_opname)
    """
    op_results = json.loads(op_results_path.read_text(encoding="utf-8"))

    # Build sets of op_names that passed each phase
    phase1_passed = {stem_to_opname[s] for s in eval_result.phase1_ops if s in stem_to_opname}
    phase2_passed = {stem_to_opname[s] for s in eval_result.phase2_ops if s in stem_to_opname}

    for op_name, entry in op_results.items():
        entry["eval_phase1"] = op_name in phase1_passed
        entry["eval_phase2"] = op_name in phase2_passed
        entry["eval_speedup"] = eval_result.phase3_speedup  # aggregate only

    op_results_path.write_text(
        json.dumps(op_results, indent=2), encoding="utf-8",
    )
