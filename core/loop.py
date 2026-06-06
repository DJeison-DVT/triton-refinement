"""Refinement loop with trajectory logging for one operator."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.llm_client import LLMClient
from core.pattern_memory import PatternMemory
from prompts import translator, reviewer, fixer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IterationLog:
    """One step in the refinement trajectory."""

    iteration: int
    stage: str  # translate | compile | test | review | fix
    prompt_hash: str
    generation: str
    outcome: str  # pass | fail
    error: str | None
    timestamp: str


@dataclass
class RefinementResult:
    """Full result from a refinement run for one operator."""

    op_name: str
    final_code: str
    test_code: str
    passed: bool
    total_iterations: int
    trajectory: list[IterationLog] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _now() -> str:
    """Return UTC ISO8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _extract_code(response: str) -> str:
    """Extract code from a chat response that may contain markdown fences.

    Looks for ```python or bare ``` fences. If none found, returns the
    response as-is (stripped). This handles both instruct models that
    wrap code in fences and models that output raw code.
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _clean_kernel(kernel_code: str) -> str:
    """Strip trailing non-code from grammar-constrained kernel output.

    The grammar may allow the model to generate trailing content after
    the kernel function (markdown fences, natural language, examples).
    This strips everything after the last indented code line.
    """
    lines = kernel_code.split("\n")
    # Find the last line that looks like code (starts with space/def/import/@)
    last_code = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (line.startswith("    ") or line.startswith("def ")
                         or line.startswith("import ") or line.startswith("from ")
                         or line.startswith("@")):
            last_code = i
    return "\n".join(lines[:last_code + 1])


def _run_code(code: str, timeout: int = 120) -> tuple[bool, str]:
    """Write code to a temp file, run it, and return (success, stderr).

    Args:
        code: Python source code to execute.
        timeout: Maximum seconds to wait for the subprocess.

    Returns:
        Tuple of (success, stderr). success is True if the process exited 0.
    """
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired: process did not finish within {timeout}s"
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _translate(
    client: LLMClient,
    pytorch_code: str,
    grammar: str | None,
) -> tuple[str, list[IterationLog]]:
    """Translate PyTorch code to Triton.

    With grammar (two-step, completion API):
      1. Generate kernel with grammar constraint via complete().
      2. Generate wrapper free-form via complete() with stop sequences.
      3. Combine: kernel_code + wrapper_code.

    Without grammar (one-step, completion API):
      1. Generate full module via complete() with stop sequences.

    Returns:
        Tuple of (triton_code, trajectory_entries).
    """
    logs: list[IterationLog] = []

    if grammar:
        # --- Step 1: grammar-constrained kernel ---
        kernel_prompt = translator.format_kernel_prompt(pytorch_code)
        kernel_raw = client.complete(kernel_prompt, grammar=grammar, max_tokens=1024)
        kernel_code = _clean_kernel(kernel_raw)
        logs.append(
            IterationLog(
                iteration=0,
                stage="translate",
                prompt_hash=hashlib.sha256(kernel_prompt.encode()).hexdigest()[:16],
                generation=kernel_code,
                outcome="pass",
                error=None,
                timestamp=_now(),
            )
        )

        # --- Step 2: wrapper via chat API ---
        wrapper_msgs = translator.format_wrapper_messages(pytorch_code, kernel_code)
        wrapper_raw = client.generate(wrapper_msgs, max_tokens=512)
        wrapper_code = _extract_code(wrapper_raw)
        logs.append(
            IterationLog(
                iteration=0,
                stage="translate",
                prompt_hash=hashlib.sha256(
                    wrapper_msgs[-1]["content"].encode()
                ).hexdigest()[:16],
                generation=wrapper_code,
                outcome="pass",
                error=None,
                timestamp=_now(),
            )
        )

        # Ensure all imports are present regardless of what grammar/clean produced
        imports = "import torch\nimport triton\nimport triton.language as tl\n\n"
        # Strip any existing imports from kernel to avoid duplicates
        kernel_lines = kernel_code.strip().split("\n")
        kernel_body = "\n".join(
            l for l in kernel_lines
            if not l.startswith("import ") and not l.startswith("from ")
        )
        triton_code = imports + kernel_body.strip() + "\n\n" + wrapper_code
    else:
        # --- One-step free-form (completion API) ---
        full_prompt = translator.format_prompt(pytorch_code)
        raw = client.complete(
            full_prompt,
            max_tokens=2048,
            stop=["\n\n# ", "\n\nif __name__"],
        )
        triton_code = "import torch\nimport triton\nimport triton.language as tl\n\n" + raw
        logs.append(
            IterationLog(
                iteration=0,
                stage="translate",
                prompt_hash=hashlib.sha256(full_prompt.encode()).hexdigest()[:16],
                generation=triton_code,
                outcome="pass",
                error=None,
                timestamp=_now(),
            )
        )

    return triton_code, logs


# ---------------------------------------------------------------------------
# Fix (multi-turn chat)
# ---------------------------------------------------------------------------


def _chat_fix(
    client: LLMClient,
    pytorch_code: str,
    triton_code: str,
    error: str,
    fixer_messages: list[dict[str, str]],
    trajectory: list[IterationLog],
    iteration: int,
) -> tuple[str, list[dict[str, str]]]:
    """Ask the fixer to fix triton_code via multi-turn chat.

    On the first call, initializes fixer_messages with the system prompt
    and initial context. On subsequent calls, appends a follow-up user
    message. The assistant's response is appended to the thread.

    Returns:
        Tuple of (new_triton_code, updated fixer_messages).
    """
    if not fixer_messages:
        # First fix call — initialize the conversation
        fixer_messages = fixer.format_messages(pytorch_code, triton_code, error)
    else:
        # Subsequent call — append follow-up
        fixer_messages.append(fixer.format_followup(triton_code, error))

    raw = client.generate(fixer_messages, temperature=0.4, max_tokens=1024)

    # Append assistant response to conversation thread
    fixer_messages.append({"role": "assistant", "content": raw})

    # Extract code from potential markdown fences
    new_code = _extract_code(raw)

    trajectory.append(
        IterationLog(
            iteration=iteration,
            stage="fix",
            prompt_hash=hashlib.sha256(
                fixer_messages[-2]["content"].encode()
            ).hexdigest()[:16],
            generation=new_code,
            outcome="pass",
            error=None,
            timestamp=_now(),
        )
    )
    return new_code, fixer_messages


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _log(msg: str, verbose: bool) -> None:
    """Print a progress message if verbose is enabled."""
    if verbose:
        print(msg, flush=True)


def _write_work_file(work_dir: Path | None, name: str, code: str) -> None:
    """Write code to a work file so it can be inspected live."""
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / name).write_text(code, encoding="utf-8")


def generate_with_refinement(
    op_name: str,
    pytorch_code: str,
    test_code: str,
    client: LLMClient,
    *,
    grammar: str | None = None,
    max_iters: int = 5,
    pattern_memory: PatternMemory | None = None,
    verbose: bool = False,
    work_dir: Path | None = None,
) -> RefinementResult:
    """Orchestrate the full refinement pipeline for one operator.

    Pipeline:
        translate → [compile → test → review → fix]* → result

    Args:
        op_name: Name of the operator (used for logging and pattern memory).
        pytorch_code: The reference PyTorch operator source.
        test_code: Test code to run against the Triton implementation.
        client: LLMClient instance.
        grammar: Optional EBNF grammar string for constrained kernel generation.
        max_iters: Maximum number of refinement iterations.
        pattern_memory: Optional PatternMemory for storing/retrieving patterns.
        verbose: If True, print step-by-step progress to stdout.
        work_dir: If set, write triton code and tests to files here so they
                  can be inspected live during the run.

    Returns:
        RefinementResult with final_code, trajectory, pass/fail status.
    """
    trajectory: list[IterationLog] = []

    # Step 1: Initial translation
    _log(f"  [translate] Generating initial Triton code...", verbose)
    triton_code, translate_logs = _translate(client, pytorch_code, grammar)
    trajectory.extend(translate_logs)
    _log(f"  [translate] Done ({len(triton_code.splitlines())} lines)", verbose)
    _write_work_file(work_dir, f"{op_name}.py", triton_code)
    _write_work_file(work_dir, f"{op_name}_test.py", test_code)

    # Step 2: Refinement loop
    fixer_messages: list[dict[str, str]] = []  # built on first fix call
    i = 0
    for i in range(1, max_iters + 1):
        _log(f"  [iter {i}/{max_iters}] Compile check...", verbose)

        # --- Compile check ---
        compile_ok, compile_err = _run_code(triton_code)
        trajectory.append(
            IterationLog(
                iteration=i,
                stage="compile",
                prompt_hash="",
                generation=triton_code,
                outcome="pass" if compile_ok else "fail",
                error=compile_err if not compile_ok else None,
                timestamp=_now(),
            )
        )
        if not compile_ok:
            err_first_line = compile_err.strip().split("\n")[-1][:120]
            _log(f"  [iter {i}/{max_iters}] Compile FAIL: {err_first_line}", verbose)
            if i == max_iters:
                break
            _log(f"  [iter {i}/{max_iters}] Calling fixer...", verbose)
            old_code = triton_code
            triton_code, fixer_messages = _chat_fix(
                client, pytorch_code, triton_code, compile_err,
                fixer_messages, trajectory, i,
            )
            if triton_code == old_code:
                _log(f"  [iter {i}/{max_iters}] Fixer produced identical code — stopping early", verbose)
                break
            _write_work_file(work_dir, f"{op_name}.py", triton_code)
            continue

        _log(f"  [iter {i}/{max_iters}] Compile PASS. Running tests...", verbose)

        # --- Test run ---
        combined_code = triton_code + "\n\n" + test_code
        test_ok, test_err = _run_code(combined_code)
        trajectory.append(
            IterationLog(
                iteration=i,
                stage="test",
                prompt_hash="",
                generation=triton_code,
                outcome="pass" if test_ok else "fail",
                error=test_err if not test_ok else None,
                timestamp=_now(),
            )
        )
        if not test_ok:
            err_first_line = test_err.strip().split("\n")[-1][:120]
            _log(f"  [iter {i}/{max_iters}] Test FAIL: {err_first_line}", verbose)
            if i == max_iters:
                break
            _log(f"  [iter {i}/{max_iters}] Calling fixer...", verbose)
            old_code = triton_code
            triton_code, fixer_messages = _chat_fix(
                client, pytorch_code, triton_code, test_err,
                fixer_messages, trajectory, i,
            )
            if triton_code == old_code:
                _log(f"  [iter {i}/{max_iters}] Fixer produced identical code — stopping early", verbose)
                break
            _write_work_file(work_dir, f"{op_name}.py", triton_code)
            continue

        _log(f"  [iter {i}/{max_iters}] Tests PASS. Reviewing...", verbose)

        # --- Review (stateless chat call) ---
        patterns = pattern_memory.retrieve(pytorch_code) if pattern_memory else None
        review_msgs = reviewer.format_messages(triton_code, pytorch_code, patterns=patterns)
        review_response = client.generate(review_msgs)
        approved = "APPROVED" in review_response.strip().split("\n")[0]
        trajectory.append(
            IterationLog(
                iteration=i,
                stage="review",
                prompt_hash=hashlib.sha256(
                    review_msgs[-1]["content"].encode()
                ).hexdigest()[:16],
                generation=review_response,
                outcome="pass" if approved else "fail",
                error=None if approved else review_response,
                timestamp=_now(),
            )
        )

        if approved:
            _log(f"  [iter {i}/{max_iters}] Review APPROVED", verbose)
            if pattern_memory is not None:
                pattern_memory.store(op_name, triton_code, "pass")
            return RefinementResult(
                op_name=op_name,
                final_code=triton_code,
                test_code=test_code,
                passed=True,
                total_iterations=i,
                trajectory=trajectory,
            )

        # Reviewer rejected — inject feedback into fixer thread
        feedback_preview = review_response.strip().split("\n")[0][:120]
        _log(f"  [iter {i}/{max_iters}] Review REJECTED: {feedback_preview}", verbose)
        if i == max_iters:
            break
        _log(f"  [iter {i}/{max_iters}] Calling fixer...", verbose)
        old_code = triton_code
        triton_code, fixer_messages = _chat_fix(
            client, pytorch_code, triton_code, review_response,
            fixer_messages, trajectory, i,
        )
        if triton_code == old_code:
            _log(f"  [iter {i}/{max_iters}] Fixer produced identical code — stopping early", verbose)
            break
        _write_work_file(work_dir, f"{op_name}.py", triton_code)

    # Loop exhausted without approval
    _log(f"  Max iterations ({max_iters}) reached", verbose)
    if pattern_memory is not None:
        pattern_memory.store(op_name, triton_code, "fail")

    return RefinementResult(
        op_name=op_name,
        final_code=triton_code,
        test_code=test_code,
        passed=False,
        total_iterations=i,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# Trajectory persistence
# ---------------------------------------------------------------------------


def save_trajectory(result: RefinementResult, output_dir: Path) -> Path:
    """Write the trajectory to a JSONL file.

    Each line is a JSON object with keys:
        op_name, iteration, stage, prompt_hash, generation, outcome, error, timestamp.

    Args:
        result: The RefinementResult containing the trajectory.
        output_dir: Directory to write the file into (created if needed).

    Returns:
        Path to the written JSONL file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.op_name}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in result.trajectory:
            record = {
                "op_name": result.op_name,
                "iteration": entry.iteration,
                "stage": entry.stage,
                "prompt_hash": entry.prompt_hash,
                "generation": entry.generation,
                "outcome": entry.outcome,
                "error": entry.error,
                "timestamp": entry.timestamp,
            }
            f.write(json.dumps(record) + "\n")

    return out_path


# ---------------------------------------------------------------------------
# Per-op outcome extraction
# ---------------------------------------------------------------------------


def build_op_result(result: RefinementResult) -> dict:
    """Extract per-op outcome summary from a RefinementResult.

    Returns a dict suitable for writing to op_results.json:
        phase1_compile: True/False/None (None = never attempted, unlikely)
        phase2_test:    True/False/None (None = skipped, compile never passed)
        review_approved: True/False/None (None = skipped, test never passed)
        iterations_used: int
        repaired: bool (failed on first iteration, passed later)
        phases_reached: list of stage names that were executed
        phases_skipped: list of stage names never reached
        final_status: "pass" or "fail"
    """
    compiles = [e for e in result.trajectory if e.stage == "compile"]
    tests = [e for e in result.trajectory if e.stage == "test"]
    reviews = [e for e in result.trajectory if e.stage == "review"]

    phase1 = compiles[-1].outcome == "pass" if compiles else None
    phase2 = tests[-1].outcome == "pass" if tests else None
    review = reviews[-1].outcome == "pass" if reviews else None

    reached = []
    if compiles:
        reached.append("compile")
    if tests:
        reached.append("test")
    if reviews:
        reached.append("review")

    all_phases = ["compile", "test", "review"]
    skipped = [p for p in all_phases if p not in reached]

    first_iter_compiles = [e for e in compiles if e.iteration == 1]
    first_iter_tests = [e for e in tests if e.iteration == 1]
    first_failed = False
    if first_iter_compiles and first_iter_compiles[0].outcome == "fail":
        first_failed = True
    if first_iter_tests and first_iter_tests[0].outcome == "fail":
        first_failed = True
    repaired = first_failed and result.passed

    return {
        "phase1_compile": phase1,
        "phase2_test": phase2,
        "review_approved": review,
        "iterations_used": result.total_iterations,
        "repaired": repaired,
        "phases_reached": reached,
        "phases_skipped": skipped,
        "final_status": "pass" if result.passed else "fail",
    }
