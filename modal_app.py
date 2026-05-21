"""Modal orchestrator: run the full refinement experiment on Modal GPUs.

Usage:
    modal run modal_app.py::run --model Qwen/Qwen2.5-Coder-7B \
      --vllm-url https://<user>--triton-refinement-vllm-serve.modal.run/v1 \
      --limit 5 --seed 42
"""

import json
import modal

app = modal.App("triton-refinement")

experiment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "triton>=3.0.0",
        "openai>=1.50",
    )
    .add_local_dir("core", "/app/core")
    .add_local_dir("prompts", "/app/prompts")
    .add_local_dir("adapters", "/app/adapters")
    .add_local_dir("scripts", "/app/scripts")
    .add_local_dir("triton_lexeme", "/app/triton_lexeme")
    .add_local_dir("TritonBench/data", "/data")
)

volume = modal.Volume.from_name(
    "triton-refinement-results", create_if_missing=True,
)


CONDITIONS = {
    "single-shot": {"max_iterations": 1, "grammar_constrained": True, "pattern_memory": False},
    "refinement": {"max_iterations": 5, "grammar_constrained": True, "pattern_memory": "inmemory"},
}


@app.function(
    image=experiment_image,
    gpu="T4",
    timeout=3600 * 6,
    volumes={"/results": volume},
)
def run(
    model: str = "Qwen/Qwen2.5-Coder-7B",
    vllm_url: str = "",
    condition: str = "refinement",
    limit: int | None = None,
    seed: int = 42,
):
    """Run the refinement experiment on Modal."""
    import sys
    sys.path.insert(0, "/app")

    import adapters.dataset as ds
    from pathlib import Path
    ds._DATA_DIR = Path("/data")
    ds._ALPACA_PATH = ds._DATA_DIR / "TritonBench_T_simp_alpac_v1.json"
    ds._METADATA_PATH = ds._DATA_DIR / "TritonBench_T_v1.jsonl"
    ds._PYTORCH_DIR = ds._DATA_DIR / "TritonBench_T_v1"

    from adapters.dataset import load_ops
    from adapters.tritonbench import write_predictions
    from core.grammar import load_grammar
    from core.llm_client import LLMClient
    from core.loop import generate_with_refinement, save_trajectory, build_op_result
    from core.memory_inmemory import InMemoryPatternMemory
    from prompts import extract_code, test_generator

    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    settings = CONDITIONS[condition]
    max_iters = settings["max_iterations"]
    use_grammar = settings["grammar_constrained"]
    mem_type = settings["pattern_memory"]

    client = LLMClient(base_url=vllm_url, model=model, api_key="EMPTY")
    ops = load_ops(limit=limit)
    ebnf = load_grammar() if use_grammar else None
    mem = InMemoryPatternMemory() if mem_type == "inmemory" else None

    model_short = model.split("/")[-1]
    run_id = f"{model_short}_{condition}_seed{seed}"
    output_dir = Path(f"/results/{run_id}")
    traj_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = []
    predictions = []

    for idx, op in enumerate(ops):
        print(f"[{idx + 1}/{len(ops)}] {op.op_name}")

        test_msgs = test_generator.format_messages(op.pytorch_code)
        raw_tests = client.generate(test_msgs)
        test_code = extract_code(raw_tests)

        result = generate_with_refinement(
            op_name=op.op_name,
            pytorch_code=op.pytorch_code,
            test_code=test_code,
            client=client,
            grammar=ebnf,
            max_iters=max_iters,
            pattern_memory=mem,
        )

        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} after {result.total_iterations} iterations")

        save_trajectory(result, traj_dir)
        results_list.append(result)
        predictions.append({
            "instruction": op.instruction,
            "predict": result.final_code,
        })

    pred_path = output_dir / "predictions.jsonl"
    write_predictions(predictions, pred_path)

    op_results = {}
    for result_item, op in zip(results_list, ops):
        op_results[op.op_name] = build_op_result(result_item)
    (output_dir / "op_results.json").write_text(
        json.dumps(op_results, indent=2), encoding="utf-8",
    )

    n_passed = sum(1 for r in results_list if r.passed)
    summary = {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "seed": seed,
        "max_iters": max_iters,
        "grammar": use_grammar,
        "pattern_memory": mem_type if mem_type else "none",
        "n_ops": len(results_list),
        "n_passed": n_passed,
        "pass_rate": n_passed / len(results_list) if results_list else 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    volume.commit()

    print(f"\nDone: {n_passed}/{len(results_list)} passed")
    print(f"Results saved to Modal volume: /results/{run_id}/")
    return summary
