"""Deploy vLLM on Modal as an OpenAI-compatible API server.

Serves any of the 3 experiment models with XGrammar constrained decoding.

Usage:
    modal deploy modal_vllm.py                                   # deploy default (Qwen2.5-Coder-7B)
    MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct modal deploy modal_vllm.py

    modal serve modal_vllm.py                                    # dev mode (hot reload)

Once deployed, point LLMClient at the Modal URL:
    LLMClient(base_url="https://<your-app>.modal.run/v1", api_key="not-needed")
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "triton-refinement-vllm"

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
GPU_TYPE = os.environ.get("GPU_TYPE", "A10G")
VLLM_PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm",
        "hf_transfer",
    )
)

app = modal.App(APP_NAME, image=image)

# Persist model weights across cold starts
model_volume = modal.Volume.from_name(
    "vllm-model-cache", create_if_missing=True
)


@app.function(
    gpu=GPU_TYPE,
    timeout=2 * 60 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes={"/root/.cache/huggingface": model_volume},
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    """Launch vLLM's OpenAI-compatible server as a subprocess."""
    cmd = " ".join([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
    ])
    subprocess.Popen(cmd, shell=True)
