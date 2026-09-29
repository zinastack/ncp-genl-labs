"""Device helpers shared by every gpu_lab.py.

Cheap lab GPUs differ: L4 (Ada, sm_89) has bf16 + FP8, T4 (Turing, sm_75) has neither and needs
fp16. These helpers pick the right device/dtype so the same script runs on L4, T4, A10G, an
Apple-silicon Mac (MPS) or plain CPU.
"""

import gc
import os
import sys

import torch


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def half_dtype() -> torch.dtype:
    """bf16 where the hardware supports it, else fp16 (T4, V100, MPS), else fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def require_cuda(min_gpus: int = 1) -> None:
    """Exit with a helpful message when a GPU-only cell/script runs without enough GPUs."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < min_gpus and os.environ.get("ALLOW_NO_GPU") != "1":
        sys.exit(f"This part needs {min_gpus} NVIDIA GPU(s), found {n}. Run it on the Brev/AWS instance "
                 f"(see brev/README.md), or set ALLOW_NO_GPU=1 to try anyway.")


def free() -> None:
    """Release memory from deleted models before the next measurement."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def summary() -> str:
    if torch.cuda.is_available():
        rows = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            rows.append(f"cuda:{i} {p.name} · {p.total_memory / 1e9:.0f} GB · sm_{p.major}{p.minor}")
        return "\n".join(rows) + f"\nhalf dtype: {half_dtype()}"
    return f"{device()} (no NVIDIA GPU) · half dtype: {half_dtype()}"
