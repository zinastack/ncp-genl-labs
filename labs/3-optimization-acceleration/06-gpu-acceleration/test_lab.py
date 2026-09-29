import os
import pathlib
import socket
import subprocess
import sys

import numpy as np
import pytest

GB = 1e9


def test_1_zero(lab):
    # ZeRO paper example: 7.5B params on 64 GPUs
    got = [lab.zero_bytes_per_gpu(7.5e9, 64, s) / GB for s in range(4)]
    assert got == pytest.approx([120, 31.4, 16.6, 1.875], abs=0.1)


def test_2_pipeline_bubble(lab):
    assert lab.pipeline_bubble_fraction(4, 4) == pytest.approx(3 / 7)
    assert lab.pipeline_bubble_fraction(4, 32) == pytest.approx(3 / 35)
    assert lab.pipeline_bubble_fraction(4, 4, virtual_stages=3) == pytest.approx(1 / 5)
    assert lab.pipeline_bubble_fraction(1, 8) == 0


def test_3_allreduce(lab):
    size = 14e9  # 7B bf16 gradients
    assert lab.ring_allreduce_bytes(size, 8) == pytest.approx(24.5e9)
    assert lab.ring_allreduce_bytes(size, 1024) < 2 * size, "per-GPU traffic is ~constant in N"
    assert lab.allreduce_seconds(size, 8, 400e9) == pytest.approx(0.06125)


def test_4_rank_layout(lab):
    # 16 GPUs = 2 nodes x 8; TP=4, PP=2 → DP=2
    assert lab.rank_coords(0, 16, 4, 2) == (0, 0, 0)
    assert lab.rank_coords(5, 16, 4, 2) == (1, 1, 0)
    assert lab.rank_coords(13, 16, 4, 2) == (1, 1, 1)
    assert lab.tensor_parallel_group(6, 4) == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        lab.rank_coords(0, 12, 8, 1)


def test_5_tensor_parallel(lab):
    rng = np.random.default_rng(0)
    x, w1, w2 = rng.normal(size=(3, 8)), rng.normal(size=(8, 16)), rng.normal(size=(16, 8))
    parts = lab.column_parallel(x, w1, 4)
    assert len(parts) == 4 and parts[0].shape == (3, 4)
    np.testing.assert_allclose(np.concatenate(parts, axis=1), x @ w1)  # all-gather reproduces full output
    np.testing.assert_allclose(lab.megatron_mlp(x, w1, w2, 4), np.maximum(x @ w1, 0) @ w2)


def test_6_ddp_under_torchrun(lab):
    here = pathlib.Path(__file__).parent
    with socket.socket() as sock:  # free port; explicit IPv4 avoids hostname/IPv6 lookups on laptops
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={port}", "ddp_check.py"]
    if lab.__file__.endswith("solutions.py"):
        args.append("--solutions")
    proc = subprocess.run(args, cwd=here, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert "match full batch: True" in proc.stdout


def test_7_flops_and_mfu(lab):
    # Llama-2-7B: 2T tokens
    assert lab.training_flops(7e9, 2e12) == pytest.approx(8.4e22)
    # 1024 A100s (312 TFLOPS bf16) processing 3.2M tokens/s on a 7B model
    assert lab.mfu(3.2e6, 7e9, 1024, 312e12) == pytest.approx(0.42, abs=0.01)


def test_8_roofline(lab):
    h100_peak, h100_bw = 989e12, 3.35e12
    decode = lab.gemm_arithmetic_intensity(1, 8192, 8192)      # batch-1 GEMV
    prefill = lab.gemm_arithmetic_intensity(4096, 8192, 8192)
    assert decode < 1.1
    assert prefill > 1000
    assert lab.is_memory_bound(decode, h100_peak, h100_bw)
    assert not lab.is_memory_bound(prefill, h100_peak, h100_bw)
