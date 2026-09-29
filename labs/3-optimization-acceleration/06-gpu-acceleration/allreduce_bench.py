"""Measure NCCL all-reduce bandwidth (a mini nccl-tests) and compare with your cost model.

    torchrun --standalone --nproc_per_node=2 allreduce_bench.py

busbw = algbw × 2(N−1)/N is the number nccl-tests reports: it is comparable across GPU counts
and should approach the link bandwidth (PCIe Gen4 x16 ≈ 25 GB/s effective, NVLink hundreds of GB/s).
"""

import importlib.util
import os
import sys
import pathlib
import time

import torch
import torch.distributed as dist

spec = importlib.util.spec_from_file_location("impl", pathlib.Path(__file__).with_name(
    "exercises.py" if os.environ.get("USE_EXERCISES") == "1" else "solutions.py"))
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


def main():
    cuda = torch.cuda.is_available()
    if not cuda:  # CPU/gloo: bind to loopback so it works without hostname resolution
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"{'size':>10}{'time ms':>10}{'algbw GB/s':>12}{'busbw GB/s':>12}{'model ms':>10}")
    for mb in (1, 4, 16, 64, 256):
        x = torch.ones(mb * 2**20 // 4, device=device)  # fp32
        for _ in range(3):
            dist.all_reduce(x)
        if cuda:
            torch.cuda.synchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        if cuda:
            torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
        size = x.numel() * 4
        algbw = size / t
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            predicted = impl.allreduce_seconds(size, world, busbw) * 1000  # your model, fed the measured link speed
            print(f"{mb:>8}MB{t * 1000:>10.2f}{algbw / 1e9:>12.2f}{busbw / 1e9:>12.2f}{predicted:>10.2f}")
    if rank == 0:
        print("Small messages are latency-bound (low bandwidth); large ones approach the link limit.\n"
              "This is why DDP groups gradients into ~25 MB buckets.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
