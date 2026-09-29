"""Verify `average_gradients` under a real multi-process launch.

    torchrun --standalone --nproc_per_node=2 ddp_check.py [--solutions]

Each rank computes gradients on its own shard of a batch, calls average_gradients, and rank 0
checks the result equals the single-process gradient on the full batch. Uses NCCL when there is
one GPU per rank, gloo on CPU otherwise.
"""

import importlib.util
import os
import pathlib
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def load_impl():
    name = "solutions.py" if "--solutions" in sys.argv or os.environ.get("LAB_SOLUTIONS") == "1" else "exercises.py"
    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    use_nccl = torch.cuda.is_available() and torch.cuda.device_count() >= int(os.environ["WORLD_SIZE"])
    if not use_nccl:  # CPU/gloo: bind to loopback so it works without hostname resolution
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("nccl" if use_nccl else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if use_nccl else torch.device("cpu")
    if use_nccl:
        torch.cuda.set_device(device)

    torch.manual_seed(0)  # identical init and data on every rank
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
    x, y = torch.randn(32, 8, device=device), torch.randn(32, 1, device=device)

    # Reference: full-batch gradient on one process.
    F.mse_loss(model(x), y).backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    # DDP: each rank sees its shard, then gradients are averaged.
    xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]
    F.mse_loss(model(xs), ys).backward()
    load_impl().average_gradients(model)

    ok = all(torch.allclose(p.grad, r, atol=1e-6) for p, r in zip(model.parameters(), reference))
    flag = torch.tensor([int(ok)], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"backend={dist.get_backend()} world={world} averaged gradients match full batch: {bool(flag.item())}")
    dist.destroy_process_group()
    sys.exit(0 if flag.item() else 1)


if __name__ == "__main__":
    main()
