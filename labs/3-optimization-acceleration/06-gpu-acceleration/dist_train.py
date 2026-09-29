"""Train a small GPT with DDP or FSDP under torchrun; report throughput and per-GPU memory.

    torchrun --standalone --nproc_per_node=2 dist_train.py --strategy ddp
    torchrun --standalone --nproc_per_node=2 dist_train.py --strategy fsdp --size large
    nsys profile -t cuda,nvtx,osrt -o dist --force-overwrite true \
        torchrun --standalone --nproc_per_node=2 dist_train.py --nvtx --steps 10

`make gpu-06` runs the whole comparison (scaling, DDP vs FSDP memory, profiles).
Falls back to gloo on CPU so you can dry-run it on a laptop with --size tiny.
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

SIZES = {  # d_model, layers, heads
    "tiny": (128, 2, 4),
    "base": (1024, 12, 16),     # ~0.19B params: fits with DDP on a 16 GB T4
    "large": (2048, 16, 16),    # ~0.87B params: 16 B/param ≈ 14 GB of model state per GPU under DDP; FSDP shards it
}
VOCAB, SEQ = 32000, 512


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)  # GPT-2 init; unit variance + tied head → huge logits
        self.pos = nn.Parameter(torch.zeros(SEQ, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[: ids.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.emb.weight.T  # tied LM head


@contextmanager
def nvtx(name, enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        yield
        torch.cuda.nvtx.range_pop()
    else:
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    ap.add_argument("--size", choices=list(SIZES), default="base")
    ap.add_argument("--batch", type=int, default=8, help="micro-batch per GPU")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--nvtx", action="store_true", help="annotate phases for Nsight Systems")
    ap.add_argument("--torch-profile", metavar="DIR", help="write a PyTorch profiler trace (TensorBoard/Perfetto)")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    if args.strategy == "fsdp" and not cuda:
        raise SystemExit("FSDP here needs NVIDIA GPUs; use --strategy ddp for a CPU dry run.")
    if not cuda:  # CPU/gloo: bind to loopback so it works without hostname resolution
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    half = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.bfloat16

    torch.manual_seed(0)
    model = GPT(*SIZES[args.size])
    n_params = sum(p.numel() for p in model.parameters())

    if args.strategy == "ddp":
        model = DDP(model.to(device), device_ids=[local] if cuda else None)
        autocast = torch.autocast(device.type, dtype=half) if cuda else nullcontext()
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        model = FSDP(model, device_id=device if cuda else None, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     auto_wrap_policy=ModuleWrapPolicy({Block}),
                     mixed_precision=MixedPrecision(param_dtype=half, reduce_dtype=half, buffer_dtype=half) if cuda else None)
        autocast = nullcontext()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=cuda)
    scaler = torch.amp.GradScaler(enabled=cuda and half == torch.float16 and args.strategy == "ddp")
    ids = torch.randint(0, VOCAB, (args.batch, SEQ), device=device)

    profiler = None
    if args.torch_profile and rank == 0:
        from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        profiler = profile(activities=acts, schedule=schedule(wait=2, warmup=2, active=3),
                           on_trace_ready=tensorboard_trace_handler(args.torch_profile), profile_memory=True)
        profiler.start()

    warmup, t0 = 5, None
    for step in range(args.steps):
        if step == warmup:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        with nvtx(f"step {step}", args.nvtx):
            with nvtx("forward", args.nvtx), autocast:
                logits = model(ids[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
            with nvtx("backward", args.nvtx):   # DDP all-reduces gradient buckets during this phase
                scaler.scale(loss).backward()
            with nvtx("optimizer", args.nvtx):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        if profiler:
            profiler.step()
    if profiler:
        profiler.stop()
    if cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    tokens = (args.steps - warmup) * args.batch * (SEQ - 1) * world
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"RESULT strategy={args.strategy} size={args.size} params={n_params / 1e6:.0f}M world={world} "
              f"tokens_per_s={tokens / elapsed:,.0f} peak_gb_per_gpu={peak.item():.2f} loss={loss.item():.3f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
