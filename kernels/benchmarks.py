"""
Kernel Benchmarks: Fused Flash Attention JVP vs PyTorch Native SDPA

Measures wall-clock latency and peak VRAM for the custom Triton kernel
against a naive PyTorch baseline that runs F.scaled_dot_product_attention
followed by a separate forward-mode AD pass (torch.func.jvp).

Usage:
    python -m kernels.benchmarks
    python -m kernels.benchmarks --seq-lens 256 512 1024 2048 4096
    python -m kernels.benchmarks --head-dim 128 --warmup 20 --iters 100
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels.tvm_flash_jvp import flash_attention_jvp

# ── helpers ────────────────────────────────────────────────────────────

def _reset_memory_stats(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _make_inputs(n_ctx: int, head_dim: int, device: torch.device):
    """Return (q, k, v, tq, tk, tv) as contiguous fp16 tensors."""
    def _rand():
        return torch.randn(n_ctx, head_dim, device=device, dtype=torch.float16)
    return _rand(), _rand(), _rand(), _rand(), _rand(), _rand()


# ── baselines ──────────────────────────────────────────────────────────

def _pytorch_sdpa_jvp(q, k, v, tq, tk, tv):
    """Naive baseline: SDPA forward + torch.func.jvp for tangent."""
    # SDPA expects (batch, heads, seq, dim) — unsqueeze to (1, 1, N, D).
    def _sdpa_fn(q_, k_, v_):
        # Force the math backend because the fused CUDA SDPA kernels do not
        # currently expose the forward AD needed by torch.func.jvp.
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q_.unsqueeze(0).unsqueeze(0),
                k_.unsqueeze(0).unsqueeze(0),
                v_.unsqueeze(0).unsqueeze(0),
            ).squeeze(0).squeeze(0)

    out, tout = torch.func.jvp(_sdpa_fn, (q, k, v), (tq, tk, tv))
    return out, tout


# ── benchmark runner ───────────────────────────────────────────────────

def _bench_fn(fn, warmup: int, iters: int, device: torch.device):
    """Time *fn* and return summary latency, throughput, and VRAM metrics."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    _reset_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    vram = _peak_vram_mb(device)
    times.sort()
    median_ms = times[len(times) // 2]
    p95_index = max(0, math.ceil(len(times) * 0.95) - 1)
    p95_ms = times[p95_index]
    mean_ms = sum(times) / len(times)
    throughput_ops_s = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return median_ms, p95_ms, throughput_ops_s, vram


def run_benchmark(
    seq_lens: list[int],
    head_dim: int,
    warmup: int,
    iters: int,
):
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — benchmarks require a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    rows: list[dict] = []

    for n_ctx in seq_lens:
        q, k, v, tq, tk, tv = _make_inputs(n_ctx, head_dim, device)

        # --- Triton fused kernel ---
        triton_fn = partial(flash_attention_jvp, q, k, v, tq, tk, tv)
        triton_ms, triton_p95_ms, triton_throughput, triton_vram = _bench_fn(
            triton_fn, warmup, iters, device
        )

        # --- PyTorch SDPA + jvp baseline ---
        # Need float32 for torch.func.jvp
        q32, k32, v32 = q.float(), k.float(), v.float()
        tq32, tk32, tv32 = tq.float(), tk.float(), tv.float()
        pytorch_fn = partial(_pytorch_sdpa_jvp, q32, k32, v32, tq32, tk32, tv32)
        pytorch_ms, pytorch_p95_ms, pytorch_throughput, pytorch_vram = _bench_fn(
            pytorch_fn, warmup, iters, device
        )

        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
        vram_reduction = (1 - triton_vram / pytorch_vram) * 100 if pytorch_vram > 0 else 0

        rows.append({
            "n_ctx": n_ctx,
            "triton_ms": triton_ms,
            "triton_p95_ms": triton_p95_ms,
            "triton_throughput": triton_throughput,
            "pytorch_ms": pytorch_ms,
            "pytorch_p95_ms": pytorch_p95_ms,
            "pytorch_throughput": pytorch_throughput,
            "speedup": speedup,
            "triton_vram": triton_vram,
            "pytorch_vram": pytorch_vram,
            "vram_reduction": vram_reduction,
        })

    # ── Print results ──────────────────────────────────────────────────

    print(f"\nHardware: {gpu_name}")
    print(f"HEAD_DIM={head_dim}  BLOCK_SIZE=64  dtype=float16")
    print(f"Warmup={warmup}  Iters={iters}\n")

    # Markdown table for README
    header = (
        "| Seq Len | Triton p95 (ms) | PyTorch p95 (ms) | Triton Throughput (ops/s) "
        "| PyTorch Throughput (ops/s) | Triton VRAM (MB) | PyTorch VRAM (MB) | Speedup | VRAM Reduction |"
    )
    sep = (
        "|---------|-----------------|------------------|---------------------------|"
        "----------------------------|------------------|-------------------|---------|----------------|"
    )
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['n_ctx']:>7} "
            f"| {r['triton_p95_ms']:>15.2f} "
            f"| {r['pytorch_p95_ms']:>16.2f} "
            f"| {r['triton_throughput']:>25.2f} "
            f"| {r['pytorch_throughput']:>26.2f} "
            f"| {r['triton_vram']:>16.1f} "
            f"| {r['pytorch_vram']:>17.1f} "
            f"| {r['speedup']:>6.2f}x "
            f"| {r['vram_reduction']:>13.1f}% |"
        )

    print()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flash Attention JVP benchmark")
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    args = parser.parse_args()

    run_benchmark(args.seq_lens, args.head_dim, args.warmup, args.iters)


if __name__ == "__main__":
    main()
