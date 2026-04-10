# Performance

## Triton Flash Attention JVP Kernel Benchmark

Measured via `python -m kernels.benchmarks` with `HEAD_DIM=64`, `BLOCK_SIZE=64`,
`warmup=20`, `iters=200`. Median of 3 fresh runs.

| Seq Len | Triton p95 (ms) | PyTorch p95 (ms) | Speedup | VRAM Reduction |
|---------|-----------------|------------------|---------|----------------|
| 256 | 0.06 | 1.11 | 17.13x | 97.5% |
| 512 | 0.08 | 1.24 | 10.62x | 96.6% |
| 1024 | 0.11 | 1.00 | 7.33x | 96.8% |
| 2048 | 0.16 | 1.61 | 7.39x | 97.9% |
| 4096 | 0.28 | 3.99 | 12.65x | 98.8% |

The PyTorch baseline uses the math SDPA backend with `torch.func.jvp` for a valid
forward-mode AD comparison.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.11.0+cu128
- Python: 3.12.2
- Measured: April 9, 2026

## Optimization Roadmap

1. **Multi-head / batch dimension sweeps** — Current benchmarks use single-head single-batch. Profile scaling behavior.
2. **Concurrent load profiling** — Measure kernel performance under realistic multiplexed 3DGS + DiT workloads.
3. **FP8 quantization** — Test E4M3/E5M2 paths for further VRAM reduction on Ada/Hopper GPUs.
4. **End-to-end SLO compliance** — Benchmark the full router → TVM scaler → kernel path under Locust load.
5. **Memory fragmentation analysis** — Profile GPU memory fragmentation during mixed 3DGS rendering + diffusion inference.
