# cloud-inference-platform

**SLO-Aware GPU Router with Terminal Velocity Matching Dynamic Step-Scaling**

Production ML inference infrastructure for generative video and spatial AI workloads. Implements an SLO-aware routing layer that multiplexes 3D Gaussian Splatting rendering alongside diffusion model inference, with dynamic NFE step-scaling based on server load using Terminal Velocity Matching (TVM).

---

## Portfolio Context

This repo is part of [Creative AI Workflows](https://github.com/chrismado/creative-ai-workflows), a portfolio showcase connecting generative video, 3D scene review, creative QA, and enterprise deployment.

In that system, `cloud-inference-platform` is the **enterprise rollout layer**. It answers the question that comes after a great demo: how does a creative AI workflow stay fast, observable, and reliable when a real team starts using it?

### Customer-Facing Use Case

An enterprise customer wants to use custom AI media workflows across a team, not just in a one-off prototype. This repo is positioned around adoption constraints: latency, routing, service goals, quality tradeoffs, observability, and operational trust.

### Demo Narrative

- Start with a creative workflow that works well for one user but slows down under team load.
- Show routing decisions, latency targets, and quality tradeoffs during normal load versus a spike.
- Close by explaining how operational trust affects creative adoption.

---

## The Problem

vLLM and SGLang are token-centric. Serving continuous 3D Gaussian Splatting data (spherical harmonics arrays) alongside LLM inference creates GPU memory fragmentation and latency spikes. No existing open-source system handles this multiplexing.

Additionally, standard inference routers use fixed diffusion step counts regardless of server load — wasting compute during low traffic and violating SLOs during spikes. TVM enables dynamic NFE scaling (4-step → 1-step) while maintaining FID bounds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Request Router                        │
│         SLO-aware · Load-monitored · Priority queue     │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼────────┐
│  Text Prefill  │      │  3DGS Rendering  │
│  SGLang        │      │  gsplat CUDA     │
│  RadixAttention│      │  Spatial serving │
└───────┬────────┘      └────────┬────────┘
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │     TVM Step Scaler     │
        │  4-step → 1-step DiT    │
        │  based on p95 latency   │
        └─────────────────────────┘
```

### TVM Dynamic Step-Scaling
During normal load: 4-step DiT generation (full quality)
Under traffic spike: autonomously degrades to 1-step TVM generation
SLO maintained: p95 latency target never violated
FID bound: <2.5 degradation from 4-step baseline

Core TVM math (Zhou et al., Luma AI, ICLR 2026):
```
d/ds f_θ(x_t, t, s) = F_θ(x_t, t, s) + (s-t) · ∂_s F_θ(x_t, t, s)
```
The last term requires JVP through the network — implemented via custom Triton kernel for Flash Attention JVP (see `kernels/tvm_flash_jvp.py`).

---

## Stack

- **Routing:** SGLang RadixAttention for text prefill, custom gang-scheduler for 3DGS
- **Inference engines:** vLLM (PagedAttention), SGLang (RadixAttention), TensorRT
- **Step scaling:** Terminal Velocity Matching (TVM) — ICLR 2026
- **Observability:** Prometheus + Grafana, p50/p95/p99 latency tracking
- **Caching:** Redis for KV cache, S3-compatible checkpoint storage
- **Orchestration:** Kubernetes via Helm chart, Docker
- **Load testing:** Locust with concurrent user simulation

---

## Benchmarks

The repo currently ships the kernel microbenchmark in `kernels/benchmarks.py`.
The table below reports median values from three fresh runs of that benchmark on a
single NVIDIA GeForce RTX 3090 (`HEAD_DIM=64`, `BLOCK_SIZE=64`, `warmup=20`,
`iters=200`).

| Seq Len | Triton p95 (ms) | PyTorch p95 (ms) | Triton Throughput (ops/s) | PyTorch Throughput (ops/s) | Triton VRAM (MB) | PyTorch VRAM (MB) | Speedup | VRAM Reduction |
|---------|-----------------|------------------|---------------------------|----------------------------|------------------|-------------------|---------|----------------|
| 256 | 0.06 | 1.11 | 20716.22 | 1186.17 | 0.2 | 10.1 | 17.13x | 97.5% |
| 512 | 0.08 | 1.24 | 14815.69 | 1298.02 | 0.5 | 14.5 | 10.62x | 96.6% |
| 1024 | 0.11 | 1.00 | 9990.63 | 1267.91 | 1.0 | 30.9 | 7.33x | 96.8% |
| 2048 | 0.16 | 1.61 | 6677.54 | 846.65 | 2.0 | 93.6 | 7.39x | 97.9% |
| 4096 | 0.28 | 3.99 | 3825.00 | 288.16 | 4.0 | 339.2 | 12.65x | 98.8% |

**98.8% peak VRAM reduction at seq len 4096 vs the PyTorch SDPA+jvp baseline.**
Benchmarks measured on an NVIDIA GeForce RTX 3090 with PyTorch 2.11.0+cu128 and
Python 3.12.2 on April 9, 2026.

---

## Directory Structure

```
cloud-inference-platform/
├── .gitignore
├── LICENSE
├── cache/
│   ├── __init__.py
│   ├── checkpoint_store.py
│   └── redis_cache.py
├── configs/
│   └── slo_config.yaml
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── prometheus.yml
│   └── helm/
│       ├── Chart.yaml
│       ├── templates/
│       │   ├── deployment.yaml
│       │   ├── ingress.yaml
│       │   └── service.yaml
│       └── values.yaml
├── kernels/
│   ├── benchmarks.py
│   └── tvm_flash_jvp.py
├── observability/
│   ├── __init__.py
│   ├── grafana_dashboard.json
│   ├── prometheus_metrics.py
│   └── slo_tracker.py
├── router/
│   ├── __init__.py
│   ├── load_monitor.py
│   ├── priority_queue.py
│   ├── slo_router.py
│   └── tvm_scaler.py
├── serving/
│   ├── __init__.py
│   ├── gaussian_backend.py
│   ├── sglang_backend.py
│   ├── tensorrt_backend.py
│   └── vllm_backend.py
├── tests/
│   ├── __init__.py
│   ├── locust_load_test.py
│   ├── conftest.py
│   ├── test_router.py
│   └── test_tvm_scaler.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/chrismado/cloud-inference-platform
cd cloud-inference-platform
pip install -r requirements.txt

# Start with Docker
docker-compose up

# Or Kubernetes
helm install cloud-inference ./deploy/helm

# Run benchmarks
python -m kernels.benchmarks --seq-len 1024 --head-dim 64
```

---

## What I Learned

- **Flash Attention still needs explicit JVP support.** The fused SDPA kernels are fast for standard attention, but `torch.func.jvp` does not work through the efficient CUDA backend, so a TVM-style forward-mode path has to be implemented or the baseline has to fall back to the math kernel.
- **Forward-mode AD is practical when the derivative is part of the forward pass.** TVM needs the directional derivative alongside the primal output, so carrying tangents through the kernel avoids building a separate reverse-mode graph and cuts the peak VRAM cost substantially.
- **Power-of-2 block sizes are not a style preference here.** The Triton kernel relies on tile shapes that map cleanly onto SRAM usage, warp scheduling, and vectorized memory access, so sizes like `64` and `128` compile and run predictably while odd sizes hurt codegen or correctness.
- **Benchmarks need the baseline defined as carefully as the optimized path.** The repo only produced trustworthy numbers after fixing the Triton compile-time scale bug and forcing the PyTorch reference path onto the math SDPA backend so the JVP comparison was both valid and reproducible.

---

## References

1. **Terminal Velocity Matching** — Zhou et al. (Luma AI), ICLR 2026. Generalizes flow matching for single-stage generative training. 1.99 FID on ImageNet-256 with 4 NFE.
2. **SGLang: Efficient Execution of Structured Language Model Programs** — Zheng et al. RadixAttention KV cache reuse. sgl-project/sglang.
3. **Efficient Memory Management for LLM Serving with PagedAttention** — Kwon et al., SOSP 2023. vLLM foundation.
4. **SwapServeLLM** — Engine-agnostic model hot-swapping framework.
5. **Drift: SLO-Aware GPU Frequency Scaling** — arxiv 2508.16449.
6. **amorehead/jvp_flash_attention** — Community Flash Attention JVP implementation used for validation.

---

*Built on RTX 4090 + RTX 3090. Targeting Luma AI, Decart, Pika Labs inference engineering roles.*
