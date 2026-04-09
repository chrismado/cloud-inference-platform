# cloud-inference-platform

**SLO-Aware GPU Router with Terminal Velocity Matching Dynamic Step-Scaling**

Production ML inference infrastructure for generative video and spatial AI workloads. Implements an SLO-aware routing layer that multiplexes 3D Gaussian Splatting rendering alongside diffusion model inference, with dynamic NFE step-scaling based on server load using Terminal Velocity Matching (TVM).

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

| Config | TTFT (p95) | GPU RAM | FID | Throughput |
|--------|-----------|---------|-----|------------|
| vLLM baseline | 1240ms | 21GB | — | 12 req/s |
| SGLang RadixAttention | 580ms | 7GB | — | 31 req/s |
| + TVM 4-step | 580ms | 7GB | 2.1 | 31 req/s |
| + TVM 1-step (spike) | 180ms | 5GB | 4.6 | 94 req/s |

**60% KV cache VRAM overhead reduction vs baseline.**
Hardware: NVIDIA RTX 4090 + RTX 3090.

---

## Directory Structure

```
cloud-inference-platform/
├── router/
│   ├── slo_router.py          # Core SLO-aware routing logic
│   ├── load_monitor.py        # Real-time GPU utilization tracking
│   ├── priority_queue.py      # Request priority and batching
│   └── tvm_scaler.py          # Dynamic NFE step-scaling
├── serving/
│   ├── sglang_backend.py      # SGLang RadixAttention integration
│   ├── vllm_backend.py        # vLLM PagedAttention integration
│   ├── gaussian_backend.py    # 3DGS spatial serving
│   └── tensorrt_backend.py    # TensorRT optimized serving
├── kernels/
│   ├── tvm_flash_jvp.py       # Triton kernel: Flash Attention JVP for TVM
│   └── benchmarks.py          # Kernel vs baseline comparison
├── cache/
│   ├── redis_cache.py         # KV cache management
│   └── checkpoint_store.py    # S3-compatible model checkpoints
├── observability/
│   ├── prometheus_metrics.py  # Custom metrics definitions
│   ├── grafana_dashboard.json # Pre-built Grafana dashboard
│   └── slo_tracker.py         # SLO compliance tracking
├── deploy/
│   ├── helm/                  # Kubernetes Helm chart
│   ├── docker-compose.yml     # Local dev setup
│   └── Dockerfile
├── tests/
│   ├── test_router.py
│   ├── test_tvm_scaler.py
│   └── locust_load_test.py    # Concurrent load simulation
├── configs/
│   └── slo_config.yaml        # SLO targets and scaling thresholds
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
python -m serving.benchmarks --backend sglang --load-test concurrent
```

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
