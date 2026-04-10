"""
Custom Triton Kernel: Flash Attention JVP for Terminal Velocity Matching

Standard Flash Attention lacks native JVP support, causing massive VRAM overhead
during TVM training. This kernel fuses JVP directly into the forward pass.

TVM core equation (Zhou et al., Luma AI, ICLR 2026):
  d/ds f_θ(x_t, t, s) = F_θ(x_t, t, s) + (s-t) · ∂_s F_θ(x_t, t, s)

Implementation:
  - Load Q,K,V blocks + tangents (tQ, tK, tV)
  - Compute S = QKᵀ AND tS = tQKᵀ + QtKᵀ simultaneously
  - Forward-mode AD: same memory as forward pass
  - Block sizes MUST be powers of 2 (e.g. 64, 128)

Targets: 65% speedup vs naive PyTorch SDPA, major VRAM reduction
Validate: amorehead/jvp_flash_attention
"""

import math

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 64  # Must be power of 2 — do NOT use 72, 96, etc.


@triton.jit
def flash_attention_jvp_kernel(
    Q: object,
    K: object,
    V: object,
    tQ: object,
    tK: object,
    tV: object,
    O: object,
    tO: object,
    stride_qm: int,
    stride_qk: int,
    stride_km: int,
    stride_kk: int,
    N_CTX: int,
    OUTPUT_DTYPE: object,
    HEAD_DIM: int,
    BLOCK_M: int,
    BLOCK_N: int,
) -> None:
    """Fused Flash Attention + JVP with tangent propagation.

    Each program instance handles one BLOCK_M-row tile of the output.
    We iterate over BLOCK_N-column tiles of K/V, accumulating:
      - O   = softmax(QKᵀ / √d) V              (standard attention)
      - tO  = d/dε [ softmax((Q+εtQ)(K+εtK)ᵀ/√d)(V+εtV) ] |_{ε=0}

    Tangent of softmax(S)·V where S = QKᵀ/√d:
      tO = P·tV + (tP)·V
    where P = softmax(S) and tP = P*(tS - rowsum(P*tS)) with tS = (tQ·Kᵀ + Q·tKᵀ)/√d.

    All intermediate products are kept in SRAM — same memory footprint as
    a standard Flash Attention forward pass.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers for Q-block and tQ-block  [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    tq_ptrs = tQ + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    tq_block = tl.load(tq_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # HEAD_DIM is a constexpr Python int inside Triton JIT, so compute the
    # scale as a Python constant instead of calling tensor methods on it.
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Running accumulators (online softmax + tangent)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)  # row-max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # row-sum(exp)
    o_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    to_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    rowsum_ts = tl.zeros([BLOCK_M], dtype=tl.float32)  # Σ P·tS

    num_blocks_n = tl.cdiv(N_CTX, BLOCK_N)
    for j in range(num_blocks_n):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K, V, tK, tV blocks  [BLOCK_N, HEAD_DIM]
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
        v_ptrs = V + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
        tk_ptrs = tK + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
        tv_ptrs = tV + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk

        mask_n = offs_n[:, None] < N_CTX
        k_block = tl.load(k_ptrs, mask=mask_n, other=0.0)
        v_block = tl.load(v_ptrs, mask=mask_n, other=0.0)
        tk_block = tl.load(tk_ptrs, mask=mask_n, other=0.0)
        tv_block = tl.load(tv_ptrs, mask=mask_n, other=0.0)

        # S = QKᵀ / √d   [BLOCK_M, BLOCK_N]
        s_ij = tl.dot(q_block, tl.trans(k_block)) * scale

        # tS = (tQ·Kᵀ + Q·tKᵀ) / √d
        ts_ij = (tl.dot(tq_block, tl.trans(k_block)) + tl.dot(q_block, tl.trans(tk_block))) * scale

        # --- online softmax rescaling ---
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])

        l_new = alpha * l_i + tl.sum(p_ij, axis=1)

        # Rescale running accumulators
        o_acc = o_acc * alpha[:, None]
        to_acc = to_acc * alpha[:, None]
        rowsum_ts = rowsum_ts * alpha

        # Accumulate primal: O += P·V
        o_acc += tl.dot(p_ij.to(v_block.dtype), v_block)

        # Accumulate tangent pieces: tO += P·tV + (P*tS)·V
        to_acc += tl.dot(p_ij.to(tv_block.dtype), tv_block)

        pts_ij = p_ij * ts_ij  # element-wise
        to_acc += tl.dot(pts_ij.to(v_block.dtype), v_block)

        # Track Σ_j P_ij * tS_ij per row (needed for softmax tangent correction)
        rowsum_ts += tl.sum(pts_ij, axis=1)

        m_i = m_new
        l_i = l_new

    # Finalize: divide by l_i (softmax denominator)
    inv_l = 1.0 / l_i
    o_acc = o_acc * inv_l[:, None]

    # Tangent correction: subtract P·V · Σ(P·tS) term
    # tO = (tO_acc - O · rowsum_ts) / l_i
    to_acc = (to_acc - o_acc * (rowsum_ts[:, None] * inv_l[:, None])) * inv_l[:, None]

    # Store O and tO  [BLOCK_M, HEAD_DIM]
    o_ptrs = O + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    to_ptrs = tO + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    mask_m = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, o_acc.to(OUTPUT_DTYPE), mask=mask_m)
    tl.store(to_ptrs, to_acc.to(OUTPUT_DTYPE), mask=mask_m)


def flash_attention_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output, tangent_output) via fused forward + JVP.

    Parameters
    ----------
    q, k, v : torch.Tensor   — (N_CTX, HEAD_DIM), float16
    tq, tk, tv : torch.Tensor — tangent vectors, same shape

    Returns
    -------
    (O, tO) : tuple[torch.Tensor, torch.Tensor]
    """
    assert q.shape == k.shape == v.shape
    assert tq.shape == tk.shape == tv.shape
    assert q.shape == tq.shape
    N_CTX, HEAD_DIM = q.shape

    o = torch.empty_like(q)
    to_ = torch.empty_like(q)

    grid = (triton.cdiv(N_CTX, BLOCK_SIZE),)

    flash_attention_jvp_kernel[grid](
        q,
        k,
        v,
        tq,
        tk,
        tv,
        o,
        to_,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_SIZE,
        BLOCK_N=BLOCK_SIZE,
        OUTPUT_DTYPE=(
            tl.float16 if q.dtype == torch.float16 else tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
        ),
    )

    return o, to_
