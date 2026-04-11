from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
attention = pytest.importorskip("torch.nn.attention")
SDPBackend = attention.SDPBackend
sdpa_kernel = attention.sdpa_kernel

pytest.importorskip("triton")


def _reference_sdpa_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _sdpa_fn(q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel([SDPBackend.MATH]):
            return (
                F.scaled_dot_product_attention(
                    q_.unsqueeze(0).unsqueeze(0),
                    k_.unsqueeze(0).unsqueeze(0),
                    v_.unsqueeze(0).unsqueeze(0),
                )
                .squeeze(0)
                .squeeze(0)
            )

    return torch.func.jvp(_sdpa_fn, (q, k, v), (tq, tk, tv))  # type: ignore[misc]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton Flash Attention JVP requires CUDA")
def test_flash_attention_jvp_matches_pytorch_math_backend() -> None:
    from kernels.tvm_flash_jvp import flash_attention_jvp

    torch.manual_seed(7)
    device = torch.device("cuda")
    n_ctx = 64
    head_dim = 32

    q = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)
    tq = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)
    tk = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)
    tv = torch.randn(n_ctx, head_dim, device=device, dtype=torch.float32)

    expected, expected_tangent = _reference_sdpa_jvp(q, k, v, tq, tk, tv)
    actual, actual_tangent = flash_attention_jvp(
        q.half(),
        k.half(),
        v.half(),
        tq.half(),
        tk.half(),
        tv.half(),
    )

    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_tangent.float(), expected_tangent, rtol=6e-2, atol=6e-2)
