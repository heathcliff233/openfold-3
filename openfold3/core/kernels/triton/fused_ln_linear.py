# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused LayerNorm-to-Linear
# (Triton) with training backward.

"""Fused LayerNorm → Linear (Triton).

One program owns a row tile, computes LN in registers, then loops over
output-channel tiles. ``M`` is not an autotune or specialize key. Training
saves ``x`` / ``mean`` / ``rstd`` and rematerializes the LN output in exclusive
split-M ``dW`` and fused ``dX`` kernels. Ineligible shapes use
``F.layer_norm`` + ``F.linear``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# Pair-scale LN sites are <= 384; 512 is the one-pass SMEM / register budget.
_MAX_C_IN = 512
_MAX_C_OUT = 512
_MIN_M = 4096
_BWD_SPLIT_M = 16
_TRUE = {"1", "true", "yes", "on"}


def is_fused_ln_linear_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_LN_LINEAR", "1").strip().lower() in _TRUE


def is_fused_ln_linear_eligible(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    c_in = gamma.shape[0]
    masters = (gamma, weight)
    if beta is not None:
        masters += (beta,)
    if bias is not None:
        masters += (bias,)
    return (
        is_fused_ln_linear_enabled()
        and _TRITON_AVAILABLE
        and x.is_cuda
        and gamma.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and gamma.is_contiguous()
        and weight.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and all(w.dtype == torch.float32 for w in masters)
        and weight.shape[-1] == c_in
        and c_in <= _MAX_C_IN
        and weight.shape[0] <= _MAX_C_OUT
        and (x.numel() // c_in) >= _MIN_M
    )


def eager_ln_linear(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Matching-precision eager LN → Linear (bf16 downcasts masters)."""
    if x.dtype == torch.bfloat16:
        gamma = gamma.to(dtype=x.dtype)
        beta = beta.to(dtype=x.dtype) if beta is not None else None
        weight = weight.to(dtype=x.dtype)
        bias = bias.to(dtype=x.dtype) if bias is not None else None
    return F.linear(F.layer_norm(x, (gamma.shape[0],), gamma, beta, eps), weight, bias)


if _TRITON_AVAILABLE:

    def _configs(tiles):
        return [
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n},
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for block_m, block_n, num_warps, num_stages in tiles
        ]

    _GEMM_TILES = (
        (16, 32, 4, 1),
        (16, 64, 4, 1),
        (16, 64, 4, 2),
        (16, 128, 4, 1),
        (32, 32, 4, 1),
        (32, 64, 4, 1),
        (32, 64, 4, 2),
        (32, 128, 8, 1),
        (64, 64, 4, 1),
        (64, 128, 8, 1),
    )
    _DW_TILES = (
        (16, 32, 4, 1),
        (16, 64, 4, 1),
        (16, 64, 4, 2),
        (32, 32, 4, 1),
        (32, 64, 4, 1),
        (32, 64, 4, 2),
    )

    def _prune_by_precision(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        n = kwargs.get("N", named_args.get("N"))
        k = kwargs.get("K", named_args.get("K"))
        n_block = max(_next_power_of_two(int(n)), 16) if n is not None else None
        kept = []
        for cfg in configs:
            block_m = cfg.kwargs["BLOCK_M"]
            block_n = cfg.kwargs["BLOCK_N"]
            if n_block is not None and block_n > n_block:
                continue
            if k is not None and int(k) >= 256 and block_m >= 64:
                continue
            if mode == "ieee" and block_m >= 64:
                continue
            kept.append(cfg)
        return kept or configs[:1]

    def _autotuned(tiles, key, restore, strides, ptrs):
        def decorator(fn):
            return triton.autotune(
                configs=_configs(tiles),
                key=key,
                prune_configs_by={"early_config_prune": _prune_by_precision},
                restore_value=restore,
            )(
                triton.jit(
                    do_not_specialize=list(strides),
                    do_not_specialize_on_alignment=list(ptrs),
                )(fn)
            )

        return decorator

    @triton.jit
    def _round_to_tf32(x):
        ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
        return tl.inline_asm_elementwise(
            ASM, "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _dot_f32(a, b, GEMM_MODE: tl.constexpr):
        if GEMM_MODE == "bf16":
            return tl.dot(
                a.to(tl.bfloat16),
                b.to(tl.bfloat16),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        if GEMM_MODE == "tf32":
            return tl.dot(
                _round_to_tf32(a),
                _round_to_tf32(b),
                input_precision="tf32",
                out_dtype=tl.float32,
            )
        return tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)

    @triton.jit
    def _load2d(ptr, offs_m, offs_n, stride_m, m_mask, n_mask):
        return tl.load(
            ptr + offs_m[:, None] * stride_m + offs_n[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

    @triton.jit
    def _store2d(ptr, val, offs_m, offs_n, stride_m, m_mask, n_mask):
        tl.store(
            ptr + offs_m[:, None] * stride_m + offs_n[None, :],
            val.to(ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _load_w(ptr, offs_r, offs_c, row_stride, r_mask, c_mask):
        return tl.load(
            ptr + offs_r[:, None] * row_stride + offs_c[None, :],
            mask=r_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

    @_autotuned(
        _GEMM_TILES,
        ["GEMM_MODE", "K", "N", "HAS_LN_BIAS", "HAS_LIN_BIAS"],
        ["Y_ptr", "Mean_ptr", "Rstd_ptr"],
        ("stride_x_m", "stride_y_m", "M", "eps"),
        (
            "X_ptr",
            "W_ptr",
            "Y_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Bias_ptr",
            "Mean_ptr",
            "Rstd_ptr",
        ),
    )
    def _fused_ln_linear_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Gamma_ptr,
        Beta_ptr,
        Bias_ptr,
        Mean_ptr,
        Rstd_ptr,
        stride_x_m,
        stride_y_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_LIN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K

        x = _load2d(X_ptr, offs_m64, offs_k, stride_x_m, m_mask, k_mask).to(tl.float32)
        mean = tl.sum(x, axis=1) / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x_centered * x_centered, axis=1) / K + eps)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)
        tl.store(Mean_ptr + offs_m, mean, mask=m_mask)
        tl.store(Rstd_ptr + offs_m, rstd, mask=m_mask)

        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            w = _load_w(W_ptr, offs_n, offs_k, K, n_mask, k_mask)
            acc = _dot_f32(x_hat, tl.trans(w), GEMM_MODE)
            if HAS_LIN_BIAS:
                acc = acc + tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(
                    tl.float32
                )
            _store2d(Y_ptr, acc, offs_m64, offs_n, stride_y_m, m_mask, n_mask)

    @_autotuned(
        _DW_TILES,
        ["GEMM_MODE", "K", "N", "HAS_LN_BIAS"],
        ["PW_ptr", "PB_ptr"],
        ("stride_x_m", "stride_go_m", "M"),
        (
            "X_ptr",
            "GO_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "PW_ptr",
            "PB_ptr",
        ),
    )
    def _fused_ln_linear_bwd_dw_kernel(
        X_ptr,
        GO_ptr,
        Gamma_ptr,
        Beta_ptr,
        Mean_ptr,
        Rstd_ptr,
        PW_ptr,
        PB_ptr,
        stride_x_m,
        stride_go_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        HAS_LIN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        SPLIT_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        split = tl.program_id(0)
        rows_per_split = (M + SPLIT_M - 1) // SPLIT_M
        split_start = split * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = offs_k < K
        n_mask = offs_n < N
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        acc_dw = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc_db = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end
            x = _load2d(X_ptr, offs_m64, offs_k, stride_x_m, m_mask, k_mask).to(
                tl.float32
            )
            mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
            rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
            x_norm = tl.where(
                k_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0
            )
            ln_out = x_norm * gamma[None, :]
            if HAS_LN_BIAS:
                beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
                ln_out = ln_out + tl.where(k_mask[None, :], beta[None, :], 0.0)
            go = _load2d(GO_ptr, offs_m64, offs_n, stride_go_m, m_mask, n_mask).to(
                tl.float32
            )
            acc_dw += _dot_f32(tl.trans(go), ln_out, GEMM_MODE)
            if HAS_LIN_BIAS:
                acc_db += tl.sum(go, axis=0)

        tl.store(
            PW_ptr + split * N * K + offs_n[:, None] * K + offs_k[None, :],
            acc_dw,
            mask=n_mask[:, None] & k_mask[None, :],
        )
        if HAS_LIN_BIAS:
            tl.store(PB_ptr + split * N + offs_n, acc_db, mask=n_mask)

    @_autotuned(
        _GEMM_TILES,
        ["GEMM_MODE", "K", "N"],
        ["GX_ptr", "PGamma_ptr", "PBeta_ptr"],
        ("stride_x_m", "stride_go_m", "stride_gx_m", "M"),
        (
            "X_ptr",
            "W_ptr",
            "GO_ptr",
            "Gamma_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "GX_ptr",
            "PGamma_ptr",
            "PBeta_ptr",
        ),
    )
    def _fused_ln_linear_bwd_dx_kernel(
        X_ptr,
        W_ptr,
        GO_ptr,
        Gamma_ptr,
        Mean_ptr,
        Rstd_ptr,
        GX_ptr,
        PGamma_ptr,
        PBeta_ptr,
        stride_x_m,
        stride_go_m,
        stride_gx_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        x = _load2d(X_ptr, offs_m64, offs_k, stride_x_m, m_mask, k_mask).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_norm = tl.where(k_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0)
        dx_ln = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            go = _load2d(GO_ptr, offs_m64, offs_n, stride_go_m, m_mask, n_mask)
            w = _load_w(W_ptr, offs_n, offs_k, K, n_mask, k_mask)
            dx_ln += _dot_f32(go, w, GEMM_MODE)
        dx_ln = tl.where(m_mask[:, None] & k_mask[None, :], dx_ln, 0.0)
        dx_hat = dx_ln * gamma[None, :]
        grad_mean = tl.sum(dx_hat, axis=1) / K
        grad_proj = tl.sum(dx_hat * x_norm, axis=1) / K
        gx = rstd[:, None] * (
            dx_hat - grad_mean[:, None] - x_norm * grad_proj[:, None]
        )
        _store2d(GX_ptr, gx, offs_m64, offs_k, stride_gx_m, m_mask, k_mask)
        tl.store(
            PGamma_ptr + pid * K + offs_k,
            tl.sum(dx_ln * x_norm, axis=0),
            mask=k_mask,
        )
        tl.store(PBeta_ptr + pid * K + offs_k, tl.sum(dx_ln, axis=0), mask=k_mask)

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_ln_linear")

    _fused_ln_linear_fwd_kernel = _unavailable
    _fused_ln_linear_bwd_dw_kernel = _unavailable
    _fused_ln_linear_bwd_dx_kernel = _unavailable


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _launch_fused_ln_linear(
    x_2d: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_ln_linear")
    M, K = x_2d.shape
    N = weight.shape[0]
    y = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
    mean = torch.empty(M, dtype=torch.float32, device=x_2d.device)
    rstd = torch.empty_like(mean)
    dummy = x_2d
    _fused_ln_linear_fwd_kernel[(lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),))](
        x_2d,
        weight,
        y,
        gamma,
        beta if beta is not None else dummy,
        bias if bias is not None else dummy,
        mean,
        rstd,
        x_2d.stride(0),
        y.stride(0),
        M,
        K,
        N,
        float(eps),
        HAS_LN_BIAS=beta is not None,
        HAS_LIN_BIAS=bias is not None,
        GEMM_MODE=_gemm_mode(x_2d),
        BLOCK_K=max(_next_power_of_two(K), 16),
    )
    return y, mean, rstd


def _fused_ln_linear_backward(
    x_2d: torch.Tensor,
    grad_out_2d: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    compute_dx: bool,
    compute_dw: bool,
    compute_dbias: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_ln_linear")
    M, K = x_2d.shape
    N = weight.shape[0]
    x_c = x_2d.contiguous()
    go_c = grad_out_2d.contiguous()
    w_c = weight.contiguous()
    dummy = x_c
    gemm_mode = _gemm_mode(x_c)
    block_k = max(_next_power_of_two(K), 16)
    has_beta = beta is not None

    grad_w = grad_bias = None
    if compute_dw:
        partial_w = torch.empty(
            (_BWD_SPLIT_M, N, K), dtype=torch.float32, device=x_c.device
        )
        partial_b = (
            torch.empty((_BWD_SPLIT_M, N), dtype=torch.float32, device=x_c.device)
            if compute_dbias
            else dummy
        )
        _fused_ln_linear_bwd_dw_kernel[
            lambda meta: (_BWD_SPLIT_M, triton.cdiv(N, meta["BLOCK_N"]))
        ](
            x_c,
            go_c,
            gamma,
            beta if has_beta else dummy,
            mean,
            rstd,
            partial_w,
            partial_b,
            x_c.stride(0),
            go_c.stride(0),
            M,
            K,
            N,
            HAS_LN_BIAS=has_beta,
            HAS_LIN_BIAS=compute_dbias,
            GEMM_MODE=gemm_mode,
            SPLIT_M=_BWD_SPLIT_M,
            BLOCK_K=block_k,
        )
        grad_w = partial_w.sum(0)
        grad_bias = partial_b.sum(0) if compute_dbias else None

    grad_x = d_gamma = d_beta = None
    if compute_dx:
        grad_x = torch.empty_like(x_c)
        n_partial = triton.cdiv(M, 16)
        partial_gamma = torch.zeros(
            (n_partial, K), dtype=torch.float32, device=x_c.device
        )
        partial_beta = torch.zeros_like(partial_gamma)
        _fused_ln_linear_bwd_dx_kernel[
            lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        ](
            x_c,
            w_c,
            go_c,
            gamma,
            mean,
            rstd,
            grad_x,
            partial_gamma,
            partial_beta,
            x_c.stride(0),
            go_c.stride(0),
            grad_x.stride(0),
            M,
            K,
            N,
            GEMM_MODE=gemm_mode,
            BLOCK_K=block_k,
        )
        d_gamma = partial_gamma.sum(0)
        d_beta = partial_beta.sum(0) if has_beta else None
    return grad_x, d_gamma, d_beta, grad_w, grad_bias


class _FusedLNLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, weight, bias, eps):
        has_beta, has_bias = beta is not None, bias is not None
        x_2d = x.contiguous().view(-1, gamma.shape[0])
        y_2d, mean, rstd = _launch_fused_ln_linear(
            x_2d, gamma, beta, weight, bias, float(eps)
        )
        ctx.save_for_backward(
            x,
            gamma,
            beta if has_beta else x.new_empty(0),
            weight,
            mean,
            rstd,
        )
        ctx.has_beta = has_beta
        ctx.has_bias = has_bias
        ctx.bias_dtype = bias.dtype if has_bias else weight.dtype
        ctx.x_shape = x.shape
        return y_2d.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, gamma, beta_t, weight, mean, rstd = ctx.saved_tensors
        need = ctx.needs_input_grad
        gx, dgamma, dbeta, dw, dbias = _fused_ln_linear_backward(
            x.contiguous().view(-1, gamma.shape[0]),
            grad_out.contiguous().view(-1, weight.shape[0]),
            gamma,
            beta_t if ctx.has_beta else None,
            weight,
            mean,
            rstd,
            compute_dx=any(need[:3]),
            compute_dw=need[3] or (need[4] and ctx.has_bias),
            compute_dbias=bool(need[4] and ctx.has_bias),
        )
        return (
            gx.to(dtype=x.dtype).view(ctx.x_shape) if need[0] else None,
            dgamma.to(dtype=gamma.dtype) if need[1] else None,
            dbeta.to(dtype=beta_t.dtype) if need[2] and ctx.has_beta else None,
            dw.to(dtype=weight.dtype) if need[3] else None,
            dbias.to(dtype=ctx.bias_dtype) if need[4] and ctx.has_bias else None,
            None,
        )


def fused_ln_linear(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused LN → Linear. Callers must check eligibility first."""
    if not is_fused_ln_linear_eligible(x, gamma, beta, weight, bias):
        raise RuntimeError(
            "fused_ln_linear requires an eligible Triton launch; "
            "use eager_ln_linear for the fallback path"
        )

    use_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, gamma, beta, weight, bias)
    )
    if use_grad:
        return _FusedLNLinearFn.apply(x, gamma, beta, weight, bias, eps)

    y_2d, _mean, _rstd = _launch_fused_ln_linear(
        x.contiguous().view(-1, gamma.shape[0]),
        gamma,
        beta,
        weight,
        bias,
        eps,
    )
    return y_2d.view(*x.shape[:-1], weight.shape[0])
