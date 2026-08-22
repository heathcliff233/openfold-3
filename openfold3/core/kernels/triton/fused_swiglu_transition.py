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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused pair SwiGLU transition
# (Triton) with training backward.

"""Fused pair SwiGLU transition (Triton).

LN -> SwiGLU -> Linear -> (*mask) [+ residual]. ``M`` is not an autotune or
specialize key. Training saves ``x_hat`` / ``mean`` / ``rstd`` and rematerializes
``a`` / ``b``; ``dW`` uses exclusive split-M tiles. Ineligible shapes use
``SwiGLUTransition``.
"""

from __future__ import annotations

import os

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# Pair transition (c_z=128, n=4 -> H=512) with headroom; wider sites fall back.
_MAX_C_IN = 256
_MAX_HIDDEN = 512
_MIN_M = 4096
_BWD_SPLIT_M = 16
_TRUE = {"1", "true", "yes", "on"}


def is_fused_swiglu_transition_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1").strip().lower() in (
        _TRUE
    )


def is_fused_swiglu_transition_eligible(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
) -> bool:
    c_in = gamma.shape[0]
    masters = (gamma, w_a, w_b, w_out) + ((beta,) if beta is not None else ())
    return (
        is_fused_swiglu_transition_enabled()
        and _TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and all(w.dtype == torch.float32 for w in masters)
        and c_in <= _MAX_C_IN
        and w_a.shape[0] <= _MAX_HIDDEN
        and (x.numel() // c_in) >= _MIN_M
    )


if _TRITON_AVAILABLE:

    def _configs(tiles):
        return [
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_H": block_h},
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for block_m, block_h, num_warps, num_stages in tiles
        ]

    # Shared by forward / dX. dW keeps smaller tiles (3 live weight accums).
    _GEMM_TILES = (
        (16, 32, 4, 1),
        (16, 32, 4, 2),
        (16, 64, 4, 1),
        (32, 32, 4, 1),
        (32, 32, 4, 2),
        (32, 64, 4, 1),
        (64, 64, 4, 1),
        (128, 64, 8, 1),
    )
    _DW_TILES = (
        (16, 32, 4, 1),
        (16, 32, 4, 2),
        (32, 32, 4, 1),
        (32, 32, 4, 2),
    )

    def _prune_by_precision(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        H = kwargs.get("H", named_args.get("H"))
        kept = [
            cfg
            for cfg in configs
            if (H is None or cfg.kwargs["BLOCK_H"] <= int(H))
            and not (mode == "ieee" and cfg.kwargs["BLOCK_M"] >= 64)
        ]
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

    @triton.jit
    def _mask_rows(go, Mask_ptr, offs_m, stride_mask_m, m_mask, HAS_MASK: tl.constexpr):
        if not HAS_MASK:
            return go
        mask_val = tl.load(Mask_ptr + offs_m * stride_mask_m, mask=m_mask, other=0.0)
        return go.to(tl.float32) * mask_val[:, None].to(tl.float32)

    @triton.jit
    def _swiglu_grads(a, b, grad_h):
        sig = tl.sigmoid(a)
        silu_a = a * sig
        grad_a = grad_h * b * (sig + silu_a * (1.0 - sig))
        return silu_a * b, grad_a, grad_h * silu_a

    @_autotuned(
        _GEMM_TILES + ((128, 64, 8, 2),),
        ["GEMM_MODE", "K", "H", "C_OUT", "SAVE_ACTS"],
        ["Y_ptr", "XHAT_ptr", "Mean_ptr", "Rstd_ptr"],
        (
            "stride_x_m",
            "stride_res_m",
            "stride_y_m",
            "stride_mask_m",
            "stride_xhat_m",
            "M",
            "eps",
        ),
        (
            "X_ptr",
            "WA_ptr",
            "WB_ptr",
            "WOUT_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Mask_ptr",
            "Res_ptr",
            "Y_ptr",
            "XHAT_ptr",
            "Mean_ptr",
            "Rstd_ptr",
        ),
    )
    def _fused_swiglu_transition_fwd_kernel(
        X_ptr,
        WA_ptr,
        WB_ptr,
        WOUT_ptr,
        Gamma_ptr,
        Beta_ptr,
        Mask_ptr,
        Res_ptr,
        Y_ptr,
        XHAT_ptr,
        Mean_ptr,
        Rstd_ptr,
        stride_x_m,
        stride_res_m,
        stride_y_m,
        stride_mask_m,
        stride_xhat_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        SAVE_ACTS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT

        x = _load2d(X_ptr, offs_m64, offs_k, stride_x_m, m_mask, k_mask).to(tl.float32)
        mean = tl.sum(x, axis=1) / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x_centered * x_centered, axis=1) / K + eps)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)
        if SAVE_ACTS:
            _store2d(XHAT_ptr, x_hat, offs_m64, offs_k, stride_xhat_m, m_mask, k_mask)
            tl.store(Mean_ptr + offs_m, mean, mask=m_mask)
            tl.store(Rstd_ptr + offs_m, rstd, mask=m_mask)

        acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            wa = _load_w(WA_ptr, offs_h, offs_k, K, h_mask, k_mask)
            wb = _load_w(WB_ptr, offs_h, offs_k, K, h_mask, k_mask)
            wo = _load_w(WOUT_ptr, offs_n, offs_h, H, n_mask, h_mask)
            a = _dot_f32(x_hat, tl.trans(wa), GEMM_MODE)
            b = _dot_f32(x_hat, tl.trans(wb), GEMM_MODE)
            silu = (a * tl.sigmoid(a)) * b
            acc_out += _dot_f32(silu, tl.trans(wo), GEMM_MODE)

        acc_out = _mask_rows(acc_out, Mask_ptr, offs_m, stride_mask_m, m_mask, HAS_MASK)
        if HAS_RESIDUAL:
            acc_out = acc_out + _load2d(
                Res_ptr, offs_m64, offs_n, stride_res_m, m_mask, n_mask
            ).to(tl.float32)
        _store2d(Y_ptr, acc_out, offs_m64, offs_n, stride_y_m, m_mask, n_mask)

    @_autotuned(
        _GEMM_TILES,
        ["GEMM_MODE", "K", "H", "C_OUT"],
        ["GX_ptr"],
        ("stride_xhat_m", "stride_go_m", "stride_gx_m", "stride_mask_m", "M"),
        ("XHAT_ptr", "WA_ptr", "WB_ptr", "WOUT_ptr", "Mask_ptr", "GO_ptr", "GX_ptr"),
    )
    def _fused_swiglu_transition_bwd_dx_kernel(
        XHAT_ptr,
        WA_ptr,
        WB_ptr,
        WOUT_ptr,
        Mask_ptr,
        GO_ptr,
        GX_ptr,
        stride_xhat_m,
        stride_go_m,
        stride_gx_m,
        stride_mask_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT

        go = _mask_rows(
            _load2d(GO_ptr, offs_m64, offs_n, stride_go_m, m_mask, n_mask),
            Mask_ptr,
            offs_m,
            stride_mask_m,
            m_mask,
            HAS_MASK,
        )
        x_hat = _load2d(XHAT_ptr, offs_m64, offs_k, stride_xhat_m, m_mask, k_mask)
        grad_x_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            wa = _load_w(WA_ptr, offs_h, offs_k, K, h_mask, k_mask)
            wb = _load_w(WB_ptr, offs_h, offs_k, K, h_mask, k_mask)
            wo = _load_w(WOUT_ptr, offs_n, offs_h, H, n_mask, h_mask)
            a = _dot_f32(x_hat, tl.trans(wa), GEMM_MODE)
            b = _dot_f32(x_hat, tl.trans(wb), GEMM_MODE)
            _h, grad_a, grad_b = _swiglu_grads(a, b, _dot_f32(go, wo, GEMM_MODE))
            grad_x_hat += _dot_f32(grad_a, wa, GEMM_MODE)
            grad_x_hat += _dot_f32(grad_b, wb, GEMM_MODE)
        _store2d(GX_ptr, grad_x_hat, offs_m64, offs_k, stride_gx_m, m_mask, k_mask)

    @_autotuned(
        _DW_TILES,
        ["GEMM_MODE", "K", "H", "C_OUT"],
        ["PWA_ptr", "PWB_ptr", "PWO_ptr"],
        ("stride_xhat_m", "stride_go_m", "stride_mask_m", "M"),
        (
            "XHAT_ptr",
            "WA_ptr",
            "WB_ptr",
            "WOUT_ptr",
            "Mask_ptr",
            "GO_ptr",
            "PWA_ptr",
            "PWB_ptr",
            "PWO_ptr",
        ),
    )
    def _fused_swiglu_transition_bwd_dw_kernel(
        XHAT_ptr,
        WA_ptr,
        WB_ptr,
        WOUT_ptr,
        Mask_ptr,
        GO_ptr,
        PWA_ptr,
        PWB_ptr,
        PWO_ptr,
        stride_xhat_m,
        stride_go_m,
        stride_mask_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        SPLIT_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        split = tl.program_id(0)
        rows_per_split = (M + SPLIT_M - 1) // SPLIT_M
        split_start = split * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT
        h_mask = offs_h < H

        wo = _load_w(WOUT_ptr, offs_n, offs_h, H, n_mask, h_mask)
        wa = _load_w(WA_ptr, offs_h, offs_k, K, h_mask, k_mask)
        wb = _load_w(WB_ptr, offs_h, offs_k, K, h_mask, k_mask)
        acc_dwa = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        acc_dwb = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        acc_dwo = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)

        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end
            x_hat = _load2d(XHAT_ptr, offs_m64, offs_k, stride_xhat_m, m_mask, k_mask)
            go = _mask_rows(
                _load2d(GO_ptr, offs_m64, offs_n, stride_go_m, m_mask, n_mask),
                Mask_ptr,
                offs_m,
                stride_mask_m,
                m_mask,
                HAS_MASK,
            )
            a = _dot_f32(x_hat, tl.trans(wa), GEMM_MODE)
            b = _dot_f32(x_hat, tl.trans(wb), GEMM_MODE)
            h, grad_a, grad_b = _swiglu_grads(a, b, _dot_f32(go, wo, GEMM_MODE))
            acc_dwo += _dot_f32(tl.trans(go), h, GEMM_MODE)
            acc_dwa += _dot_f32(tl.trans(grad_a), x_hat, GEMM_MODE)
            acc_dwb += _dot_f32(tl.trans(grad_b), x_hat, GEMM_MODE)

        tl.store(
            PWA_ptr + split * H * K + offs_h[:, None] * K + offs_k[None, :],
            acc_dwa,
            mask=h_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            PWB_ptr + split * H * K + offs_h[:, None] * K + offs_k[None, :],
            acc_dwb,
            mask=h_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            PWO_ptr + split * C_OUT * H + offs_n[:, None] * H + offs_h[None, :],
            acc_dwo,
            mask=n_mask[:, None] & h_mask[None, :],
        )

    @triton.jit(
        do_not_specialize=["stride_gy_m", "stride_x_m", "stride_gx_m", "M"],
        do_not_specialize_on_alignment=[
            "GY_ptr",
            "X_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "Gamma_ptr",
            "GX_ptr",
            "PGamma_ptr",
            "PBeta_ptr",
        ],
    )
    def _fused_swiglu_transition_ln_bwd_kernel(
        GY_ptr,
        X_ptr,
        Mean_ptr,
        Rstd_ptr,
        Gamma_ptr,
        GX_ptr,
        PGamma_ptr,
        PBeta_ptr,
        stride_gy_m,
        stride_x_m,
        stride_gx_m,
        M,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        gy = _load2d(GY_ptr, offs_m64, offs_k, stride_gy_m, m_mask, k_mask).to(
            tl.float32
        )
        x = _load2d(X_ptr, offs_m64, offs_k, stride_x_m, m_mask, k_mask).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_norm = tl.where(
            m_mask[:, None] & k_mask[None, :],
            (x - mean[:, None]) * rstd[:, None],
            0.0,
        )
        grad_norm = gy * gamma[None, :]
        grad_mean = tl.sum(grad_norm, axis=1) / K
        grad_proj = tl.sum(grad_norm * x_norm, axis=1) / K
        gx = rstd[:, None] * (
            grad_norm - grad_mean[:, None] - x_norm * grad_proj[:, None]
        )
        _store2d(GX_ptr, gx, offs_m64, offs_k, stride_gx_m, m_mask, k_mask)
        tl.store(
            PGamma_ptr + pid * K + offs_k, tl.sum(gy * x_norm, axis=0), mask=k_mask
        )
        tl.store(PBeta_ptr + pid * K + offs_k, tl.sum(gy, axis=0), mask=k_mask)

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_swiglu_transition")

    _fused_swiglu_transition_fwd_kernel = _unavailable
    _fused_swiglu_transition_bwd_dx_kernel = _unavailable
    _fused_swiglu_transition_bwd_dw_kernel = _unavailable
    _fused_swiglu_transition_ln_bwd_kernel = _unavailable


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _block_kn(k: int, c_out: int) -> tuple[int, int]:
    return max(_next_power_of_two(k), 16), max(_next_power_of_two(c_out), 16)


def _launch_fused_swiglu_transition(
    x_2d: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
    mask_1d: torch.Tensor | None,
    residual_2d: torch.Tensor | None,
    eps: float,
    save_acts: bool = False,
):
    M, K = x_2d.shape
    H, C_OUT = w_a.shape[0], w_out.shape[0]
    w_a, w_b, w_out, gamma = (t.contiguous() for t in (w_a, w_b, w_out, gamma))
    if beta is not None:
        beta = beta.contiguous()

    in_place = (
        residual_2d is not None
        and residual_2d.data_ptr() == x_2d.data_ptr()
        and not residual_2d.requires_grad
    )
    if save_acts and in_place:
        raise RuntimeError("fused SwiGLU training forward cannot write in place")
    y = (
        x_2d
        if in_place
        else torch.empty((M, C_OUT), dtype=x_2d.dtype, device=x_2d.device)
    )
    if save_acts:
        x_hat = torch.empty((M, K), dtype=x_2d.dtype, device=x_2d.device)
        mean = torch.empty((M,), dtype=torch.float32, device=x_2d.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x_2d.device)
    else:
        x_hat = mean = rstd = x_2d

    dummy = x_2d
    block_k, block_n = _block_kn(K, C_OUT)
    _fused_swiglu_transition_fwd_kernel[
        (lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),))
    ](
        x_2d,
        w_a,
        w_b,
        w_out,
        gamma,
        beta if beta is not None else dummy,
        mask_1d if mask_1d is not None else dummy,
        residual_2d if residual_2d is not None else dummy,
        y,
        x_hat,
        mean,
        rstd,
        x_2d.stride(0),
        (residual_2d if residual_2d is not None else dummy).stride(0),
        y.stride(0),
        mask_1d.stride(0) if mask_1d is not None else 0,
        x_hat.stride(0),
        M,
        K,
        H,
        C_OUT,
        eps,
        HAS_LN_BIAS=beta is not None,
        HAS_MASK=mask_1d is not None,
        HAS_RESIDUAL=residual_2d is not None,
        SAVE_ACTS=save_acts,
        GEMM_MODE=_gemm_mode(x_2d),
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )
    return (y, x_hat, mean, rstd) if save_acts else y


def _fused_swiglu_transition_weight_grad(
    x_hat: torch.Tensor,
    grad_out_2d: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
    mask_1d: torch.Tensor | None,
    compute_dx: bool = True,
    compute_dw: bool = True,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    M, K = x_hat.shape
    H, C_OUT = w_a.shape[0], w_out.shape[0]
    x_hat_c, go_c = x_hat.contiguous(), grad_out_2d.contiguous()
    wa_c, wb_c, wo_c = w_a.contiguous(), w_b.contiguous(), w_out.contiguous()
    mask_ptr = mask_1d if mask_1d is not None else x_hat_c
    stride_mask = mask_1d.stride(0) if mask_1d is not None else 0
    block_k, block_n = _block_kn(K, C_OUT)
    gemm_mode = _gemm_mode(x_hat)
    has_mask = mask_1d is not None

    grad_wa = grad_wb = grad_wo = None
    if compute_dw:
        partial_wa = torch.empty(
            _BWD_SPLIT_M, H, K, device=x_hat.device, dtype=torch.float32
        )
        partial_wb = torch.empty_like(partial_wa)
        partial_wo = torch.empty(
            _BWD_SPLIT_M, C_OUT, H, device=x_hat.device, dtype=torch.float32
        )
        _fused_swiglu_transition_bwd_dw_kernel[
            lambda meta: (_BWD_SPLIT_M, triton.cdiv(H, meta["BLOCK_H"]))
        ](
            x_hat_c,
            wa_c,
            wb_c,
            wo_c,
            mask_ptr,
            go_c,
            partial_wa,
            partial_wb,
            partial_wo,
            x_hat_c.stride(0),
            go_c.stride(0),
            stride_mask,
            M,
            K,
            H,
            C_OUT,
            HAS_MASK=has_mask,
            GEMM_MODE=gemm_mode,
            SPLIT_M=_BWD_SPLIT_M,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
        )
        grad_wa, grad_wb, grad_wo = (
            partial_wa.sum(0),
            partial_wb.sum(0),
            partial_wo.sum(0),
        )

    grad_x_hat = None
    if compute_dx:
        grad_x_hat = (
            x_hat
            if x_hat.dtype == torch.float32
            else torch.empty((M, K), dtype=torch.float32, device=x_hat.device)
        )
        _fused_swiglu_transition_bwd_dx_kernel[
            lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        ](
            x_hat_c,
            wa_c,
            wb_c,
            wo_c,
            mask_ptr,
            go_c,
            grad_x_hat,
            x_hat_c.stride(0),
            go_c.stride(0),
            grad_x_hat.stride(0),
            stride_mask,
            M,
            K,
            H,
            C_OUT,
            HAS_MASK=has_mask,
            GEMM_MODE=gemm_mode,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
        )
    return grad_x_hat, grad_wa, grad_wb, grad_wo


def _fused_swiglu_transition_layer_norm_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = x.shape
    block_m, block_k = 16, max(_next_power_of_two(K), 16)
    n_blocks = triton.cdiv(M, block_m)
    grad_x = torch.empty_like(x)
    partial_gamma = torch.empty((n_blocks, K), dtype=torch.float32, device=x.device)
    partial_beta = torch.empty_like(partial_gamma)
    _fused_swiglu_transition_ln_bwd_kernel[(n_blocks,)](
        grad_y,
        x,
        mean,
        rstd,
        gamma,
        grad_x,
        partial_gamma,
        partial_beta,
        grad_y.stride(0),
        x.stride(0),
        grad_x.stride(0),
        M,
        K,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return grad_x, partial_gamma.sum(0), partial_beta.sum(0)


class _FusedSwiGLUTransitionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, w_a, w_b, w_out, mask, eps):
        has_beta, has_mask = beta is not None, mask is not None
        x_2d = x.contiguous().view(-1, gamma.shape[0])
        mask_1d = mask.reshape(-1).contiguous() if has_mask else None
        y_2d, x_hat, mean, rstd = _launch_fused_swiglu_transition(
            x_2d,
            gamma,
            beta,
            w_a,
            w_b,
            w_out,
            mask_1d,
            None,
            float(eps),
            save_acts=True,
        )
        ctx.save_for_backward(
            x,
            x_hat,
            mean,
            rstd,
            gamma,
            beta if has_beta else x.new_empty(0),
            w_a,
            w_b,
            w_out,
            mask if has_mask else x.new_empty(0),
        )
        ctx.has_beta = has_beta
        ctx.has_mask = has_mask
        ctx.x_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, x_hat, mean, rstd, gamma, beta_t, w_a, w_b, w_out, mask_t = ctx.saved_tensors
        need = ctx.needs_input_grad
        mask_1d = mask_t.reshape(-1).contiguous() if ctx.has_mask else None
        gxhat, gwa, gwb, gwo = _fused_swiglu_transition_weight_grad(
            x_hat,
            grad_out.contiguous().view(-1, grad_out.shape[-1]),
            w_a,
            w_b,
            w_out,
            mask_1d,
            compute_dx=any(need[:3]),
            compute_dw=any(need[3:6]),
        )
        gx = ggamma = gbeta = None
        if any(need[:3]):
            gin, ggamma, gbeta = _fused_swiglu_transition_layer_norm_backward(
                gxhat,
                x.contiguous().view(-1, gamma.shape[0]),
                mean,
                rstd,
                gamma,
            )
            gx = gin.view(ctx.x_shape) if need[0] else None
            ggamma = ggamma.to(dtype=gamma.dtype) if need[1] else None
            gbeta = gbeta.to(dtype=beta_t.dtype) if need[2] and ctx.has_beta else None
        return (
            gx,
            ggamma,
            gbeta,
            gwa.to(dtype=w_a.dtype) if need[3] else None,
            gwb.to(dtype=w_b.dtype) if need[4] else None,
            gwo.to(dtype=w_out.dtype) if need[5] else None,
            None,
            None,
        )


def fused_swiglu_transition(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused LN -> SwiGLU -> Linear -> (*mask) [-> +residual]."""
    if not is_fused_swiglu_transition_eligible(x, gamma, beta, w_a, w_b, w_out):
        raise RuntimeError(
            "fused_swiglu_transition requires an eligible Triton launch; "
            "use SwiGLUTransition for the eager path"
        )

    use_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, gamma, beta, w_a, w_b, w_out)
    )
    if use_grad:
        if residual is not None:
            return residual + fused_swiglu_transition(
                x, gamma, beta, w_a, w_b, w_out, mask=mask, eps=eps
            )
        return _FusedSwiGLUTransitionFn.apply(
            x, gamma, beta, w_a, w_b, w_out, mask, eps
        )

    c_in = gamma.shape[0]
    y_2d = _launch_fused_swiglu_transition(
        x.contiguous().view(-1, c_in),
        gamma,
        beta,
        w_a,
        w_b,
        w_out,
        mask.reshape(-1).contiguous() if mask is not None else None,
        residual.contiguous().view(-1, c_in) if residual is not None else None,
        eps,
        save_acts=False,
    )
    return y_2d.view_as(x if residual is None else residual)
