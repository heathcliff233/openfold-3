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

"""Fused SwiGLU pair transition via a length-generic Triton kernel.

Fuses

    x -> LayerNorm -> SwiGLU(SiLU(linear_a(x)) * linear_b(x)) -> linear_out
      -> * mask  [-> + residual]

GEMMs honor ``torch.backends.cuda.matmul.allow_tf32`` for fp32 activations:

- TF32 on: ``input_precision="tf32"`` with round-nearest ``f32->tf32``
  (``cvt.rna.tf32.f32``), matching cuBLAS TF32. Triton's default TF32
  *truncates*, which was the source of the extra ~2e-2 abs error.
- TF32 off: ``input_precision="ieee"`` (true FP32).

Triton requires fp32 Parameter masters (same as fused template
projection). Activations may be fp32 or bf16. bf16 activations use
``tl.dot`` in bf16 with fp32 accumulate; weight tiles are downcast
inside the GEMM only. Pure bf16 weights are ineligible.

Inference (in-place) never materializes the ``[M, hidden]`` expansion.
Training (out-of-place) writes ``x_hat``, ``a``, and ``b`` so backward can
skip LayerNorm and the up-projections. Weight grads use exclusive split-M
tiles (no atomics). Sequence length ``M`` is not an autotune / specialize
key. Ineligible shapes use ``SwiGLUTransition`` primitives.
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
# Below this row count, fused launch / tiling overhead is not worthwhile.
_MIN_M = 4096
# Exclusive row splits for deterministic Triton weight grads (no atomics).
_BWD_SPLIT_M = 16


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


def is_fused_swiglu_transition_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_fused_swiglu_transition_eligible(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
) -> bool:
    c_in = gamma.shape[0]
    hidden = w_a.shape[0]
    masters = (gamma, w_a, w_b, w_out) + ((beta,) if beta is not None else ())
    return (
        is_fused_swiglu_transition_enabled()
        and _TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and all(w.dtype == torch.float32 for w in masters)
        and c_in <= _MAX_C_IN
        and hidden <= _MAX_HIDDEN
        and (x.numel() // c_in) >= _MIN_M
    )


if _TRITON_AVAILABLE:

    def _fwd_autotune_configs():
        configs = []
        for block_m, block_h, num_warps, num_stages in (
            (16, 32, 4, 1),
            (16, 32, 4, 2),
            (16, 64, 4, 1),
            (32, 32, 4, 1),
            (32, 32, 4, 2),
            (32, 64, 4, 1),
            (64, 64, 4, 1),
            (128, 64, 8, 1),
            (128, 64, 8, 2),
        ):
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_H": block_h},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
        return configs

    def _dx_autotune_configs():
        configs = []
        for block_m, block_h, num_warps, num_stages in (
            (16, 32, 4, 1),
            (16, 32, 4, 2),
            (16, 64, 4, 1),
            (32, 32, 4, 1),
            (32, 32, 4, 2),
            (32, 64, 4, 1),
            (64, 64, 4, 1),
            (128, 64, 8, 1),
        ):
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_H": block_h},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
        return configs

    def _dw_autotune_configs():
        # dW keeps three weight accumulators live; keep tiles inside SMEM.
        configs = []
        for block_m, block_h, num_warps in (
            (16, 32, 4),
            (16, 64, 4),
            (32, 32, 4),
            (32, 64, 4),
            (64, 32, 4),
            (64, 32, 8),
        ):
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_H": block_h},
                    num_warps=num_warps,
                    num_stages=1,
                )
            )
        return configs

    def _prune_by_precision(configs, named_args, **kwargs):
        precision = kwargs.get("INPUT_PRECISION", named_args.get("INPUT_PRECISION"))
        act = named_args.get("X_ptr")
        if act is None:
            act = named_args.get("A_ptr")
        if act is None:
            act = named_args.get("XHAT_ptr")
        H = kwargs.get("H", named_args.get("H"))
        kept = []
        for cfg in configs:
            block_m = cfg.kwargs["BLOCK_M"]
            block_h = cfg.kwargs["BLOCK_H"]
            if H is not None and block_h > int(H):
                continue
            if (
                precision == "ieee"
                and getattr(act, "dtype", None) != torch.bfloat16
                and block_m >= 64
            ):
                continue
            kept.append(cfg)
        return kept if kept else configs[:1]

    @triton.jit
    def _silu_fp32(x):
        return x * tl.sigmoid(x)

    @triton.jit
    def _round_to_tf32(x):
        ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
        return tl.inline_asm_elementwise(
            ASM, "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _dot_f32(a, b, INPUT_PRECISION: tl.constexpr, ROUND_TF32: tl.constexpr, act_ptr):
        # act_ptr is the activation allocation (X / a / x_hat). Its storage
        # dtype is the input; tl.dot input_precision is only ieee/tf32.
        if act_ptr.dtype.element_ty == tl.bfloat16:
            return tl.dot(
                a.to(tl.bfloat16),
                b.to(tl.bfloat16),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        if ROUND_TF32:
            a = _round_to_tf32(a)
            b = _round_to_tf32(b)
        return tl.dot(a, b, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    @triton.autotune(
        configs=_fwd_autotune_configs(),
        key=["INPUT_PRECISION", "K", "H", "C_OUT", "SAVE_ACTS"],
        prune_configs_by={"early_config_prune": _prune_by_precision},
        restore_value=["Y_ptr", "A_ptr", "B_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "stride_x_m",
            "stride_x_k",
            "stride_wa_h",
            "stride_wa_k",
            "stride_wb_h",
            "stride_wb_k",
            "stride_wo_n",
            "stride_wo_h",
            "stride_res_m",
            "stride_res_n",
            "stride_y_m",
            "stride_y_n",
            "stride_mask_m",
            "stride_xhat_m",
            "stride_a_m",
            "stride_b_m",
            "M",
            "eps",
        ],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "WA_ptr",
            "WB_ptr",
            "WOUT_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Mask_ptr",
            "Res_ptr",
            "Y_ptr",
            "A_ptr",
            "B_ptr",
        ],
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
        A_ptr,
        B_ptr,
        stride_x_m,
        stride_x_k,
        stride_wa_h,
        stride_wa_k,
        stride_wb_h,
        stride_wb_k,
        stride_wo_n,
        stride_wo_h,
        stride_res_m,
        stride_res_n,
        stride_y_m,
        stride_y_n,
        stride_mask_m,
        stride_xhat_m,
        stride_a_m,
        stride_b_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        SAVE_ACTS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        ROUND_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT

        x_ptrs = X_ptr + offs_m_64[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float32
        )
        mean = tl.sum(x, axis=1) / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) / K
        rstd = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)

        if SAVE_ACTS:
            xhat_ptrs = (
                XHAT_ptr
                + offs_m_64[:, None] * stride_xhat_m
                + offs_k[None, :]
            )
            tl.store(
                xhat_ptrs,
                x_hat.to(XHAT_ptr.dtype.element_ty),
                mask=m_mask[:, None] & k_mask[None, :],
            )

        acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H

            wa_ptrs = (
                WA_ptr + offs_h[:, None] * stride_wa_h + offs_k[None, :] * stride_wa_k
            )
            wa = tl.load(wa_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_a = _dot_f32(x_hat, tl.trans(wa), INPUT_PRECISION, ROUND_TF32, X_ptr)

            wb_ptrs = (
                WB_ptr + offs_h[:, None] * stride_wb_h + offs_k[None, :] * stride_wb_k
            )
            wb = tl.load(wb_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_b = _dot_f32(x_hat, tl.trans(wb), INPUT_PRECISION, ROUND_TF32, X_ptr)

            if SAVE_ACTS:
                a_ptrs = A_ptr + offs_m_64[:, None] * stride_a_m + offs_h[None, :]
                b_ptrs = B_ptr + offs_m_64[:, None] * stride_b_m + offs_h[None, :]
                tl.store(
                    a_ptrs,
                    acc_a.to(A_ptr.dtype.element_ty),
                    mask=m_mask[:, None] & h_mask[None, :],
                )
                tl.store(
                    b_ptrs,
                    acc_b.to(B_ptr.dtype.element_ty),
                    mask=m_mask[:, None] & h_mask[None, :],
                )

            silu = _silu_fp32(acc_a) * acc_b
            wo_ptrs = (
                WOUT_ptr + offs_n[:, None] * stride_wo_n + offs_h[None, :] * stride_wo_h
            )
            wo = tl.load(wo_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_out += _dot_f32(silu, tl.trans(wo), INPUT_PRECISION, ROUND_TF32, X_ptr)

        if HAS_MASK:
            mask_val = tl.load(
                Mask_ptr + offs_m * stride_mask_m, mask=m_mask, other=0.0
            )
            acc_out = acc_out * mask_val[:, None].to(tl.float32)

        if HAS_RESIDUAL:
            res_ptrs = (
                Res_ptr
                + offs_m_64[:, None] * stride_res_m
                + offs_n[None, :] * stride_res_n
            )
            res = tl.load(
                res_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0
            ).to(tl.float32)
            acc_out = acc_out + res

        y_ptrs = Y_ptr + offs_m_64[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
        tl.store(
            y_ptrs,
            acc_out.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.autotune(
        configs=_dx_autotune_configs(),
        key=["INPUT_PRECISION", "K", "H", "C_OUT"],
        prune_configs_by={"early_config_prune": _prune_by_precision},
        restore_value=["GX_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "stride_a_m",
            "stride_b_m",
            "stride_go_m",
            "stride_gx_m",
            "stride_mask_m",
            "M",
        ],
        do_not_specialize_on_alignment=[
            "A_ptr",
            "B_ptr",
            "WA_ptr",
            "WB_ptr",
            "WOUT_ptr",
            "Mask_ptr",
            "GO_ptr",
            "GX_ptr",
        ],
    )
    def _fused_swiglu_transition_bwd_dx_kernel(
        A_ptr,
        B_ptr,
        WA_ptr,
        WB_ptr,
        WOUT_ptr,
        Mask_ptr,
        GO_ptr,
        GX_ptr,
        stride_a_m,
        stride_b_m,
        stride_go_m,
        stride_gx_m,
        stride_mask_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        ROUND_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """d(x_hat) from saved up-projection activations; grid over rows."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT

        go_ptrs = GO_ptr + offs_m_64[:, None] * stride_go_m + offs_n[None, :]
        go = tl.load(go_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(
            tl.float32
        )
        if HAS_MASK:
            mask_val = tl.load(
                Mask_ptr + offs_m * stride_mask_m, mask=m_mask, other=0.0
            ).to(tl.float32)
            go = go * mask_val[:, None]

        grad_x_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            a = tl.load(
                A_ptr + offs_m_64[:, None] * stride_a_m + offs_h[None, :],
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                B_ptr + offs_m_64[:, None] * stride_b_m + offs_h[None, :],
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wa = tl.load(
                WA_ptr + offs_h[:, None] * K + offs_k[None, :],
                mask=h_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wb = tl.load(
                WB_ptr + offs_h[:, None] * K + offs_k[None, :],
                mask=h_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wo = tl.load(
                WOUT_ptr + offs_n[:, None] * H + offs_h[None, :],
                mask=n_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            grad_h = _dot_f32(go, wo, INPUT_PRECISION, ROUND_TF32, A_ptr)
            sig = tl.sigmoid(a)
            silu_a = a * sig
            grad_a = grad_h * b * (sig + silu_a * (1.0 - sig))
            grad_b = grad_h * silu_a
            grad_x_hat += _dot_f32(grad_a, wa, INPUT_PRECISION, ROUND_TF32, A_ptr)
            grad_x_hat += _dot_f32(grad_b, wb, INPUT_PRECISION, ROUND_TF32, A_ptr)

        gx_ptrs = GX_ptr + offs_m_64[:, None] * stride_gx_m + offs_k[None, :]
        tl.store(
            gx_ptrs,
            grad_x_hat.to(GX_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(
        configs=_dw_autotune_configs(),
        key=["INPUT_PRECISION", "K", "H", "C_OUT"],
        prune_configs_by={"early_config_prune": _prune_by_precision},
        restore_value=["PWA_ptr", "PWB_ptr", "PWO_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "stride_xhat_m",
            "stride_a_m",
            "stride_b_m",
            "stride_go_m",
            "stride_mask_m",
            "M",
        ],
        do_not_specialize_on_alignment=[
            "XHAT_ptr",
            "A_ptr",
            "B_ptr",
            "WOUT_ptr",
            "Mask_ptr",
            "GO_ptr",
            "PWA_ptr",
            "PWB_ptr",
            "PWO_ptr",
        ],
    )
    def _fused_swiglu_transition_bwd_dw_kernel(
        XHAT_ptr,
        A_ptr,
        B_ptr,
        WOUT_ptr,
        Mask_ptr,
        GO_ptr,
        PWA_ptr,
        PWB_ptr,
        PWO_ptr,
        stride_xhat_m,
        stride_a_m,
        stride_b_m,
        stride_go_m,
        stride_mask_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        ROUND_TF32: tl.constexpr,
        SPLIT_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program: one row-split x one H-tile; dW accumulated in registers."""
        split = tl.program_id(0)
        h_tile = tl.program_id(1)
        rows_per_split = (M + SPLIT_M - 1) // SPLIT_M
        split_start = split * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, M)

        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT
        h_mask = offs_h < H

        wo = tl.load(
            WOUT_ptr + offs_n[:, None] * H + offs_h[None, :],
            mask=n_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc_dwa = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        acc_dwb = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        acc_dwo = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)

        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m_64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end

            x_hat = tl.load(
                XHAT_ptr + offs_m_64[:, None] * stride_xhat_m + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            go = tl.load(
                GO_ptr + offs_m_64[:, None] * stride_go_m + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_MASK:
                mask_val = tl.load(
                    Mask_ptr + offs_m * stride_mask_m, mask=m_mask, other=0.0
                ).to(tl.float32)
                go = go * mask_val[:, None]
            a = tl.load(
                A_ptr + offs_m_64[:, None] * stride_a_m + offs_h[None, :],
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                B_ptr + offs_m_64[:, None] * stride_b_m + offs_h[None, :],
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            sig = tl.sigmoid(a)
            silu_a = a * sig
            h = silu_a * b
            grad_h = _dot_f32(go, wo, INPUT_PRECISION, ROUND_TF32, A_ptr)
            acc_dwo += _dot_f32(tl.trans(go), h, INPUT_PRECISION, ROUND_TF32, A_ptr)
            grad_a = grad_h * b * (sig + silu_a * (1.0 - sig))
            grad_b = grad_h * silu_a
            acc_dwa += _dot_f32(tl.trans(grad_a), x_hat, INPUT_PRECISION, ROUND_TF32, A_ptr)
            acc_dwb += _dot_f32(tl.trans(grad_b), x_hat, INPUT_PRECISION, ROUND_TF32, A_ptr)

        pwa_ptrs = PWA_ptr + split * H * K + offs_h[:, None] * K + offs_k[None, :]
        pwb_ptrs = PWB_ptr + split * H * K + offs_h[:, None] * K + offs_k[None, :]
        pwo_ptrs = PWO_ptr + split * C_OUT * H + offs_n[:, None] * H + offs_h[None, :]
        tl.store(pwa_ptrs, acc_dwa, mask=h_mask[:, None] & k_mask[None, :])
        tl.store(pwb_ptrs, acc_dwb, mask=h_mask[:, None] & k_mask[None, :])
        tl.store(pwo_ptrs, acc_dwo, mask=n_mask[:, None] & h_mask[None, :])

else:  # pragma: no cover

    def _fused_swiglu_transition_fwd_kernel(*_a, **_k):
        raise RuntimeError("Triton is required for fused_swiglu_transition")

    def _fused_swiglu_transition_bwd_dx_kernel(*_a, **_k):
        raise RuntimeError("Triton is required for fused_swiglu_transition")

    def _fused_swiglu_transition_bwd_dw_kernel(*_a, **_k):
        raise RuntimeError("Triton is required for fused_swiglu_transition")


def _matmul_precision_args(act: torch.Tensor) -> tuple[str, bool]:
    """``tl.dot`` input_precision from the TF32 context. bf16 inputs ignore it."""
    if act.dtype == torch.bfloat16 or not torch.backends.cuda.matmul.allow_tf32:
        return "ieee", False
    return "tf32", True


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


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
    """Launch fused forward.

    Inference (``save_acts=False``) may write in place when ``residual`` aliases
    ``x``. Training (``save_acts=True``) writes a fresh ``y`` plus ``x_hat``,
    ``a``, ``b`` for the backward kernels.
    """
    M, K = x_2d.shape
    H = w_a.shape[0]
    C_OUT = w_out.shape[0]

    in_place = residual_2d is not None and residual_2d.data_ptr() == x_2d.data_ptr()
    if save_acts and in_place:
        raise RuntimeError("fused SwiGLU training forward cannot write in place")
    y = x_2d if in_place else torch.empty((M, C_OUT), dtype=x_2d.dtype, device=x_2d.device)

    if save_acts:
        x_hat = torch.empty((M, K), dtype=x_2d.dtype, device=x_2d.device)
        a = torch.empty((M, H), dtype=x_2d.dtype, device=x_2d.device)
        b = torch.empty((M, H), dtype=x_2d.dtype, device=x_2d.device)
    else:
        x_hat = a = b = y

    BLOCK_K = max(_next_power_of_two(K), 16)
    BLOCK_N = max(_next_power_of_two(C_OUT), 16)
    input_precision, round_tf32 = _matmul_precision_args(x_2d)

    beta_ptr = beta if beta is not None else x_2d
    mask_ptr = mask_1d if mask_1d is not None else x_2d
    stride_mask_m = mask_1d.stride(0) if mask_1d is not None else 0
    res = residual_2d if residual_2d is not None else x_2d

    def _grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    _fused_swiglu_transition_fwd_kernel[_grid](
        x_2d,
        w_a,
        w_b,
        w_out,
        gamma,
        beta_ptr,
        mask_ptr,
        res,
        y,
        x_hat,
        a,
        b,
        x_2d.stride(0),
        x_2d.stride(1),
        w_a.stride(0),
        w_a.stride(1),
        w_b.stride(0),
        w_b.stride(1),
        w_out.stride(0),
        w_out.stride(1),
        res.stride(0),
        res.stride(1),
        y.stride(0),
        y.stride(1),
        stride_mask_m,
        x_hat.stride(0),
        a.stride(0),
        b.stride(0),
        M,
        K,
        H,
        C_OUT,
        eps,
        HAS_LN_BIAS=beta is not None,
        HAS_MASK=mask_1d is not None,
        HAS_RESIDUAL=residual_2d is not None,
        SAVE_ACTS=save_acts,
        INPUT_PRECISION=input_precision,
        ROUND_TF32=round_tf32,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    if save_acts:
        return y, x_hat, a, b
    return y


def fused_swiglu_transition_weight_grad(
    x_hat: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_out_2d: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_out: torch.Tensor,
    mask_1d: torch.Tensor | None,
    compute_dx: bool = True,
    compute_dw: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Triton backward from saved ``x_hat``, ``a``, ``b``.

    Returns ``d(x_hat), dWa, dWb, dWo``. Split-M ``dW`` runs first so the
    ``x_hat`` buffer can be reused for ``d(x_hat)`` when it is fp32.
    """
    M, K = x_hat.shape
    H = a.shape[1]
    C_OUT = w_out.shape[0]
    input_precision, round_tf32 = _matmul_precision_args(a)
    BLOCK_K = max(_next_power_of_two(K), 16)
    BLOCK_N = max(_next_power_of_two(C_OUT), 16)

    a_c = a.contiguous()
    b_c = b.contiguous()
    go_c = grad_out_2d.contiguous()
    wo_c = w_out.contiguous()
    mask_ptr = mask_1d if mask_1d is not None else a_c
    stride_mask_m = mask_1d.stride(0) if mask_1d is not None else 0
    has_mask = mask_1d is not None

    grad_wa = grad_wb = grad_wo = None
    if compute_dw:
        x_hat_c = x_hat.contiguous()
        partial_wa = torch.zeros(
            _BWD_SPLIT_M, H, K, device=x_hat.device, dtype=torch.float32
        )
        partial_wb = torch.zeros(
            _BWD_SPLIT_M, H, K, device=x_hat.device, dtype=torch.float32
        )
        partial_wo = torch.zeros(
            _BWD_SPLIT_M, C_OUT, H, device=x_hat.device, dtype=torch.float32
        )

        def _grid_dw(meta):
            return (_BWD_SPLIT_M, triton.cdiv(H, meta["BLOCK_H"]))

        _fused_swiglu_transition_bwd_dw_kernel[_grid_dw](
            x_hat_c,
            a_c,
            b_c,
            wo_c,
            mask_ptr,
            go_c,
            partial_wa,
            partial_wb,
            partial_wo,
            x_hat_c.stride(0),
            a_c.stride(0),
            b_c.stride(0),
            go_c.stride(0),
            stride_mask_m,
            M,
            K,
            H,
            C_OUT,
            HAS_MASK=has_mask,
            INPUT_PRECISION=input_precision,
            ROUND_TF32=round_tf32,
            SPLIT_M=_BWD_SPLIT_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )
        grad_wa = partial_wa.sum(dim=0)
        grad_wb = partial_wb.sum(dim=0)
        grad_wo = partial_wo.sum(dim=0)
        del partial_wa, partial_wb, partial_wo

    grad_x_hat = None
    if compute_dx:
        # fp32 x_hat is dead after dW; reuse it for d(x_hat) instead of +1U.
        if x_hat.dtype == torch.float32:
            grad_x_hat = x_hat
        else:
            grad_x_hat = torch.empty((M, K), dtype=torch.float32, device=x_hat.device)
        wa_c = w_a.contiguous()
        wb_c = w_b.contiguous()

        def _grid_dx(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _fused_swiglu_transition_bwd_dx_kernel[_grid_dx](
            a_c,
            b_c,
            wa_c,
            wb_c,
            wo_c,
            mask_ptr,
            go_c,
            grad_x_hat,
            a_c.stride(0),
            b_c.stride(0),
            go_c.stride(0),
            grad_x_hat.stride(0),
            stride_mask_m,
            M,
            K,
            H,
            C_OUT,
            HAS_MASK=has_mask,
            INPUT_PRECISION=input_precision,
            ROUND_TF32=round_tf32,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )

    return grad_x_hat, grad_wa, grad_wb, grad_wo


class _FusedSwiGLUTransitionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, w_a, w_b, w_out, mask, eps):
        has_beta = beta is not None
        has_mask = mask is not None
        beta_t = beta if has_beta else x.new_empty(0)
        mask_t = mask if has_mask else x.new_empty(0)

        c_in = gamma.shape[0]
        x_2d = x.contiguous().view(-1, c_in)
        mask_1d = mask.reshape(-1).contiguous() if has_mask else None
        y_2d, x_hat, a, b = _launch_fused_swiglu_transition(
            x_2d,
            gamma,
            beta if has_beta else None,
            w_a,
            w_b,
            w_out,
            mask_1d,
            None,
            float(eps),
            save_acts=True,
        )
        y = y_2d.view_as(x)

        ctx.save_for_backward(x, x_hat, a, b, gamma, beta_t, w_a, w_b, w_out, mask_t)
        ctx.eps = float(eps)
        ctx.has_beta = has_beta
        ctx.has_mask = has_mask
        ctx.x_shape = x.shape
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, x_hat, a, b, gamma, beta_t, w_a, w_b, w_out, mask_t = ctx.saved_tensors
        beta = beta_t if ctx.has_beta else None
        mask = mask_t if ctx.has_mask else None
        eps = ctx.eps
        c_in = gamma.shape[0]

        go_2d = grad_out.contiguous().view(-1, grad_out.shape[-1])
        mask_1d = mask.reshape(-1).contiguous() if mask is not None else None

        need_dx = any(ctx.needs_input_grad[:3])
        need_dw = any(ctx.needs_input_grad[3:6])
        grad_x_hat, grad_wa, grad_wb, grad_wo = fused_swiglu_transition_weight_grad(
            x_hat,
            a,
            b,
            go_2d,
            w_a,
            w_b,
            w_out,
            mask_1d,
            compute_dx=need_dx,
            compute_dw=need_dw,
        )

        grad_x = grad_gamma = grad_beta = None
        if need_dx:
            x_2d = x.contiguous().view(-1, c_in)
            x_f = x_2d.float()
            mean = x_f.mean(dim=-1)
            var = x_f.var(dim=-1, unbiased=False)
            rstd = torch.rsqrt(var + eps)
            ln_weight = gamma.float()
            ln_bias = beta.float() if beta is not None else None
            grad_input, grad_gamma, grad_beta = torch.ops.aten.native_layer_norm_backward(
                grad_x_hat,
                x_f,
                (c_in,),
                mean,
                rstd,
                ln_weight,
                ln_bias,
                (
                    ctx.needs_input_grad[0],
                    ctx.needs_input_grad[1],
                    ctx.needs_input_grad[2],
                ),
            )
            if grad_input is not None:
                grad_x = grad_input.view(ctx.x_shape).to(dtype=x.dtype)
            if grad_gamma is not None:
                grad_gamma = grad_gamma.to(dtype=gamma.dtype)
            if grad_beta is not None:
                grad_beta = grad_beta.to(dtype=beta.dtype)

        if not ctx.needs_input_grad[3]:
            grad_wa = None
        else:
            grad_wa = grad_wa.to(dtype=w_a.dtype)
        if not ctx.needs_input_grad[4]:
            grad_wb = None
        else:
            grad_wb = grad_wb.to(dtype=w_b.dtype)
        if not ctx.needs_input_grad[5]:
            grad_wo = None
        else:
            grad_wo = grad_wo.to(dtype=w_out.dtype)

        return grad_x, grad_gamma, grad_beta, grad_wa, grad_wb, grad_wo, None, None


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
    """Fused LN -> SwiGLU -> Linear -> (*mask) [-> +residual].

    Triton only. The model uses ``SwiGLUTransition`` primitives when this
    launch is ineligible. Inference may write in place when ``residual is
    x``. Training is always out-of-place and saves activations for the
    Triton backward. Triton keeps fp32 masters; bf16 activations use bf16
    GEMMs.
    """
    if not is_fused_swiglu_transition_eligible(x, gamma, beta, w_a, w_b, w_out):
        raise RuntimeError(
            "fused_swiglu_transition requires an eligible Triton launch; "
            "use SwiGLUTransition for the eager path"
        )

    c_in = gamma.shape[0]
    use_grad = torch.is_grad_enabled() and (
        x.requires_grad
        or gamma.requires_grad
        or (beta is not None and beta.requires_grad)
        or w_a.requires_grad
        or w_b.requires_grad
        or w_out.requires_grad
    )
    if use_grad:
        if residual is not None:
            update = fused_swiglu_transition(
                x, gamma, beta, w_a, w_b, w_out, mask=mask, eps=eps, residual=None
            )
            return residual + update
        return _FusedSwiGLUTransitionFn.apply(
            x, gamma, beta, w_a, w_b, w_out, mask, eps
        )

    x_2d = x.contiguous().view(-1, c_in)
    mask_1d = mask.reshape(-1) if mask is not None else None
    res_2d = residual.view(-1, c_in) if residual is not None else None
    y_2d = _launch_fused_swiglu_transition(
        x_2d, gamma, beta, w_a, w_b, w_out, mask_1d, res_2d, eps, save_acts=False
    )
    return y_2d.view_as(x if residual is None else residual)
