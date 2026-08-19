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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused triangle multiplicative
# update (Triton) reconstructed from the original inference kernel.

"""Fused triangle multiplicative update (Triton).

LN_in → gated A/B projections → triangle einsum → LN_out → gated output
[+ residual]. Sequence length is not an autotune or specialize key.
Ineligible shapes use the matching-precision eager primitives.
In-place residual writes are inference-only.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# Production PairBlock is c_z = c_hidden = 128. Wider sites fall back.
_MAX_C = 128
_MIN_M = 4096
_TRUE = {"1", "true", "yes", "on"}


def is_fused_trimul_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_TRIMUL", "1").strip().lower() in _TRUE


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


def _all_fp32_masters(*tensors: torch.Tensor | None) -> bool:
    return all(t is None or t.dtype == torch.float32 for t in tensors)


def is_fused_trimul_eligible(
    z: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
) -> bool:
    if z.ndim != 4:
        return False
    _batch, n, n2, c_z = z.shape
    c_hidden = wa_p.shape[0]
    masters = (wa_p, wa_g, wb_p, wb_g, wz, wg, ln_in_w, ln_out_w, ln_in_b, ln_out_b)
    return (
        is_fused_trimul_enabled()
        and _TRITON_AVAILABLE
        and z.is_cuda
        and z.is_contiguous()
        and n == n2
        and z.dtype in (torch.float32, torch.bfloat16)
        and _all_fp32_masters(*masters)
        and c_z <= _MAX_C
        and c_hidden <= _MAX_C
        and wz.shape[0] == c_z
        and wz.shape[1] == c_hidden
        and wg.shape == (c_z, c_z)
        and (z.numel() // c_z) >= _MIN_M
    )


def _downcast_masters(act_dtype: torch.dtype, *tensors: torch.Tensor | None):
    if act_dtype != torch.bfloat16:
        return tensors
    return tuple(None if t is None else t.to(dtype=act_dtype) for t in tensors)


def eager_trimul(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    outgoing: bool,
    *,
    ln_in_eps: float = 1e-5,
    ln_out_eps: float = 1e-5,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Matching-precision eager AF3 trimul (bf16 downcasts masters)."""
    (
        wa_p,
        wa_g,
        wb_p,
        wb_g,
        wz,
        wg,
        ln_in_w,
        ln_in_b,
        ln_out_w,
        ln_out_b,
    ) = _downcast_masters(
        z.dtype, wa_p, wa_g, wb_p, wb_g, wz, wg, ln_in_w, ln_in_b, ln_out_w, ln_out_b
    )
    if mask is None:
        mask_u = z.new_ones(z.shape[:-1] + (1,))
    else:
        mask_u = mask.unsqueeze(-1)
    z_n = F.layer_norm(z, (ln_in_w.shape[0],), ln_in_w, ln_in_b, ln_in_eps)
    a = mask_u * torch.sigmoid(F.linear(z_n, wa_g)) * F.linear(z_n, wa_p)
    b = mask_u * torch.sigmoid(F.linear(z_n, wb_g)) * F.linear(z_n, wb_p)
    if outgoing:
        x = torch.einsum("...ikc,...jkc->...ijc", a, b)
    else:
        x = torch.einsum("...kic,...kjc->...ijc", a, b)
    x = F.layer_norm(x, (ln_out_w.shape[0],), ln_out_w, ln_out_b, ln_out_eps)
    x = torch.sigmoid(F.linear(z_n, wg)) * F.linear(x, wz)
    return z + x if residual is not None else x


if _TRITON_AVAILABLE:

    _DUAL_GEMM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2
    )
    _DUAL_GEMM_CFG_N64 = dict(
        TILE_M=64, TILE_N=64, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2
    )
    _OUT_DM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2
    )
    _OUT_GEMM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2
    )
    _LN_STATS_TILE_M = 64
    _LN_TRANSPOSE_TILE_M = 64
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
    def _pid_mn(M, N, TILE_M: tl.constexpr, TILE_N: tl.constexpr, GROUP_M: tl.constexpr):
        pid_m_raw = tl.program_id(0)
        pid_n_raw = tl.program_id(1)
        num_pid_m = tl.cdiv(M, TILE_M)
        num_pid_n = tl.cdiv(N, TILE_N)
        pid = pid_n_raw * num_pid_m + pid_m_raw
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        return tl.cast(pid_m, tl.int64), tl.cast(pid_n, tl.int64)

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "x_ptr",
            "wp_ptr",
            "wg_ptr",
            "mask_ptr",
            "gamma_ptr",
            "beta_ptr",
            "ln_stats_ptr",
            "out_ptr",
        ],
    )
    def _gated_dual_gemm_kernel(
        x_ptr,
        wp_ptr,
        wg_ptr,
        mask_ptr,
        gamma_ptr,
        beta_ptr,
        ln_stats_ptr,
        out_ptr,
        M,
        Nproj,
        K,
        eps,
        HAS_MASK: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid_m, pid_n = _pid_mn(M, Nproj, TILE_M, TILE_N, GROUP_M)
        M64 = tl.cast(M, tl.int64)
        nproj64 = tl.cast(Nproj, tl.int64)
        k64 = tl.cast(K, tl.int64)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < M64
        mask_n = offs_n < nproj64

        mean = tl.load(ln_stats_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        rstd = tl.load(ln_stats_ptr + M64 + offs_m, mask=mask_m, other=0.0).to(
            tl.float32
        )
        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, K, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < k64
            z_k = tl.load(
                x_ptr + offs_m[:, None] * k64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_k = tl.load(gamma_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)
            x_tile = (z_k - mean[:, None]) * rstd[:, None] * gamma_k[None, :]
            if HAS_LN_BIAS:
                beta_k = tl.load(beta_ptr + k_range, mask=k_mask, other=0.0).to(
                    tl.float32
                )
                x_tile = x_tile + tl.where(k_mask[None, :], beta_k[None, :], 0.0)
            wp = tl.load(
                wp_ptr + offs_n[None, :] * k64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            wg = tl.load(
                wg_ptr + offs_n[None, :] * k64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            val_acc += _dot_f32(x_tile, wp, GEMM_MODE)
            gate_acc += _dot_f32(x_tile, wg, GEMM_MODE)

        delta = tl.sigmoid(gate_acc) * val_acc
        if HAS_MASK:
            mtile = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            delta = delta * mtile[:, None]
        tl.store(
            out_ptr + offs_n[:, None] * M64 + offs_m[None, :],
            tl.trans(delta).to(out_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_m[None, :],
        )

    @triton.jit(
        do_not_specialize=["M", "eps_out"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_dm_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
            "gamma_in_ptr",
            "beta_in_ptr",
            "ln_stats_ptr",
            "gamma_out_ptr",
            "beta_out_ptr",
            "out_ptr",
        ],
    )
    def _gated_out_from_dm_kernel(
        x_in_ptr,
        x_dm_ptr,
        wg_ptr,
        wp_ptr,
        residual_ptr,
        gamma_in_ptr,
        beta_in_ptr,
        ln_stats_ptr,
        gamma_out_ptr,
        beta_out_ptr,
        out_ptr,
        M,
        CZ,
        CH,
        eps_out,
        WITH_ADD: tl.constexpr,
        HAS_LN_IN_BIAS: tl.constexpr,
        HAS_LN_OUT_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid_m, pid_n = _pid_mn(M, CZ, TILE_M, TILE_N, GROUP_M)
        m64 = tl.cast(M, tl.int64)
        cz64 = tl.cast(CZ, tl.int64)
        ch64 = tl.cast(CH, tl.int64)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < m64
        mask_n = offs_n < cz64

        out_sum = tl.zeros((TILE_M,), dtype=tl.float32)
        out_sumsq = tl.zeros((TILE_M,), dtype=tl.float32)
        for k_off in range(0, CH, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < ch64
            x_k = tl.load(
                x_dm_ptr + k_range[None, :] * m64 + offs_m[:, None],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_k = tl.where(k_mask[None, :], x_k, 0.0)
            out_sum += tl.sum(x_k, axis=1)
            out_sumsq += tl.sum(x_k * x_k, axis=1)
        out_mean = out_sum / ch64
        out_var = tl.maximum(out_sumsq / ch64 - out_mean * out_mean, 0.0)
        out_rstd = 1.0 / tl.sqrt(out_var + eps_out)

        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CH, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < ch64
            x_k = tl.load(
                x_dm_ptr + k_range[None, :] * m64 + offs_m[:, None],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_out = tl.load(gamma_out_ptr + k_range, mask=k_mask, other=0.0).to(
                tl.float32
            )
            x_value = (x_k - out_mean[:, None]) * out_rstd[:, None] * gamma_out[None, :]
            if HAS_LN_OUT_BIAS:
                beta_out = tl.load(beta_out_ptr + k_range, mask=k_mask, other=0.0).to(
                    tl.float32
                )
                x_value += tl.where(k_mask[None, :], beta_out[None, :], 0.0)
            wp = tl.load(
                wp_ptr + offs_n[None, :] * ch64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            val_acc += _dot_f32(x_value, wp, GEMM_MODE)

        in_mean = tl.load(ln_stats_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        in_rstd = tl.load(ln_stats_ptr + m64 + offs_m, mask=mask_m, other=0.0).to(
            tl.float32
        )
        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < cz64
            z_k = tl.load(
                x_in_ptr + offs_m[:, None] * cz64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_in = tl.load(gamma_in_ptr + k_range, mask=k_mask, other=0.0).to(
                tl.float32
            )
            x_gate = (z_k - in_mean[:, None]) * in_rstd[:, None] * gamma_in[None, :]
            if HAS_LN_IN_BIAS:
                beta_in = tl.load(beta_in_ptr + k_range, mask=k_mask, other=0.0).to(
                    tl.float32
                )
                x_gate += tl.where(k_mask[None, :], beta_in[None, :], 0.0)
            wgate = tl.load(
                wg_ptr + offs_n[None, :] * cz64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            gate_acc += _dot_f32(x_gate, wgate, GEMM_MODE)

        out_val = tl.sigmoid(gate_acc) * val_acc
        if WITH_ADD:
            resid = tl.load(
                residual_ptr + offs_m[:, None] * cz64 + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            out_val += resid
        tl.store(
            out_ptr + offs_m[:, None] * cz64 + offs_n[None, :],
            out_val.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=["x_ptr", "out_ptr"],
    )
    def _ln_stats_kernel(
        x_ptr,
        out_ptr,
        M,
        D: tl.constexpr,
        eps,
        TILE_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        m64 = tl.cast(M, tl.int64)
        offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        mask_m = offs_m < m64
        mask_d = offs_d < D
        x = tl.load(
            x_ptr + offs_m[:, None] * D + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.where(mask_d[None, :], x, 0.0)
        mean = tl.sum(x, axis=1) / D
        x_c = tl.where(mask_d[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x_c * x_c, axis=1) / D + eps)
        tl.store(out_ptr + offs_m, mean, mask=mask_m)
        tl.store(out_ptr + m64 + offs_m, rstd, mask=mask_m)

    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["x_ptr", "w_ptr", "b_ptr", "out_ptr"],
    )
    def _ln_transpose_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        M,
        D: tl.constexpr,
        EPS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        m64 = tl.cast(M, tl.int64)
        offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_d = tl.arange(0, D).to(tl.int64)
        mask_m = offs_m < m64
        x = tl.load(
            x_ptr + offs_d[None, :] * m64 + offs_m[:, None],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        x_c = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(x_c * x_c, axis=1) / D + EPS)
        y = x_c * rstd[:, None] * tl.load(w_ptr + offs_d).to(tl.float32)[None, :]
        if HAS_BIAS:
            y = y + tl.load(b_ptr + offs_d).to(tl.float32)[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * D + offs_d[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_out_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
            "gamma_ptr",
            "beta_ptr",
            "ln_stats_ptr",
            "out_ptr",
        ],
    )
    def _gated_out_gemm_residual_kernel(
        x_in_ptr,
        x_out_ptr,
        wg_ptr,
        wp_ptr,
        residual_ptr,
        gamma_ptr,
        beta_ptr,
        ln_stats_ptr,
        out_ptr,
        M,
        CZ,
        CH,
        eps,
        WITH_ADD: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid_m, pid_n = _pid_mn(M, CZ, TILE_M, TILE_N, GROUP_M)
        m64 = tl.cast(M, tl.int64)
        cz64 = tl.cast(CZ, tl.int64)
        ch64 = tl.cast(CH, tl.int64)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < m64
        mask_n = offs_n < cz64

        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CH, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < ch64
            xo = tl.load(
                x_out_ptr + offs_m[:, None] * ch64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wp = tl.load(
                wp_ptr + offs_n[None, :] * ch64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            val_acc += _dot_f32(xo, wp, GEMM_MODE)

        mean = tl.load(ln_stats_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        rstd = tl.load(ln_stats_ptr + m64 + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < cz64
            z_k = tl.load(
                x_in_ptr + offs_m[:, None] * cz64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_k = tl.load(gamma_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)
            xi = (z_k - mean[:, None]) * rstd[:, None] * gamma_k[None, :]
            if HAS_LN_BIAS:
                beta_k = tl.load(beta_ptr + k_range, mask=k_mask, other=0.0).to(
                    tl.float32
                )
                xi = xi + tl.where(k_mask[None, :], beta_k[None, :], 0.0)
            wgate = tl.load(
                wg_ptr + offs_n[None, :] * cz64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            gate_acc += _dot_f32(xi, wgate, GEMM_MODE)

        out_val = tl.sigmoid(gate_acc) * val_acc
        if WITH_ADD:
            resid = tl.load(
                residual_ptr + offs_m[:, None] * cz64 + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            out_val += resid
        tl.store(
            out_ptr + offs_m[:, None] * cz64 + offs_n[None, :],
            out_val.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_trimul")

    _gated_dual_gemm_kernel = _unavailable
    _gated_out_from_dm_kernel = _unavailable
    _ln_stats_kernel = _unavailable
    _ln_transpose_kernel = _unavailable
    _gated_out_gemm_residual_kernel = _unavailable


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _dual_cfg(nproj: int) -> dict:
    return _DUAL_GEMM_CFG_N64 if nproj < 128 else _DUAL_GEMM_CFG


def ln_stats(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return per-row LayerNorm mean/rstd as float32 ``[2, M]``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    m, d = x.shape
    x = x.contiguous()
    out = torch.empty((2, m), device=x.device, dtype=torch.float32)
    _ln_stats_kernel[(triton.cdiv(m, _LN_STATS_TILE_M),)](
        x,
        out,
        m,
        D=d,
        eps=float(eps),
        TILE_M=_LN_STATS_TILE_M,
        BLOCK_D=_next_power_of_two(d),
        num_warps=8,
        num_stages=2,
    )
    return out


def gated_dual_gemm(
    x: torch.Tensor,
    wp: torch.Tensor,
    wg: torch.Tensor,
    mask: torch.Tensor | None,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor | None,
    ln_stats_t: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``sigmoid(LN(x)@wg^T) * (LN(x)@wp^T) * mask`` as ``[Nproj, M]``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    m, k = x.shape
    nproj = wp.shape[0]
    x = x.contiguous()
    out = torch.empty((nproj, m), device=x.device, dtype=x.dtype)
    dummy = x
    cfg = _dual_cfg(nproj)

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(nproj, meta["TILE_N"]))

    _gated_dual_gemm_kernel[grid](
        x,
        wp.contiguous(),
        wg.contiguous(),
        mask.contiguous().view(-1) if mask is not None else dummy,
        ln_weight.contiguous(),
        ln_bias.contiguous() if ln_bias is not None else dummy,
        ln_stats_t.contiguous(),
        out,
        m,
        nproj,
        k,
        float(eps),
        HAS_MASK=mask is not None,
        HAS_LN_BIAS=ln_bias is not None,
        GEMM_MODE=_gemm_mode(x),
        **cfg,
    )
    return out


def ln_transpose(
    x_dm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm with layout transpose: ``(D, M)`` → ``(M, D)``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    d, m = x_dm.shape
    out = torch.empty((m, d), device=x_dm.device, dtype=x_dm.dtype)
    dummy = x_dm
    _ln_transpose_kernel[(triton.cdiv(m, _LN_TRANSPOSE_TILE_M),)](
        x_dm.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else dummy,
        out,
        m,
        D=d,
        EPS=float(eps),
        HAS_BIAS=bias is not None,
        TILE_M=_LN_TRANSPOSE_TILE_M,
        num_warps=8,
        num_stages=2,
    )
    return out


def gated_out_gemm_residual(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    wg: torch.Tensor,
    wp: torch.Tensor,
    residual: torch.Tensor | None,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor | None,
    ln_stats_t: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    m, cz = x_in.shape
    ch = x_out.shape[1]
    if out is None:
        out = torch.empty((m, cz), device=x_in.device, dtype=x_in.dtype)
    dummy = x_in

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(cz, meta["TILE_N"]))

    _gated_out_gemm_residual_kernel[grid](
        x_in.contiguous(),
        x_out.contiguous(),
        wg.contiguous(),
        wp.contiguous(),
        residual.contiguous() if residual is not None else dummy,
        ln_weight.contiguous(),
        ln_bias.contiguous() if ln_bias is not None else dummy,
        ln_stats_t.contiguous(),
        out,
        m,
        cz,
        ch,
        1e-5,
        WITH_ADD=residual is not None,
        HAS_LN_BIAS=ln_bias is not None,
        GEMM_MODE=_gemm_mode(x_in),
        **_OUT_GEMM_CFG,
    )
    return out


def gated_out_from_dm(
    x_in: torch.Tensor,
    x_dm: torch.Tensor,
    wg: torch.Tensor,
    wp: torch.Tensor,
    residual: torch.Tensor | None,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_stats_t: torch.Tensor,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    ln_out_eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse LN_out on D-major contraction output with gated output projection."""
    m, cz = x_in.shape
    ch, x_m = x_dm.shape
    if x_m != m:
        raise ValueError(f"x_dm has M={x_m}, expected {m}")
    if ch < 128:
        x_out = ln_transpose(x_dm, ln_out_w, ln_out_b, eps=ln_out_eps)
        return gated_out_gemm_residual(
            x_in,
            x_out,
            wg,
            wp,
            residual,
            ln_in_w,
            ln_in_b,
            ln_stats_t,
            out=out,
        )
    if out is None:
        out = torch.empty((m, cz), device=x_in.device, dtype=x_in.dtype)
    dummy = x_in
    if cz > 128 or ch > 128:
        raise ValueError("fused D-major output supports CZ/CH <= 128")
    if out.data_ptr() == x_in.data_ptr() and _OUT_DM_CFG["TILE_N"] < cz:
        raise ValueError("in-place D-major output requires one channel tile")
    grid = (
        triton.cdiv(m, _OUT_DM_CFG["TILE_M"]),
        triton.cdiv(cz, _OUT_DM_CFG["TILE_N"]),
    )
    _gated_out_from_dm_kernel[grid](
        x_in.contiguous(),
        x_dm.contiguous(),
        wg.contiguous(),
        wp.contiguous(),
        residual.contiguous() if residual is not None else dummy,
        ln_in_w.contiguous(),
        ln_in_b.contiguous() if ln_in_b is not None else dummy,
        ln_stats_t.contiguous(),
        ln_out_w.contiguous(),
        ln_out_b.contiguous() if ln_out_b is not None else dummy,
        out,
        m,
        cz,
        ch,
        float(ln_out_eps),
        WITH_ADD=residual is not None,
        HAS_LN_IN_BIAS=ln_in_b is not None,
        HAS_LN_OUT_BIAS=ln_out_b is not None,
        GEMM_MODE=_gemm_mode(x_in),
        **_OUT_DM_CFG,
    )
    return out


def _contract(a: torch.Tensor, b: torch.Tensor, outgoing: bool) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if outgoing:
            return torch.einsum("cbik,cbjk->cbij", a, b)
        return torch.einsum("cbki,cbkj->cbij", a, b)


def _trimul_whole(
    z: torch.Tensor,
    mask_flat: torch.Tensor,
    stats: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    outgoing: bool,
    residual: torch.Tensor | None,
    ln_in_eps: float,
    ln_out_eps: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    batch, n, _, c_z = z.shape
    c_hidden = wa_p.shape[0]
    z_2d = z.reshape(-1, c_z)
    wp_ab = torch.cat([wa_p, wb_p], dim=0)
    wg_ab = torch.cat([wa_g, wb_g], dim=0)
    ab = gated_dual_gemm(
        z_2d, wp_ab, wg_ab, mask_flat, ln_in_w, ln_in_b, stats, eps=ln_in_eps
    )
    a = ab[:c_hidden].view(c_hidden, batch, n, n)
    b = ab[c_hidden:].view(c_hidden, batch, n, n)
    x = _contract(a, b, outgoing)
    del a, b, ab
    x_dm = x.reshape(c_hidden, batch * n * n)
    out_2d = gated_out_from_dm(
        z_2d,
        x_dm,
        wg,
        wz,
        residual.reshape(-1, c_z) if residual is not None else None,
        ln_in_w,
        ln_in_b,
        stats,
        ln_out_w,
        ln_out_b,
        ln_out_eps=ln_out_eps,
        out=out.reshape(-1, c_z) if out is not None else None,
    )
    y = out_2d.view(batch, n, n, c_z)
    del x, x_dm
    return y


def _trimul_chunked_outgoing(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    stats: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    residual: torch.Tensor | None,
    ln_in_eps: float,
    ln_out_eps: float,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    m = n * n
    b_full = gated_dual_gemm(
        z_2d, wb_p, wb_g, mask_flat, ln_in_w, ln_in_b, stats, eps=ln_in_eps
    )
    b_4d = b_full.view(c_hidden, 1, n, n)
    out_2d = out.reshape(m, c_z) if out is not None else torch.empty(
        (m, c_z), device=z_2d.device, dtype=z_2d.dtype
    )
    resid_2d = residual.reshape(m, c_z) if residual is not None else None
    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        a_c = gated_dual_gemm(
            z_2d[m0:m1],
            wa_p,
            wa_g,
            mask_flat[m0:m1],
            ln_in_w,
            ln_in_b,
            stats[:, m0:m1].contiguous(),
            eps=ln_in_eps,
        )
        x_c = _contract(a_c.view(c_hidden, 1, rows, n), b_4d, outgoing=True)
        del a_c
        gated_out_from_dm(
            z_2d[m0:m1],
            x_c.reshape(c_hidden, rows * n),
            wg,
            wz,
            resid_2d[m0:m1] if resid_2d is not None else None,
            ln_in_w,
            ln_in_b,
            stats[:, m0:m1].contiguous(),
            ln_out_w,
            ln_out_b,
            ln_out_eps=ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_c
    return out_2d.view(1, n, n, c_z)


def _trimul_chunked_incoming(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    stats: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    residual: torch.Tensor | None,
    ln_in_eps: float,
    ln_out_eps: float,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    m = n * n
    group_size = min(64, c_hidden)
    grouped_k_cap = min(n, chunk_cap * c_hidden // group_size)
    x_accum = torch.empty((c_hidden, n, n), device=z_2d.device, dtype=z_2d.dtype)
    for c_start in range(0, c_hidden, group_size):
        c_end = min(c_hidden, c_start + group_size)
        channels = c_end - c_start
        wp_ab = torch.cat([wa_p[c_start:c_end], wb_p[c_start:c_end]], dim=0)
        wg_ab = torch.cat([wa_g[c_start:c_end], wb_g[c_start:c_end]], dim=0)
        x_group = x_accum[c_start:c_end]
        first_chunk = True
        for k_start in range(0, n, grouped_k_cap):
            k_end = min(n, k_start + grouped_k_cap)
            k_rows = k_end - k_start
            m0, m1 = k_start * n, k_end * n
            ab_k = gated_dual_gemm(
                z_2d[m0:m1],
                wp_ab,
                wg_ab,
                mask_flat[m0:m1],
                ln_in_w,
                ln_in_b,
                stats[:, m0:m1].contiguous(),
                eps=ln_in_eps,
            )
            a_k = ab_k[:channels].view(channels, k_rows, n)
            b_k = ab_k[channels:].view(channels, k_rows, n)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                if first_chunk:
                    torch.bmm(a_k.transpose(1, 2), b_k, out=x_group)
                    first_chunk = False
                else:
                    torch.baddbmm(x_group, a_k.transpose(1, 2), b_k, out=x_group)
            del ab_k, a_k, b_k

    out_2d = out.reshape(m, c_z) if out is not None else torch.empty(
        (m, c_z), device=z_2d.device, dtype=z_2d.dtype
    )
    resid_2d = residual.reshape(m, c_z) if residual is not None else None
    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        x_dm = x_accum[:, i_start:i_end, :].reshape(c_hidden, rows * n).contiguous()
        gated_out_from_dm(
            z_2d[m0:m1],
            x_dm,
            wg,
            wz,
            resid_2d[m0:m1] if resid_2d is not None else None,
            ln_in_w,
            ln_in_b,
            stats[:, m0:m1].contiguous(),
            ln_out_w,
            ln_out_b,
            ln_out_eps=ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_dm
    return out_2d.view(1, n, n, c_z)


def _launch_fused_trimul(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    outgoing: bool,
    residual: torch.Tensor | None,
    ln_in_eps: float,
    ln_out_eps: float,
    chunk_cap: int | None,
) -> torch.Tensor:
    batch, n, _, c_z = z.shape
    c_hidden = wa_p.shape[0]
    z_c = z.contiguous()
    z_2d = z_c.reshape(-1, c_z)
    mask_flat = (
        mask.reshape(-1).contiguous()
        if mask is not None
        else z_c.new_ones(z_2d.shape[0])
    )
    stats = ln_stats(z_2d, ln_in_eps)
    inplace = (
        residual is not None
        and residual.data_ptr() == z_c.data_ptr()
        and not torch.is_grad_enabled()
    )
    out = z_c if inplace else None
    resid = z_c if residual is not None else None
    kwargs = dict(
        outgoing=outgoing,
        residual=resid,
        ln_in_eps=ln_in_eps,
        ln_out_eps=ln_out_eps,
        out=out,
    )
    if chunk_cap is not None and chunk_cap < n and batch == 1:
        chunk_kwargs = dict(
            n=n,
            c_z=c_z,
            c_hidden=c_hidden,
            residual=resid,
            ln_in_eps=ln_in_eps,
            ln_out_eps=ln_out_eps,
            out=out,
            chunk_cap=chunk_cap,
        )
        if outgoing:
            return _trimul_chunked_outgoing(
                z_2d,
                mask_flat,
                stats,
                wa_p,
                wa_g,
                wb_p,
                wb_g,
                wz,
                wg,
                ln_in_w,
                ln_in_b,
                ln_out_w,
                ln_out_b,
                **chunk_kwargs,
            )
        return _trimul_chunked_incoming(
            z_2d,
            mask_flat,
            stats,
            wa_p,
            wa_g,
            wb_p,
            wb_g,
            wz,
            wg,
            ln_in_w,
            ln_in_b,
            ln_out_w,
            ln_out_b,
            **chunk_kwargs,
        )
    return _trimul_whole(
        z_c,
        mask_flat,
        stats,
        wa_p,
        wa_g,
        wb_p,
        wb_g,
        wz,
        wg,
        ln_in_w,
        ln_in_b,
        ln_out_w,
        ln_out_b,
        **kwargs,
    )


def fused_trimul(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_in_w: torch.Tensor,
    ln_in_b: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    outgoing: bool,
    *,
    ln_in_eps: float = 1e-5,
    ln_out_eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    chunk_cap: int | None = None,
) -> torch.Tensor:
    """Fused AF3 trimul. Callers must check eligibility first.

    Inference may write ``z + update`` in place when ``residual is z``.
    """
    if not is_fused_trimul_eligible(
        z, wa_p, wa_g, wb_p, wb_g, wz, wg, ln_in_w, ln_in_b, ln_out_w, ln_out_b
    ):
        raise RuntimeError(
            "fused_trimul requires an eligible Triton launch; "
            "use eager_trimul for the fallback path"
        )
    return _launch_fused_trimul(
        z,
        mask,
        wa_p,
        wa_g,
        wb_p,
        wb_g,
        wz,
        wg,
        ln_in_w,
        ln_in_b,
        ln_out_w,
        ln_out_b,
        outgoing,
        residual,
        ln_in_eps,
        ln_out_eps,
        chunk_cap,
    )


def fused_trimul_from_module(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    residual: torch.Tensor | None = None,
    chunk_cap: int | None = None,
) -> torch.Tensor | None:
    """Adapt a ``TriangleMultiplicativeUpdate`` module; ``None`` if ineligible."""
    if not hasattr(module, "linear_a_p"):
        return None
    weights = (
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_z.weight,
        module.linear_g.weight,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
    )
    if any(linear.bias is not None for linear in (
        module.linear_a_p,
        module.linear_a_g,
        module.linear_b_p,
        module.linear_b_g,
        module.linear_z,
        module.linear_g,
    )):
        return None
    if not is_fused_trimul_eligible(z, *weights):
        return None
    return fused_trimul(
        z,
        mask,
        *weights,
        outgoing=module._outgoing,
        ln_in_eps=module.layer_norm_in.eps,
        ln_out_eps=module.layer_norm_out.eps,
        residual=residual,
        chunk_cap=chunk_cap,
    )

