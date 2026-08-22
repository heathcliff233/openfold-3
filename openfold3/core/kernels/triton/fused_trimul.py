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
[+ residual]. ``M`` / sequence length is not an autotune or specialize key.
GEMM and LN tiles autotune on ``GEMM_MODE`` (LN on an empty key) so each
GPU picks its own winner. Packed ``[C,B,N,N]`` panels keep a constexpr
inner stride of 1 (layout, not target length). Forward materializes LN_in
once, then dual-gemm and the output-gate MMAs load matching dtypes (bf16
masters are downcast once; no in-loop LN).
Backward keeps ``X`` / ``dX`` / ``dA`` / ``dB`` CH and rematerializes
LN_out stats in the output-gate kernels. Gated ``dW``/``dX`` load ``dA``/``dB``
as ``[C,M]``.
Training saves A/B/X from the forward (non-reentrant checkpoint packs the
first-forward copies so they do not stack). Exclusive split-M ``dW`` plus
a fused output-gate backward that never stores ``g`` / ``val`` /
``x_hat``. Pair-sized intermediates are freed as soon as they are
consumed. Ineligible shapes use the matching-precision eager primitives.
In-place residual writes are inference-only.
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

# Production PairBlock is c_z = c_hidden = 128. Wider sites fall back.
_MAX_C = 128
_MIN_M = 4096
_BWD_SPLIT_M = 64
# Large-N biased; N is not an autotune key. IEEE drops 128/K=64 in prune.
_TRI_GEMM_TILES = (
    (64, 64, 32, 4, 3),
    (64, 64, 32, 8, 3),
    (64, 128, 32, 8, 3),
    (128, 64, 32, 8, 3),
    (128, 128, 32, 8, 3),
)
# Dual-gemm: no 64×64 — a GEMM_MODE-only cache would otherwise keep the
# small-M winner for every later length. Do not mark pointer args
# do_not_specialize_on_alignment — that spills 64×128×32. TILE_K must
# divide K (no K-mask in the inner loop). M stays do_not_specialize.
_DUAL_GEMM_TILES = (
    (64, 128, 16, 4, 2),
    (64, 128, 16, 4, 3),
    (64, 128, 16, 8, 2),
    (64, 128, 32, 4, 2),
    (64, 128, 32, 8, 2),
    (64, 128, 32, 8, 3),
)
# D-major output. TILE_N<=128 keeps in-place CZ<=128 safe. No 64×64.
_FWD_GEMM_TILES = (
    (128, 64, 16, 4, 2),
    (128, 64, 16, 4, 3),
    (128, 64, 32, 8, 2),
    (64, 128, 16, 4, 2),
    (64, 128, 32, 4, 2),
    (64, 128, 32, 8, 2),
    (64, 128, 16, 8, 3),
)
# LN row tiles (TILE_M / BLOCK_M, warps, stages). Partials size to min M.
_LN_ROW_TILES = (
    (32, 4, 2),
    (64, 4, 2),
    (64, 8, 2),
    (128, 8, 2),
)
_LN_BWD_TILES = (
    (16, 4, 1),
    (32, 4, 1),
    (32, 8, 1),
    (64, 4, 1),
)
_BWD_GEMM_TILES = (
    (16, 32, 4, 1),
    (16, 64, 4, 1),
    (32, 32, 4, 1),
    (32, 64, 4, 1),
    (32, 64, 4, 2),
    (64, 32, 4, 1),
    (64, 32, 8, 1),
)
_BWD_MIN_BLOCK_M = 16
# First dual-gemm launch below this M warms autotune on a large dummy
# grid so a test-sized call cannot lock the GEMM_MODE winner.
_DUAL_WARM_M = 65536
_TRUE = {"1", "true", "yes", "on"}


def is_fused_trimul_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_TRIMUL", "1").strip().lower() in _TRUE


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

    def _gemm_configs(tiles):
        return [
            triton.Config(
                {"TILE_M": tm, "TILE_N": tn, "TILE_K": tk, "GROUP_M": 8},
                num_warps=warps,
                num_stages=stages,
            )
            for tm, tn, tk, warps, stages in tiles
        ]

    def _dual_gemm_configs():
        return [
            triton.Config(
                {"TILE_M": tm, "TILE_N": tn, "TILE_K": tk},
                num_warps=warps,
                num_stages=stages,
            )
            for tm, tn, tk, warps, stages in _DUAL_GEMM_TILES
        ]

    def _ln_row_configs(tiles):
        return [
            triton.Config(
                {"TILE_M": tile_m},
                num_warps=warps,
                num_stages=stages,
            )
            for tile_m, warps, stages in tiles
        ]

    def _ln_bwd_configs():
        return [
            triton.Config(
                {"BLOCK_M": block_m},
                num_warps=warps,
                num_stages=stages,
            )
            for block_m, warps, stages in _LN_BWD_TILES
        ]

    def _bwd_gemm_configs():
        return [
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n},
                num_warps=warps,
                num_stages=stages,
            )
            for block_m, block_n, warps, stages in _BWD_GEMM_TILES
        ]

    def _tri_gemm_configs():
        return _gemm_configs(_TRI_GEMM_TILES)

    def _prune_tri_gemm(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        kept = [
            cfg
            for cfg in configs
            if not (
                mode == "ieee"
                and (cfg.kwargs["TILE_M"] >= 128 or cfg.kwargs["TILE_K"] >= 64)
            )
        ]
        return kept or configs[:1]

    def _fwd_gemm_configs():
        return _gemm_configs(_FWD_GEMM_TILES)

    def _prune_fwd_gemm(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        n_out = named_args.get("Nproj", named_args.get("CZ"))
        kept = []
        for cfg in configs:
            if cfg.kwargs["TILE_K"] >= 64 or (
                cfg.kwargs["TILE_M"] >= 128 and cfg.kwargs["TILE_N"] >= 128
            ):
                continue
            if mode == "ieee" and cfg.kwargs["TILE_M"] >= 128:
                continue
            if n_out is not None and cfg.kwargs["TILE_N"] > n_out:
                continue
            k_in = named_args.get("K", named_args.get("CH"))
            if k_in is not None and cfg.kwargs["TILE_K"] > k_in:
                continue
            kept.append(cfg)
        return kept or configs[:1]

    def _prune_dual_gemm(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        nproj = named_args.get("Nproj")
        k = named_args.get("K")
        kept = []
        for cfg in configs:
            tile_k = cfg.kwargs["TILE_K"]
            tile_n = cfg.kwargs["TILE_N"]
            if tile_k >= 64:
                continue
            if mode == "ieee" and tile_k >= 32:
                continue
            if nproj is not None and tile_n > nproj:
                continue
            if k is not None and (tile_k > k or k % tile_k != 0):
                continue
            kept.append(cfg)
        return kept or configs[:1]

    def _prune_bwd_gemm(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        n = named_args.get("N", named_args.get("CZ"))
        kept = []
        for cfg in configs:
            block_m = cfg.kwargs["BLOCK_M"]
            block_n = cfg.kwargs["BLOCK_N"]
            if mode == "ieee" and (block_m >= 64 or block_n >= 64):
                continue
            if n is not None and block_n > n:
                continue
            kept.append(cfg)
        return kept or configs[:1]

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
    def _mma(a, b, acc, GEMM_MODE: tl.constexpr):
        """Accumulate ``a @ b`` into ``acc``. bf16 uses default tensor-core MMA."""
        if GEMM_MODE == "bf16":
            return tl.dot(
                a.to(tl.bfloat16),
                b.to(tl.bfloat16),
                acc,
                out_dtype=tl.float32,
            )
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        if GEMM_MODE == "tf32":
            return tl.dot(
                _round_to_tf32(a),
                _round_to_tf32(b),
                acc,
                input_precision="tf32",
                out_dtype=tl.float32,
            )
        return tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)

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

    @triton.autotune(
        configs=_dual_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_dual_gemm},
        restore_value=["out_ptr"],
    )
    @triton.jit(do_not_specialize=["M"])
    def _gated_dual_gemm_kernel(
        x_ptr,
        wp_ptr,
        wg_ptr,
        mask_ptr,
        out_ptr,
        M,
        Nproj,
        K,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        """``x`` is already LayerNorm'd. Simple pid; K must divide ``TILE_K``."""
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        M64 = tl.cast(M, tl.int64)
        k64 = tl.cast(K, tl.int64)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < M64
        mask_n = offs_n < Nproj
        x_ptrs = x_ptr + offs_m[:, None] * k64 + offs_k[None, :]
        wp_ptrs = wp_ptr + offs_n[None, :] * k64 + offs_k[:, None]
        wg_ptrs = wg_ptr + offs_n[None, :] * k64 + offs_k[:, None]
        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for _k in range(0, K, TILE_K):
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            wp = tl.load(wp_ptrs, mask=mask_n[None, :], other=0.0)
            wg = tl.load(wg_ptrs, mask=mask_n[None, :], other=0.0)
            if GEMM_MODE == "bf16":
                val_acc = tl.dot(x, wp, val_acc, out_dtype=tl.float32)
                gate_acc = tl.dot(x, wg, gate_acc, out_dtype=tl.float32)
            elif GEMM_MODE == "tf32":
                val_acc = tl.dot(
                    _round_to_tf32(x.to(tl.float32)),
                    _round_to_tf32(wp.to(tl.float32)),
                    val_acc,
                    input_precision="tf32",
                    out_dtype=tl.float32,
                )
                gate_acc = tl.dot(
                    _round_to_tf32(x.to(tl.float32)),
                    _round_to_tf32(wg.to(tl.float32)),
                    gate_acc,
                    input_precision="tf32",
                    out_dtype=tl.float32,
                )
            else:
                val_acc = tl.dot(
                    x.to(tl.float32),
                    wp.to(tl.float32),
                    val_acc,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
                gate_acc = tl.dot(
                    x.to(tl.float32),
                    wg.to(tl.float32),
                    gate_acc,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )
            x_ptrs += TILE_K
            wp_ptrs += TILE_K
            wg_ptrs += TILE_K

        delta = tl.sigmoid(gate_acc) * val_acc
        if HAS_MASK:
            mtile = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            delta = delta * mtile[:, None]
        tl.store(
            out_ptr + offs_n[:, None] * M64 + offs_m[None, :],
            tl.trans(delta).to(out_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_m[None, :],
        )

    @triton.autotune(
        configs=_fwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_fwd_gemm},
        restore_value=["out_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M", "eps_out"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_dm_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
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
        gamma_out_ptr,
        beta_out_ptr,
        out_ptr,
        M,
        CZ,
        CH,
        eps_out,
        WITH_ADD: tl.constexpr,
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
            # [C,M] is contiguous along M; load [TILE_K, TILE_M] then transpose.
            x_k = tl.trans(
                tl.load(
                    x_dm_ptr + k_range[:, None] * m64 + offs_m[None, :],
                    mask=k_mask[:, None] & mask_m[None, :],
                    other=0.0,
                ).to(tl.float32)
            )
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
            x_k = tl.trans(
                tl.load(
                    x_dm_ptr + k_range[:, None] * m64 + offs_m[None, :],
                    mask=k_mask[:, None] & mask_m[None, :],
                    other=0.0,
                ).to(tl.float32)
            )
            x_k = tl.where(k_mask[None, :], x_k, 0.0)
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
            )
            val_acc = _mma(x_value, wp, val_acc, GEMM_MODE)

        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < cz64
            x_gate = tl.load(
                x_in_ptr + offs_m[:, None] * cz64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            )
            wgate = tl.load(
                wg_ptr + offs_n[None, :] * cz64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            gate_acc = _mma(x_gate, wgate, gate_acc, GEMM_MODE)

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

    @triton.autotune(
        configs=_ln_row_configs(_LN_ROW_TILES),
        key=[],
        restore_value=["y_ptr", "stats_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "x_ptr",
            "w_ptr",
            "b_ptr",
            "y_ptr",
            "stats_ptr",
        ],
    )
    def _ln_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        stats_ptr,
        M,
        D: tl.constexpr,
        eps,
        HAS_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Write LayerNorm ``y`` and ``[mean, rstd]`` in one pass."""
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
        gamma = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        y = x_c * rstd[:, None] * gamma[None, :]
        if HAS_BIAS:
            beta = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            y = y + tl.where(mask_d[None, :], beta[None, :], 0.0)
        tl.store(
            y_ptr + offs_m[:, None] * D + offs_d[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )
        tl.store(stats_ptr + offs_m, mean, mask=mask_m)
        tl.store(stats_ptr + m64 + offs_m, rstd, mask=mask_m)

    @triton.autotune(
        configs=_ln_row_configs(_LN_ROW_TILES),
        key=[],
        restore_value=["out_ptr"],
    )
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

    @triton.autotune(
        configs=_fwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_fwd_gemm},
        restore_value=["out_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_out_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
            "out_ptr",
        ],
    )
    def _gated_out_gemm_residual_kernel(
        x_in_ptr,
        x_out_ptr,
        wg_ptr,
        wp_ptr,
        residual_ptr,
        out_ptr,
        M,
        CZ,
        CH,
        WITH_ADD: tl.constexpr,
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
            )
            wp = tl.load(
                wp_ptr + offs_n[None, :] * ch64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            val_acc = _mma(xo, wp, val_acc, GEMM_MODE)

        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < cz64
            xi = tl.load(
                x_in_ptr + offs_m[:, None] * cz64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            )
            wgate = tl.load(
                wg_ptr + offs_n[None, :] * cz64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            gate_acc = _mma(xi, wgate, gate_acc, GEMM_MODE)

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

    @triton.autotune(
        configs=_ln_bwd_configs(),
        key=[],
        restore_value=["GX_ptr", "PGamma_ptr", "PBeta_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "DY_ptr",
            "GX_ptr",
            "PGamma_ptr",
            "PBeta_ptr",
        ],
    )
    def _ln_bwd_kernel(
        X_ptr,
        DY_ptr,
        Mean_ptr,
        Rstd_ptr,
        Gamma_ptr,
        GX_ptr,
        PGamma_ptr,
        PBeta_ptr,
        stride_x,
        stride_dy,
        stride_gx,
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
        x = tl.load(
            X_ptr + offs_m64[:, None] * stride_x + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY_ptr + offs_m64[:, None] * stride_dy + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_norm = tl.where(k_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0)
        dx_hat = dy * gamma[None, :]
        grad_mean = tl.sum(dx_hat, axis=1) / K
        grad_proj = tl.sum(dx_hat * x_norm, axis=1) / K
        gx = rstd[:, None] * (dx_hat - grad_mean[:, None] - x_norm * grad_proj[:, None])
        tl.store(
            GX_ptr + offs_m64[:, None] * stride_gx + offs_k[None, :],
            gx.to(GX_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            PGamma_ptr + pid * K + offs_k,
            tl.sum(dy * x_norm, axis=0),
            mask=k_mask,
        )
        tl.store(PBeta_ptr + pid * K + offs_k, tl.sum(dy, axis=0), mask=k_mask)

    @triton.autotune(
        configs=_bwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_bwd_gemm},
        restore_value=["PWp_ptr", "PWg_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "GO_ptr",
            "Wp_ptr",
            "Wg_ptr",
            "Mask_ptr",
            "PWp_ptr",
            "PWg_ptr",
        ],
    )
    def _gated_dw_kernel(
        X_ptr,
        GO_ptr,
        Wp_ptr,
        Wg_ptr,
        Mask_ptr,
        PWp_ptr,
        PWg_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        HAS_MASK: tl.constexpr,
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
        wp = tl.load(
            Wp_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        wg = tl.load(
            Wg_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_p = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc_g = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end
            x = tl.load(
                X_ptr + offs_m64[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            go = tl.load(
                GO_ptr + offs_n[None, :] * M + offs_m64[:, None],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            p = _dot_f32(x, tl.trans(wp), GEMM_MODE)
            s = tl.sigmoid(_dot_f32(x, tl.trans(wg), GEMM_MODE))
            scale = go
            if HAS_MASK:
                mtile = tl.load(Mask_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
                scale = scale * mtile[:, None]
            d_p = scale * s
            d_s = scale * p * s * (1.0 - s)
            acc_p += _dot_f32(tl.trans(d_p), x, GEMM_MODE)
            acc_g += _dot_f32(tl.trans(d_s), x, GEMM_MODE)
        tl.store(
            PWp_ptr + split * N * K + offs_n[:, None] * K + offs_k[None, :],
            acc_p,
            mask=n_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            PWg_ptr + split * N * K + offs_n[:, None] * K + offs_k[None, :],
            acc_g,
            mask=n_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(
        configs=_bwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_bwd_gemm},
        restore_value=["DX_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "GO_ptr",
            "Wp_ptr",
            "Wg_ptr",
            "Mask_ptr",
            "DX_ptr",
        ],
    )
    def _gated_dx_kernel(
        X_ptr,
        GO_ptr,
        Wp_ptr,
        Wg_ptr,
        Mask_ptr,
        DX_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        HAS_MASK: tl.constexpr,
        ACCUM: tl.constexpr,
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
        x = tl.load(
            X_ptr + offs_m64[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if ACCUM:
            dx = tl.load(
                DX_ptr + offs_m64[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            dx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        mtile = (
            tl.load(Mask_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
            if HAS_MASK
            else 1.0
        )
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            go = tl.load(
                GO_ptr + offs_n[None, :] * M + offs_m64[:, None],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wp = tl.load(
                Wp_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wg = tl.load(
                Wg_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            p = _dot_f32(x, tl.trans(wp), GEMM_MODE)
            s = tl.sigmoid(_dot_f32(x, tl.trans(wg), GEMM_MODE))
            scale = go * mtile[:, None] if HAS_MASK else go
            d_p = scale * s
            d_s = scale * p * s * (1.0 - s)
            dx += _dot_f32(d_p, wp, GEMM_MODE) + _dot_f32(d_s, wg, GEMM_MODE)
        tl.store(
            DX_ptr + offs_m64[:, None] * K + offs_k[None, :],
            dx.to(DX_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(
        configs=_bwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_bwd_gemm},
        restore_value=["DX_ptr", "DZ_ptr", "PGamma_ptr", "PBeta_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "ZH_ptr",
            "X_ptr",
            "GO_ptr",
            "Wz_ptr",
            "Wg_ptr",
            "DX_ptr",
            "DZ_ptr",
        ],
    )
    def _gated_out_bwd_dx_kernel(
        ZH_ptr,
        X_ptr,
        GO_ptr,
        Wz_ptr,
        Wg_ptr,
        Gamma_ptr,
        Beta_ptr,
        DX_ptr,
        DZ_ptr,
        PGamma_ptr,
        PBeta_ptr,
        M,
        eps,
        CZ: tl.constexpr,
        CH: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_CZ: tl.constexpr,
        BLOCK_CH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        m64 = tl.cast(M, tl.int64)
        offs_cz = tl.arange(0, BLOCK_CZ)
        offs_ch = tl.arange(0, BLOCK_CH)
        m_mask = offs_m < M
        cz_mask = offs_cz < CZ
        ch_mask = offs_ch < CH
        zh = tl.load(
            ZH_ptr + offs_m64[:, None] * CZ + offs_cz[None, :],
            mask=m_mask[:, None] & cz_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # [C,M] is contiguous along M; load [BLOCK_CH, BLOCK_M] then transpose.
        x = tl.trans(
            tl.load(
                X_ptr + offs_ch[:, None] * m64 + offs_m64[None, :],
                mask=ch_mask[:, None] & m_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        )
        x = tl.where(ch_mask[None, :], x, 0.0)
        mean = tl.sum(x, axis=1) / CH
        x_norm = tl.where(ch_mask[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x_norm * x_norm, axis=1) / CH + eps)
        x_norm = x_norm * rstd[:, None]
        gamma = tl.load(Gamma_ptr + offs_ch, mask=ch_mask, other=0.0).to(tl.float32)
        x_hat = x_norm * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_ch, mask=ch_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(ch_mask[None, :], beta[None, :], 0.0)
        dx_hat = tl.zeros((BLOCK_M, BLOCK_CH), dtype=tl.float32)
        dz = tl.zeros((BLOCK_M, BLOCK_CZ), dtype=tl.float32)
        for n0 in range(0, CZ, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < CZ
            go = tl.load(
                GO_ptr + offs_m64[:, None] * CZ + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wg = tl.load(
                Wg_ptr + offs_n[:, None] * CZ + offs_cz[None, :],
                mask=n_mask[:, None] & cz_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wz = tl.load(
                Wz_ptr + offs_n[:, None] * CH + offs_ch[None, :],
                mask=n_mask[:, None] & ch_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            g = tl.sigmoid(_dot_f32(zh, tl.trans(wg), GEMM_MODE))
            val = _dot_f32(x_hat, tl.trans(wz), GEMM_MODE)
            d_val = go * g
            d_pre = go * val * g * (1.0 - g)
            dx_hat += _dot_f32(d_val, wz, GEMM_MODE)
            dz += _dot_f32(d_pre, wg, GEMM_MODE)
        dxh = dx_hat * gamma[None, :]
        grad_mean = tl.sum(dxh, axis=1) / CH
        grad_proj = tl.sum(dxh * x_norm, axis=1) / CH
        dx = rstd[:, None] * (
            dxh - grad_mean[:, None] - x_norm * grad_proj[:, None]
        )
        tl.store(
            DX_ptr + offs_ch[:, None] * m64 + offs_m64[None, :],
            tl.trans(dx).to(DX_ptr.dtype.element_ty),
            mask=ch_mask[:, None] & m_mask[None, :],
        )
        tl.store(
            DZ_ptr + offs_m64[:, None] * CZ + offs_cz[None, :],
            dz.to(DZ_ptr.dtype.element_ty),
            mask=m_mask[:, None] & cz_mask[None, :],
        )
        tl.store(
            PGamma_ptr + pid * CH + offs_ch,
            tl.sum(dx_hat * x_norm, axis=0),
            mask=ch_mask,
        )
        tl.store(PBeta_ptr + pid * CH + offs_ch, tl.sum(dx_hat, axis=0), mask=ch_mask)

    @triton.autotune(
        configs=_bwd_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_bwd_gemm},
        restore_value=["PWz_ptr", "PWg_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "ZH_ptr",
            "X_ptr",
            "GO_ptr",
            "Wz_ptr",
            "Wg_ptr",
            "PWz_ptr",
            "PWg_ptr",
        ],
    )
    def _gated_out_bwd_dw_kernel(
        ZH_ptr,
        X_ptr,
        GO_ptr,
        Wz_ptr,
        Wg_ptr,
        Gamma_ptr,
        Beta_ptr,
        PWz_ptr,
        PWg_ptr,
        M,
        eps,
        CZ: tl.constexpr,
        CH: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        SPLIT_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_CZ: tl.constexpr,
        BLOCK_CH: tl.constexpr,
    ):
        split = tl.program_id(0)
        rows_per_split = (M + SPLIT_M - 1) // SPLIT_M
        split_start = split * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_cz = tl.arange(0, BLOCK_CZ)
        offs_ch = tl.arange(0, BLOCK_CH)
        n_mask = offs_n < CZ
        cz_mask = offs_cz < CZ
        ch_mask = offs_ch < CH
        acc_wz = tl.zeros((BLOCK_N, BLOCK_CH), dtype=tl.float32)
        acc_wg = tl.zeros((BLOCK_N, BLOCK_CZ), dtype=tl.float32)
        gamma = tl.load(Gamma_ptr + offs_ch, mask=ch_mask, other=0.0).to(tl.float32)
        beta = (
            tl.load(Beta_ptr + offs_ch, mask=ch_mask, other=0.0).to(tl.float32)
            if HAS_LN_BIAS
            else 0.0
        )
        wg_n = tl.load(
            Wg_ptr + offs_n[:, None] * CZ + offs_cz[None, :],
            mask=n_mask[:, None] & cz_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        wz_n = tl.load(
            Wz_ptr + offs_n[:, None] * CH + offs_ch[None, :],
            mask=n_mask[:, None] & ch_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        m64 = tl.cast(M, tl.int64)
        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end
            zh = tl.load(
                ZH_ptr + offs_m64[:, None] * CZ + offs_cz[None, :],
                mask=m_mask[:, None] & cz_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.trans(
                tl.load(
                    X_ptr + offs_ch[:, None] * m64 + offs_m64[None, :],
                    mask=ch_mask[:, None] & m_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            )
            x = tl.where(ch_mask[None, :], x, 0.0)
            mean = tl.sum(x, axis=1) / CH
            x_c = tl.where(ch_mask[None, :], x - mean[:, None], 0.0)
            rstd = 1.0 / tl.sqrt(tl.sum(x_c * x_c, axis=1) / CH + eps)
            x_hat = x_c * rstd[:, None] * gamma[None, :]
            if HAS_LN_BIAS:
                x_hat = x_hat + tl.where(ch_mask[None, :], beta[None, :], 0.0)
            go = tl.load(
                GO_ptr + offs_m64[:, None] * CZ + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            g = tl.sigmoid(_dot_f32(zh, tl.trans(wg_n), GEMM_MODE))
            val = _dot_f32(x_hat, tl.trans(wz_n), GEMM_MODE)
            d_val = go * g
            d_pre = go * val * g * (1.0 - g)
            acc_wz += _dot_f32(tl.trans(d_val), x_hat, GEMM_MODE)
            acc_wg += _dot_f32(tl.trans(d_pre), zh, GEMM_MODE)
        tl.store(
            PWz_ptr + split * CZ * CH + offs_n[:, None] * CH + offs_ch[None, :],
            acc_wz,
            mask=n_mask[:, None] & ch_mask[None, :],
        )
        tl.store(
            PWg_ptr + split * CZ * CZ + offs_n[:, None] * CZ + offs_cz[None, :],
            acc_wg,
            mask=n_mask[:, None] & cz_mask[None, :],
        )


    @triton.autotune(
        configs=_ln_bwd_configs(),
        key=[],
        restore_value=["Y_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["X_ptr", "Y_ptr"],
    )
    def _ln_apply_kernel(
        X_ptr,
        Mean_ptr,
        Rstd_ptr,
        Gamma_ptr,
        Beta_ptr,
        Y_ptr,
        stride_x,
        stride_y,
        M,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        x = tl.load(
            X_ptr + offs_m64[:, None] * stride_x + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * rstd[:, None] * gamma[None, :]
        if HAS_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            y = y + tl.where(k_mask[None, :], beta[None, :], 0.0)
        tl.store(
            Y_ptr + offs_m64[:, None] * stride_y + offs_k[None, :],
            y.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(
        configs=_tri_gemm_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_tri_gemm},
        restore_value=["C_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M", "N", "K", "C", "BATCH"],
        do_not_specialize_on_alignment=["A_ptr", "B_ptr", "C_ptr"],
    )
    def _batched_gemm_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        C,
        BATCH,
        GEMM_MODE: tl.constexpr,
        TRANS_A: tl.constexpr,
        TRANS_B: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # Contiguous [C,B,N,N] panels. Inner stride is constexpr 1; row pitch is N.
        g = tl.program_id(2)
        pid_m, pid_n = _pid_mn(M, N, TILE_M, TILE_N, GROUP_M)
        c = (g % C).to(tl.int64)
        b = (g // C).to(tl.int64)
        offs_m = (pid_m * TILE_M + tl.arange(0, TILE_M)).to(tl.int64)
        offs_n = (pid_n * TILE_N + tl.arange(0, TILE_N)).to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N
        nn = (N * N).to(tl.int64)
        a_ptr = A_ptr + (c * BATCH + b) * nn
        b_ptr = B_ptr + (c * BATCH + b) * nn
        acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k0 in range(0, K, TILE_K):
            offs_k = (k0 + tl.arange(0, TILE_K)).to(tl.int64)
            k_mask = offs_k < K
            if TRANS_A:
                a = tl.load(
                    a_ptr + offs_k[None, :] * N + offs_m[:, None],
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                a = tl.load(
                    a_ptr + offs_m[:, None] * N + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            if TRANS_B:
                bb = tl.load(
                    b_ptr + offs_n[None, :] * N + offs_k[:, None],
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                bb = tl.load(
                    b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            acc += _dot_f32(a, bb, GEMM_MODE)
        tl.store(
            C_ptr + (c * BATCH + b) * nn + offs_m[:, None] * N + offs_n[None, :],
            acc.to(C_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_trimul")

    _gated_dual_gemm_kernel = _unavailable
    _gated_out_from_dm_kernel = _unavailable
    _ln_fwd_kernel = _unavailable
    _ln_transpose_kernel = _unavailable
    _gated_out_gemm_residual_kernel = _unavailable
    _ln_bwd_kernel = _unavailable
    _gated_dw_kernel = _unavailable
    _gated_dx_kernel = _unavailable
    _gated_out_bwd_dx_kernel = _unavailable
    _gated_out_bwd_dw_kernel = _unavailable
    _ln_apply_kernel = _unavailable
    _batched_gemm_kernel = _unavailable


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def ln_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm ``y`` plus float32 ``[mean, rstd]``. ``y`` matches ``x.dtype``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    x = x.contiguous()
    m, d = x.shape
    y = torch.empty_like(x)
    stats = torch.empty((2, m), device=x.device, dtype=torch.float32)
    dummy = x
    _ln_fwd_kernel[(lambda meta: (triton.cdiv(m, meta["TILE_M"]),))](
        x,
        weight.contiguous(),
        bias.contiguous() if bias is not None else dummy,
        y,
        stats,
        m,
        D=d,
        eps=float(eps),
        HAS_BIAS=bias is not None,
        BLOCK_D=_next_power_of_two(d),
    )
    return y, stats


_dual_autotune_ready: set[tuple] = set()


def _ensure_dual_autotune(
    sample: torch.Tensor,
    wp: torch.Tensor,
    nproj: int,
    k: int,
    mode: str,
) -> None:
    """Time dual-gemm on a large dummy grid so a small first ``M`` cannot lock tiles."""
    key = (mode, sample.device.type, sample.device.index, nproj, k, sample.dtype)
    if key in _dual_autotune_ready:
        return
    m = _DUAL_WARM_M
    dummy_x = torch.empty((m, k), device=sample.device, dtype=sample.dtype)
    dummy_out = torch.empty((nproj, m), device=sample.device, dtype=sample.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(nproj, meta["TILE_N"]))

    _gated_dual_gemm_kernel[grid](
        dummy_x,
        wp,
        wp,
        dummy_x,
        dummy_out,
        m,
        nproj,
        k,
        HAS_MASK=False,
        GEMM_MODE=mode,
    )
    _dual_autotune_ready.add(key)


def gated_dual_gemm(
    x_hat: torch.Tensor,
    wp: torch.Tensor,
    wg: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """``sigmoid(x_hat@wg^T) * (x_hat@wp^T) * mask`` as ``[Nproj, M]``.

    ``x_hat`` is already LayerNorm'd. bf16 activations downcast masters once.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    m, k = x_hat.shape
    nproj = wp.shape[0]
    x_hat = x_hat.contiguous()
    wp, wg = _downcast_masters(x_hat.dtype, wp.contiguous(), wg.contiguous())
    out = torch.empty((nproj, m), device=x_hat.device, dtype=x_hat.dtype)
    dummy = x_hat
    mode = _gemm_mode(x_hat)
    key = (mode, x_hat.device.type, x_hat.device.index, nproj, k, x_hat.dtype)
    if key not in _dual_autotune_ready and m < _DUAL_WARM_M:
        _ensure_dual_autotune(x_hat, wp, nproj, k, mode)

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(nproj, meta["TILE_N"]))

    _gated_dual_gemm_kernel[grid](
        x_hat,
        wp,
        wg,
        mask.contiguous().view(-1) if mask is not None else dummy,
        out,
        m,
        nproj,
        k,
        HAS_MASK=mask is not None,
        GEMM_MODE=mode,
    )
    _dual_autotune_ready.add(key)
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
    _ln_transpose_kernel[(lambda meta: (triton.cdiv(m, meta["TILE_M"]),))](
        x_dm.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else dummy,
        out,
        m,
        D=d,
        EPS=float(eps),
        HAS_BIAS=bias is not None,
    )
    return out


def gated_out_gemm_residual(
    x_hat: torch.Tensor,
    x_out: torch.Tensor,
    wg: torch.Tensor,
    wp: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    m, cz = x_hat.shape
    ch = x_out.shape[1]
    if out is None:
        out = torch.empty((m, cz), device=x_hat.device, dtype=x_hat.dtype)
    dummy = x_hat
    wg, wp = _downcast_masters(x_hat.dtype, wg.contiguous(), wp.contiguous())

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(cz, meta["TILE_N"]))

    _gated_out_gemm_residual_kernel[grid](
        x_hat.contiguous(),
        x_out.contiguous(),
        wg,
        wp,
        residual.contiguous() if residual is not None else dummy,
        out,
        m,
        cz,
        ch,
        WITH_ADD=residual is not None,
        GEMM_MODE=_gemm_mode(x_hat),
    )
    return out


def gated_out_from_dm(
    x_hat: torch.Tensor,
    x_dm: torch.Tensor,
    wg: torch.Tensor,
    wp: torch.Tensor,
    residual: torch.Tensor | None,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    ln_out_eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse LN_out on D-major contraction output with gated output projection.

    ``x_hat`` is already LN_in; the gate GEMM is a clean MMA.
    """
    m, cz = x_hat.shape
    ch, x_m = x_dm.shape
    if x_m != m:
        raise ValueError(f"x_dm has M={x_m}, expected {m}")
    if ch < 128:
        x_out = ln_transpose(x_dm, ln_out_w, ln_out_b, eps=ln_out_eps)
        return gated_out_gemm_residual(
            x_hat, x_out, wg, wp, residual, out=out
        )
    if cz > 128 or ch > 128:
        raise ValueError("fused D-major output supports CZ/CH <= 128")
    if out is None:
        out = torch.empty((m, cz), device=x_hat.device, dtype=x_hat.dtype)
    dummy = x_hat
    wg, wp = _downcast_masters(x_hat.dtype, wg.contiguous(), wp.contiguous())

    def grid(meta):
        return (triton.cdiv(m, meta["TILE_M"]), triton.cdiv(cz, meta["TILE_N"]))

    _gated_out_from_dm_kernel[grid](
        x_hat.contiguous(),
        x_dm.contiguous(),
        wg,
        wp,
        residual.contiguous() if residual is not None else dummy,
        ln_out_w.contiguous(),
        ln_out_b.contiguous() if ln_out_b is not None else dummy,
        out,
        m,
        cz,
        ch,
        float(ln_out_eps),
        WITH_ADD=residual is not None,
        HAS_LN_OUT_BIAS=ln_out_b is not None,
        GEMM_MODE=_gemm_mode(x_hat),
    )
    return out


def _contract(a: torch.Tensor, b: torch.Tensor, outgoing: bool) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if outgoing:
            return torch.einsum("cbik,cbjk->cbij", a, b)
        return torch.einsum("cbki,cbkj->cbij", a, b)


def _trimul_whole(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask_flat: torch.Tensor,
    stats: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    outgoing: bool,
    residual: torch.Tensor | None,
    ln_out_eps: float,
    out: torch.Tensor | None,
    return_acts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n, _, c_z = z.shape
    c_hidden = wa_p.shape[0]
    wp_ab = torch.cat([wa_p, wb_p], dim=0)
    wg_ab = torch.cat([wa_g, wb_g], dim=0)
    ab = gated_dual_gemm(z_hat, wp_ab, wg_ab, mask_flat)
    a = ab[:c_hidden].view(c_hidden, batch, n, n)
    b = ab[c_hidden:].view(c_hidden, batch, n, n)
    x = _contract(a, b, outgoing)
    if not return_acts:
        del a, b, ab
    x_dm = x.reshape(c_hidden, batch * n * n)
    out_2d = gated_out_from_dm(
        z_hat,
        x_dm,
        wg,
        wz,
        residual.reshape(-1, c_z) if residual is not None else None,
        ln_out_w,
        ln_out_b,
        ln_out_eps=ln_out_eps,
        out=out.reshape(-1, c_z) if out is not None else None,
    )
    y = out_2d.view(batch, n, n, c_z)
    if return_acts:
        return y, stats, ab, x
    del x, x_dm
    return y


def _trimul_chunked_outgoing(
    z_hat: torch.Tensor,
    mask_flat: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    residual: torch.Tensor | None,
    ln_out_eps: float,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    m = n * n
    b_full = gated_dual_gemm(z_hat, wb_p, wb_g, mask_flat)
    b_4d = b_full.view(c_hidden, 1, n, n)
    out_2d = out.reshape(m, c_z) if out is not None else torch.empty(
        (m, c_z), device=z_hat.device, dtype=z_hat.dtype
    )
    resid_2d = residual.reshape(m, c_z) if residual is not None else None
    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        a_c = gated_dual_gemm(z_hat[m0:m1], wa_p, wa_g, mask_flat[m0:m1])
        x_c = _contract(a_c.view(c_hidden, 1, rows, n), b_4d, outgoing=True)
        del a_c
        gated_out_from_dm(
            z_hat[m0:m1],
            x_c.reshape(c_hidden, rows * n),
            wg,
            wz,
            resid_2d[m0:m1] if resid_2d is not None else None,
            ln_out_w,
            ln_out_b,
            ln_out_eps=ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_c
    return out_2d.view(1, n, n, c_z)


def _trimul_chunked_incoming(
    z_hat: torch.Tensor,
    mask_flat: torch.Tensor,
    wa_p: torch.Tensor,
    wa_g: torch.Tensor,
    wb_p: torch.Tensor,
    wb_g: torch.Tensor,
    wz: torch.Tensor,
    wg: torch.Tensor,
    ln_out_w: torch.Tensor,
    ln_out_b: torch.Tensor | None,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    residual: torch.Tensor | None,
    ln_out_eps: float,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    m = n * n
    group_size = min(64, c_hidden)
    grouped_k_cap = min(n, chunk_cap * c_hidden // group_size)
    x_accum = torch.empty((c_hidden, n, n), device=z_hat.device, dtype=z_hat.dtype)
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
            ab_k = gated_dual_gemm(z_hat[m0:m1], wp_ab, wg_ab, mask_flat[m0:m1])
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
        (m, c_z), device=z_hat.device, dtype=z_hat.dtype
    )
    resid_2d = residual.reshape(m, c_z) if residual is not None else None
    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        x_dm = x_accum[:, i_start:i_end, :].reshape(c_hidden, rows * n).contiguous()
        gated_out_from_dm(
            z_hat[m0:m1],
            x_dm,
            wg,
            wz,
            resid_2d[m0:m1] if resid_2d is not None else None,
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
    return_acts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n, _, c_z = z.shape
    c_hidden = wa_p.shape[0]
    z_c = z.contiguous()
    z_2d = z_c.reshape(-1, c_z)
    mask_flat = (
        mask.reshape(-1).contiguous()
        if mask is not None
        else z_c.new_ones(z_2d.shape[0])
    )
    z_hat, stats = ln_fwd(z_2d, ln_in_w, ln_in_b, ln_in_eps)
    inplace = (
        residual is not None
        and residual.data_ptr() == z_c.data_ptr()
        and not return_acts
        and not torch.is_grad_enabled()
        and not residual.requires_grad
    )
    out = z_c if inplace else None
    resid = z_c if residual is not None else None
    kwargs = dict(
        outgoing=outgoing,
        residual=resid,
        ln_out_eps=ln_out_eps,
        out=out,
    )
    if chunk_cap is not None and chunk_cap < n and batch == 1:
        chunk_kwargs = dict(
            n=n,
            c_z=c_z,
            c_hidden=c_hidden,
            residual=resid,
            ln_out_eps=ln_out_eps,
            out=out,
            chunk_cap=chunk_cap,
        )
        if return_acts:
            raise RuntimeError("fused_trimul training acts are not used with chunk_cap")
        if outgoing:
            return _trimul_chunked_outgoing(
                z_hat,
                mask_flat,
                wa_p,
                wa_g,
                wb_p,
                wb_g,
                wz,
                wg,
                ln_out_w,
                ln_out_b,
                **chunk_kwargs,
            )
        return _trimul_chunked_incoming(
            z_hat,
            mask_flat,
            wa_p,
            wa_g,
            wb_p,
            wb_g,
            wz,
            wg,
            ln_out_w,
            ln_out_b,
            **chunk_kwargs,
        )
    return _trimul_whole(
        z_c,
        z_hat,
        mask_flat,
        stats,
        wa_p,
        wa_g,
        wb_p,
        wb_g,
        wz,
        wg,
        ln_out_w,
        ln_out_b,
        return_acts=return_acts,
        **kwargs,
    )


def _ln_bwd(x, dy, mean, rstd, gamma, has_beta: bool, *, out=None):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    x_c, dy_c = x.contiguous(), dy.contiguous()
    m, k = x_c.shape
    # The kernel loads ``dy`` into registers before writing ``gx``, so
    # in-place ``out is dy`` is safe and avoids a pair-sized extra buffer.
    if out is dy:
        gx = dy_c
    elif out is not None:
        gx = out
    else:
        gx = torch.empty_like(x_c)
    # Zero so unused rows stay valid if autotune picks BLOCK_M > min.
    n_partial = triton.cdiv(m, _BWD_MIN_BLOCK_M)
    partial_g = torch.zeros((n_partial, k), dtype=torch.float32, device=x.device)
    partial_b = torch.zeros_like(partial_g)
    _ln_bwd_kernel[(lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),))](
        x_c,
        dy_c,
        mean,
        rstd,
        gamma,
        gx,
        partial_g,
        partial_b,
        x_c.stride(0),
        dy_c.stride(0),
        gx.stride(0),
        m,
        K=k,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    d_gamma = partial_g.sum(0)
    d_beta = partial_b.sum(0) if has_beta else None
    return gx, d_gamma, d_beta


def _gated_dw(x, go, wp, wg, mask):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    x_c, go_c = x.contiguous(), go.contiguous()
    n, k = wp.shape
    dummy = x_c
    p_wp = torch.empty((_BWD_SPLIT_M, n, k), dtype=torch.float32, device=x.device)
    p_wg = torch.empty_like(p_wp)

    def grid(meta):
        return (_BWD_SPLIT_M, triton.cdiv(n, meta["BLOCK_N"]))

    _gated_dw_kernel[grid](
        x_c,
        go_c,
        wp.contiguous(),
        wg.contiguous(),
        mask.contiguous() if mask is not None else dummy,
        p_wp,
        p_wg,
        x_c.shape[0],
        K=k,
        N=n,
        HAS_MASK=mask is not None,
        GEMM_MODE=_gemm_mode(x_c),
        SPLIT_M=_BWD_SPLIT_M,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return p_wp.sum(0), p_wg.sum(0)


def _gated_dx(x, go, wp, wg, mask, dx=None):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    x_c, go_c = x.contiguous(), go.contiguous()
    n, k = wp.shape
    dummy = x_c
    accum = dx is not None
    if dx is None:
        dx = torch.empty((x_c.shape[0], k), dtype=x_c.dtype, device=x.device)
    elif not dx.is_contiguous():
        dx = dx.contiguous()
    _gated_dx_kernel[(lambda meta: (triton.cdiv(x_c.shape[0], meta["BLOCK_M"]),))](
        x_c,
        go_c,
        wp.contiguous(),
        wg.contiguous(),
        mask.contiguous() if mask is not None else dummy,
        dx,
        x_c.shape[0],
        K=k,
        N=n,
        HAS_MASK=mask is not None,
        ACCUM=accum,
        GEMM_MODE=_gemm_mode(x_c),
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return dx


def _gated_out_bwd(z_hat, x_ch, go, wz, wg, ln_out_w, ln_out_b, eps: float = 1e-5):
    """Output-gate backward. ``x_ch`` is ``[C, M]``; no stored g / val / x_hat."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    z_c, x_c, go_c = z_hat.contiguous(), x_ch.contiguous(), go.contiguous()
    m, cz = z_c.shape
    ch = x_c.shape[0]
    if x_c.shape[1] != m:
        raise ValueError(f"x_ch has M={x_c.shape[1]}, expected {m}")
    dummy = z_c
    d_x = torch.empty((ch, m), device=z_c.device, dtype=x_c.dtype)
    d_z_hat = torch.empty_like(z_c)
    p_wz = torch.empty((_BWD_SPLIT_M, cz, ch), dtype=torch.float32, device=z_c.device)
    p_wg = torch.empty((_BWD_SPLIT_M, cz, cz), dtype=torch.float32, device=z_c.device)
    n_partial = triton.cdiv(m, _BWD_MIN_BLOCK_M)
    p_gamma = torch.zeros((n_partial, ch), dtype=torch.float32, device=z_c.device)
    p_beta = torch.zeros_like(p_gamma)
    shared = dict(
        CZ=cz,
        CH=ch,
        HAS_LN_BIAS=ln_out_b is not None,
        GEMM_MODE=_gemm_mode(z_c),
        BLOCK_CZ=max(_next_power_of_two(cz), 16),
        BLOCK_CH=max(_next_power_of_two(ch), 16),
    )

    def dw_grid(meta):
        return (_BWD_SPLIT_M, triton.cdiv(cz, meta["BLOCK_N"]))

    _gated_out_bwd_dw_kernel[dw_grid](
        z_c,
        x_c,
        go_c,
        wz.contiguous(),
        wg.contiguous(),
        ln_out_w.contiguous(),
        ln_out_b.contiguous() if ln_out_b is not None else dummy,
        p_wz,
        p_wg,
        m,
        float(eps),
        SPLIT_M=_BWD_SPLIT_M,
        **shared,
    )
    _gated_out_bwd_dx_kernel[(lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),))](
        z_c,
        x_c,
        go_c,
        wz.contiguous(),
        wg.contiguous(),
        ln_out_w.contiguous(),
        ln_out_b.contiguous() if ln_out_b is not None else dummy,
        d_x,
        d_z_hat,
        p_gamma,
        p_beta,
        m,
        float(eps),
        **shared,
    )
    d_wz = p_wz.sum(0)
    d_wg = p_wg.sum(0)
    d_ln_w = p_gamma.sum(0)
    d_ln_b = p_beta.sum(0) if ln_out_b is not None else None
    del p_wz, p_wg, p_gamma, p_beta
    return d_x, d_z_hat, d_wz, d_wg, d_ln_w, d_ln_b


def _flat_pair(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1, t.shape[-1])
    return flat if flat.is_contiguous() else t.contiguous().view(-1, t.shape[-1])


def _ln_apply(x, mean, rstd, gamma, beta, out=None):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_trimul")
    x_c = x.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
    dummy = x_c
    _ln_apply_kernel[(lambda meta: (triton.cdiv(x_c.shape[0], meta["BLOCK_M"]),))](
        x_c,
        mean,
        rstd,
        gamma.contiguous(),
        beta.contiguous() if beta is not None else dummy,
        out,
        x_c.stride(0),
        out.stride(0),
        x_c.shape[0],
        K=x_c.shape[1],
        HAS_BIAS=beta is not None,
        BLOCK_K=max(_next_power_of_two(x_c.shape[1]), 16),
    )
    return out


def _tri_gemm(
    a,
    b,
    out,
    *,
    n: int,
    hidden: int,
    batch: int,
    trans_a: bool,
    trans_b: bool,
    gemm_mode: str,
) -> None:
    def grid(meta):
        return (
            triton.cdiv(n, meta["TILE_M"]),
            triton.cdiv(n, meta["TILE_N"]),
            hidden * batch,
        )

    _batched_gemm_kernel[grid](
        a,
        b,
        out,
        n,
        n,
        n,
        hidden,
        batch,
        GEMM_MODE=gemm_mode,
        TRANS_A=trans_a,
        TRANS_B=trans_b,
    )


def _trimul_da_ch(d_x_ch: torch.Tensor, b: torch.Tensor, outgoing: bool) -> torch.Tensor:
    """``dA`` as ``[C, M]`` so gated ``dW``/``dX`` skip a pair transpose."""
    hidden, batch, n, _ = b.shape
    d_a_ch = torch.empty_like(d_x_ch)
    mode = _gemm_mode(d_x_ch)
    if outgoing:
        _tri_gemm(
            d_x_ch,
            b,
            d_a_ch,
            n=n,
            hidden=hidden,
            batch=batch,
            trans_a=False,
            trans_b=False,
            gemm_mode=mode,
        )
    else:
        _tri_gemm(
            b,
            d_x_ch,
            d_a_ch,
            n=n,
            hidden=hidden,
            batch=batch,
            trans_a=False,
            trans_b=True,
            gemm_mode=mode,
        )
    return d_a_ch.view(hidden, -1)


def _trimul_db_ch(d_x_ch: torch.Tensor, a: torch.Tensor, outgoing: bool) -> torch.Tensor:
    hidden, batch, n, _ = a.shape
    d_b_ch = torch.empty_like(d_x_ch)
    mode = _gemm_mode(d_x_ch)
    if outgoing:
        _tri_gemm(
            d_x_ch,
            a,
            d_b_ch,
            n=n,
            hidden=hidden,
            batch=batch,
            trans_a=True,
            trans_b=False,
            gemm_mode=mode,
        )
    else:
        _tri_gemm(
            a,
            d_x_ch,
            d_b_ch,
            n=n,
            hidden=hidden,
            batch=batch,
            trans_a=False,
            trans_b=False,
            gemm_mode=mode,
        )
    return d_b_ch.view(hidden, -1)


def _fused_trimul_backward(
    z,
    mask,
    go,
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
    mean_in,
    rstd_in,
    outgoing: bool,
    has_residual: bool,
    ln_out_eps: float,
    ab,
    x_ch,
):
    batch, n, _, c_z = z.shape
    c_h = wa_p.shape[0]
    z_2d = _flat_pair(z)
    go_2d = _flat_pair(go)
    mask_flat = mask.reshape(-1).contiguous() if mask is not None else None
    z_hat = _ln_apply(z_2d, mean_in, rstd_in, ln_in_w, ln_in_b)
    a = ab[:c_h].view(c_h, batch, n, n)
    b = ab[c_h:].view(c_h, batch, n, n)
    x_cm = x_ch.reshape(c_h, -1)
    d_x_cm, d_z_hat, d_wz, d_wg, d_ln_out_w, d_ln_out_b = _gated_out_bwd(
        z_hat, x_cm, go_2d, wz, wg, ln_out_w, ln_out_b, ln_out_eps
    )
    del x_ch, x_cm

    d_x_ch = d_x_cm.view(c_h, batch, n, n)
    del d_x_cm
    d_a_ch = _trimul_da_ch(d_x_ch, b, outgoing)
    d_wa_p, d_wa_g = _gated_dw(z_hat, d_a_ch, wa_p, wa_g, mask_flat)
    d_z_hat = _gated_dx(z_hat, d_a_ch, wa_p, wa_g, mask_flat, dx=d_z_hat)
    del d_a_ch

    d_b_ch = _trimul_db_ch(d_x_ch, a, outgoing)
    del d_x_ch, ab, a, b
    d_wb_p, d_wb_g = _gated_dw(z_hat, d_b_ch, wb_p, wb_g, mask_flat)
    d_z_hat = _gated_dx(z_hat, d_b_ch, wb_p, wb_g, mask_flat, dx=d_z_hat)
    del d_b_ch, z_hat

    d_z, d_ln_in_w, d_ln_in_b = _ln_bwd(
        z_2d, d_z_hat, mean_in, rstd_in, ln_in_w, ln_in_b is not None, out=d_z_hat
    )
    if has_residual:
        d_z.add_(go_2d)
    return (
        d_z.view(batch, n, n, c_z),
        d_wa_p,
        d_wa_g,
        d_wb_p,
        d_wb_g,
        d_wz,
        d_wg,
        d_ln_in_w,
        d_ln_in_b,
        d_ln_out_w,
        d_ln_out_b,
    )


class _FusedTrimulFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        ln_in_eps,
        ln_out_eps,
        has_residual,
    ):
        residual = z if has_residual else None
        y, stats, ab, x_ch = _launch_fused_trimul(
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
            float(ln_in_eps),
            float(ln_out_eps),
            None,
            return_acts=True,
        )
        ctx.save_for_backward(
            z,
            mask if mask is not None else z.new_empty(0),
            wa_p,
            wa_g,
            wb_p,
            wb_g,
            wz,
            wg,
            ln_in_w,
            ln_in_b if ln_in_b is not None else z.new_empty(0),
            ln_out_w,
            ln_out_b if ln_out_b is not None else z.new_empty(0),
            stats[0],
            stats[1],
            ab,
            x_ch,
        )
        ctx.outgoing = bool(outgoing)
        ctx.has_mask = mask is not None
        ctx.has_ln_in_b = ln_in_b is not None
        ctx.has_ln_out_b = ln_out_b is not None
        ctx.has_residual = bool(has_residual)
        ctx.ln_out_eps = float(ln_out_eps)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        (
            z,
            mask_t,
            wa_p,
            wa_g,
            wb_p,
            wb_g,
            wz,
            wg,
            ln_in_w,
            ln_in_b_t,
            ln_out_w,
            ln_out_b_t,
            mean_in,
            rstd_in,
            ab,
            x_ch,
        ) = ctx.saved_tensors
        grads = _fused_trimul_backward(
            z,
            mask_t if ctx.has_mask else None,
            grad_out,
            wa_p,
            wa_g,
            wb_p,
            wb_g,
            wz,
            wg,
            ln_in_w,
            ln_in_b_t if ctx.has_ln_in_b else None,
            ln_out_w,
            ln_out_b_t if ctx.has_ln_out_b else None,
            mean_in,
            rstd_in,
            ctx.outgoing,
            ctx.has_residual,
            ctx.ln_out_eps,
            ab,
            x_ch,
        )
        need = ctx.needs_input_grad
        packed = (
            grads[0] if need[0] else None,
            None,
            grads[1].to(dtype=wa_p.dtype) if need[2] else None,
            grads[2].to(dtype=wa_g.dtype) if need[3] else None,
            grads[3].to(dtype=wb_p.dtype) if need[4] else None,
            grads[4].to(dtype=wb_g.dtype) if need[5] else None,
            grads[5].to(dtype=wz.dtype) if need[6] else None,
            grads[6].to(dtype=wg.dtype) if need[7] else None,
            grads[7].to(dtype=ln_in_w.dtype) if need[8] else None,
            grads[8].to(dtype=ln_in_b_t.dtype) if need[9] and ctx.has_ln_in_b else None,
            grads[9].to(dtype=ln_out_w.dtype) if need[10] else None,
            grads[10].to(dtype=ln_out_b_t.dtype) if need[11] and ctx.has_ln_out_b else None,
            None,
            None,
            None,
            None,
        )
        return packed


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

    Training saves A/B/X from the forward. The output-gate backward stays
    fused (no stored ``g`` / ``val`` / ``x_hat``) with exclusive split-M
    ``dW``. Inference may write ``z + update`` in place when
    ``residual is z``.
    """
    if not is_fused_trimul_eligible(
        z, wa_p, wa_g, wb_p, wb_g, wz, wg, ln_in_w, ln_in_b, ln_out_w, ln_out_b
    ):
        raise RuntimeError(
            "fused_trimul requires an eligible Triton launch; "
            "use eager_trimul for the fallback path"
        )
    use_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (
            z,
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
            residual,
        )
    )
    if use_grad:
        if residual is not None and residual.data_ptr() != z.data_ptr():
            raise RuntimeError("fused_trimul training residual must be z or None")
        return _FusedTrimulFn.apply(
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
            float(ln_in_eps),
            float(ln_out_eps),
            residual is not None,
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
