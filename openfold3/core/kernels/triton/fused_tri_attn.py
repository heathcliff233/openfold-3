# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused triangle attention
# reconstructed from the original inference kernel, with training backward.

"""Fused triangle attention (Triton).

LN → pair-bias Linear → gated MHA → ``W_o`` [+ residual]. ``I`` / ``J`` /
sequence length are not autotune or specialize keys. GEMM tiles autotune on
``GEMM_MODE`` only so each GPU picks its own winner. The pair-bias prologue
decodes ``(i, j)`` from independent strides so contig starting-node ``z`` and
PairBlock's transposed ending-node view share one kernel (no 1U ``z_norm``).
Attention is flash / online-softmax: the ``[I, H, J, J]`` score matrix is
never stored. The forward may group two pair-rows per program so they share
one triangle-bias load. Training rematerializes QKV from saved ``z`` / LN
stats, keeps ``LSE`` + ungated ``O`` + triangle bias, and uses a pure-Triton
backward (LN apply / LN bwd / Linear / gate / flash / exclusive split-M
``dW``, no ATen math). In-place residual writes are inference-only.
Ineligible shapes use the matching-precision eager primitives.
"""

from __future__ import annotations

import math
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

_MAX_C = 128
_MIN_M = 4096
_BWD_SPLIT_M = 16
_DEFAULT_ROW_BLOCK = 128
# First launch below these sizes warms autotune on a large dummy grid so a
# test-sized call cannot lock the GEMM_MODE winner (same contract as trimul).
_FLASH_WARM_J = 1024
_PAIR_LN_WARM_M = 65536
_LINEAR_WARM_M = 65536
_TRUE = {"1", "true", "yes", "on"}
_SUPPORTED_CH = (16, 32, 64, 128)
_flash_autotune_ready: set[tuple] = set()
_flash_bwd_autotune_ready: set[tuple] = set()
_pair_ln_autotune_ready: set[tuple] = set()
_linear_autotune_ready: set[tuple] = set()

# Large-N biased; J is not an autotune key. Forward can afford larger K tiles
# than backward (fewer live accumulators). ``I_TILE`` groups pair-rows that
# share the same triangle-bias tile (bias does not depend on ``i``).
# Tuples are ``(BLOCK_M, BLOCK_N, I_TILE, warps, stages)``.
_FLASH_FWD_TILES = (
    (64, 16, 1, 4, 2),
    (128, 16, 1, 4, 2),
    (64, 32, 1, 4, 2),
    (128, 32, 1, 4, 2),
    (128, 16, 1, 8, 2),
    (128, 32, 1, 8, 2),
    (128, 32, 1, 4, 3),
    (64, 64, 1, 4, 2),
    (128, 64, 1, 4, 2),
    (64, 16, 2, 4, 2),
    (128, 16, 2, 4, 2),
    (64, 32, 2, 4, 2),
    (128, 32, 2, 8, 2),
    (128, 16, 2, 8, 2),
)
_FLASH_BWD_TILES = (
    (64, 16, 4, 2),
    (128, 16, 4, 2),
    (64, 32, 4, 2),
    (128, 32, 4, 2),
    (128, 16, 8, 2),
)
_PAIR_LN_TILES = (
    (32, 16, 4, 1),
    (64, 16, 4, 1),
    (64, 16, 8, 1),
    (128, 16, 4, 1),
)
_DW_TILES = (
    (16, 32, 4, 1),
    (32, 32, 4, 1),
    (32, 64, 4, 1),
    (64, 32, 4, 1),
)
_LN_TILES = (
    (32, 4, 1),
    (64, 4, 1),
    (128, 4, 1),
)
_LINEAR_TILES = (
    (32, 32, 4, 1),
    (64, 32, 4, 1),
    (64, 64, 4, 1),
    (128, 32, 4, 1),
)


def is_fused_tri_attn_v1_enabled() -> bool:
    return os.environ.get("OPENFOLD3_FUSED_TRI_ATTN_V1", "1").strip().lower() in _TRUE


def _all_fp32_masters(*tensors: torch.Tensor | None) -> bool:
    return all(t is None or t.dtype == torch.float32 for t in tensors)


def is_fused_tri_attn_eligible(
    z: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    wz: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    *,
    c_hidden: int,
    no_heads: int,
) -> bool:
    if z.ndim != 4:
        return False
    batch, n_i, n_j, c_in = z.shape
    total = c_hidden * no_heads
    masters = (ln_w, wz, wq, wk, wv, wg, wo, ln_b)
    return (
        is_fused_tri_attn_v1_enabled()
        and _TRITON_AVAILABLE
        and z.is_cuda
        and batch == 1
        and n_i == n_j
        and z.dtype in (torch.float32, torch.bfloat16)
        and _all_fp32_masters(*masters)
        and c_in <= _MAX_C
        and total == c_in
        and c_hidden in _SUPPORTED_CH
        and wz.shape == (no_heads, c_in)
        and wq.shape == (total, c_in)
        and wk.shape == (total, c_in)
        and wv.shape == (total, c_in)
        and wg.shape == (total, c_in)
        and wo.shape == (c_in, total)
        and (n_i * n_j) >= _MIN_M
    )


def _downcast_masters(act_dtype: torch.dtype, *tensors: torch.Tensor | None):
    if act_dtype != torch.bfloat16:
        return tensors
    return tuple(None if t is None else t.to(dtype=act_dtype) for t in tensors)


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def eager_tri_attn(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    wz: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    *,
    c_hidden: int,
    no_heads: int,
    starting: bool,
    inf: float = 1e9,
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Matching-precision eager AF3 triangle attention (bf16 downcasts masters)."""
    ln_w, ln_b, wz, wq, wk, wv, wg, wo = _downcast_masters(
        z.dtype, ln_w, ln_b, wz, wq, wk, wv, wg, wo
    )
    if not starting:
        z = z.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        if residual is not None:
            residual = residual.transpose(-2, -3)
    z_n = F.layer_norm(z, (ln_w.shape[0],), ln_w, ln_b, eps)
    heads = no_heads
    ch = c_hidden
    scale = 1.0 / math.sqrt(ch)
    tb = F.linear(z_n, wz).permute(0, 3, 1, 2).unsqueeze(1)
    q = F.linear(z_n, wq).view(*z_n.shape[:-1], heads, ch).transpose(-2, -3) * scale
    k = F.linear(z_n, wk).view(*z_n.shape[:-1], heads, ch).transpose(-2, -3)
    v = F.linear(z_n, wv).view(*z_n.shape[:-1], heads, ch).transpose(-2, -3)
    scores = torch.einsum("...qc,...kc->...qk", q, k) + tb
    if mask is not None:
        scores = scores + (inf * (mask - 1))[..., :, None, None, :]
    attn_dtype = z.dtype
    if attn_dtype == torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            probs = torch.nn.functional.softmax(scores, dim=-1)
    else:
        probs = torch.nn.functional.softmax(scores, dim=-1)
    o = torch.einsum("...qk,...kc->...qc", probs.to(dtype=v.dtype), v)
    o = o.transpose(-2, -3)
    gate = torch.sigmoid(F.linear(z_n, wg)).view(*z_n.shape[:-1], heads, ch)
    o = (o * gate).reshape(*z_n.shape[:-1], heads * ch)
    out = F.linear(o, wo)
    if residual is not None:
        out = residual + out
    if not starting:
        out = out.transpose(-2, -3)
    return out


if _TRITON_AVAILABLE:
    RCP_LN2 = tl.constexpr(1.4426950408889634)

    def _configs(tiles, names):
        return [
            triton.Config(
                {names[0]: a, names[1]: b},
                num_warps=warps,
                num_stages=stages,
            )
            for a, b, warps, stages in tiles
        ]

    def _flash_fwd_configs():
        return [
            triton.Config(
                {"BLOCK_M": m, "BLOCK_N": n, "I_TILE": it},
                num_warps=warps,
                num_stages=stages,
            )
            for m, n, it, warps, stages in _FLASH_FWD_TILES
        ]

    def _prune_flash(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        kept = []
        for cfg in configs:
            it = cfg.kwargs.get("I_TILE", 1)
            if mode == "ieee" and (
                cfg.kwargs["BLOCK_M"] >= 128
                or cfg.kwargs["BLOCK_N"] >= 32
                or it > 1
            ):
                continue
            kept.append(cfg)
        return kept or configs[:1]

    def _prune_dw(configs, named_args, **kwargs):
        mode = kwargs.get("GEMM_MODE", named_args.get("GEMM_MODE"))
        n = named_args.get("N")
        kept = []
        for cfg in configs:
            if mode == "ieee" and cfg.kwargs["BLOCK_N"] >= 64:
                continue
            if n is not None and cfg.kwargs["BLOCK_N"] > n:
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

    @triton.autotune(
        configs=_configs(_PAIR_LN_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        restore_value=["Y_ptr"],
    )
    @triton.jit(
        do_not_specialize=["I_dim", "J_dim", "eps"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "W_ptr",
            "Y_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Mean_ptr",
            "Rstd_ptr",
        ],
    )
    def _pair_ln_linear_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Gamma_ptr,
        Beta_ptr,
        Mean_ptr,
        Rstd_ptr,
        I_dim,
        J_dim,
        stride_x_i,
        stride_x_j,
        stride_x_k,
        N,
        K,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        WRITE_STATS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """LN + Linear over logical ``[1, I, J, K]`` rows (stride-decoded)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m64 = tl.cast(I_dim * J_dim, tl.int64)
        m_mask = offs_m < m64
        n_mask = offs_n < N
        k_mask = offs_k < K
        offs_i = offs_m // J_dim
        offs_j = offs_m - offs_i * J_dim
        x_base = offs_i.to(tl.int64) * stride_x_i + offs_j.to(tl.int64) * stride_x_j
        x = tl.load(
            X_ptr + x_base[:, None] + offs_k[None, :] * stride_x_k,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.where(k_mask[None, :], x, 0.0)
        mean = tl.sum(x, axis=1) / K
        x_c = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x_c * x_c, axis=1) / K + eps)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_c * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)
        if WRITE_STATS:
            tl.store(Mean_ptr + offs_m, mean, mask=m_mask)
            tl.store(Rstd_ptr + offs_m, rstd, mask=m_mask)
        w = tl.load(
            W_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = _dot_f32(x_hat, tl.trans(w), GEMM_MODE)
        # Write ``[H, I, J]`` directly so flash can consume it without a permute.
        tl.store(
            Y_ptr
            + offs_n.to(tl.int64)[None, :]
            * (tl.cast(I_dim, tl.int64) * tl.cast(J_dim, tl.int64))
            + offs_m.to(tl.int64)[:, None],
            acc.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.autotune(
        configs=_flash_fwd_configs(),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_flash},
        restore_value=["O_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "I_dim",
            "J_dim",
            "row_start",
            "softmax_scale",
            "mask_inf",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "G_ptr",
            "TB_ptr",
            "Mask_ptr",
            "O_ptr",
            "LSE_ptr",
        ],
    )
    def _flash_tri_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        G_ptr,
        TB_ptr,
        Mask_ptr,
        O_ptr,
        LSE_ptr,
        I_dim,
        J_dim,
        row_start,
        softmax_scale,
        mask_inf,
        stride_q_i,
        stride_q_j,
        stride_q_h,
        stride_q_c,
        stride_tb_h,
        stride_tb_j,
        stride_tb_k,
        stride_mb_i,
        stride_mb_j,
        stride_o_i,
        stride_o_j,
        stride_o_h,
        stride_o_c,
        H: tl.constexpr,
        CH: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_GATE: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        I_TILE: tl.constexpr,
    ):
        """One program owns a Q-row tile of ``I_TILE`` pair-rows × one head."""
        pid_m = tl.program_id(0)
        pid_ih = tl.program_id(1)
        h = pid_ih % H
        i0 = (pid_ih // H) * I_TILE
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, CH)
        mask_m = offs_m < J_dim
        rcp = RCP_LN2
        tb_row = h * stride_tb_h + offs_m * stride_tb_j
        q0_base = i0 * stride_q_i + h * stride_q_h
        q0 = tl.load(
            Q_ptr + q0_base + offs_m[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_m[:, None],
            other=0.0,
        )
        m0 = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        o0 = tl.zeros((BLOCK_M, CH), dtype=tl.float32)
        mb0 = (i0 + row_start) * stride_mb_i
        if I_TILE == 2:
            i1 = i0 + 1
            live1 = i1 < I_dim
            q1_base = i1 * stride_q_i + h * stride_q_h
            q1 = tl.load(
                Q_ptr
                + q1_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None] & live1,
                other=0.0,
            )
            m1 = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
            l1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
            o1 = tl.zeros((BLOCK_M, CH), dtype=tl.float32)
            mb1 = (i1 + row_start) * stride_mb_i
        for n0 in range(0, J_dim, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < J_dim
            tb = tl.load(
                TB_ptr + tb_row[:, None] + offs_n[None, :] * stride_tb_k,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            tb = tb * rcp
            k0 = tl.load(
                K_ptr
                + q0_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            qk0 = _dot_f32(q0, tl.trans(k0), GEMM_MODE) * (softmax_scale * rcp) + tb
            if HAS_MASK:
                mb = tl.load(Mask_ptr + mb0 + offs_n * stride_mb_j, mask=mask_n, other=0.0)
                qk0 = qk0 + (mask_inf * (mb.to(tl.float32) - 1.0))[None, :] * rcp
            qk0 = tl.where(mask_n[None, :], qk0, float("-inf"))
            m0n = tl.maximum(m0, tl.max(qk0, 1))
            p0 = tl.math.exp2(qk0 - m0n[:, None])
            a0 = tl.math.exp2(m0 - m0n)
            l0 = l0 * a0 + tl.sum(p0, 1)
            o0 = o0 * a0[:, None]
            m0 = m0n
            v0 = tl.load(
                V_ptr
                + q0_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            o0 = o0 + _dot_f32(p0, v0, GEMM_MODE)
            if I_TILE == 2:
                k1 = tl.load(
                    K_ptr
                    + q1_base
                    + offs_n[:, None] * stride_q_j
                    + offs_c[None, :] * stride_q_c,
                    mask=mask_n[:, None] & live1,
                    other=0.0,
                )
                qk1 = _dot_f32(q1, tl.trans(k1), GEMM_MODE) * (softmax_scale * rcp) + tb
                if HAS_MASK:
                    mb = tl.load(
                        Mask_ptr + mb1 + offs_n * stride_mb_j, mask=mask_n & live1, other=0.0
                    )
                    qk1 = qk1 + (mask_inf * (mb.to(tl.float32) - 1.0))[None, :] * rcp
                qk1 = tl.where(mask_n[None, :] & live1, qk1, float("-inf"))
                m1n = tl.maximum(m1, tl.max(qk1, 1))
                p1 = tl.math.exp2(qk1 - m1n[:, None])
                a1 = tl.math.exp2(m1 - m1n)
                l1 = l1 * a1 + tl.sum(p1, 1)
                o1 = o1 * a1[:, None]
                m1 = m1n
                v1 = tl.load(
                    V_ptr
                    + q1_base
                    + offs_n[:, None] * stride_q_j
                    + offs_c[None, :] * stride_q_c,
                    mask=mask_n[:, None] & live1,
                    other=0.0,
                )
                o1 = o1 + _dot_f32(p1, v1, GEMM_MODE)
        o0 = o0 / l0[:, None]
        if WRITE_LSE:
            tl.store(
                LSE_ptr + (i0 * H + h) * J_dim + offs_m,
                m0 + tl.math.log2(l0),
                mask=mask_m,
            )
        if HAS_GATE:
            g0 = tl.load(
                G_ptr
                + q0_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            o0 = o0 * tl.sigmoid(g0)
        tl.store(
            O_ptr
            + i0 * stride_o_i
            + h * stride_o_h
            + offs_m[:, None] * stride_o_j
            + offs_c[None, :] * stride_o_c,
            o0.to(O_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
        if I_TILE == 2:
            o1 = o1 / l1[:, None]
            if WRITE_LSE:
                tl.store(
                    LSE_ptr + (i1 * H + h) * J_dim + offs_m,
                    m1 + tl.math.log2(l1),
                    mask=mask_m & live1,
                )
            if HAS_GATE:
                g1 = tl.load(
                    G_ptr
                    + q1_base
                    + offs_m[:, None] * stride_q_j
                    + offs_c[None, :] * stride_q_c,
                    mask=mask_m[:, None] & live1,
                    other=0.0,
                ).to(tl.float32)
                o1 = o1 * tl.sigmoid(g1)
            tl.store(
                O_ptr
                + i1 * stride_o_i
                + h * stride_o_h
                + offs_m[:, None] * stride_o_j
                + offs_c[None, :] * stride_o_c,
                o1.to(O_ptr.dtype.element_ty),
                mask=mask_m[:, None] & live1,
            )

    @triton.autotune(
        configs=_configs(_FLASH_BWD_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_flash},
        restore_value=["DQ_ptr"],
    )
    @triton.jit(
        do_not_specialize=["I_dim", "J_dim", "row_start", "softmax_scale", "mask_inf"],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "O_ptr",
            "DO_ptr",
            "LSE_ptr",
            "TB_ptr",
            "Mask_ptr",
            "DQ_ptr",
        ],
    )
    def _flash_tri_attn_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        DO_ptr,
        LSE_ptr,
        TB_ptr,
        Mask_ptr,
        DQ_ptr,
        I_dim,
        J_dim,
        row_start,
        softmax_scale,
        mask_inf,
        stride_q_i,
        stride_q_j,
        stride_q_h,
        stride_q_c,
        stride_tb_h,
        stride_tb_j,
        stride_tb_k,
        stride_mb_i,
        stride_mb_j,
        H: tl.constexpr,
        CH: tl.constexpr,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_ih = tl.program_id(1)
        h = pid_ih % H
        i = pid_ih // H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, CH)
        mask_m = offs_m < J_dim
        q_base = i * stride_q_i + h * stride_q_h
        q = tl.load(
            Q_ptr + q_base + offs_m[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            DO_ptr + q_base + offs_m[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        o = tl.load(
            O_ptr + q_base + offs_m[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(LSE_ptr + (i * H + h) * J_dim + offs_m, mask=mask_m, other=0.0)
        delta = tl.sum(do * o, axis=1)
        dq = tl.zeros((BLOCK_M, CH), dtype=tl.float32)
        rcp = RCP_LN2
        tb_row = h * stride_tb_h + offs_m * stride_tb_j
        mb_base = (i + row_start) * stride_mb_i
        for n0 in range(0, J_dim, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < J_dim
            k = tl.load(
                K_ptr
                + q_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            v = tl.load(
                V_ptr
                + q_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            qk = _dot_f32(q, tl.trans(k), GEMM_MODE) * (softmax_scale * rcp)
            tb = tl.load(
                TB_ptr + tb_row[:, None] + offs_n[None, :] * stride_tb_k,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            qk = qk + tb * rcp
            if HAS_MASK:
                mb = tl.load(Mask_ptr + mb_base + offs_n * stride_mb_j, mask=mask_n, other=0.0)
                qk = qk + (mask_inf * (mb.to(tl.float32) - 1.0))[None, :] * rcp
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            p = tl.math.exp2(qk - lse[:, None])
            dp = _dot_f32(do, tl.trans(v), GEMM_MODE)
            ds = p * (dp - delta[:, None])
            dq += _dot_f32(ds, k, GEMM_MODE) * softmax_scale
        tl.store(
            DQ_ptr
            + q_base
            + offs_m[:, None] * stride_q_j
            + offs_c[None, :] * stride_q_c,
            dq.to(DQ_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

    @triton.autotune(
        configs=_configs(_FLASH_BWD_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_flash},
        restore_value=["DK_ptr", "DV_ptr"],
    )
    @triton.jit(
        do_not_specialize=["I_dim", "J_dim", "row_start", "softmax_scale", "mask_inf"],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "O_ptr",
            "DO_ptr",
            "LSE_ptr",
            "TB_ptr",
            "Mask_ptr",
            "DK_ptr",
            "DV_ptr",
        ],
    )
    def _flash_tri_attn_bwd_dkv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        DO_ptr,
        LSE_ptr,
        TB_ptr,
        Mask_ptr,
        DK_ptr,
        DV_ptr,
        I_dim,
        J_dim,
        row_start,
        softmax_scale,
        mask_inf,
        stride_q_i,
        stride_q_j,
        stride_q_h,
        stride_q_c,
        stride_tb_h,
        stride_tb_j,
        stride_tb_k,
        stride_mb_i,
        stride_mb_j,
        H: tl.constexpr,
        CH: tl.constexpr,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exclusive dK/dV: one program owns a K-tile of one ``(i, h)``."""
        pid_n = tl.program_id(0)
        pid_ih = tl.program_id(1)
        h = pid_ih % H
        i = pid_ih // H
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, CH)
        mask_n = offs_n < J_dim
        q_base = i * stride_q_i + h * stride_q_h
        k = tl.load(
            K_ptr + q_base + offs_n[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V_ptr + q_base + offs_n[:, None] * stride_q_j + offs_c[None, :] * stride_q_c,
            mask=mask_n[:, None],
            other=0.0,
        )
        dk = tl.zeros((BLOCK_N, CH), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, CH), dtype=tl.float32)
        rcp = RCP_LN2
        mb_base = (i + row_start) * stride_mb_i
        tb_col = h * stride_tb_h + offs_n * stride_tb_k
        for m0 in range(0, J_dim, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < J_dim
            q = tl.load(
                Q_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            )
            do = tl.load(
                DO_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            o = tl.load(
                O_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(LSE_ptr + (i * H + h) * J_dim + offs_m, mask=mask_m, other=0.0)
            delta = tl.sum(do * o, axis=1)
            qk = _dot_f32(q, tl.trans(k), GEMM_MODE) * (softmax_scale * rcp)
            tb = tl.load(
                TB_ptr + offs_m[:, None] * stride_tb_j + tb_col[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            qk = qk + tb * rcp
            if HAS_MASK:
                mb = tl.load(Mask_ptr + mb_base + offs_n * stride_mb_j, mask=mask_n, other=0.0)
                qk = qk + (mask_inf * (mb.to(tl.float32) - 1.0))[None, :] * rcp
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))
            p = tl.math.exp2(qk - lse[:, None])
            dp = _dot_f32(do, tl.trans(v), GEMM_MODE)
            ds = p * (dp - delta[:, None])
            ds = tl.where(mask_m[:, None] & mask_n[None, :], ds, 0.0)
            dk += _dot_f32(tl.trans(ds), q, GEMM_MODE) * softmax_scale
            dv += _dot_f32(tl.trans(p), do, GEMM_MODE)
        tl.store(
            DK_ptr
            + q_base
            + offs_n[:, None] * stride_q_j
            + offs_c[None, :] * stride_q_c,
            dk.to(DK_ptr.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            DV_ptr
            + q_base
            + offs_n[:, None] * stride_q_j
            + offs_c[None, :] * stride_q_c,
            dv.to(DV_ptr.dtype.element_ty),
            mask=mask_n[:, None],
        )

    @triton.autotune(
        configs=_configs(_FLASH_BWD_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_flash},
        restore_value=["DTB_ptr"],
    )
    @triton.jit(
        do_not_specialize=["I_dim", "J_dim", "row_start", "softmax_scale", "mask_inf"],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "O_ptr",
            "DO_ptr",
            "LSE_ptr",
            "TB_ptr",
            "Mask_ptr",
            "DTB_ptr",
        ],
    )
    def _flash_tri_attn_bwd_dbias_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        DO_ptr,
        LSE_ptr,
        TB_ptr,
        Mask_ptr,
        DTB_ptr,
        I_dim,
        J_dim,
        row_start,
        softmax_scale,
        mask_inf,
        stride_q_i,
        stride_q_j,
        stride_q_h,
        stride_q_c,
        stride_tb_h,
        stride_tb_j,
        stride_tb_k,
        stride_mb_i,
        stride_mb_j,
        H: tl.constexpr,
        CH: tl.constexpr,
        HAS_MASK: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exclusive ``d`` triangle-bias: one program owns a ``(h, q-tile, k-tile)``."""
        pid_m = tl.program_id(0)
        pid_nh = tl.program_id(1)
        h = pid_nh % H
        pid_n = pid_nh // H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, CH)
        mask_m = offs_m < J_dim
        mask_n = offs_n < J_dim
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        rcp = RCP_LN2
        tb_row = h * stride_tb_h + offs_m * stride_tb_j
        # Triangle bias is independent of pair-row ``i``; load once.
        tb = tl.load(
            TB_ptr + tb_row[:, None] + offs_n[None, :] * stride_tb_k,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        tb = tb * rcp
        for i in range(0, I_dim):
            q_base = i * stride_q_i + h * stride_q_h
            q = tl.load(
                Q_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            )
            k = tl.load(
                K_ptr
                + q_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            v = tl.load(
                V_ptr
                + q_base
                + offs_n[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_n[:, None],
                other=0.0,
            )
            do = tl.load(
                DO_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            o = tl.load(
                O_ptr
                + q_base
                + offs_m[:, None] * stride_q_j
                + offs_c[None, :] * stride_q_c,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(LSE_ptr + (i * H + h) * J_dim + offs_m, mask=mask_m, other=0.0)
            delta = tl.sum(do * o, axis=1)
            qk = _dot_f32(q, tl.trans(k), GEMM_MODE) * (softmax_scale * rcp) + tb
            if HAS_MASK:
                mb = tl.load(
                    Mask_ptr + (i + row_start) * stride_mb_i + offs_n * stride_mb_j,
                    mask=mask_n,
                    other=0.0,
                )
                qk = qk + (mask_inf * (mb.to(tl.float32) - 1.0))[None, :] * rcp
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))
            p = tl.math.exp2(qk - lse[:, None])
            dp = _dot_f32(do, tl.trans(v), GEMM_MODE)
            acc += p * (dp - delta[:, None])
        tl.store(
            DTB_ptr + h * J_dim * J_dim + offs_m[:, None] * J_dim + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.autotune(
        configs=_configs(_DW_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_dw},
        restore_value=["P_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["X_ptr", "GO_ptr", "P_ptr"],
    )
    def _split_m_dw_kernel(
        X_ptr,
        GO_ptr,
        P_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        SPLIT_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        split = tl.program_id(0)
        rows = (M + SPLIT_M - 1) // SPLIT_M
        start = split * rows
        end = tl.minimum(start + rows, M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        n_mask = offs_n < N
        k_mask = offs_k < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m0 in range(start, end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_mask = offs_m < end
            x = tl.load(
                X_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            go = tl.load(
                GO_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += _dot_f32(tl.trans(go), x, GEMM_MODE)
        tl.store(
            P_ptr + split * N * K + offs_n[:, None] * K + offs_k[None, :],
            acc,
            mask=n_mask[:, None] & k_mask[None, :],
        )

    def _ln_configs():
        return [
            triton.Config({"BLOCK_M": m}, num_warps=warps, num_stages=stages)
            for m, warps, stages in _LN_TILES
        ]

    @triton.autotune(configs=_ln_configs(), key=["HAS_LN_BIAS"], restore_value=["Y_ptr"])
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "Y_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "Gamma_ptr",
            "Beta_ptr",
        ],
    )
    def _ln_apply_kernel(
        X_ptr,
        Y_ptr,
        Mean_ptr,
        Rstd_ptr,
        Gamma_ptr,
        Beta_ptr,
        M,
        K: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        x = tl.load(
            X_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            y = y + tl.where(k_mask[None, :], beta[None, :], 0.0)
        tl.store(
            Y_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            y.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(
        configs=_ln_configs(),
        key=["HAS_LN_BIAS"],
        restore_value=["DX_ptr", "PGamma_ptr", "PBeta_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "DY_ptr",
            "DX_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "Gamma_ptr",
            "PGamma_ptr",
            "PBeta_ptr",
        ],
    )
    def _ln_bwd_kernel(
        X_ptr,
        DY_ptr,
        DX_ptr,
        Mean_ptr,
        Rstd_ptr,
        Gamma_ptr,
        PGamma_ptr,
        PBeta_ptr,
        M,
        K: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        x = tl.load(
            X_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_norm = tl.where(k_mask[None, :], (x - mean[:, None]) * rstd[:, None], 0.0)
        dx_hat = dy * gamma[None, :]
        dx_hat = tl.where(k_mask[None, :], dx_hat, 0.0)
        dmean = tl.sum(dx_hat, axis=1) / K
        dproj = tl.sum(dx_hat * x_norm, axis=1) / K
        dx = rstd[:, None] * (dx_hat - dmean[:, None] - x_norm * dproj[:, None])
        tl.store(
            DX_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            dx.to(DX_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            PGamma_ptr + pid * K + offs_k,
            tl.sum(dy * x_norm, axis=0),
            mask=k_mask,
        )
        if HAS_LN_BIAS:
            tl.store(PBeta_ptr + pid * K + offs_k, tl.sum(dy, axis=0), mask=k_mask)

    @triton.autotune(
        configs=_configs(_LINEAR_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_dw},
        restore_value=["Y_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["X_ptr", "W_ptr", "Y_ptr"],
    )
    def _linear_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``Y = X @ W^T`` with ``X[M,K]``, ``W[N,K]``."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(
                X_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                W_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc += _dot_f32(x, tl.trans(w), GEMM_MODE)
        tl.store(
            Y_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :],
            acc.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.autotune(
        configs=_configs(_LINEAR_TILES, ("BLOCK_M", "BLOCK_N")),
        key=["GEMM_MODE"],
        prune_configs_by={"early_config_prune": _prune_dw},
        restore_value=["DX_ptr"],
    )
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["DY_ptr", "W_ptr", "DX_ptr"],
    )
    def _linear_dx_kernel(
        DY_ptr,
        W_ptr,
        DX_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``dX = dY @ W`` with ``dY[M,N]``, ``W[N,K]``."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        k_mask = offs_k < K
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            dy = tl.load(
                DY_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                W_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc += _dot_f32(dy, w, GEMM_MODE)
        tl.store(
            DX_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :],
            acc.to(DX_ptr.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )

    @triton.autotune(configs=_ln_configs(), key=["GEMM_MODE"], restore_value=["DO_ptr"])
    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "O_ptr",
            "G_ptr",
            "DATT_ptr",
            "DO_ptr",
            "DG_ptr",
            "GATED_ptr",
        ],
    )
    def _gate_bwd_kernel(
        O_ptr,
        G_ptr,
        DATT_ptr,
        DO_ptr,
        DG_ptr,
        GATED_ptr,
        M,
        C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GEMM_MODE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        c_mask = offs_c < C
        o = tl.load(
            O_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            mask=m_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        g = tl.load(
            G_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            mask=m_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        datt = tl.load(
            DATT_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            mask=m_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        gated = o * sig
        d_o = datt * sig
        d_g = datt * o * sig * (1.0 - sig)
        tl.store(
            GATED_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            gated.to(GATED_ptr.dtype.element_ty),
            mask=m_mask[:, None] & c_mask[None, :],
        )
        tl.store(
            DO_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            d_o.to(DO_ptr.dtype.element_ty),
            mask=m_mask[:, None] & c_mask[None, :],
        )
        tl.store(
            DG_ptr + offs_m.to(tl.int64)[:, None] * C + offs_c[None, :],
            d_g.to(DG_ptr.dtype.element_ty),
            mask=m_mask[:, None] & c_mask[None, :],
        )

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_tri_attn")

    _pair_ln_linear_fwd_kernel = _unavailable
    _flash_tri_attn_fwd_kernel = _unavailable
    _flash_tri_attn_bwd_dq_kernel = _unavailable
    _flash_tri_attn_bwd_dkv_kernel = _unavailable
    _flash_tri_attn_bwd_dbias_kernel = _unavailable
    _split_m_dw_kernel = _unavailable
    _ln_apply_kernel = _unavailable
    _ln_bwd_kernel = _unavailable
    _linear_fwd_kernel = _unavailable
    _linear_dx_kernel = _unavailable
    _gate_bwd_kernel = _unavailable


def _pair_ln_autotune_key(z, wz, ln_b, write_stats, mode):
    return (
        mode,
        z.device.type,
        z.device.index,
        wz.shape[0],
        z.shape[-1],
        z.dtype,
        ln_b is not None,
        write_stats,
    )


def _ensure_pair_ln_autotune(
    z: torch.Tensor,
    wz: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    write_stats: bool,
    mode: str,
) -> None:
    """Time pair-LN on a large dummy grid so a small first ``M`` cannot lock tiles."""
    key = _pair_ln_autotune_key(z, wz, ln_b, write_stats, mode)
    if key in _pair_ln_autotune_ready:
        return
    k = z.shape[-1]
    heads = wz.shape[0]
    side = int(_PAIR_LN_WARM_M**0.5)
    dummy_x = torch.empty((1, side, side, k), device=z.device, dtype=z.dtype)
    dummy_y = torch.empty((heads, side, side), device=z.device, dtype=z.dtype)
    dummy_stats = (
        torch.empty(side * side, device=z.device, dtype=torch.float32)
        if write_stats
        else z.new_empty(1)
    )
    dummy = z.new_empty(1)

    def grid(meta):
        return (
            triton.cdiv(side * side, meta["BLOCK_M"]),
            triton.cdiv(heads, meta["BLOCK_N"]),
        )

    _pair_ln_linear_fwd_kernel[grid](
        dummy_x,
        wz,
        dummy_y,
        ln_w,
        ln_b if ln_b is not None else dummy,
        dummy_stats,
        dummy_stats,
        side,
        side,
        dummy_x.stride(1),
        dummy_x.stride(2),
        dummy_x.stride(3),
        heads,
        k,
        1e-5,
        HAS_LN_BIAS=ln_b is not None,
        WRITE_STATS=write_stats,
        GEMM_MODE=mode,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    _pair_ln_autotune_ready.add(key)


def _pair_ln_linear(
    z: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    wz: torch.Tensor,
    eps: float,
    *,
    write_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``LN(z) @ wz^T`` as ``[H, I, J]`` without a contiguous 1U ``z_norm``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    _, n_i, n_j, c_in = z.shape
    heads = wz.shape[0]
    ln_w, ln_b, wz = _downcast_masters(z.dtype, ln_w, ln_b, wz)
    y = torch.empty((heads, n_i, n_j), device=z.device, dtype=z.dtype)
    mean = rstd = None
    if write_stats:
        mean = torch.empty(n_i * n_j, device=z.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
    dummy = z.new_empty(1)
    mode = _gemm_mode(z)
    if n_i * n_j < _PAIR_LN_WARM_M:
        _ensure_pair_ln_autotune(z, wz, ln_w, ln_b, write_stats, mode)

    def grid(meta):
        return (
            triton.cdiv(n_i * n_j, meta["BLOCK_M"]),
            triton.cdiv(heads, meta["BLOCK_N"]),
        )

    _pair_ln_linear_fwd_kernel[grid](
        z,
        wz,
        y,
        ln_w,
        ln_b if ln_b is not None else dummy,
        mean if mean is not None else dummy,
        rstd if rstd is not None else dummy,
        n_i,
        n_j,
        z.stride(1),
        z.stride(2),
        z.stride(3),
        heads,
        c_in,
        float(eps),
        HAS_LN_BIAS=ln_b is not None,
        WRITE_STATS=write_stats,
        GEMM_MODE=mode,
        BLOCK_K=max(_next_power_of_two(c_in), 16),
    )
    return y, mean, rstd


def _flash_fwd_key(q, heads, ch, has_mask, has_gate, write_lse, mode):
    return (
        mode,
        q.device.type,
        q.device.index,
        heads,
        ch,
        has_mask,
        has_gate,
        write_lse,
        q.dtype,
    )


def _ensure_flash_fwd_autotune(
    q,
    heads: int,
    ch: int,
    has_mask: bool,
    has_gate: bool,
    write_lse: bool,
    mode: str,
) -> None:
    """Time flash-fwd on a large dummy ``J`` so a small first length cannot lock tiles."""
    key = _flash_fwd_key(q, heads, ch, has_mask, has_gate, write_lse, mode)
    if key in _flash_autotune_ready:
        return
    j = _FLASH_WARM_J
    # Two live pair-rows so ``I_TILE=2`` is timed fairly (a single dummy row
    # makes the second slot look like wasted work).
    rows = 2
    dummy_q = torch.empty((rows, j, heads, ch), device=q.device, dtype=q.dtype)
    dummy_tb = torch.empty((heads, j, j), device=q.device, dtype=q.dtype)
    dummy_mask = torch.ones((rows, j), device=q.device, dtype=q.dtype)
    dummy_out = torch.empty_like(dummy_q)
    dummy_lse = (
        torch.empty((rows, heads, j), device=q.device, dtype=torch.float32)
        if write_lse
        else q.new_empty(1)
    )

    def grid(meta):
        return (
            triton.cdiv(j, meta["BLOCK_M"]),
            triton.cdiv(rows, meta["I_TILE"]) * heads,
        )

    _flash_tri_attn_fwd_kernel[grid](
        dummy_q,
        dummy_q,
        dummy_q,
        dummy_q,
        dummy_tb,
        dummy_mask,
        dummy_out,
        dummy_lse,
        rows,
        j,
        0,
        1.0,
        1e9,
        dummy_q.stride(0),
        dummy_q.stride(1),
        dummy_q.stride(2),
        dummy_q.stride(3),
        dummy_tb.stride(0),
        dummy_tb.stride(1),
        dummy_tb.stride(2),
        dummy_mask.stride(0),
        dummy_mask.stride(1),
        dummy_out.stride(0),
        dummy_out.stride(1),
        dummy_out.stride(2),
        dummy_out.stride(3),
        H=heads,
        CH=ch,
        HAS_MASK=has_mask,
        HAS_GATE=has_gate,
        WRITE_LSE=write_lse,
        GEMM_MODE=mode,
    )
    _flash_autotune_ready.add(key)


def _flash_fwd(
    q,
    k,
    v,
    gate,
    triangle_bias,
    mask,
    row_start: int,
    scale: float,
    mask_inf: float,
    *,
    write_lse: bool,
    has_gate: bool,
    out=None,
):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    rows, j, heads, ch = q.shape
    if out is None:
        out = torch.empty((rows, j, heads, ch), device=q.device, dtype=q.dtype)
    lse = (
        torch.empty((rows, heads, j), device=q.device, dtype=torch.float32)
        if write_lse
        else q.new_empty(1)
    )
    dummy = q
    mode = _gemm_mode(q)
    has_mask = mask is not None
    if j < _FLASH_WARM_J:
        _ensure_flash_fwd_autotune(q, heads, ch, has_mask, has_gate, write_lse, mode)

    def grid(meta):
        return (
            triton.cdiv(j, meta["BLOCK_M"]),
            triton.cdiv(rows, meta["I_TILE"]) * heads,
        )

    # Q/K/V/G/O share the ``[I, J, H, CH]`` layout after the packed split.
    _flash_tri_attn_fwd_kernel[grid](
        q,
        k,
        v,
        gate if gate is not None else dummy,
        triangle_bias,
        mask if mask is not None else dummy,
        out,
        lse,
        rows,
        j,
        row_start,
        float(scale),
        float(mask_inf),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        triangle_bias.stride(0),
        triangle_bias.stride(1),
        triangle_bias.stride(2),
        mask.stride(-2) if mask is not None else 0,
        mask.stride(-1) if mask is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H=heads,
        CH=ch,
        HAS_MASK=has_mask,
        HAS_GATE=has_gate,
        WRITE_LSE=write_lse,
        GEMM_MODE=mode,
    )
    return out, lse if write_lse else None


def _flash_bwd_key(q, heads, ch, has_mask, mode):
    return (
        mode,
        q.device.type,
        q.device.index,
        heads,
        ch,
        has_mask,
        q.dtype,
    )


def _ensure_flash_bwd_autotune(
    q,
    heads: int,
    ch: int,
    has_mask: bool,
    mode: str,
) -> None:
    """Time flash-bwd on a large dummy ``J`` so a small first length cannot lock tiles."""
    key = _flash_bwd_key(q, heads, ch, has_mask, mode)
    if key in _flash_bwd_autotune_ready:
        return
    j = _FLASH_WARM_J
    dummy_q = torch.empty((1, j, heads, ch), device=q.device, dtype=q.dtype)
    dummy_tb = torch.empty((heads, j, j), device=q.device, dtype=q.dtype)
    dummy_mask = torch.ones((1, j), device=q.device, dtype=q.dtype)
    dummy_lse = torch.empty((1, heads, j), device=q.device, dtype=torch.float32)
    dummy_dq = torch.empty_like(dummy_q)
    dummy_dk = torch.empty_like(dummy_q)
    dummy_dv = torch.empty_like(dummy_q)
    dummy_dtb = torch.zeros((heads, j, j), device=q.device, dtype=torch.float32)
    shared = dict(
        I_dim=1,
        J_dim=j,
        row_start=0,
        softmax_scale=1.0,
        mask_inf=1e9,
        stride_q_i=dummy_q.stride(0),
        stride_q_j=dummy_q.stride(1),
        stride_q_h=dummy_q.stride(2),
        stride_q_c=dummy_q.stride(3),
        stride_tb_h=dummy_tb.stride(0),
        stride_tb_j=dummy_tb.stride(1),
        stride_tb_k=dummy_tb.stride(2),
        stride_mb_i=dummy_mask.stride(0),
        stride_mb_j=dummy_mask.stride(1),
        H=heads,
        CH=ch,
        HAS_MASK=has_mask,
        GEMM_MODE=mode,
    )
    args = (
        dummy_q,
        dummy_q,
        dummy_q,
        dummy_q,
        dummy_q,
        dummy_lse,
        dummy_tb,
        dummy_mask,
    )

    def q_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_M"]), heads)

    def kv_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_N"]), heads)

    def bias_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_M"]), triton.cdiv(j, meta["BLOCK_N"]) * heads)

    _flash_tri_attn_bwd_dq_kernel[q_grid](*args, dummy_dq, **shared)
    _flash_tri_attn_bwd_dkv_kernel[kv_grid](*args, dummy_dk, dummy_dv, **shared)
    _flash_tri_attn_bwd_dbias_kernel[bias_grid](*args, dummy_dtb, **shared)
    _flash_bwd_autotune_ready.add(key)


def _flash_bwd(
    q,
    k,
    v,
    o,
    do,
    lse,
    triangle_bias,
    mask,
    row_start: int,
    scale: float,
    mask_inf: float,
):
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    o = o if o.is_contiguous() else o.contiguous()
    do = do if do.is_contiguous() else do.contiguous()
    rows, j, heads, ch = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dtb = torch.zeros(
        (heads, j, j), device=q.device, dtype=torch.float32
    )
    dummy = q
    mode = _gemm_mode(q)
    has_mask = mask is not None
    if j < _FLASH_WARM_J:
        _ensure_flash_bwd_autotune(q, heads, ch, has_mask, mode)
    shared = dict(
        I_dim=rows,
        J_dim=j,
        row_start=row_start,
        softmax_scale=float(scale),
        mask_inf=float(mask_inf),
        stride_q_i=q.stride(0),
        stride_q_j=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_c=q.stride(3),
        stride_tb_h=triangle_bias.stride(0),
        stride_tb_j=triangle_bias.stride(1),
        stride_tb_k=triangle_bias.stride(2),
        stride_mb_i=mask.stride(-2) if mask is not None else 0,
        stride_mb_j=mask.stride(-1) if mask is not None else 0,
        H=heads,
        CH=ch,
        HAS_MASK=has_mask,
        GEMM_MODE=mode,
    )

    def q_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_M"]), rows * heads)

    def kv_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_N"]), rows * heads)

    def bias_grid(meta):
        return (triton.cdiv(j, meta["BLOCK_M"]), triton.cdiv(j, meta["BLOCK_N"]) * heads)

    args = (q, k, v, o, do, lse, triangle_bias, mask if mask is not None else dummy)
    _flash_tri_attn_bwd_dq_kernel[q_grid](*args, dq, **shared)
    _flash_tri_attn_bwd_dkv_kernel[kv_grid](*args, dk, dv, **shared)
    _flash_tri_attn_bwd_dbias_kernel[bias_grid](*args, dtb, **shared)
    return dq, dk, dv, dtb


def _split_m_dw(x: torch.Tensor, go: torch.Tensor) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    m, k = x.shape
    n = go.shape[1]
    p = torch.empty((_BWD_SPLIT_M, n, k), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (_BWD_SPLIT_M, triton.cdiv(n, meta["BLOCK_N"]))

    _split_m_dw_kernel[grid](
        x.contiguous(),
        go.contiguous(),
        p,
        m,
        K=k,
        N=n,
        GEMM_MODE=_gemm_mode(x),
        SPLIT_M=_BWD_SPLIT_M,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return p.sum(0)


def _linear_autotune_key(x, w, kind: str):
    return (
        _gemm_mode(x),
        kind,
        x.device.type,
        x.device.index,
        x.dtype,
        x.shape[-1],
        w.shape[0],
    )


def _ensure_linear_autotune(x: torch.Tensor, w: torch.Tensor, kind: str) -> None:
    """Time linear tiles on a large dummy ``M`` so a small first call cannot lock."""
    key = _linear_autotune_key(x, w, kind)
    if key in _linear_autotune_ready:
        return
    m = _LINEAR_WARM_M
    k = w.shape[1]
    n = w.shape[0]
    dummy_x = torch.empty((m, k), device=x.device, dtype=x.dtype)
    dummy_w = torch.empty_like(w)
    dummy_y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    mode = _gemm_mode(x)
    block_k = max(_next_power_of_two(k), 16)
    if kind == "fwd":

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

        _linear_fwd_kernel[grid](
            dummy_x, dummy_w, dummy_y, m, K=k, N=n, GEMM_MODE=mode, BLOCK_K=block_k
        )
    else:

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]),)

        _linear_dx_kernel[grid](
            dummy_y, dummy_w, dummy_x, m, K=k, N=n, GEMM_MODE=mode, BLOCK_K=block_k
        )
    _linear_autotune_ready.add(key)


def _linear_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``Y = X @ W^T`` with Triton ``GEMM_MODE``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    x = x if x.is_contiguous() else x.contiguous()
    w = w if w.is_contiguous() else w.contiguous()
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m < _LINEAR_WARM_M:
        _ensure_linear_autotune(x, w, "fwd")

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

    _linear_fwd_kernel[grid](
        x,
        w,
        y,
        m,
        K=k,
        N=n,
        GEMM_MODE=_gemm_mode(x),
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return y


def _linear_dx(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``dX = dY @ W`` with Triton ``GEMM_MODE``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    dy = dy if dy.is_contiguous() else dy.contiguous()
    w = w if w.is_contiguous() else w.contiguous()
    m, n = dy.shape
    k = w.shape[1]
    dx = torch.empty((m, k), device=dy.device, dtype=dy.dtype)
    if m < _LINEAR_WARM_M:
        _ensure_linear_autotune(dy, w, "dx")

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]),)

    _linear_dx_kernel[grid](
        dy,
        w,
        dx,
        m,
        K=k,
        N=n,
        GEMM_MODE=_gemm_mode(dy),
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return dx


def _ln_apply(
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
) -> torch.Tensor:
    """Apply saved LN stats: ``(x-mean)*rstd*γ[+β]``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    x = x if x.is_contiguous() else x.contiguous()
    m, k = x.shape
    y = torch.empty_like(x)
    dummy = x.new_empty(1)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]),)

    _ln_apply_kernel[grid](
        x,
        y,
        mean,
        rstd,
        gamma,
        beta if beta is not None else dummy,
        m,
        K=k,
        HAS_LN_BIAS=beta is not None,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    return y


def _ln_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
):
    """LN backward with exclusive partial ``dγ`` / ``dβ`` (no atomics)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    x = x if x.is_contiguous() else x.contiguous()
    dy = dy if dy.is_contiguous() else dy.contiguous()
    m, k = x.shape
    dx = torch.zeros_like(x)
    n_prog = triton.cdiv(m, min(t[0] for t in _LN_TILES))
    p_gamma = torch.zeros((n_prog, k), dtype=torch.float32, device=x.device)
    p_beta = (
        torch.zeros_like(p_gamma)
        if beta is not None
        else x.new_empty(1)
    )
    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]),)

    _ln_bwd_kernel[grid](
        x,
        dy,
        dx,
        mean,
        rstd,
        gamma,
        p_gamma,
        p_beta,
        m,
        K=k,
        HAS_LN_BIAS=beta is not None,
        BLOCK_K=max(_next_power_of_two(k), 16),
    )
    d_gamma = p_gamma.sum(0).to(dtype=gamma.dtype)
    d_beta = None if beta is None else p_beta.sum(0).to(dtype=beta.dtype)
    return dx, d_gamma, d_beta


def _gate_bwd(o: torch.Tensor, g: torch.Tensor, d_attn: torch.Tensor):
    """``sig = σ(g)``; write gated ``O``, ``dO``, ``dG``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_tri_attn")
    o = o if o.is_contiguous() else o.contiguous()
    g = g if g.is_contiguous() else g.contiguous()
    d_attn = d_attn if d_attn.is_contiguous() else d_attn.contiguous()
    m, c = o.reshape(o.shape[0], -1).shape
    o2 = o.reshape(m, c)
    g2 = g.reshape(m, c)
    da2 = d_attn.reshape(m, c)
    d_o = torch.empty_like(o2)
    d_g = torch.empty_like(g2)
    gated = torch.empty_like(o2)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]),)

    _gate_bwd_kernel[grid](
        o2,
        g2,
        da2,
        d_o,
        d_g,
        gated,
        m,
        C=c,
        BLOCK_K=max(_next_power_of_two(c), 16),
        GEMM_MODE=_gemm_mode(o),
    )
    return d_o, d_g, gated


def _row_block(n_i: int, chunk_size: int | None) -> int:
    if chunk_size is None:
        return min(n_i, _DEFAULT_ROW_BLOCK)
    return max(1, min(n_i, int(chunk_size)))


def _project_qkvg(z_norm: torch.Tensor, w_qkvg: torch.Tensor, heads: int, ch: int):
    rows, j, c_in = z_norm.shape
    qkvg = _linear_fwd(z_norm.reshape(-1, c_in), w_qkvg)
    qkvg = qkvg.view(rows, j, 4, heads, ch).permute(2, 0, 1, 3, 4).contiguous()
    return qkvg[0], qkvg[1], qkvg[2], qkvg[3]


def _project_block_qkvg(
    z_block: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    w_qkvg: torch.Tensor,
    heads: int,
    ch: int,
    eps: float,
):
    """LN → packed QKVG for one I-row block. Reuses 3a when eligible."""
    if not z_block.is_contiguous():
        z_block = z_block.contiguous()
    rows, j, c_in = z_block.shape[1], z_block.shape[2], z_block.shape[3]
    x_2d = z_block.reshape(-1, c_in)
    used_fused = False
    try:
        from openfold3.core.kernels.triton.fused_ln_linear import (
            _launch_fused_ln_linear,
            is_fused_ln_linear_eligible,
        )

        if is_fused_ln_linear_eligible(z_block, ln_w, ln_b, w_qkvg, None):
            qkvg, _, _ = _launch_fused_ln_linear(x_2d, ln_w, ln_b, w_qkvg, None, eps)
            used_fused = True
    except ImportError:  # pragma: no cover
        pass
    if not used_fused:
        ln_w_d, ln_b_d, w_d = _downcast_masters(z_block.dtype, ln_w, ln_b, w_qkvg)
        qkvg = F.linear(F.layer_norm(x_2d, (c_in,), ln_w_d, ln_b_d, eps), w_d)
    qkvg = qkvg.view(rows, j, 4, heads, ch)
    return qkvg[:, :, 0], qkvg[:, :, 1], qkvg[:, :, 2], qkvg[:, :, 3]


def _write_wo(
    attn: torch.Tensor,
    wo: torch.Tensor,
    residual_wv: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    rows, j, heads, ch = attn.shape
    attn_2d = attn.reshape(-1, heads * ch)
    wo_t = wo.t()
    if residual_wv is None:
        return torch.mm(attn_2d, wo_t).view(1, rows, j, wo.shape[0])
    if residual_wv.is_contiguous():
        out_2d = residual_wv[0, start:end].reshape(-1, wo.shape[0])
        torch.addmm(out_2d, attn_2d, wo_t, out=out_2d)
        return None
    scratch = torch.mm(attn_2d, wo_t).view(1, rows, j, wo.shape[0])
    residual_wv[:, start:end].add_(scratch)
    return None


def _launch_fused_tri_attn(
    z,
    mask,
    ln_w,
    ln_b,
    wz,
    wq,
    wk,
    wv,
    wg,
    wo,
    *,
    c_hidden: int,
    no_heads: int,
    starting: bool,
    inf: float,
    eps: float,
    residual: torch.Tensor | None,
    chunk_size: int | None,
    return_acts: bool,
):
    if not starting:
        z = z.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        if residual is not None:
            residual = residual.transpose(-2, -3)
    if not z.is_contiguous():
        z = z.contiguous()
    _, n_i, n_j, c_in = z.shape
    heads, ch = no_heads, c_hidden
    scale = 1.0 / math.sqrt(ch)
    write_stats = return_acts
    tb, mean, rstd = _pair_ln_linear(z, ln_w, ln_b, wz, eps, write_stats=write_stats)
    w_qkvg = torch.cat([wq, wk, wv, wg], dim=0)
    wo_d = _downcast_masters(z.dtype, wo)[0]
    residual_wv = residual
    if residual_wv is not None:
        out = residual_wv
    else:
        out = torch.empty((1, n_i, n_j, c_in), device=z.device, dtype=z.dtype)
    block = _row_block(n_i, chunk_size)
    mask_c = None
    if mask is not None:
        mask_c = mask if mask.is_contiguous() else mask.contiguous()
        if mask_c.dim() == 3:
            mask_c = mask_c[0]
    lse_all = []
    o_all = []
    for start in range(0, n_i, block):
        end = min(n_i, start + block)
        q, k, v, g = _project_block_qkvg(
            z[:, start:end], ln_w, ln_b, w_qkvg, heads, ch, eps
        )
        o, lse = _flash_fwd(
            q,
            k,
            v,
            g,
            tb,
            mask_c,
            start,
            scale,
            inf,
            write_lse=return_acts,
            has_gate=not return_acts,
            out=q,
        )
        if return_acts:
            gated = o * torch.sigmoid(g)
            lse_all.append(lse)
            # ``o`` aliases the packed QKVG view when ``out=q``; pack so the
            # 4-wide projection can be freed after the block.
            o_all.append(o.contiguous())
            written = _write_wo(gated, wo_d, residual_wv, start, end)
        else:
            written = _write_wo(o, wo_d, residual_wv, start, end)
        if residual_wv is None and written is not None:
            out[:, start:end] = written
        del q, k, v, g, o
    del w_qkvg
    if residual is not None:
        y = residual if starting else residual.transpose(-2, -3)
    else:
        y = out if starting else out.transpose(-2, -3)
    if not return_acts:
        return y, None, None, None, None, None
    return y, mean, rstd, torch.cat(lse_all, 0), torch.cat(o_all, 0), tb


def _fused_tri_attn_backward(
    z,
    mask,
    grad_out,
    ln_w,
    ln_b,
    wz,
    wq,
    wk,
    wv,
    wg,
    wo,
    mean,
    rstd,
    lse,
    o_ungated,
    tb,
    *,
    c_hidden: int,
    no_heads: int,
    starting: bool,
    inf: float,
    has_residual: bool,
):
    if not starting:
        z = z.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        grad_out = grad_out.transpose(-2, -3)
    if not z.is_contiguous():
        z = z.contiguous()
    go = grad_out.contiguous()
    _, n_i, n_j, c_in = z.shape
    heads, ch = no_heads, c_hidden
    scale = 1.0 / math.sqrt(ch)
    ln_w_d, ln_b_d, wz_d, wq, wk, wv, wg, wo = _downcast_masters(
        z.dtype, ln_w, ln_b, wz, wq, wk, wv, wg, wo
    )
    z_hat = _ln_apply(z.reshape(-1, c_in), mean, rstd, ln_w_d, ln_b_d)
    z_hat = z_hat.view(1, n_i, n_j, c_in)
    w_qkvg = torch.cat([wq, wk, wv, wg], dim=0)
    mask_c = None
    if mask is not None:
        mask_c = mask if mask.is_contiguous() else mask.contiguous()
        if mask_c.dim() == 3:
            mask_c = mask_c[0]
    d_z_hat = torch.zeros_like(z_hat)
    d_qkvg_w = torch.zeros_like(w_qkvg, dtype=torch.float32)
    d_wo = torch.zeros_like(wo, dtype=torch.float32)
    d_tb = torch.zeros((heads, n_j, n_j), device=z.device, dtype=torch.float32)
    block = _DEFAULT_ROW_BLOCK
    for start in range(0, n_i, block):
        end = min(n_i, start + block)
        rows = end - start
        zn = z_hat[0, start:end]
        q, k, v, g = _project_qkvg(zn, w_qkvg, heads, ch)
        o = o_ungated[start:end]
        go_block = go[0, start:end].reshape(rows * n_j, c_in)
        d_attn = _linear_dx(go_block, wo)
        d_o, d_g_pre, attn_gated = _gate_bwd(
            o.reshape(rows * n_j, heads * ch),
            g.reshape(rows * n_j, heads * ch),
            d_attn,
        )
        d_wo += _split_m_dw(attn_gated, go_block)
        dq, dk, dv, dtb = _flash_bwd(
            q,
            k,
            v,
            o,
            d_o.view(rows, n_j, heads, ch),
            lse[start:end],
            tb,
            mask_c,
            start,
            scale,
            inf,
        )
        d_tb += dtb
        d_qkvg = torch.stack(
            [dq, dk, dv, d_g_pre.view(rows, n_j, heads, ch)], dim=2
        ).reshape(rows * n_j, 4 * heads * ch)
        d_qkvg_w += _split_m_dw(zn.reshape(rows * n_j, c_in), d_qkvg)
        d_zn = _linear_dx(d_qkvg, w_qkvg)
        d_z_hat[0, start:end] += d_zn.view(rows, n_j, c_in)
        del q, k, v, g, o, dq, dk, dv, d_qkvg, d_zn
    d_tb_flat = d_tb.permute(1, 2, 0).reshape(-1, heads).to(dtype=z.dtype)
    d_wz = _split_m_dw(z_hat.reshape(-1, c_in), d_tb_flat)
    d_z_hat = d_z_hat + _linear_dx(d_tb_flat, wz_d).view(1, n_i, n_j, c_in)
    d_z, d_ln_w, d_ln_b = _ln_bwd(
        z.reshape(-1, c_in),
        d_z_hat.reshape(-1, c_in),
        mean,
        rstd,
        ln_w_d,
        ln_b_d,
    )
    d_z = d_z.view(1, n_i, n_j, c_in)
    if has_residual:
        d_z = d_z + go
    if not starting:
        d_z = d_z.transpose(-2, -3)
    d_wq, d_wk, d_wv, d_wg = d_qkvg_w.view(4, heads * ch, c_in).unbind(0)
    return (
        d_z,
        d_ln_w.to(dtype=ln_w.dtype),
        d_ln_b.to(dtype=ln_b.dtype) if d_ln_b is not None else None,
        d_wz.to(dtype=wz.dtype),
        d_wq.to(dtype=wq.dtype),
        d_wk.to(dtype=wk.dtype),
        d_wv.to(dtype=wv.dtype),
        d_wg.to(dtype=wg.dtype),
        d_wo.to(dtype=wo.dtype),
    )


class _FusedTriAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        mask,
        ln_w,
        ln_b,
        wz,
        wq,
        wk,
        wv,
        wg,
        wo,
        c_hidden,
        no_heads,
        starting,
        inf,
        eps,
        has_residual,
    ):
        residual = z if has_residual else None
        y, mean, rstd, lse, o_ungated, tb = _launch_fused_tri_attn(
            z,
            mask,
            ln_w,
            ln_b,
            wz,
            wq,
            wk,
            wv,
            wg,
            wo,
            c_hidden=int(c_hidden),
            no_heads=int(no_heads),
            starting=bool(starting),
            inf=float(inf),
            eps=float(eps),
            residual=residual,
            chunk_size=None,
            return_acts=True,
        )
        ctx.save_for_backward(
            z,
            mask if mask is not None else z.new_empty(0),
            ln_w,
            ln_b if ln_b is not None else z.new_empty(0),
            wz,
            wq,
            wk,
            wv,
            wg,
            wo,
            mean,
            rstd,
            lse,
            o_ungated,
            tb,
        )
        ctx.c_hidden = int(c_hidden)
        ctx.no_heads = int(no_heads)
        ctx.starting = bool(starting)
        ctx.inf = float(inf)
        ctx.has_mask = mask is not None
        ctx.has_ln_b = ln_b is not None
        ctx.has_residual = bool(has_residual)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        (
            z,
            mask_t,
            ln_w,
            ln_b_t,
            wz,
            wq,
            wk,
            wv,
            wg,
            wo,
            mean,
            rstd,
            lse,
            o_ungated,
            tb,
        ) = ctx.saved_tensors
        grads = _fused_tri_attn_backward(
            z,
            mask_t if ctx.has_mask else None,
            grad_out,
            ln_w,
            ln_b_t if ctx.has_ln_b else None,
            wz,
            wq,
            wk,
            wv,
            wg,
            wo,
            mean,
            rstd,
            lse,
            o_ungated,
            tb,
            c_hidden=ctx.c_hidden,
            no_heads=ctx.no_heads,
            starting=ctx.starting,
            inf=ctx.inf,
            has_residual=ctx.has_residual,
        )
        need = ctx.needs_input_grad
        packed = (
            grads[0] if need[0] else None,
            None,
            grads[1] if need[2] else None,
            grads[2] if need[3] and ctx.has_ln_b else None,
            grads[3] if need[4] else None,
            grads[4] if need[5] else None,
            grads[5] if need[6] else None,
            grads[6] if need[7] else None,
            grads[7] if need[8] else None,
            grads[8] if need[9] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return packed


def fused_tri_attn(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor | None,
    wz: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    *,
    c_hidden: int,
    no_heads: int,
    starting: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Fused AF3 triangle attention. Callers must check eligibility first."""
    if not is_fused_tri_attn_eligible(
        z, ln_w, ln_b, wz, wq, wk, wv, wg, wo, c_hidden=c_hidden, no_heads=no_heads
    ):
        raise RuntimeError(
            "fused_tri_attn requires an eligible Triton launch; "
            "use eager_tri_attn for the fallback path"
        )
    use_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (z, ln_w, ln_b, wz, wq, wk, wv, wg, wo, residual)
    )
    if use_grad:
        if residual is not None and residual.data_ptr() != z.data_ptr():
            raise RuntimeError("fused_tri_attn training residual must be z or None")
        return _FusedTriAttnFn.apply(
            z,
            mask,
            ln_w,
            ln_b,
            wz,
            wq,
            wk,
            wv,
            wg,
            wo,
            int(c_hidden),
            int(no_heads),
            bool(starting),
            float(inf),
            float(eps),
            residual is not None,
        )
    y, *_ = _launch_fused_tri_attn(
        z,
        mask,
        ln_w,
        ln_b,
        wz,
        wq,
        wk,
        wv,
        wg,
        wo,
        c_hidden=c_hidden,
        no_heads=no_heads,
        starting=starting,
        inf=inf,
        eps=eps,
        residual=residual,
        chunk_size=chunk_size,
        return_acts=False,
    )
    return y


def fused_tri_attn_from_module(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    residual: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor | None:
    """Adapt a ``TriangleAttention`` module; ``None`` if ineligible."""
    if not hasattr(module, "linear_z") or not hasattr(module, "mha"):
        return None
    mha = module.mha
    if mha.linear_g is None:
        return None
    linears = (
        module.linear_z,
        mha.linear_q,
        mha.linear_k,
        mha.linear_v,
        mha.linear_g,
        mha.linear_o,
    )
    if any(lin.bias is not None for lin in linears):
        return None
    weights = (
        module.layer_norm.weight,
        module.layer_norm.bias,
        module.linear_z.weight,
        mha.linear_q.weight,
        mha.linear_k.weight,
        mha.linear_v.weight,
        mha.linear_g.weight,
        mha.linear_o.weight,
    )
    if not is_fused_tri_attn_eligible(
        z,
        *weights,
        c_hidden=module.c_hidden,
        no_heads=module.no_heads,
    ):
        return None
    return fused_tri_attn(
        z,
        mask,
        *weights,
        c_hidden=module.c_hidden,
        no_heads=module.no_heads,
        starting=module.starting,
        inf=module.inf,
        eps=module.layer_norm.eps,
        residual=residual,
        chunk_size=chunk_size,
    )
