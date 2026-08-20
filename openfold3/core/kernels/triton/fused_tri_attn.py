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
# reconstructed from the original inference kernel.

"""Fused triangle attention (Triton).

LN → pair-bias Linear → gated MHA → ``W_o`` [+ residual]. ``I`` / ``J`` /
sequence length are not autotune or specialize keys. GEMM tiles autotune on
``GEMM_MODE`` only so each GPU picks its own winner. The pair-bias prologue
decodes ``(i, j)`` from independent strides so contig starting-node ``z`` and
PairBlock's transposed ending-node view share one kernel (no 1U ``z_norm``).
Attention is flash / online-softmax: the ``[I, H, J, J]`` score matrix is
never stored. The forward may group two pair-rows per program so they share
one triangle-bias load. In-place residual writes are inference-only.
Ineligible shapes use the matching-precision eager primitives.
"""

from __future__ import annotations

import math
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

_MAX_C = 128
_MIN_M = 4096
_DEFAULT_ROW_BLOCK = 128
# First launch below these sizes warms autotune on a large dummy grid so a
# test-sized call cannot lock the GEMM_MODE winner (same contract as trimul).
_FLASH_WARM_J = 1024
_PAIR_LN_WARM_M = 65536
_TRUE = {"1", "true", "yes", "on"}
_SUPPORTED_CH = (16, 32, 64, 128)
_flash_autotune_ready: set[tuple] = set()
_pair_ln_autotune_ready: set[tuple] = set()

# Large-N biased; J is not an autotune key. ``I_TILE`` groups pair-rows that
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
_PAIR_LN_TILES = (
    (32, 16, 4, 1),
    (64, 16, 4, 1),
    (64, 16, 8, 1),
    (128, 16, 4, 1),
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
        and not torch.is_grad_enabled()
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

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_tri_attn")

    _pair_ln_linear_fwd_kernel = _unavailable
    _flash_tri_attn_fwd_kernel = _unavailable


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


def _row_block(n_i: int, chunk_size: int | None) -> int:
    if chunk_size is None:
        return min(n_i, _DEFAULT_ROW_BLOCK)
    return max(1, min(n_i, int(chunk_size)))


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
    write_stats = False
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
            write_lse=False,
            has_gate=True,
            out=q,
        )
        written = _write_wo(o, wo_d, residual_wv, start, end)
        if residual_wv is None and written is not None:
            out[:, start:end] = written
        del q, k, v, g, o
    del w_qkvg
    if residual is not None:
        y = residual if starting else residual.transpose(-2, -3)
    else:
        y = out if starting else out.transpose(-2, -3)
    return y


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
    return _launch_fused_tri_attn(
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
    )


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
