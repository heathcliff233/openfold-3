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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused diffusion attention
# reconstructed from the original inference flash kernel, with training
# backward.

"""Fused diffusion token attention (Triton).

Flash / online-softmax over ``[B, S, H, N_Q, C]`` Q × ``[B, S, H, N_K, C]``
K/V with an additive pair bias and mask. ``N_Q`` / ``N_K`` / ``S`` /
strides are not autotune or specialize keys. The ``[B, S, H, N_Q, N_K]``
score matrix is never stored.

One program owns a Q-tile of one ``(b, s, h)`` (FlashAttention-2
partitioning): ``Q`` and the online-softmax state stay in SRAM for the
K-loop. Pair bias is the one-shot ``LN_z → Linear_z`` output
(``[B, 1, H, N, N]``). Training writes exclusive ``dBias`` — no atomics.

AdaLN stays eager. Inference packs ``QKVG`` in one GEMM and applies
gate ⊙ ``W_o`` in place; training keeps separate projections so
autograd hits the fp32 Parameter masters. Ineligible shapes use the
matching-precision eager einsum.
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

_TRUE = {"1", "true", "yes", "on"}
_MAX_CH = 128
_MAX_CZ = 128
_MIN_BLOCK_C = 16
_DEFAULT_MIN_TOKENS = 0


def is_fused_diffusion_attn_enabled() -> bool:
    return (
        os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN", "1").strip().lower() in _TRUE
    )


def is_diffusion_pair_bias_cache_enabled() -> bool:
    return os.environ.get(
        "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE", "0"
    ).strip().lower() in (_TRUE)


def fused_diffusion_attn_min_tokens() -> int:
    raw = os.environ.get(
        "OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", str(_DEFAULT_MIN_TOKENS)
    )
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MIN_TOKENS


def _gemm_mode(act: torch.Tensor) -> str:
    if act.dtype == torch.bfloat16:
        return "bf16"
    if torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _block_c(ch: int) -> int:
    # tl.arange requires a power of two; CH=48 pads to 64 and is masked.
    return max(_MIN_BLOCK_C, 1 << (ch - 1).bit_length()) if ch > 1 else 1


def is_fused_diffusion_attn_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
) -> bool:
    if q.ndim != 5 or k.ndim != 5 or v.shape != k.shape:
        return False
    if q.shape[:3] != k.shape[:3] or q.shape[-1] != k.shape[-1]:
        return False
    bsz, samples, heads, n_q, ch = q.shape
    n_k = k.shape[-2]
    pb_s = pair_bias.shape[1]
    mb_s = mask_bias.shape[1]
    return (
        is_fused_diffusion_attn_enabled()
        and _TRITON_AVAILABLE
        and q.is_cuda
        and q.dtype in (torch.float32, torch.bfloat16)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and pair_bias.is_cuda
        and mask_bias.is_cuda
        and pair_bias.shape[0] == bsz
        and mask_bias.shape[0] == bsz
        and pb_s in (1, samples)
        and mb_s in (1, samples)
        and pair_bias.shape[-3:] == (heads, n_q, n_k)
        and mask_bias.shape[-3:] == (1, 1, n_k)
        and ch <= _MAX_CH
        and _block_c(ch) <= _MAX_CH
        and n_q >= fused_diffusion_attn_min_tokens()
        and n_k >= fused_diffusion_attn_min_tokens()
    )


def can_use_fused_diffusion_attention(
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    biases: list[torch.Tensor] | None,
    no_heads: int,
) -> bool:
    """Eligibility for ``Attention.forward`` (pre-QKV)."""
    if (
        not is_fused_diffusion_attn_enabled()
        or not _TRITON_AVAILABLE
        or not q_x.is_cuda
        or kv_x is not q_x
        or biases is None
        or len(biases) != 2
        or q_x.ndim != 4
        or q_x.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    bsz, samples, n_tok, c_a = q_x.shape
    if c_a % no_heads != 0:
        return False
    ch = c_a // no_heads
    if ch > _MAX_CH or _block_c(ch) > _MAX_CH:
        return False
    if n_tok < fused_diffusion_attn_min_tokens():
        return False
    mask_bias, pair_bias = biases
    if mask_bias.ndim != 5 or pair_bias.ndim != 5:
        return False
    return (
        mask_bias.shape[0] == bsz
        and pair_bias.shape[0] == bsz
        and mask_bias.shape[1] in (1, samples)
        and pair_bias.shape[1] in (1, samples)
        and mask_bias.shape[-3:] == (1, 1, n_tok)
        and pair_bias.shape[-3:] == (no_heads, n_tok, n_tok)
    )


def eager_diffusion_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Matching-precision eager QK + pair/mask + softmax + PV."""
    scores = torch.einsum("bshqc,bshkc->bshqk", q, k) * softmax_scale
    scores = scores + pair_bias + mask_bias
    if q.dtype == torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            probs = torch.nn.functional.softmax(scores, dim=-1)
    else:
        probs = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bshqk,bshkc->bshqc", probs.to(dtype=v.dtype), v)


if _TRITON_AVAILABLE:
    RCP_LN2 = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _round_to_tf32(x):
        ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
        return tl.inline_asm_elementwise(
            ASM, "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _dot_f32(a, b, GEMM_MODE: tl.constexpr):
        if GEMM_MODE == "bf16":
            # Native Tensor-Core MMA. ``input_precision="ieee"`` forces the
            # slow IEEE path and is why fused bf16 did not scale vs SDPA.
            return tl.dot(
                a.to(tl.bfloat16),
                b.to(tl.bfloat16),
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

    def _fwd_tiles(mode: str):
        # Measured RTX 4090 table. Large-N autotune lock-in loses S=1
        # short sequences; keep one compile per GEMM_MODE.
        if mode == "bf16":
            return 128, 64, 4, 2
        return 64, 32, 4, 1

    def _bwd_dq_tiles(mode: str):
        if mode == "bf16":
            return 64, 32, 4, 2
        return 64, 16, 4, 1

    def _bwd_dkv_tiles(mode: str):
        if mode == "bf16":
            return 64, 32, 4, 2
        return 32, 16, 4, 1

    @triton.jit(
        do_not_specialize=[
            "N_Q",
            "S_dim",
            "stride_ob",
            "stride_os",
            "stride_oh",
            "stride_on",
            "stride_oc",
            "stride_dob",
            "stride_dos",
            "stride_doh",
            "stride_don",
            "stride_doc",
            "stride_db",
            "stride_ds",
            "stride_dh",
            "stride_dn",
        ],
        do_not_specialize_on_alignment=["O_ptr", "DO_ptr", "D_ptr"],
    )
    def _attn_delta_kernel(
        O_ptr,
        DO_ptr,
        D_ptr,
        N_Q,
        S_dim,
        stride_ob,
        stride_os,
        stride_oh,
        stride_on,
        stride_oc,
        stride_dob,
        stride_dos,
        stride_doh,
        stride_don,
        stride_doc,
        stride_db,
        stride_ds,
        stride_dh,
        stride_dn,
        H: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """``delta = (dO * O).sum(-1)`` in fp32 without an ``[B,S,H,N,C]`` product."""
        pid = tl.program_id(0)
        n = pid % N_Q
        t = pid // N_Q
        h = t % H
        t = t // H
        s = t % S_dim
        b = t // S_dim
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < CH
        o = tl.load(
            O_ptr
            + b * stride_ob
            + s * stride_os
            + h * stride_oh
            + n * stride_on
            + offs_c * stride_oc,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            DO_ptr
            + b * stride_dob
            + s * stride_dos
            + h * stride_doh
            + n * stride_don
            + offs_c * stride_doc,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            D_ptr + b * stride_db + s * stride_ds + h * stride_dh + n * stride_dn,
            tl.sum(o * do, 0),
        )

    @triton.jit(
        do_not_specialize=[
            "N_Q",
            "N_K",
            "S_dim",
            "softmax_scale",
            "stride_qb",
            "stride_qs",
            "stride_qh",
            "stride_qn",
            "stride_qc",
            "stride_kb",
            "stride_ks",
            "stride_kh",
            "stride_kn",
            "stride_kc",
            "stride_vb",
            "stride_vs",
            "stride_vh",
            "stride_vn",
            "stride_vc",
            "stride_ob",
            "stride_os",
            "stride_oh",
            "stride_on",
            "stride_oc",
            "stride_pb_b",
            "stride_pb_s",
            "stride_pb_h",
            "stride_pb_q",
            "stride_pb_k",
            "stride_mb_b",
            "stride_mb_s",
            "stride_mb_k",
            "stride_lse_b",
            "stride_lse_s",
            "stride_lse_h",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "PB_ptr",
            "MB_ptr",
            "O_ptr",
            "LSE_ptr",
        ],
    )
    def _flash_diffusion_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        PB_ptr,
        MB_ptr,
        O_ptr,
        LSE_ptr,
        N_Q,
        N_K,
        S_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qn,
        stride_qc,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kn,
        stride_kc,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vn,
        stride_vc,
        stride_ob,
        stride_os,
        stride_oh,
        stride_on,
        stride_oc,
        stride_pb_b,
        stride_pb_s,
        stride_pb_h,
        stride_pb_q,
        stride_pb_k,
        stride_mb_b,
        stride_mb_s,
        stride_mb_k,
        stride_lse_b,
        stride_lse_s,
        stride_lse_h,
        H: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_C: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One ``(b, s, h)`` Q-tile. ``Q`` / ``m`` / ``ℓ`` / ``O`` stay in SRAM."""
        pid_m = tl.program_id(0)
        pid_bsh = tl.program_id(1)
        h = pid_bsh % H
        s = (pid_bsh // H) % S_dim
        b = (pid_bsh // H) // S_dim
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        mask_m = offs_m < N_Q
        mask_c = offs_c < CH
        rcp = RCP_LN2
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
        q = tl.load(
            Q_ptr
            + b * stride_qb
            + s * stride_qs
            + h * stride_qh
            + offs_m[:, None] * stride_qn
            + offs_c[None, :] * stride_qc,
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        )
        k_base = b * stride_kb + s * stride_ks + h * stride_kh
        v_base = b * stride_vb + s * stride_vs + h * stride_vh
        o_base = b * stride_ob + s * stride_os + h * stride_oh
        pb_base = b * stride_pb_b + s * stride_pb_s + h * stride_pb_h
        mb_base = b * stride_mb_b + s * stride_mb_s
        for n0 in tl.range(0, N_K, BLOCK_N, num_stages=2):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_K
            k = tl.load(
                K_ptr
                + k_base
                + offs_n[:, None] * stride_kn
                + offs_c[None, :] * stride_kc,
                mask=mask_n[:, None] & mask_c[None, :],
                other=0.0,
            )
            qk = _dot_f32(q, tl.trans(k), GEMM_MODE) * (softmax_scale * rcp)
            pb = tl.load(
                PB_ptr
                + pb_base
                + offs_m[:, None] * stride_pb_q
                + offs_n[None, :] * stride_pb_k,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            mb = tl.load(
                MB_ptr + mb_base + offs_n * stride_mb_k, mask=mask_n, other=0.0
            )
            qk = qk + (pb + mb.to(tl.float32)[None, :]) * rcp
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            m_i = m_ij
            v = tl.load(
                V_ptr
                + v_base
                + offs_n[:, None] * stride_vn
                + offs_c[None, :] * stride_vc,
                mask=mask_n[:, None] & mask_c[None, :],
                other=0.0,
            )
            acc = acc + _dot_f32(p, v, GEMM_MODE)
        acc = acc / l_i[:, None]
        if WRITE_LSE:
            tl.store(
                LSE_ptr
                + b * stride_lse_b
                + s * stride_lse_s
                + h * stride_lse_h
                + offs_m,
                m_i + tl.math.log2(l_i),
                mask=mask_m,
            )
        tl.store(
            O_ptr + o_base + offs_m[:, None] * stride_on + offs_c[None, :] * stride_oc,
            acc.to(O_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_c[None, :],
        )

    @triton.jit(
        do_not_specialize=[
            "N_Q",
            "N_K",
            "S_dim",
            "softmax_scale",
            "stride_qb",
            "stride_qs",
            "stride_qh",
            "stride_qn",
            "stride_qc",
            "stride_kb",
            "stride_ks",
            "stride_kh",
            "stride_kn",
            "stride_kc",
            "stride_vb",
            "stride_vs",
            "stride_vh",
            "stride_vn",
            "stride_vc",
            "stride_dob",
            "stride_dos",
            "stride_doh",
            "stride_don",
            "stride_doc",
            "stride_dqb",
            "stride_dqs",
            "stride_dqh",
            "stride_dqn",
            "stride_dqc",
            "stride_pb_b",
            "stride_pb_s",
            "stride_pb_h",
            "stride_pb_q",
            "stride_pb_k",
            "stride_mb_b",
            "stride_mb_s",
            "stride_mb_k",
            "stride_dpb_b",
            "stride_dpb_s",
            "stride_dpb_h",
            "stride_dpb_q",
            "stride_dpb_k",
            "stride_lse_b",
            "stride_lse_s",
            "stride_lse_h",
            "stride_del_b",
            "stride_del_s",
            "stride_del_h",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "DO_ptr",
            "LSE_ptr",
            "Delta_ptr",
            "PB_ptr",
            "MB_ptr",
            "DQ_ptr",
            "DPB_ptr",
        ],
    )
    def _flash_diffusion_attn_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        LSE_ptr,
        Delta_ptr,
        PB_ptr,
        MB_ptr,
        DQ_ptr,
        DPB_ptr,
        N_Q,
        N_K,
        S_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qn,
        stride_qc,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kn,
        stride_kc,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vn,
        stride_vc,
        stride_dob,
        stride_dos,
        stride_doh,
        stride_don,
        stride_doc,
        stride_dqb,
        stride_dqs,
        stride_dqh,
        stride_dqn,
        stride_dqc,
        stride_pb_b,
        stride_pb_s,
        stride_pb_h,
        stride_pb_q,
        stride_pb_k,
        stride_mb_b,
        stride_mb_s,
        stride_mb_k,
        stride_dpb_b,
        stride_dpb_s,
        stride_dpb_h,
        stride_dpb_q,
        stride_dpb_k,
        stride_lse_b,
        stride_lse_s,
        stride_lse_h,
        stride_del_b,
        stride_del_s,
        stride_del_h,
        H: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_C: tl.constexpr,
        WRITE_DBIAS: tl.constexpr,
        ACCUM_DBIAS: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exclusive ``dQ``. Pair path may write ``dBias``."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        h = pid_bh % H
        b = pid_bh // H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        mask_m = offs_m < N_Q
        mask_c = offs_c < CH
        rcp = RCP_LN2
        for s in range(0, S_dim):
            q_base = b * stride_qb + s * stride_qs + h * stride_qh
            k_base = b * stride_kb + s * stride_ks + h * stride_kh
            v_base = b * stride_vb + s * stride_vs + h * stride_vh
            do_base = b * stride_dob + s * stride_dos + h * stride_doh
            dq_base = b * stride_dqb + s * stride_dqs + h * stride_dqh
            pb_base = b * stride_pb_b + s * stride_pb_s + h * stride_pb_h
            mb_base = b * stride_mb_b + s * stride_mb_s
            lse_base = b * stride_lse_b + s * stride_lse_s + h * stride_lse_h
            del_base = b * stride_del_b + s * stride_del_s + h * stride_del_h
            q = tl.load(
                Q_ptr
                + q_base
                + offs_m[:, None] * stride_qn
                + offs_c[None, :] * stride_qc,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            )
            do = tl.load(
                DO_ptr
                + do_base
                + offs_m[:, None] * stride_don
                + offs_c[None, :] * stride_doc,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(LSE_ptr + lse_base + offs_m, mask=mask_m, other=0.0)
            delta = tl.load(Delta_ptr + del_base + offs_m, mask=mask_m, other=0.0)
            dq = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
            for n0 in tl.range(0, N_K, BLOCK_N, num_stages=2):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_K
                tile = mask_m[:, None] & mask_n[None, :]
                k = tl.load(
                    K_ptr
                    + k_base
                    + offs_n[:, None] * stride_kn
                    + offs_c[None, :] * stride_kc,
                    mask=mask_n[:, None] & mask_c[None, :],
                    other=0.0,
                )
                k_scaled = k * softmax_scale
                v = tl.load(
                    V_ptr
                    + v_base
                    + offs_n[:, None] * stride_vn
                    + offs_c[None, :] * stride_vc,
                    mask=mask_n[:, None] & mask_c[None, :],
                    other=0.0,
                )
                pb = tl.load(
                    PB_ptr
                    + pb_base
                    + offs_m[:, None] * stride_pb_q
                    + offs_n[None, :] * stride_pb_k,
                    mask=tile,
                    other=0.0,
                ).to(tl.float32)
                mb = tl.load(
                    MB_ptr + mb_base + offs_n * stride_mb_k, mask=mask_n, other=0.0
                )
                qk = _dot_f32(q, tl.trans(k_scaled), GEMM_MODE) * rcp
                qk = qk + (pb + mb.to(tl.float32)[None, :]) * rcp
                qk = tl.where(tile, qk, float("-inf"))
                p = tl.math.exp2(qk - lse[:, None])
                ds = p * (_dot_f32(do, tl.trans(v), GEMM_MODE) - delta[:, None])
                ds = tl.where(tile, ds, 0.0)
                dq += _dot_f32(ds, k_scaled, GEMM_MODE)
                if WRITE_DBIAS:
                    dpb_ptr = (
                        DPB_ptr
                        + b * stride_dpb_b
                        + s * stride_dpb_s
                        + h * stride_dpb_h
                        + offs_m[:, None] * stride_dpb_q
                        + offs_n[None, :] * stride_dpb_k
                    )
                    if ACCUM_DBIAS:
                        prev = tl.load(dpb_ptr, mask=tile, other=0.0)
                        tl.store(dpb_ptr, prev + ds, mask=tile)
                    else:
                        tl.store(dpb_ptr, ds, mask=tile)
            tl.store(
                DQ_ptr
                + dq_base
                + offs_m[:, None] * stride_dqn
                + offs_c[None, :] * stride_dqc,
                dq.to(DQ_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_c[None, :],
            )

    @triton.jit(
        do_not_specialize=[
            "N_Q",
            "N_K",
            "S_dim",
            "softmax_scale",
            "stride_qb",
            "stride_qs",
            "stride_qh",
            "stride_qn",
            "stride_qc",
            "stride_kb",
            "stride_ks",
            "stride_kh",
            "stride_kn",
            "stride_kc",
            "stride_vb",
            "stride_vs",
            "stride_vh",
            "stride_vn",
            "stride_vc",
            "stride_dob",
            "stride_dos",
            "stride_doh",
            "stride_don",
            "stride_doc",
            "stride_dkb",
            "stride_dks",
            "stride_dkh",
            "stride_dkn",
            "stride_dkc",
            "stride_dvb",
            "stride_dvs",
            "stride_dvh",
            "stride_dvn",
            "stride_dvc",
            "stride_pb_b",
            "stride_pb_s",
            "stride_pb_h",
            "stride_pb_q",
            "stride_pb_k",
            "stride_mb_b",
            "stride_mb_s",
            "stride_mb_k",
            "stride_lse_b",
            "stride_lse_s",
            "stride_lse_h",
            "stride_del_b",
            "stride_del_s",
            "stride_del_h",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "DO_ptr",
            "LSE_ptr",
            "Delta_ptr",
            "PB_ptr",
            "MB_ptr",
            "DK_ptr",
            "DV_ptr",
        ],
    )
    def _flash_diffusion_attn_bwd_dkv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        LSE_ptr,
        Delta_ptr,
        PB_ptr,
        MB_ptr,
        DK_ptr,
        DV_ptr,
        N_Q,
        N_K,
        S_dim,
        softmax_scale,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qn,
        stride_qc,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kn,
        stride_kc,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vn,
        stride_vc,
        stride_dob,
        stride_dos,
        stride_doh,
        stride_don,
        stride_doc,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkn,
        stride_dkc,
        stride_dvb,
        stride_dvs,
        stride_dvh,
        stride_dvn,
        stride_dvc,
        stride_pb_b,
        stride_pb_s,
        stride_pb_h,
        stride_pb_q,
        stride_pb_k,
        stride_mb_b,
        stride_mb_s,
        stride_mb_k,
        stride_lse_b,
        stride_lse_s,
        stride_lse_h,
        stride_del_b,
        stride_del_s,
        stride_del_h,
        H: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_C: tl.constexpr,
        GEMM_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exclusive ``dK`` / ``dV`` for one K-tile of one ``(b, s, h)``."""
        pid_n = tl.program_id(0)
        pid_bsh = tl.program_id(1)
        h = pid_bsh % H
        s = (pid_bsh // H) % S_dim
        b = (pid_bsh // H) // S_dim
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)
        mask_n = offs_n < N_K
        mask_c = offs_c < CH
        rcp = RCP_LN2
        q_base = b * stride_qb + s * stride_qs + h * stride_qh
        k_base = b * stride_kb + s * stride_ks + h * stride_kh
        v_base = b * stride_vb + s * stride_vs + h * stride_vh
        do_base = b * stride_dob + s * stride_dos + h * stride_doh
        dk_base = b * stride_dkb + s * stride_dks + h * stride_dkh
        dv_base = b * stride_dvb + s * stride_dvs + h * stride_dvh
        pb_base = b * stride_pb_b + s * stride_pb_s + h * stride_pb_h
        mb_base = b * stride_mb_b + s * stride_mb_s
        lse_base = b * stride_lse_b + s * stride_lse_s + h * stride_lse_h
        del_base = b * stride_del_b + s * stride_del_s + h * stride_del_h
        k = tl.load(
            K_ptr + k_base + offs_n[:, None] * stride_kn + offs_c[None, :] * stride_kc,
            mask=mask_n[:, None] & mask_c[None, :],
            other=0.0,
        )
        v = tl.load(
            V_ptr + v_base + offs_n[:, None] * stride_vn + offs_c[None, :] * stride_vc,
            mask=mask_n[:, None] & mask_c[None, :],
            other=0.0,
        )
        mb = tl.load(MB_ptr + mb_base + offs_n * stride_mb_k, mask=mask_n, other=0.0)
        dk = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
        for m0 in tl.range(0, N_Q, BLOCK_M, num_stages=2):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N_Q
            tile = mask_m[:, None] & mask_n[None, :]
            q = tl.load(
                Q_ptr
                + q_base
                + offs_m[:, None] * stride_qn
                + offs_c[None, :] * stride_qc,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            )
            q_scaled = q * softmax_scale
            do = tl.load(
                DO_ptr
                + do_base
                + offs_m[:, None] * stride_don
                + offs_c[None, :] * stride_doc,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(LSE_ptr + lse_base + offs_m, mask=mask_m, other=0.0)
            delta = tl.load(Delta_ptr + del_base + offs_m, mask=mask_m, other=0.0)
            pb = tl.load(
                PB_ptr
                + pb_base
                + offs_m[:, None] * stride_pb_q
                + offs_n[None, :] * stride_pb_k,
                mask=tile,
                other=0.0,
            ).to(tl.float32)
            qk = _dot_f32(q_scaled, tl.trans(k), GEMM_MODE) * rcp
            qk = qk + (pb + mb.to(tl.float32)[None, :]) * rcp
            qk = tl.where(tile, qk, float("-inf"))
            p = tl.math.exp2(qk - lse[:, None])
            ds = p * (_dot_f32(do, tl.trans(v), GEMM_MODE) - delta[:, None])
            ds = tl.where(tile, ds, 0.0)
            dk += _dot_f32(tl.trans(ds), q_scaled, GEMM_MODE)
            dv += _dot_f32(tl.trans(p), do, GEMM_MODE)
        tl.store(
            DK_ptr
            + dk_base
            + offs_n[:, None] * stride_dkn
            + offs_c[None, :] * stride_dkc,
            dk.to(DK_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_c[None, :],
        )
        tl.store(
            DV_ptr
            + dv_base
            + offs_n[:, None] * stride_dvn
            + offs_c[None, :] * stride_dvc,
            dv.to(DV_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_c[None, :],
        )

else:  # pragma: no cover

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("Triton is not available")

    _flash_diffusion_attn_fwd_kernel = _unavailable
    _flash_diffusion_attn_bwd_dq_kernel = _unavailable
    _flash_diffusion_attn_bwd_dkv_kernel = _unavailable
    _attn_delta_kernel = _unavailable


def _sample_stride(tensor: torch.Tensor, samples: int, dim: int = 1) -> int:
    if tensor.shape[dim] != samples:
        return 0
    return tensor.stride(dim)


def _launch_delta(out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """Row-wise ``(dO * O).sum(-1)`` in fp32. Does not materialize a C-wide product."""
    bsz, samples, heads, n_q, ch = out.shape
    delta = torch.empty(
        bsz, samples, heads, n_q, device=out.device, dtype=torch.float32
    )
    _attn_delta_kernel[(bsz * samples * heads * n_q,)](
        out,
        grad_out,
        delta,
        n_q,
        samples,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        grad_out.stride(4),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        delta.stride(3),
        H=heads,
        CH=ch,
        BLOCK_C=_block_c(ch),
        num_warps=2,
    )
    return delta


def _launch_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
    softmax_scale: float,
    *,
    write_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, samples, heads, n_q, ch = q.shape
    n_k = k.shape[-2]
    gemm_mode = _gemm_mode(q)
    block_m, block_n, warps, stages = _fwd_tiles(gemm_mode)
    block_c = _block_c(ch)
    out = torch.empty_like(q)
    lse = None
    if write_lse:
        lse = torch.empty(
            bsz, samples, heads, n_q, device=q.device, dtype=torch.float32
        )
    grid = (triton.cdiv(n_q, block_m), bsz * samples * heads)
    _flash_diffusion_attn_fwd_kernel[grid](
        q,
        k,
        v,
        pair_bias,
        mask_bias,
        out,
        lse if lse is not None else q,
        n_q,
        n_k,
        samples,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        pair_bias.stride(0),
        _sample_stride(pair_bias, samples),
        pair_bias.stride(2),
        pair_bias.stride(3),
        pair_bias.stride(4),
        mask_bias.stride(0),
        _sample_stride(mask_bias, samples),
        mask_bias.stride(-1),
        lse.stride(0) if lse is not None else 0,
        lse.stride(1) if lse is not None else 0,
        lse.stride(2) if lse is not None else 0,
        H=heads,
        CH=ch,
        BLOCK_C=block_c,
        WRITE_LSE=write_lse,
        GEMM_MODE=gemm_mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    return out, lse


def _launch_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
    softmax_scale: float,
    *,
    write_dbias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    bsz, samples, heads, n_q, ch = q.shape
    n_k = k.shape[-2]
    gemm_mode = _gemm_mode(q)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    pb_s = pair_bias.shape[1]
    accum_dbias = bool(write_dbias and pb_s == 1 and samples > 1)
    d_pair_full = None
    if write_dbias:
        d_pair_full = torch.zeros(
            (bsz, pb_s, heads, n_q, n_k),
            device=q.device,
            dtype=torch.float32,
        )
    dummy = q
    delta = _launch_delta(out, grad_out)
    block_c = _block_c(ch)
    block_m, block_n, warps, stages = _bwd_dq_tiles(gemm_mode)
    n_tiles = triton.cdiv(n_q, block_m)
    q_grid = (n_tiles, bsz * heads)
    _flash_diffusion_attn_bwd_dq_kernel[q_grid](
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        pair_bias,
        mask_bias,
        dq,
        d_pair_full if d_pair_full is not None else dummy,
        n_q,
        n_k,
        samples,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v.stride(4),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        grad_out.stride(4),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dq.stride(4),
        pair_bias.stride(0),
        _sample_stride(pair_bias, samples),
        pair_bias.stride(2),
        pair_bias.stride(3),
        pair_bias.stride(4),
        mask_bias.stride(0),
        _sample_stride(mask_bias, samples),
        mask_bias.stride(-1),
        d_pair_full.stride(0) if d_pair_full is not None else 0,
        (_sample_stride(d_pair_full, samples) if d_pair_full is not None else 0),
        d_pair_full.stride(2) if d_pair_full is not None else 0,
        d_pair_full.stride(3) if d_pair_full is not None else 0,
        d_pair_full.stride(4) if d_pair_full is not None else 1,
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        H=heads,
        CH=ch,
        BLOCK_C=block_c,
        WRITE_DBIAS=write_dbias,
        ACCUM_DBIAS=accum_dbias,
        GEMM_MODE=gemm_mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    block_m, block_n, warps, stages = _bwd_dkv_tiles(gemm_mode)
    kv_grid = (triton.cdiv(n_k, block_n), bsz * samples * heads)
    _flash_diffusion_attn_bwd_dkv_kernel[kv_grid](
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        pair_bias,
        mask_bias,
        dk,
        dv,
        n_q,
        n_k,
        samples,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v.stride(4),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        grad_out.stride(4),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dk.stride(4),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        dv.stride(4),
        pair_bias.stride(0),
        _sample_stride(pair_bias, samples),
        pair_bias.stride(2),
        pair_bias.stride(3),
        pair_bias.stride(4),
        mask_bias.stride(0),
        _sample_stride(mask_bias, samples),
        mask_bias.stride(-1),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        H=heads,
        CH=ch,
        BLOCK_C=block_c,
        GEMM_MODE=gemm_mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    d_pair = None
    if d_pair_full is not None:
        d_pair = d_pair_full.to(dtype=pair_bias.dtype)
    return dq, dk, dv, d_pair


class _FusedDiffusionAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask_bias, pair_bias, softmax_scale):
        out, lse = _launch_fwd(
            q, k, v, mask_bias, pair_bias, float(softmax_scale), write_lse=True
        )
        ctx.save_for_backward(q, k, v, out, lse, mask_bias, pair_bias)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        q, k, v, out, lse, mask_bias, pair_bias = ctx.saved_tensors
        need = ctx.needs_input_grad
        write_dbias = bool(need[4])
        dq, dk, dv, d_pair = _launch_bwd(
            q,
            k,
            v,
            out,
            grad_out.contiguous(),
            lse,
            mask_bias,
            pair_bias,
            ctx.softmax_scale,
            write_dbias=write_dbias,
        )
        return (
            dq if need[0] else None,
            dk if need[1] else None,
            dv if need[2] else None,
            None,
            d_pair if write_dbias else None,
            None,
        )


def fused_diffusion_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Flash diffusion attention. Callers must check eligibility first."""
    if not is_fused_diffusion_attn_eligible(q, k, v, mask_bias, pair_bias):
        raise RuntimeError(
            "fused_diffusion_attn requires an eligible Triton launch; "
            "use eager_diffusion_attn for the fallback path"
        )
    use_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, pair_bias)
    )
    if use_grad:
        return _FusedDiffusionAttnFn.apply(
            q, k, v, mask_bias, pair_bias, float(softmax_scale)
        )
    out, _lse = _launch_fwd(
        q, k, v, mask_bias, pair_bias, float(softmax_scale), write_lse=False
    )
    return out


def _pair_ln_linear_rows(z_rows: torch.Tensor, layer_norm, linear) -> torch.Tensor:
    from openfold3.core.kernels.triton.fused_ln_linear import (
        fused_ln_linear,
        is_fused_ln_linear_eligible,
    )

    gamma = layer_norm.weight
    if gamma is not None and is_fused_ln_linear_eligible(
        z_rows, gamma, layer_norm.bias, linear.weight, linear.bias
    ):
        return fused_ln_linear(
            z_rows, gamma, layer_norm.bias, linear.weight, linear.bias, layer_norm.eps
        )
    return linear(layer_norm(z_rows))


def _pair_bias_from_z(
    z: torch.Tensor,
    layer_norm,
    linear,
    no_heads: int,
) -> torch.Tensor:
    """``LN_z → Linear_z`` as ``[B, 1, H, N_Q, N_K]`` without copying ``z``."""
    bsz, n_q, n_k, c_z = z.shape
    z_2d = z.reshape(bsz * n_q * n_k, c_z)
    if not z_2d.is_contiguous():
        z_2d = z_2d.contiguous()
    pb = _pair_ln_linear_rows(z_2d, layer_norm, linear)
    return pb.view(bsz, n_q, n_k, no_heads).permute(0, 3, 1, 2).unsqueeze(1)


def _shared_pair_z(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor | None:
    """Normalize pair to ``[B, N, N, C_z]`` for the fused module path.

    Diffusion often broadcasts a sample axis onto ``z`` (``[B, S, N, N, C]``).
    Pair conditioning does not vary across denoising samples at a step, so
    ``z[:, 0]`` is the shared pair. Returns ``None`` if the layout is wrong.
    """
    if a.ndim != 4:
        return None
    bsz, _samples, n_tok, _c_a = a.shape
    if z.ndim == 4:
        if (
            z.shape[0] == bsz
            and z.shape[1] == n_tok
            and z.shape[2] == n_tok
            and z.shape[-1] <= _MAX_CZ
        ):
            return z
        return None
    if (
        z.ndim == 5
        and z.shape[0] == bsz
        and z.shape[1] >= 1
        and z.shape[2] == n_tok
        and z.shape[3] == n_tok
        and z.shape[-1] <= _MAX_CZ
    ):
        return z[:, 0]
    return None


def can_use_fused_diffusion_mha(
    a: torch.Tensor,
    z: torch.Tensor,
    no_heads: int,
    *,
    cached_pair_bias_h: torch.Tensor | None = None,
) -> bool:
    """Eligibility for the module-level fused MHA path (post-AdaLN ``a``)."""
    if (
        not is_fused_diffusion_attn_enabled()
        or not _TRITON_AVAILABLE
        or not a.is_cuda
        or a.ndim != 4
        or a.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    if _shared_pair_z(z, a) is None and cached_pair_bias_h is None:
        return False
    _bsz, _samples, n_tok, c_a = a.shape
    if c_a % no_heads != 0:
        return False
    ch = c_a // no_heads
    if ch > _MAX_CH or _block_c(ch) > _MAX_CH:
        return False
    if n_tok < fused_diffusion_attn_min_tokens():
        return False
    if cached_pair_bias_h is not None:
        return cached_pair_bias_h.shape[-3:] == (no_heads, n_tok, n_tok)
    return True


def _project_heads(
    x: torch.Tensor,
    linear,
    no_heads: int,
    c_hidden: int,
) -> torch.Tensor:
    y = linear(x)
    # Flash is length-generic in the ``N`` stride; do not clone Q/K/V.
    return y.view(*y.shape[:-1], no_heads, c_hidden).permute(0, 1, 3, 2, 4)


def _qkvg_pack_wins(a: torch.Tensor, linear_q) -> bool:
    """Skip packing when the cat weight buffer would dominate the peak.

    Packed activations and the cat both scale with the number of
    projections, so the test is ``numel(a) >= numel(W_q)``. Do not cache
    the packed weights: each diffusion layer would keep another full
    ``QKVG`` copy resident.
    """
    return a.numel() >= linear_q.weight.numel()


def _project_qkvg(
    a: torch.Tensor,
    linear_q,
    linear_k,
    linear_v,
    linear_g,
    no_heads: int,
    c_hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One GEMM for ``Q/K/V[/G]``. Inference-only; weights may be downcast."""
    weights = [linear_q.weight, linear_k.weight, linear_v.weight]
    linears = [linear_q, linear_k, linear_v]
    if linear_g is not None:
        weights.append(linear_g.weight)
        linears.append(linear_g)
    w = torch.cat(weights, dim=0)
    if w.dtype != a.dtype:
        w = w.to(dtype=a.dtype)
    bias = None
    if any(lin.bias is not None for lin in linears):
        chunks = []
        for lin in linears:
            rows = lin.weight.shape[0]
            if lin.bias is None:
                chunks.append(w.new_zeros(rows))
            elif lin.bias.dtype != a.dtype:
                chunks.append(lin.bias.to(dtype=a.dtype))
            else:
                chunks.append(lin.bias)
        bias = torch.cat(chunks, dim=0)
    packed = F.linear(a, w, bias)
    width = no_heads * c_hidden
    parts = packed.split(width, dim=-1)

    def as_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(*t.shape[:-1], no_heads, c_hidden).permute(0, 1, 3, 2, 4)

    g = as_heads(parts[3]) if linear_g is not None else None
    return as_heads(parts[0]), as_heads(parts[1]), as_heads(parts[2]), g


def _gated_wo(
    attn: torch.Tensor,
    g: torch.Tensor | None,
    linear_o,
    no_heads: int,
    c_hidden: int,
) -> torch.Tensor:
    """``W_o(sigmoid(G) ⊙ O)`` without a second gated copy of ``O``."""
    # attn / g are [B, S, H, N, C]; wrap_up uses [B, S, N, H, C].
    # permute().contiguous() is a no-op (and aliases ``attn``) when ``attn``
    # already has wrap_up storage — e.g. ``randn_like`` of a Q permute-view.
    o = attn.permute(0, 1, 3, 2, 4).contiguous()
    if o.data_ptr() == attn.data_ptr():
        o = o.clone()
    if g is not None:
        gate = torch.sigmoid(g.permute(0, 1, 3, 2, 4).contiguous())
        o.mul_(gate)
    return linear_o(o.reshape(*o.shape[:-2], no_heads * c_hidden))


def fused_diffusion_mha_from_module(
    a: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    linear_q,
    linear_k,
    linear_v,
    linear_g,
    linear_o,
    layer_norm_z,
    linear_z,
    no_heads: int,
    c_hidden: int,
    inf: float,
    cached_pair_bias_h: torch.Tensor | None = None,
) -> torch.Tensor:
    """MHA: one-shot pair ``LN → Linear`` then fused flash, then gate / ``W_o``.

    ``a`` is post-AdaLN. Builds a transient ``[B, 1, H, N, N]`` pair
    (~0.125U fp32 / ~0.0625U bf16) unless ``cached_pair_bias_h`` is
    supplied — not the 3U rollout cache. Inference packs ``QKVG`` when
    the cat weight buffer is not the peak, and fuses gate ⊙ ``W_o``.
    AdaLN and ada-out stay on the eager module.
    """
    scale = 1.0 / math.sqrt(c_hidden)
    if mask is None:
        mask = a.new_ones(a.shape[:-1])
    mask = mask.expand(a.shape[:-1])
    mask_bias = (inf * (mask - 1))[..., None, None, :]

    infer = not torch.is_grad_enabled()
    g = None
    if infer and _qkvg_pack_wins(a, linear_q):
        q, k, v, g = _project_qkvg(
            a, linear_q, linear_k, linear_v, linear_g, no_heads, c_hidden
        )
    else:
        q = _project_heads(a, linear_q, no_heads, c_hidden)
        k = _project_heads(a, linear_k, no_heads, c_hidden)
        v = _project_heads(a, linear_v, no_heads, c_hidden)
        if infer and linear_g is not None:
            g = _project_heads(a, linear_g, no_heads, c_hidden)

    if cached_pair_bias_h is not None:
        pair = cached_pair_bias_h.unsqueeze(1)
    else:
        z_pair = _shared_pair_z(z, a)
        if z_pair is None:
            raise RuntimeError(
                "fused_diffusion_mha_from_module requires z as [B,N,N,C] "
                "or broadcast [B,S,N,N,C]"
            )
        pair = _pair_bias_from_z(z_pair, layer_norm_z, linear_z, no_heads)
    attn = fused_diffusion_attn(q, k, v, mask_bias, pair, scale)
    if infer:
        return _gated_wo(attn, g, linear_o, no_heads, c_hidden)
    o = attn.transpose(-2, -3)
    if linear_g is not None:
        gate = torch.sigmoid(linear_g(a))
        o = o * gate.view(*gate.shape[:-1], no_heads, c_hidden)
    return linear_o(o.reshape(*o.shape[:-2], no_heads * c_hidden))
