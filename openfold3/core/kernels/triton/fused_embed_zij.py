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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: DiffusionConditioning._embed_zij
# wrapper over the shared LN→Linear backbone.

"""``LN(cat(z, relpos_complex)) → Linear`` without materializing the one-hot.

Synthesizes the concat row in registers from trunk ``z`` plus the 2c relpos
indices, then calls the shared LN / GEMM / ``dW`` / ``dX`` helpers in
``fused_ln_linear``. Matches the production site: no LN offset, no Linear
bias. ``M`` is not an autotune or specialize key.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from openfold3.core.kernels.triton.fused_ln_linear import (
    _BWD_SPLIT_M,
    _MIN_M,
    _TRITON_AVAILABLE,
    _gemm_mode,
    _next_power_of_two,
    eager_ln_linear,
    is_fused_ln_linear_enabled,
)
from openfold3.core.kernels.triton.fused_relpos_embed import _build_indices
from openfold3.core.utils.relpos import relpos_complex

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from openfold3.core.kernels.triton.fused_ln_linear import (
        _DW_TILES,
        _GEMM_TILES,
        _autotuned,
        _dot_f32,
        _dx_ln_from_go,
        _linear_from_xhat,
        _ln_bwd_store,
        _ln_fwd,
        _ln_out_from_x,
        _load2d,
    )

    @triton.jit
    def _synth_relpos_x(
        Z_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ptr,
        offs_m,
        offs_m64,
        offs_k,
        stride_z_m,
        C_Z,
        ENT_OFF,
        m_mask,
        k_mask,
    ):
        x = _load2d(Z_ptr, offs_m64, offs_k, stride_z_m, m_mask, offs_k < C_Z).to(
            tl.float32
        )
        idx1 = tl.load(IDX1_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)
        idx2 = tl.load(IDX2_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)
        idx3 = tl.load(IDX3_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)
        same = tl.load(SAME_ptr + offs_m, mask=m_mask, other=0).to(tl.float32)
        k = offs_k.to(tl.int32)[None, :]
        x = tl.where(k == (C_Z + idx1)[:, None], 1.0, x)
        x = tl.where(k == (C_Z + idx2)[:, None], 1.0, x)
        x = tl.where(k == (C_Z + idx3)[:, None], 1.0, x)
        x = tl.where(k == (C_Z + ENT_OFF), same[:, None], x)
        return tl.where(k_mask[None, :], x, 0.0)

    @_autotuned(
        _GEMM_TILES,
        ["GEMM_MODE", "K", "N", "C_Z"],
        ["Y_ptr", "Mean_ptr", "Rstd_ptr"],
        ("stride_z_m", "stride_y_m", "M", "eps"),
        (
            "Z_ptr",
            "W_ptr",
            "Y_ptr",
            "Gamma_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "IDX1_ptr",
            "IDX2_ptr",
            "IDX3_ptr",
            "SAME_ptr",
        ),
    )
    def _fused_embed_zij_fwd_kernel(
        Z_ptr,
        W_ptr,
        Y_ptr,
        Gamma_ptr,
        Mean_ptr,
        Rstd_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ptr,
        stride_z_m,
        stride_y_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        C_Z: tl.constexpr,
        ENT_OFF: tl.constexpr,
        eps,
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
        x = _synth_relpos_x(
            Z_ptr,
            IDX1_ptr,
            IDX2_ptr,
            IDX3_ptr,
            SAME_ptr,
            offs_m,
            offs_m64,
            offs_k,
            stride_z_m,
            C_Z,
            ENT_OFF,
            m_mask,
            k_mask,
        )
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        mean, rstd, x_hat = _ln_fwd(x, gamma, 0.0, k_mask, K, eps, False)
        tl.store(Mean_ptr + offs_m, mean, mask=m_mask)
        tl.store(Rstd_ptr + offs_m, rstd, mask=m_mask)
        _linear_from_xhat(
            Y_ptr,
            W_ptr,
            Z_ptr,
            x_hat,
            offs_m64,
            offs_k,
            m_mask,
            k_mask,
            stride_y_m,
            N,
            K,
            False,
            GEMM_MODE,
            BLOCK_N,
        )

    @_autotuned(
        _DW_TILES,
        ["GEMM_MODE", "K", "N", "C_Z"],
        ["PW_ptr"],
        ("stride_z_m", "stride_go_m", "M"),
        (
            "Z_ptr",
            "GO_ptr",
            "Gamma_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "IDX1_ptr",
            "IDX2_ptr",
            "IDX3_ptr",
            "SAME_ptr",
            "PW_ptr",
        ),
    )
    def _fused_embed_zij_bwd_dw_kernel(
        Z_ptr,
        GO_ptr,
        Gamma_ptr,
        Mean_ptr,
        Rstd_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ptr,
        PW_ptr,
        stride_z_m,
        stride_go_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        C_Z: tl.constexpr,
        ENT_OFF: tl.constexpr,
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
        for m0 in range(split_start, split_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_m64 = offs_m.to(tl.int64)
            m_mask = offs_m < split_end
            x = _synth_relpos_x(
                Z_ptr,
                IDX1_ptr,
                IDX2_ptr,
                IDX3_ptr,
                SAME_ptr,
                offs_m,
                offs_m64,
                offs_k,
                stride_z_m,
                C_Z,
                ENT_OFF,
                m_mask,
                k_mask,
            )
            mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
            rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
            ln_out = _ln_out_from_x(x, mean, rstd, gamma, 0.0, k_mask, False)
            go = _load2d(GO_ptr, offs_m64, offs_n, stride_go_m, m_mask, n_mask).to(
                tl.float32
            )
            acc_dw += _dot_f32(tl.trans(go), ln_out, GEMM_MODE)
        tl.store(
            PW_ptr + split * N * K + offs_n[:, None] * K + offs_k[None, :],
            acc_dw,
            mask=n_mask[:, None] & k_mask[None, :],
        )

    @_autotuned(
        _GEMM_TILES,
        ["GEMM_MODE", "K", "N", "C_Z"],
        ["GX_ptr", "PGamma_ptr", "PBeta_ptr"],
        ("stride_z_m", "stride_go_m", "stride_gx_m", "M"),
        (
            "Z_ptr",
            "W_ptr",
            "GO_ptr",
            "Gamma_ptr",
            "Mean_ptr",
            "Rstd_ptr",
            "IDX1_ptr",
            "IDX2_ptr",
            "IDX3_ptr",
            "SAME_ptr",
            "GX_ptr",
            "PGamma_ptr",
            "PBeta_ptr",
        ),
    )
    def _fused_embed_zij_bwd_dx_kernel(
        Z_ptr,
        W_ptr,
        GO_ptr,
        Gamma_ptr,
        Mean_ptr,
        Rstd_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ptr,
        GX_ptr,
        PGamma_ptr,
        PBeta_ptr,
        stride_z_m,
        stride_go_m,
        stride_gx_m,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        C_Z: tl.constexpr,
        ENT_OFF: tl.constexpr,
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
        x = _synth_relpos_x(
            Z_ptr,
            IDX1_ptr,
            IDX2_ptr,
            IDX3_ptr,
            SAME_ptr,
            offs_m,
            offs_m64,
            offs_k,
            stride_z_m,
            C_Z,
            ENT_OFF,
            m_mask,
            k_mask,
        )
        mean = tl.load(Mean_ptr + offs_m, mask=m_mask, other=0.0)
        rstd = tl.load(Rstd_ptr + offs_m, mask=m_mask, other=0.0)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        dx_ln = _dx_ln_from_go(
            GO_ptr,
            W_ptr,
            offs_m64,
            offs_k,
            m_mask,
            k_mask,
            stride_go_m,
            N,
            K,
            GEMM_MODE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        _ln_bwd_store(
            GX_ptr,
            PGamma_ptr,
            PBeta_ptr,
            x,
            dx_ln,
            mean,
            rstd,
            gamma,
            offs_m,
            offs_m64,
            offs_k,
            m_mask,
            k_mask,
            stride_gx_m,
            K,
        )

else:  # pragma: no cover

    def _unavailable(*_a, **_k):
        raise RuntimeError("Triton is required for fused_embed_zij")

    _fused_embed_zij_fwd_kernel = _unavailable
    _fused_embed_zij_bwd_dw_kernel = _unavailable
    _fused_embed_zij_bwd_dx_kernel = _unavailable


def is_fused_embed_zij_eligible(
    z: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    c_z = z.shape[-1]
    k = gamma.shape[0]
    return (
        is_fused_ln_linear_enabled()
        and _TRITON_AVAILABLE
        and z.is_cuda
        and z.is_contiguous()
        and gamma.is_cuda
        and weight.is_cuda
        and gamma.is_contiguous()
        and weight.is_contiguous()
        and z.dtype in (torch.float32, torch.bfloat16)
        and gamma.dtype == torch.float32
        and weight.dtype == torch.float32
        and weight.shape == (c_z, k)
        and k > c_z
        and (z.numel() // c_z) >= _MIN_M
    )


def eager_embed_zij(
    z: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    feats = relpos_complex(batch, max_relative_idx, max_relative_chain).to(
        dtype=z.dtype
    )
    return eager_ln_linear(
        torch.cat([z, feats], dim=-1), gamma, None, weight, None, eps
    )


def _flat_indices(idx1, idx2, idx3, same_entity):
    return (
        idx1.reshape(-1).contiguous(),
        idx2.reshape(-1).contiguous(),
        idx3.reshape(-1).contiguous(),
        same_entity.reshape(-1).to(dtype=torch.float32).contiguous(),
    )


def _launch_fused_embed_zij(
    z_2d: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    same_entity: torch.Tensor,
    entity_offset: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_embed_zij")
    M, c_z = z_2d.shape
    K, N = gamma.shape[0], weight.shape[0]
    y = torch.empty((M, N), dtype=z_2d.dtype, device=z_2d.device)
    mean = torch.empty(M, dtype=torch.float32, device=z_2d.device)
    rstd = torch.empty_like(mean)
    _fused_embed_zij_fwd_kernel[(lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),))](
        z_2d,
        weight,
        y,
        gamma,
        mean,
        rstd,
        idx1,
        idx2,
        idx3,
        same_entity,
        z_2d.stride(0),
        y.stride(0),
        M,
        K,
        N,
        c_z,
        int(entity_offset),
        float(eps),
        GEMM_MODE=_gemm_mode(z_2d),
        BLOCK_K=max(_next_power_of_two(K), 16),
    )
    return y, mean, rstd


def _fused_embed_zij_backward(
    z_2d: torch.Tensor,
    grad_out_2d: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    same_entity: torch.Tensor,
    entity_offset: int,
    compute_dx: bool,
    compute_dw: bool,
):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_embed_zij")
    M, c_z = z_2d.shape
    K, N = gamma.shape[0], weight.shape[0]
    gemm_mode = _gemm_mode(z_2d)
    block_k = max(_next_power_of_two(K), 16)
    idx = (idx1, idx2, idx3, same_entity)

    grad_w = None
    if compute_dw:
        partial_w = torch.empty(
            (_BWD_SPLIT_M, N, K), dtype=torch.float32, device=z_2d.device
        )
        _fused_embed_zij_bwd_dw_kernel[
            lambda meta: (_BWD_SPLIT_M, triton.cdiv(N, meta["BLOCK_N"]))
        ](
            z_2d,
            grad_out_2d,
            gamma,
            mean,
            rstd,
            *idx,
            partial_w,
            z_2d.stride(0),
            grad_out_2d.stride(0),
            M,
            K,
            N,
            c_z,
            int(entity_offset),
            GEMM_MODE=gemm_mode,
            SPLIT_M=_BWD_SPLIT_M,
            BLOCK_K=block_k,
        )
        grad_w = partial_w.sum(0)

    grad_x = d_gamma = None
    if compute_dx:
        grad_x = torch.empty((M, K), dtype=z_2d.dtype, device=z_2d.device)
        n_partial = triton.cdiv(M, 16)
        partial_gamma = torch.zeros(
            (n_partial, K), dtype=torch.float32, device=z_2d.device
        )
        partial_beta = torch.empty_like(partial_gamma)
        _fused_embed_zij_bwd_dx_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)](
            z_2d,
            weight,
            grad_out_2d,
            gamma,
            mean,
            rstd,
            *idx,
            grad_x,
            partial_gamma,
            partial_beta,
            z_2d.stride(0),
            grad_out_2d.stride(0),
            grad_x.stride(0),
            M,
            K,
            N,
            c_z,
            int(entity_offset),
            GEMM_MODE=gemm_mode,
            BLOCK_K=block_k,
        )
        d_gamma = partial_gamma.sum(0)
    return grad_x, d_gamma, grad_w


class _FusedEmbedZijFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, z, gamma, weight, idx1, idx2, idx3, same_entity, entity_offset, eps
    ):
        z_2d = z.contiguous().view(-1, z.shape[-1])
        y_2d, mean, rstd = _launch_fused_embed_zij(
            z_2d,
            gamma,
            weight,
            idx1,
            idx2,
            idx3,
            same_entity,
            int(entity_offset),
            eps,
        )
        ctx.save_for_backward(
            z, gamma, weight, mean, rstd, idx1, idx2, idx3, same_entity
        )
        ctx.entity_offset = int(entity_offset)
        ctx.z_shape = z.shape
        return y_2d.view(*z.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        z, gamma, weight, mean, rstd, i1, i2, i3, same = ctx.saved_tensors
        need = ctx.needs_input_grad
        gx, dgamma, dw = _fused_embed_zij_backward(
            z.contiguous().view(-1, z.shape[-1]),
            grad_out.contiguous().view(-1, weight.shape[0]),
            gamma,
            weight,
            mean,
            rstd,
            i1,
            i2,
            i3,
            same,
            ctx.entity_offset,
            compute_dx=need[0] or need[1],
            compute_dw=bool(need[2]),
        )
        dz = (
            gx[:, : z.shape[-1]].to(dtype=z.dtype).view(ctx.z_shape)
            if need[0]
            else None
        )
        return (
            dz,
            dgamma.to(dtype=gamma.dtype) if need[1] else None,
            dw.to(dtype=weight.dtype) if need[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_embed_zij(
    z: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    same_entity: torch.Tensor,
    entity_offset: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused ``LN(cat(z, onehot(idx))) → Linear``. Check eligibility first."""
    if not is_fused_embed_zij_eligible(z, gamma, weight):
        raise RuntimeError(
            "fused_embed_zij requires an eligible Triton launch; "
            "use eager_embed_zij for the fallback path"
        )
    i1, i2, i3, same = _flat_indices(idx1, idx2, idx3, same_entity)
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (z, gamma, weight)
    ):
        return _FusedEmbedZijFn.apply(
            z, gamma, weight, i1, i2, i3, same, int(entity_offset), float(eps)
        )
    y_2d, _mean, _rstd = _launch_fused_embed_zij(
        z.contiguous().view(-1, z.shape[-1]),
        gamma,
        weight,
        i1,
        i2,
        i3,
        same,
        int(entity_offset),
        eps,
    )
    return y_2d.view(*z.shape[:-1], weight.shape[0])


def try_fused_embed_zij(
    z: torch.Tensor,
    batch: dict,
    layer_norm,
    linear,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor | None:
    """Module wrapper. Returns ``None`` when the fused path is ineligible."""
    gamma = layer_norm.weight
    if gamma is None or not is_fused_embed_zij_eligible(z, gamma, linear.weight):
        return None
    idx1, idx2, idx3, same_entity, offset = _build_indices(
        batch, max_relative_idx, max_relative_chain
    )
    return fused_embed_zij(
        z,
        gamma,
        linear.weight,
        idx1,
        idx2,
        idx3,
        same_entity,
        offset,
        layer_norm.eps,
    )
