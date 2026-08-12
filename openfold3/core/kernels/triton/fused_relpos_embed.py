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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused Triton gather-add for the
# relative-position embedding tables used by the input embedder.

"""Fused relpos gather-add for the input embedder (inference path).

In-place / out-of-place forward and a thin input-embedder wrapper. Sequence
length is only the launch grid, so one compile serves all N. Training falls
back to a differentiable eager gather-add until a Triton backward lands.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_fused_relpos_embed_enabled() -> bool:
    if not _TRITON_AVAILABLE:
        return False
    return os.environ.get("OPENFOLD3_FUSED_RELPOS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _can_use_triton(z: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        is_fused_relpos_embed_enabled()
        and z.is_cuda
        and w.is_cuda
        and z.is_contiguous()
        and w.is_contiguous()
        and z.dtype in (torch.float32, torch.bfloat16)
        and w.dtype == z.dtype
        and z.shape[-1] == w.shape[-1]
    )


if _TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize_on_alignment=[
            "Z_ptr",
            "OUT_ptr",
            "W_ptr",
            "IDX1_ptr",
            "IDX2_ptr",
            "IDX3_ptr",
            "SAME_ENTITY_ptr",
        ],
    )
    def _fused_relpos_embed_kernel(
        Z_ptr,
        OUT_ptr,
        W_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ENTITY_ptr,
        C: tl.constexpr,
        SAME_ENTITY_OFFSET: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        ij = tl.program_id(0).to(tl.int64)
        idx1 = tl.load(IDX1_ptr + ij)
        idx2 = tl.load(IDX2_ptr + ij)
        idx3 = tl.load(IDX3_ptr + ij)
        same_ent = tl.load(SAME_ENTITY_ptr + ij).to(tl.float32)

        c = tl.arange(0, BLOCK_C)
        mask = c < C
        off = ij * C + c
        z_vals = tl.load(Z_ptr + off, mask=mask).to(tl.float32)
        z_vals += tl.load(W_ptr + idx1 * C + c, mask=mask).to(tl.float32)
        z_vals += tl.load(W_ptr + idx2 * C + c, mask=mask).to(tl.float32)
        z_vals += tl.load(W_ptr + idx3 * C + c, mask=mask).to(tl.float32)
        z_vals += same_ent * tl.load(
            W_ptr + SAME_ENTITY_OFFSET * C + c, mask=mask
        ).to(tl.float32)
        tl.store(OUT_ptr + off, z_vals.to(OUT_ptr.dtype.element_ty), mask=mask)

else:  # pragma: no cover

    def _fused_relpos_embed_kernel(*_a, **_k):
        raise RuntimeError("Triton is required for fused_relpos_embed")


def _launch_forward(
    z: torch.Tensor,
    w: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
    out: torch.Tensor,
) -> torch.Tensor:
    C = z.shape[-1]
    M = z.numel() // C
    _fused_relpos_embed_kernel[(M,)](
        z.view(-1, C),
        out.view(-1, C),
        w,
        rel_pos_idx.reshape(-1),
        rel_token_idx.reshape(-1),
        rel_chain_idx.reshape(-1),
        same_entity.reshape(-1),
        C=C,
        SAME_ENTITY_OFFSET=same_entity_offset,
        BLOCK_C=triton.next_power_of_2(C),
    )
    return out


def fused_relpos_embed_add_(
    z: torch.Tensor,
    w: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
) -> None:
    """In-place fused gather-add into ``z``."""
    assert z.is_contiguous(), "z must be contiguous"
    _launch_forward(
        z, w, rel_pos_idx, rel_token_idx, rel_chain_idx, same_entity,
        same_entity_offset, out=z,
    )


def fused_relpos_embed(
    z: torch.Tensor,
    w: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
) -> torch.Tensor:
    """Out-of-place fused gather-add."""
    assert z.is_contiguous(), "z must be contiguous"
    out = torch.empty_like(z)
    return _launch_forward(
        z, w, rel_pos_idx, rel_token_idx, rel_chain_idx, same_entity,
        same_entity_offset, out=out,
    )


def eager_relpos_embed_add_(
    z: torch.Tensor,
    w: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
) -> None:
    z.add_(w[rel_pos_idx])
    z.add_(w[rel_token_idx])
    z.add_(w[rel_chain_idx])
    z.add_(same_entity[..., None].to(dtype=z.dtype) * w[same_entity_offset])


def _build_indices(batch, max_relative_idx, max_relative_chain):
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def relpos_idx(pos, condition, clip):
        offset = pos[..., None] - pos[..., None, :]
        clipped = torch.clamp(offset + clip, min=0, max=2 * clip)
        return torch.where(
            condition,
            clipped,
            (2 * clip + 1) * torch.ones_like(clipped),
        ).long()

    rel_pos_bins = 2 * max_relative_idx + 2
    same_entity_offset = 2 * rel_pos_bins
    rel_pos_idx = relpos_idx(res_idx, same_chain, max_relative_idx)
    rel_token_idx = relpos_idx(
        batch["token_index"], same_chain & same_res, max_relative_idx
    ) + rel_pos_bins
    rel_chain_idx = relpos_idx(
        batch["sym_id"], same_entity, max_relative_chain
    ) + (same_entity_offset + 1)
    return rel_pos_idx, rel_token_idx, rel_chain_idx, same_entity, same_entity_offset


def add_relpos_pair_embedding(
    z: torch.Tensor,
    linear_relpos: torch.nn.Linear,
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Input-embedder wrapper: classic path, fused inference, or eager training."""
    if linear_relpos.bias is not None:
        from openfold3.core.utils.relpos import relpos_complex
        from openfold3.core.utils.tensor_utils import add

        feats = relpos_complex(
            batch=batch,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
        ).to(dtype=z.dtype)
        return add(z, linear_relpos(feats), inplace=inplace_safe)

    idx1, idx2, idx3, same_entity, offset = _build_indices(
        batch, max_relative_idx, max_relative_chain
    )
    weight = linear_relpos.weight
    w = weight.to(dtype=z.dtype).t().contiguous()
    use_grad = torch.is_grad_enabled() and (
        z.requires_grad or weight.requires_grad
    )

    if use_grad:
        # Differentiable eager fallback until Triton weight backward lands.
        wt = weight.to(dtype=z.dtype).t()
        return (
            z
            + wt[idx1]
            + wt[idx2]
            + wt[idx3]
            + same_entity[..., None].to(dtype=z.dtype) * wt[offset]
        )

    if _can_use_triton(z, w):
        if inplace_safe:
            fused_relpos_embed_add_(z, w, idx1, idx2, idx3, same_entity, offset)
            return z
        return fused_relpos_embed(z, w, idx1, idx2, idx3, same_entity, offset)

    if inplace_safe:
        eager_relpos_embed_add_(z, w, idx1, idx2, idx3, same_entity, offset)
        return z
    out = z.clone()
    eager_relpos_embed_add_(out, w, idx1, idx2, idx3, same_entity, offset)
    return out
