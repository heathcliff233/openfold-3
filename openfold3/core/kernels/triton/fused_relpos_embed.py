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

"""Fused relpos gather-add (forward + split-K weight backward).

In-place forward, out-of-place forward for training, and a thin input-embedder
wrapper. Sequence length is only the launch grid, so one compile serves all N.

Precision: IEEE fp32 and TF32 use fp32 activations (gather-add has no
``tl.dot``, so TF32 does not change the math). bf16-mixed uses bf16
activations with fp32 Parameter masters; weight tiles are upcast on load
and ``dW`` accumulates in fp32.
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

# Exclusive pair chunks (no atomics). Fixed so compiles reuse across lengths.
_BWD_SPLIT_K = 256


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
    # fp32 masters + bf16 activations (mixed), or matching fp32 / bf16.
    return (
        is_fused_relpos_embed_enabled()
        and z.is_cuda
        and w.is_cuda
        and z.is_contiguous()
        and w.is_contiguous()
        and z.dtype in (torch.float32, torch.bfloat16)
        and w.dtype in (torch.float32, z.dtype)
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
        z_vals += same_ent * tl.load(W_ptr + SAME_ENTITY_OFFSET * C + c, mask=mask).to(
            tl.float32
        )
        tl.store(OUT_ptr + off, z_vals.to(OUT_ptr.dtype.element_ty), mask=mask)

    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=[
            "GRAD_ptr",
            "IDX1_ptr",
            "IDX2_ptr",
            "IDX3_ptr",
            "SAME_ENTITY_ptr",
            "PARTIAL_ptr",
        ],
    )
    def _fused_relpos_weight_grad_kernel(
        GRAD_ptr,
        IDX1_ptr,
        IDX2_ptr,
        IDX3_ptr,
        SAME_ENTITY_ptr,
        PARTIAL_ptr,
        M,
        C: tl.constexpr,
        VOCAB: tl.constexpr,
        SAME_ENTITY_OFFSET: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Deterministic per-split accumulate (no atomics).

        One program owns ``partial[split]`` and walks its pairs in order.
        Threads only parallelize the channel axis, so load/add/store is race-free.
        """
        split = tl.program_id(0)
        pairs_per_split = (M + SPLIT_K - 1) // SPLIT_K
        split_start = split * pairs_per_split
        split_end = tl.minimum(split_start + pairs_per_split, M)

        c = tl.arange(0, BLOCK_C)
        c_mask = c < C
        base = PARTIAL_ptr + split * VOCAB * C

        for pair in range(split_start, split_end):
            idx1 = tl.load(IDX1_ptr + pair)
            idx2 = tl.load(IDX2_ptr + pair)
            idx3 = tl.load(IDX3_ptr + pair)
            same_ent = tl.load(SAME_ENTITY_ptr + pair).to(tl.float32)
            g = tl.load(GRAD_ptr + pair * C + c, mask=c_mask, other=0.0).to(tl.float32)

            p1 = base + idx1 * C + c
            tl.store(p1, tl.load(p1, mask=c_mask, other=0.0) + g, mask=c_mask)
            p2 = base + idx2 * C + c
            tl.store(p2, tl.load(p2, mask=c_mask, other=0.0) + g, mask=c_mask)
            p3 = base + idx3 * C + c
            tl.store(p3, tl.load(p3, mask=c_mask, other=0.0) + g, mask=c_mask)
            pe = base + SAME_ENTITY_OFFSET * C + c
            tl.store(
                pe,
                tl.load(pe, mask=c_mask, other=0.0) + g * same_ent,
                mask=c_mask,
            )

else:  # pragma: no cover

    def _fused_relpos_embed_kernel(*_a, **_k):
        raise RuntimeError("Triton is required for fused_relpos_embed")

    def _fused_relpos_weight_grad_kernel(*_a, **_k):
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
        z,
        w,
        rel_pos_idx,
        rel_token_idx,
        rel_chain_idx,
        same_entity,
        same_entity_offset,
        out=z,
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
    """Out-of-place fused gather-add (for training)."""
    assert z.is_contiguous(), "z must be contiguous"
    out = torch.empty_like(z)
    return _launch_forward(
        z,
        w,
        rel_pos_idx,
        rel_token_idx,
        rel_chain_idx,
        same_entity,
        same_entity_offset,
        out=out,
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


def eager_relpos_weight_grad(
    grad_z: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
    vocab: int,
) -> torch.Tensor:
    """fp32 weight-table ``dW`` baseline (same math as gather-add forward)."""
    C = grad_z.shape[-1]
    g = grad_z.reshape(-1, C).float()
    grad_w = torch.zeros(vocab, C, device=grad_z.device, dtype=torch.float32)
    grad_w.index_add_(0, rel_pos_idx.reshape(-1), g)
    grad_w.index_add_(0, rel_token_idx.reshape(-1), g)
    grad_w.index_add_(0, rel_chain_idx.reshape(-1), g)
    se = same_entity.reshape(-1).to(dtype=g.dtype)
    grad_w[same_entity_offset] += (g * se[:, None]).sum(dim=0)
    return grad_w


def fused_relpos_weight_grad(
    grad_z: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
    vocab: int,
) -> torch.Tensor:
    """Split-K Triton ``[vocab, C]`` weight gradient (fp32 accumulate)."""
    grad_z = grad_z.contiguous()
    C = grad_z.shape[-1]
    M = grad_z.numel() // C
    partial = torch.zeros(
        _BWD_SPLIT_K, vocab, C, device=grad_z.device, dtype=torch.float32
    )
    _fused_relpos_weight_grad_kernel[(_BWD_SPLIT_K,)](
        grad_z.view(-1, C),
        rel_pos_idx.reshape(-1).contiguous(),
        rel_token_idx.reshape(-1).contiguous(),
        rel_chain_idx.reshape(-1).contiguous(),
        same_entity.reshape(-1).contiguous(),
        partial,
        M,
        C=C,
        VOCAB=vocab,
        SAME_ENTITY_OFFSET=same_entity_offset,
        SPLIT_K=_BWD_SPLIT_K,
        BLOCK_C=triton.next_power_of_2(C),
    )
    return partial.sum(dim=0)


class _FusedRelposEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weight, idx1, idx2, idx3, same_entity, offset):
        w = weight.t().contiguous()
        out = fused_relpos_embed(
            z, w, idx1, idx2, idx3, same_entity, int(offset.item())
        )
        ctx.save_for_backward(idx1, idx2, idx3, same_entity)
        ctx.offset = int(offset.item())
        ctx.vocab = weight.shape[1]
        ctx.weight_dtype = weight.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        idx1, idx2, idx3, same_entity = ctx.saved_tensors
        grad_z = grad_out if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_w = fused_relpos_weight_grad(
                grad_out, idx1, idx2, idx3, same_entity, ctx.offset, ctx.vocab
            )
            grad_weight = grad_w.t().to(dtype=ctx.weight_dtype).contiguous()
        return grad_z, grad_weight, None, None, None, None, None


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
    rel_token_idx = (
        relpos_idx(batch["token_index"], same_chain & same_res, max_relative_idx)
        + rel_pos_bins
    )
    rel_chain_idx = relpos_idx(batch["sym_id"], same_entity, max_relative_chain) + (
        same_entity_offset + 1
    )
    return rel_pos_idx, rel_token_idx, rel_chain_idx, same_entity, same_entity_offset


def add_relpos_pair_embedding(
    z: torch.Tensor,
    linear_relpos: torch.nn.Linear,
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Input-embedder wrapper: classic path, fused inference, or fused training."""
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
    z = z.contiguous()
    w = weight.t().contiguous()
    use_grad = torch.is_grad_enabled() and (z.requires_grad or weight.requires_grad)

    if use_grad:
        if _can_use_triton(z, w):
            return _FusedRelposEmbedFn.apply(
                z,
                weight,
                idx1,
                idx2,
                idx3,
                same_entity,
                torch.tensor(offset, dtype=torch.int64),
            )
        # Eager differentiable fallback (e.g. CPU).
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

    # Eager gather requires matching dtypes; mixed training downcasts here.
    if w.dtype != z.dtype:
        w = w.to(dtype=z.dtype)
    if inplace_safe:
        eager_relpos_embed_add_(z, w, idx1, idx2, idx3, same_entity, offset)
        return z
    out = z.clone()
    eager_relpos_embed_add_(out, w, idx1, idx2, idx3, same_entity, offset)
    return out
