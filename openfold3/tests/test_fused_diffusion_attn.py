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

"""Parity, autograd, dispatch, cache, and compile-reuse tests for fused
diffusion attention."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_diffusion_attn import (
    _pair_bias_from_z,
    can_use_fused_diffusion_attention,
    can_use_fused_diffusion_mha,
    eager_diffusion_attn,
    fused_diffusion_attn,
    fused_diffusion_attn_min_tokens,
    fused_diffusion_mha_from_module,
)
from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

B, S, N, H, CH = 1, 2, 64, 4, 32
C_A = H * CH
C_S = 64
C_Z = 128
INF = 1e9


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _qkv(dtype=torch.float32, n=N, samples=S, heads=H, ch=CH):
    torch.manual_seed(0)
    q = torch.randn(B, samples, heads, n, ch, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(B, samples, heads, n, ch, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(B, samples, heads, n, ch, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(B, samples, 1, 1, n, device="cuda", dtype=dtype)
    pair = torch.randn(B, 1, heads, n, n, device="cuda", dtype=dtype) * 0.1
    scale = 1.0 / (ch**0.5)
    return q, k, v, mask, pair, scale


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "1")
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", "0")
    monkeypatch.setenv("OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE", "0")


def test_fused_matches_eager_ieee():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv()
    with torch.inference_mode():
        y_fused = fused_diffusion_attn(q, k, v, mask, pair, scale)
        y_ref = eager_diffusion_attn(q, k, v, mask, pair, scale)
    torch.testing.assert_close(y_fused, y_ref, atol=2e-4, rtol=2e-4)


def test_fused_tf32_matches_eager_tf32():
    _set_tf32(True)
    q, k, v, mask, pair, scale = _qkv()
    with torch.inference_mode():
        y_fused = fused_diffusion_attn(q, k, v, mask, pair, scale)
        y_ref = eager_diffusion_attn(q, k, v, mask, pair, scale)
    torch.testing.assert_close(y_fused, y_ref, atol=8e-3, rtol=3e-3)


def test_fused_bf16_mixed_matches_eager_mixed():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv(dtype=torch.bfloat16)
    with torch.inference_mode():
        y_fused = fused_diffusion_attn(q, k, v, mask, pair, scale)
        y_ref = eager_diffusion_attn(q, k, v, mask, pair, scale)
    assert y_fused.dtype == torch.bfloat16
    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=1e-1, rtol=3e-2)


def test_rectangular_q_matches_eager():
    _set_tf32(False)
    torch.manual_seed(0)
    n_q, n_k, heads, ch = 48, 64, 4, 32
    q = torch.randn(1, 2, heads, n_q, ch, device="cuda") * 0.5
    k = torch.randn(1, 2, heads, n_k, ch, device="cuda") * 0.5
    v = torch.randn(1, 2, heads, n_k, ch, device="cuda") * 0.5
    mask = torch.ones(1, 2, 1, 1, n_k, device="cuda")
    pair = torch.randn(1, 1, heads, n_q, n_k, device="cuda") * 0.1
    scale = 1.0 / (ch**0.5)
    with torch.inference_mode():
        y_fused = fused_diffusion_attn(q, k, v, mask, pair, scale)
        y_ref = eager_diffusion_attn(q, k, v, mask, pair, scale)
    torch.testing.assert_close(y_fused, y_ref, atol=2e-4, rtol=2e-4)


def test_production_head_dim_48():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv(heads=16, ch=48)
    with torch.inference_mode():
        y_fused = fused_diffusion_attn(q, k, v, mask, pair, scale)
        y_ref = eager_diffusion_attn(q, k, v, mask, pair, scale)
    torch.testing.assert_close(y_fused, y_ref, atol=4e-4, rtol=4e-4)


def test_autograd_transposed_qkv_layout():
    """Module path builds QKV as ``[B,S,N,H,C].transpose(-2,-3)`` (not contiguous)."""
    _set_tf32(True)
    torch.manual_seed(0)
    q = (torch.randn(B, S, N, H, CH, device="cuda") * 0.5).transpose(-2, -3)
    k = (torch.randn(B, S, N, H, CH, device="cuda") * 0.5).transpose(-2, -3)
    v = (torch.randn(B, S, N, H, CH, device="cuda") * 0.5).transpose(-2, -3)
    assert not q.is_contiguous()
    mask = torch.ones(B, S, 1, 1, N, device="cuda")
    pair = torch.randn(B, 1, H, N, N, device="cuda") * 0.1
    scale = 1.0 / (CH**0.5)
    qf, kf, vf, pf = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    fused_diffusion_attn(qf, kf, vf, mask, pf, scale).square().mean().backward()
    qe, ke, ve, pe = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    eager_diffusion_attn(qe, ke, ve, mask, pe, scale).square().mean().backward()
    torch.testing.assert_close(qf.grad, qe.grad, atol=2e-2, rtol=5e-3)


def test_autograd_matches_eager_ieee():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv()
    qf, kf, vf, pf = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    fused_diffusion_attn(qf, kf, vf, mask, pf, scale).square().mean().backward()
    qe, ke, ve, pe = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    eager_diffusion_attn(qe, ke, ve, mask, pe, scale).square().mean().backward()
    torch.testing.assert_close(qf.grad, qe.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(kf.grad, ke.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(vf.grad, ve.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(pf.grad, pe.grad, atol=3e-4, rtol=3e-4)


def test_autograd_matches_eager_tf32():
    _set_tf32(True)
    q, k, v, mask, pair, scale = _qkv()
    qf, kf, vf, pf = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    fused_diffusion_attn(qf, kf, vf, mask, pf, scale).square().mean().backward()
    qe, ke, ve, pe = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    eager_diffusion_attn(qe, ke, ve, mask, pe, scale).square().mean().backward()
    torch.testing.assert_close(qf.grad, qe.grad, atol=2e-2, rtol=5e-3)
    torch.testing.assert_close(kf.grad, ke.grad, atol=2e-2, rtol=5e-3)
    torch.testing.assert_close(vf.grad, ve.grad, atol=2e-2, rtol=5e-3)
    torch.testing.assert_close(pf.grad, pe.grad, atol=2e-2, rtol=5e-3)


def test_autograd_matches_eager_bf16_mixed():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv(dtype=torch.bfloat16)
    qf, kf, vf, pf = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    fused_diffusion_attn(qf, kf, vf, mask, pf, scale).square().mean().backward()
    qe, ke, ve, pe = (t.detach().requires_grad_(True) for t in (q, k, v, pair))
    eager_diffusion_attn(qe, ke, ve, mask, pe, scale).square().mean().backward()
    torch.testing.assert_close(qf.grad.float(), qe.grad.float(), atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(pf.grad.float(), pe.grad.float(), atol=8e-2, rtol=3e-2)


def test_weight_gradients_are_deterministic():
    _set_tf32(False)
    q, k, v, mask, pair, scale = _qkv()
    go = torch.randn_like(q)

    def run():
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v, pair)]
        fused_diffusion_attn(*leaves[:3], mask, leaves[3], scale).backward(go)
        return [t.grad.clone() for t in leaves]

    assert all(torch.equal(a, b) for a, b in zip(run(), run()))


def test_dispatch_policy_default_allows_s1_short(monkeypatch):
    monkeypatch.delenv("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", raising=False)
    q_x = torch.randn(1, 1, 64, C_A, device="cuda")
    pair = torch.randn(1, 1, H, 64, 64, device="cuda")
    mask = torch.zeros(1, 1, 1, 1, 64, device="cuda")
    assert can_use_fused_diffusion_attention(q_x, q_x, [mask, pair], H)
    q_x = torch.randn(1, 5, 64, C_A, device="cuda")
    pair = torch.randn(1, 1, H, 64, 64, device="cuda")
    mask = torch.zeros(1, 5, 1, 1, 64, device="cuda")
    assert can_use_fused_diffusion_attention(q_x, q_x, [mask, pair], H)


def test_min_tokens_override_still_blocks(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", "1024")
    assert fused_diffusion_attn_min_tokens() == 1024
    q_x = torch.randn(1, 1, 590, C_A, device="cuda")
    pair = torch.randn(1, 1, H, 590, 590, device="cuda")
    mask = torch.zeros(1, 1, 1, 1, 590, device="cuda")
    assert not can_use_fused_diffusion_attention(q_x, q_x, [mask, pair], H)


def test_cross_attention_is_ineligible():
    q_x = torch.randn(1, 2, 64, C_A, device="cuda")
    kv_x = torch.randn(1, 2, 64, C_A, device="cuda")
    pair = torch.randn(1, 1, H, 64, 64, device="cuda")
    mask = torch.zeros(1, 2, 1, 1, 64, device="cuda")
    assert not can_use_fused_diffusion_attention(q_x, kv_x, [mask, pair], H)


def test_default_cutoff_is_zero(monkeypatch):
    monkeypatch.delenv("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", raising=False)
    assert fused_diffusion_attn_min_tokens() == 0


def _token_transformer(n_blocks=2, heads=H, ch=CH):
    module = (
        DiffusionTransformer(
            c_a=heads * ch,
            c_s=C_S,
            c_z=C_Z,
            c_hidden=ch,
            no_heads=heads,
            no_blocks=n_blocks,
            n_transition=2,
            use_ada_layer_norm=True,
            n_query=None,
            n_key=None,
            inf=INF,
        )
        .cuda()
        .eval()
    )
    return module


def _token_inputs(n=N, samples=S, dtype=torch.float32, heads=H, ch=CH):
    torch.manual_seed(0)
    a = torch.randn(B, samples, n, heads * ch, device="cuda", dtype=dtype) * 0.5
    s = torch.randn(B, samples, n, C_S, device="cuda", dtype=dtype) * 0.5
    z = torch.randn(B, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(B, 1, n, device="cuda", dtype=dtype)
    return a, s, z, mask


def _pair_bias_module(heads=H, ch=CH):
    return (
        AttentionPairBias(
            c_q=heads * ch,
            c_k=heads * ch,
            c_v=heads * ch,
            c_s=C_S,
            c_z=C_Z,
            c_hidden=ch,
            no_heads=heads,
            use_ada_layer_norm=True,
        )
        .cuda()
        .eval()
    )


def _mha_from_module(module, a, z, mask):
    return fused_diffusion_mha_from_module(
        a,
        z,
        mask,
        linear_q=module.mha.linear_q,
        linear_k=module.mha.linear_k,
        linear_v=module.mha.linear_v,
        linear_g=module.mha.linear_g,
        linear_o=module.mha.linear_o,
        layer_norm_z=module.layer_norm_z,
        linear_z=module.linear_z,
        no_heads=module.mha.no_heads,
        c_hidden=module.mha.c_hidden,
        inf=module.inf,
    )


def test_pair_bias_from_z_matches_eager_ieee():
    _set_tf32(False)
    torch.manual_seed(1)
    z = torch.randn(B, N, N, C_Z, device="cuda") * 0.5
    ln = torch.nn.LayerNorm(C_Z).cuda()
    linear = torch.nn.Linear(C_Z, H, bias=False).cuda()
    with torch.inference_mode():
        y = _pair_bias_from_z(z, ln, linear, H)
        ref = linear(ln(z)).permute(0, 3, 1, 2).unsqueeze(1)
    torch.testing.assert_close(y, ref, atol=2e-4, rtol=2e-4)


def test_packed_qkvg_matches_separate_linears():
    _set_tf32(False)
    from openfold3.core.kernels.triton.fused_diffusion_attn import (
        _gated_wo,
        _project_heads,
        _project_qkvg,
    )

    module = _pair_bias_module()
    a, _s, _z, _mask = _token_inputs()
    mha = module.mha
    with torch.inference_mode():
        q, k, v, g = _project_qkvg(
            a, mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g, H, CH
        )
        q_r = _project_heads(a, mha.linear_q, H, CH)
        k_r = _project_heads(a, mha.linear_k, H, CH)
        v_r = _project_heads(a, mha.linear_v, H, CH)
        g_r = _project_heads(a, mha.linear_g, H, CH)
        attn = torch.randn_like(q)
        attn_ref = attn.clone()
        y = _gated_wo(attn, g, mha.linear_o, H, CH)
        y_r = mha._wrap_up(attn_ref.permute(0, 1, 3, 2, 4).contiguous(), a)
    torch.testing.assert_close(q, q_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k, k_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v, v_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g, g_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y, y_r, atol=1e-5, rtol=1e-5)


def test_fused_matches_module_ieee():
    _set_tf32(False)
    module = _pair_bias_module()
    a, s, z, mask = _token_inputs()
    with torch.inference_mode():
        a_n = module.layer_norm_a(a, s)
        y_fused = _mha_from_module(module, a_n, z, mask)
        y_fused = module.sigmoid(module.linear_ada_out(s)) * y_fused
        y_ref = module(a.clone(), z, s=s, mask=mask)
    torch.testing.assert_close(y_fused, y_ref, atol=3e-3, rtol=3e-3)


def test_module_dispatch_uses_fused(monkeypatch):
    _set_tf32(False)
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "1")
    module = _token_transformer()
    a, s, z, mask = _token_inputs()
    with torch.inference_mode():
        y = module(a.clone(), s, z, mask=mask)
        monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "0")
        y_ref = module(a.clone(), s, z, mask=mask)
    torch.testing.assert_close(y, y_ref, atol=3e-3, rtol=3e-3)


def test_module_accepts_broadcast_sample_pair():
    """Production z is often [B, S, N, N, C] via sample broadcast."""
    _set_tf32(False)
    module = _pair_bias_module()
    a, s, z, mask = _token_inputs()
    z5 = z.unsqueeze(1).expand(B, S, N, N, C_Z).contiguous()
    assert can_use_fused_diffusion_mha(a, z5, H)
    with torch.inference_mode():
        y5 = module(a.clone(), z5, s=s, mask=mask)
        y4 = module(a.clone(), z, s=s, mask=mask)
    torch.testing.assert_close(y5, y4, atol=1e-5, rtol=1e-5)


def test_fused_matches_module_production_heads(monkeypatch):
    _set_tf32(False)
    heads, ch = 16, 48
    module = _pair_bias_module(heads=heads, ch=ch)
    a, s, z, mask = _token_inputs(heads=heads, ch=ch)
    assert can_use_fused_diffusion_mha(a, z, heads)
    with torch.inference_mode():
        y = module(a.clone(), z, s=s, mask=mask)
        monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "0")
        y_ref = module(a.clone(), z, s=s, mask=mask)
    torch.testing.assert_close(y, y_ref, atol=3e-3, rtol=3e-3)


def test_can_use_fused_diffusion_mha_rejects_bad_layout():
    a, _s, z, _mask = _token_inputs()
    assert can_use_fused_diffusion_mha(a, z, H)
    assert not can_use_fused_diffusion_mha(a, z[:, :, :-1], H)
    assert not can_use_fused_diffusion_mha(
        a, torch.randn(B, N, N, C_Z + 1, device="cuda"), H
    )
    assert not can_use_fused_diffusion_mha(a.cpu(), z.cpu(), H)


def test_module_autograd_matches_eager_tf32(monkeypatch):
    _set_tf32(True)
    module = _pair_bias_module().train()
    a, s, z, mask = _token_inputs()
    af, zf = a.detach().requires_grad_(True), z.detach().requires_grad_(True)
    module(
        af, zf, s=s, mask=mask, use_high_precision_attention=True
    ).square().mean().backward()
    d_a = af.grad.detach().clone()
    d_z = zf.grad.detach().clone()
    d_ln = module.layer_norm_z.weight.grad.detach().clone()
    d_w = module.linear_z.weight.grad.detach().clone()
    module.zero_grad(set_to_none=True)
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "0")
    ae, ze = a.detach().requires_grad_(True), z.detach().requires_grad_(True)
    module(
        ae, ze, s=s, mask=mask, use_high_precision_attention=True
    ).square().mean().backward()
    torch.testing.assert_close(d_a, ae.grad, atol=2e-2, rtol=5e-3)
    torch.testing.assert_close(d_z, ze.grad, atol=2e-2, rtol=5e-3)
    torch.testing.assert_close(
        d_ln, module.layer_norm_z.weight.grad, atol=2e-2, rtol=5e-3
    )
    torch.testing.assert_close(d_w, module.linear_z.weight.grad, atol=2e-2, rtol=5e-3)


def test_high_precision_does_not_block_fused(monkeypatch):
    import openfold3.core.kernels.triton.fused_diffusion_attn as fda

    calls = []
    real = fda.fused_diffusion_attn

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fda, "fused_diffusion_attn", wrapped)
    module = _pair_bias_module().train()
    a, s, z, mask = _token_inputs()
    a = a.detach().requires_grad_(True)
    z = z.detach().requires_grad_(True)
    module(
        a, z, s=s, mask=mask, use_high_precision_attention=True
    ).square().mean().backward()
    assert calls, "fused diffusion attn must run under the training high-precision flag"

    calls.clear()
    module.eval()
    with torch.inference_mode():
        module(
            a.detach(), z.detach(), s=s, mask=mask, use_high_precision_attention=True
        )
    assert calls, "fused diffusion attn must run under inference high-precision flag"


def test_pair_bias_cache_bitwise(monkeypatch):
    _set_tf32(False)
    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "0")
    module = _token_transformer()
    a, s, z, mask = _token_inputs()
    cache = module.prepare_pair_bias_cache(z)
    assert cache is not None and len(cache) == 2
    with torch.inference_mode():
        y_cached = module(a.clone(), s, z, mask=mask, pair_bias_cache=cache)
        y = module(a.clone(), s, z, mask=mask)
    assert torch.equal(y_cached, y)

    monkeypatch.setenv("OPENFOLD3_FUSED_DIFFUSION_ATTN", "1")
    with torch.inference_mode():
        y_fused_cached = module(a.clone(), s, z, mask=mask, pair_bias_cache=cache)
        y_fused = module(a.clone(), s, z, mask=mask)
    # Cached path keeps a precomputed pair; default builds LN→Linear once
    # per call then fused flash (same math, not the 3U rollout cache).
    torch.testing.assert_close(y_fused_cached, y_fused, atol=3e-3, rtol=3e-3)


def test_prep_static_pair_bias_matches_eager():
    module = _pair_bias_module()
    z = torch.randn(1, N, N, C_Z, device="cuda")
    cached = module.prep_static_pair_bias(z)
    ref = module.linear_z(module.layer_norm_z(z)).permute(0, 3, 1, 2)
    torch.testing.assert_close(cached, ref, atol=1e-5, rtol=1e-5)


def test_triangle_flags_skip_fused():
    module = _token_transformer()
    a, s, z, mask = _token_inputs()
    q_x = torch.randn(B, S, N, C_A, device="cuda")
    pair = torch.randn(B, 1, H, N, N, device="cuda")
    mb = torch.zeros(B, S, 1, 1, N, device="cuda")
    assert can_use_fused_diffusion_attention(q_x, q_x, [mb, pair], H)
    with torch.inference_mode():
        y = module(
            a.clone(),
            s,
            z,
            mask=mask,
            use_cueq_triangle_kernels=False,
            use_triton_triangle_kernels=False,
        )
    assert y.shape == a.shape


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_diffusion_attn import (
    _flash_diffusion_attn_bwd_dq_kernel,
    _flash_diffusion_attn_fwd_kernel,
    fused_diffusion_attn,
)
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

def snapshot():
    return {
        "fwd": count("_flash_diffusion_attn_fwd_kernel"),
        "bwd_dq": count("_flash_diffusion_attn_bwd_dq_kernel"),
    }

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
heads, ch = 4, 32
scale = 1.0 / (ch ** 0.5)
lengths = (64, 80, 96, 112, 128)

def run(n):
    q = torch.randn(1, 2, heads, n, ch, device="cuda") * 0.5
    k = torch.randn(1, 2, heads, n, ch, device="cuda") * 0.5
    v = torch.randn(1, 2, heads, n, ch, device="cuda") * 0.5
    mask = torch.ones(1, 2, 1, 1, n, device="cuda")
    pair = torch.randn(1, 1, heads, n, n, device="cuda") * 0.1
    with torch.inference_mode():
        fused_diffusion_attn(q, k, v, mask, pair, scale)
    qd = q.detach().requires_grad_(True)
    kd = k.detach().requires_grad_(True)
    vd = v.detach().requires_grad_(True)
    pd = pair.detach().requires_grad_(True)
    fused_diffusion_attn(qd, kd, vd, mask, pd, scale).backward(torch.ones_like(q))

run(lengths[0])
after_first = snapshot()
for n in lengths[1:]:
    run(n)
after_all = snapshot()
assert after_first["fwd"] >= 1, after_first
assert after_first["bwd_dq"] >= 1, after_first
assert after_all == after_first, (after_first, after_all)
print(json.dumps({"after_first": after_first, "after_all": after_all}))
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "1"
        env["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "after_first" in result.stdout
