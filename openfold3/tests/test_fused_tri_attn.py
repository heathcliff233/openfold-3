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

"""Parity and compile-reuse tests for fused triangle attention."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_tri_attn import (
    eager_tri_attn,
    fused_tri_attn,
    fused_tri_attn_from_module,
    is_fused_tri_attn_eligible,
)
from openfold3.core.model.layers.triangular_attention import (
    TriangleAttention,
    TriangleAttentionEndingNode,
)

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

C_Z = 128
C_HIDDEN = 32
NO_HEADS = 4
N_TOKEN = 64


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _randomize(module):
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.mha.linear_o.weight.normal_(0, 0.02)
        module.mha.linear_g.weight.normal_(0, 0.02)


def _module(starting=True, c_z=C_Z, c_hidden=C_HIDDEN, heads=NO_HEADS):
    cls = TriangleAttention if starting else TriangleAttentionEndingNode
    module = cls(c_z, c_hidden, heads).cuda().eval()
    _randomize(module)
    return module


def _weights(module):
    return (
        module.layer_norm.weight,
        module.layer_norm.bias,
        module.linear_z.weight,
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
        module.mha.linear_g.weight,
        module.mha.linear_o.weight,
    )


def _pair(n=N_TOKEN, c_z=C_Z, dtype=torch.float32, use_mask=True):
    torch.manual_seed(0)
    z = torch.randn(1, n, n, c_z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype) if use_mask else None
    return z, mask


def _kwargs(module):
    return dict(
        c_hidden=module.c_hidden,
        no_heads=module.no_heads,
        starting=module.starting,
        inf=module.inf,
        eps=module.layer_norm.eps,
    )


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_TRI_ATTN_V1", "1")
    monkeypatch.delenv("OPENFOLD3_TRI_ATTN_CHUNK_CAP", raising=False)


def test_fused_matches_eager_ieee():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y_fused = fused_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        y_ref = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
    torch.testing.assert_close(y_fused, y_ref, atol=1e-4, rtol=1e-4)


def test_fused_tf32_matches_eager_tf32():
    _set_tf32(True)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y_fused = fused_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        y_ref = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
    torch.testing.assert_close(y_fused, y_ref, atol=8e-3, rtol=3e-3)


def test_fused_bf16_mixed_matches_eager_mixed():
    _set_tf32(False)
    module = _module()
    z, mask = _pair(dtype=torch.bfloat16)
    with torch.inference_mode():
        y_fused = fused_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        y_ref = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
    assert y_fused.dtype == torch.bfloat16
    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=1e-1, rtol=3e-2)


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_fused_matches_module_ieee(starting, use_mask):
    _set_tf32(False)
    module = _module(starting=starting)
    z, mask = _pair(use_mask=use_mask)
    with torch.inference_mode():
        y_ref = module.forward(z.clone(), mask=mask, inplace_safe=False)
        y_fused = fused_tri_attn_from_module(module, z.clone(), mask)
    assert y_fused is not None
    torch.testing.assert_close(y_fused, y_ref, atol=2e-3, rtol=2e-3)


def test_template_shape_ieee():
    _set_tf32(False)
    module = _module(c_z=64, c_hidden=16, heads=4)
    z, mask = _pair(c_z=64)
    with torch.inference_mode():
        y_fused = fused_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        y_ref = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
    torch.testing.assert_close(y_fused, y_ref, atol=1e-4, rtol=1e-4)


def test_with_add_inplace():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        upd = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        ref = z + upd
        z_in = z.clone()
        fused = fused_tri_attn(
            z_in, mask, *_weights(module), residual=z_in, **_kwargs(module)
        )
    assert fused.data_ptr() == z_in.data_ptr()
    torch.testing.assert_close(fused, ref, atol=2e-3, rtol=2e-3)


def test_ending_node_transposed_residual():
    _set_tf32(False)
    module = _module(starting=True)
    z, mask = _pair()
    z_t = z.transpose(-2, -3)
    mask_t = mask.transpose(-1, -2)
    with torch.inference_mode():
        ref = z_t + eager_tri_attn(
            z_t, mask_t, *_weights(module), **_kwargs(module)
        )
        z_in = z.transpose(-2, -3)
        fused = fused_tri_attn(
            z_in, mask_t, *_weights(module), residual=z_in, **_kwargs(module)
        )
    assert fused.data_ptr() == z_in.data_ptr()
    torch.testing.assert_close(fused, ref, atol=2e-3, rtol=2e-3)


def test_chunked_matches_whole():
    _set_tf32(False)
    module = _module()
    z, mask = _pair(n=80)
    with torch.inference_mode():
        whole = fused_tri_attn(z, mask, *_weights(module), **_kwargs(module))
        chunked = fused_tri_attn(
            z, mask, *_weights(module), chunk_size=32, **_kwargs(module)
        )
    torch.testing.assert_close(chunked, whole, atol=1e-4, rtol=1e-4)


def test_bf16_weights_are_ineligible():
    module = _module()
    z, mask = _pair()
    wz = module.linear_z.weight.to(torch.bfloat16)
    ln_w, ln_b, _, wq, wk, wv, wg, wo = _weights(module)
    assert not is_fused_tri_attn_eligible(
        z, ln_w, ln_b, wz, wq, wk, wv, wg, wo, c_hidden=C_HIDDEN, no_heads=NO_HEADS
    )


def test_small_n_is_ineligible():
    module = _module()
    z, mask = _pair(n=32)
    assert not is_fused_tri_attn_eligible(
        z, *_weights(module), c_hidden=C_HIDDEN, no_heads=NO_HEADS
    )


def test_module_dispatch_uses_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_TRI_ATTN_V1", "1")
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y = module.forward(z.clone(), mask=mask, inplace_safe=False)
        y_ref = eager_tri_attn(z, mask, *_weights(module), **_kwargs(module))
    torch.testing.assert_close(y, y_ref, atol=2e-3, rtol=2e-3)


def test_autograd_matches_eager_tf32():
    _set_tf32(True)
    module = _module()
    z, mask = _pair()
    z_f = z.detach().requires_grad_(True)
    leaves = [t.detach().requires_grad_(True) for t in _weights(module)]
    y = fused_tri_attn(z_f, mask, *leaves, **_kwargs(module))
    y.square().mean().backward()
    z_e = z.detach().requires_grad_(True)
    leaves_e = [t.detach().requires_grad_(True) for t in _weights(module)]
    y_e = eager_tri_attn(z_e, mask, *leaves_e, **_kwargs(module))
    y_e.square().mean().backward()
    torch.testing.assert_close(z_f.grad, z_e.grad, atol=2e-2, rtol=5e-3)
    for a, b in zip(leaves, leaves_e):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-2, rtol=5e-3)


def test_autograd_matches_eager_bf16_mixed():
    _set_tf32(False)
    module = _module()
    z, mask = _pair(dtype=torch.bfloat16)
    z_f = z.detach().requires_grad_(True)
    leaves = [t.detach().requires_grad_(True) for t in _weights(module)]
    fused_tri_attn(z_f, mask, *leaves, **_kwargs(module)).square().mean().backward()
    z_e = z.detach().requires_grad_(True)
    leaves_e = [t.detach().requires_grad_(True) for t in _weights(module)]
    eager_tri_attn(z_e, mask, *leaves_e, **_kwargs(module)).square().mean().backward()
    torch.testing.assert_close(z_f.grad.float(), z_e.grad.float(), atol=8e-2, rtol=3e-2)


def test_autograd_incoming_ieee():
    _set_tf32(False)
    module = _module(starting=False)
    z, mask = _pair()
    z_f = z.detach().requires_grad_(True)
    leaves = [t.detach().requires_grad_(True) for t in _weights(module)]
    fused_tri_attn(z_f, mask, *leaves, **_kwargs(module)).square().mean().backward()
    z_e = z.detach().requires_grad_(True)
    leaves_e = [t.detach().requires_grad_(True) for t in _weights(module)]
    eager_tri_attn(z_e, mask, *leaves_e, **_kwargs(module)).square().mean().backward()
    torch.testing.assert_close(z_f.grad, z_e.grad, atol=8e-3, rtol=3e-3)


def test_weight_gradients_are_deterministic():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    grad_out = torch.randn_like(z)

    def run():
        leaves = [z.detach().requires_grad_(True)] + [
            t.detach().requires_grad_(True) for t in _weights(module)
        ]
        fused_tri_attn(leaves[0], mask, *leaves[1:], **_kwargs(module)).backward(
            grad_out
        )
        return [t.grad.clone() for t in leaves]

    assert all(torch.equal(a, b) for a, b in zip(run(), run()))


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_tri_attn import (
    _flash_tri_attn_fwd_kernel,
    _flash_tri_attn_bwd_dq_kernel,
    _pair_ln_linear_fwd_kernel,
    fused_tri_attn,
)
from openfold3.core.model.layers.triangular_attention import TriangleAttention
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

def snapshot():
    return {
        "flash": count("_flash_tri_attn_fwd_kernel"),
        "bwd_dq": count("_flash_tri_attn_bwd_dq_kernel"),
        "pair_ln": count("_pair_ln_linear_fwd_kernel"),
    }

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
m = TriangleAttention(128, 32, 4).cuda().eval()
weights = (
    m.layer_norm.weight,
    m.layer_norm.bias,
    m.linear_z.weight,
    m.mha.linear_q.weight,
    m.mha.linear_k.weight,
    m.mha.linear_v.weight,
    m.mha.linear_g.weight,
    m.mha.linear_o.weight,
)
kwargs = dict(c_hidden=32, no_heads=4, starting=True, inf=m.inf, eps=m.layer_norm.eps)
lengths = (64, 80, 96, 112, 128)

def run(n):
    z = torch.randn(1, n, n, 128, device="cuda") * 0.5
    mask = torch.ones(1, n, n, device="cuda")
    with torch.inference_mode():
        fused_tri_attn(z, mask, *weights, **kwargs)
    zd = z.detach().requires_grad_(True)
    leaves = [t.detach().requires_grad_(True) for t in weights]
    fused_tri_attn(zd, mask, *leaves, **kwargs).backward(torch.ones_like(z))

run(lengths[0])
after_first = snapshot()
for n in lengths[1:]:
    run(n)
after_all = snapshot()
assert after_first["flash"] >= 1, after_first
assert after_first["bwd_dq"] >= 1, after_first
assert after_first["pair_ln"] >= 1, after_first
assert after_all == after_first, (after_first, after_all)
print(json.dumps({"after_first": after_first, "after_all": after_all}))
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd="/workspace/hongliang/of3_dev",
            capture_output=True,
            text=True,
            check=True,
        )
        assert "after_first" in result.stdout
