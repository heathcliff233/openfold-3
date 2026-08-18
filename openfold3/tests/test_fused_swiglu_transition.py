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

"""Parity, autograd, and compile-reuse tests for fused SwiGLU transition."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_swiglu_transition import (
    fused_swiglu_transition,
    is_fused_swiglu_transition_eligible,
)
from openfold3.core.model.layers.transition import SwiGLUTransition

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

C, N_HIDDEN, N_TOKEN = 128, 4, 128


def _weights(c, n, dtype, device, requires_grad=False):
    torch.manual_seed(0)
    hidden = n * c
    g = torch.randn(c, dtype=dtype, device=device, requires_grad=requires_grad)
    b = torch.randn(c, dtype=dtype, device=device, requires_grad=requires_grad)
    wa = (torch.randn(hidden, c, dtype=dtype, device=device) / c**0.5).requires_grad_(
        requires_grad
    )
    wb = (torch.randn(hidden, c, dtype=dtype, device=device) / c**0.5).requires_grad_(
        requires_grad
    )
    wo = (
        torch.randn(c, hidden, dtype=dtype, device=device) / hidden**0.5
    ).requires_grad_(requires_grad)
    return g, b, wa, wb, wo


def _bind_module(c, n, g, b, wa, wb, wo):
    module = SwiGLUTransition(c_in=c, n=n).to(device=g.device, dtype=g.dtype)
    with torch.no_grad():
        module.layer_norm.weight.copy_(g)
        module.layer_norm.bias.copy_(b)
        module.swiglu.linear_a.weight.copy_(wa)
        module.swiglu.linear_b.weight.copy_(wb)
        module.linear_out.weight.copy_(wo)
    return module


def _eager(module, x, mask, residual=None):
    y = module._eager_transition(x, mask)
    return y if residual is None else residual + y


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _pair(c=C, n=N_HIDDEN, n_token=N_TOKEN, act_dtype=torch.float32, w_dtype=None):
    w_dtype = act_dtype if w_dtype is None else w_dtype
    g, b, wa, wb, wo = _weights(c, n, w_dtype, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo)
    x = torch.randn(1, n_token, n_token, c, dtype=act_dtype, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, 1, dtype=act_dtype, device="cuda")
    return g, b, wa, wb, wo, module, x, mask


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1")


def test_fused_matches_eager_and_inplace():
    _set_tf32(False)
    g, b, wa, wb, wo, module, x, mask = _pair()
    module.eval()
    with torch.inference_mode():
        y_ref = _eager(module, x, mask)
        y_fused = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        z = x.clone()
        y_ip = fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z)
        y_ref_ip = _eager(module, x, mask, residual=x)
    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-5)
    assert y_ip.data_ptr() == z.data_ptr()
    torch.testing.assert_close(y_ip, y_ref_ip, atol=1e-5, rtol=1e-5)


def test_fused_tf32_matches_eager_tf32():
    g, b, wa, wb, wo, module, x, mask = _pair()
    module.eval()
    _set_tf32(True)
    with torch.inference_mode():
        y_fused = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        y_eager = _eager(module, x, mask)
    torch.testing.assert_close(y_fused, y_eager, atol=5e-3, rtol=2e-3)


def test_autograd_matches_eager_tf32():
    _set_tf32(True)
    g, b, wa, wb, wo, module, x, mask = _pair()
    module.train()
    x = x.requires_grad_(True)
    y_ref = _eager(module, x, mask)
    y_ref.square().mean().backward()
    grads_ref = [
        x.grad.detach().clone(),
        module.layer_norm.weight.grad.detach().clone(),
        module.layer_norm.bias.grad.detach().clone(),
        module.swiglu.linear_a.weight.grad.detach().clone(),
        module.swiglu.linear_b.weight.grad.detach().clone(),
        module.linear_out.weight.grad.detach().clone(),
    ]
    leaves = [t.detach().requires_grad_(True) for t in (x, g, b, wa, wb, wo)]
    y = fused_swiglu_transition(*leaves, mask=mask)
    y.square().mean().backward()
    torch.testing.assert_close(y, y_ref, atol=5e-3, rtol=2e-3)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad, ref, atol=5e-3, rtol=2e-3)
    assert y.data_ptr() != leaves[0].data_ptr()


def test_autograd_matches_eager_bf16_mixed():
    _set_tf32(False)
    g, b, wa, wb, wo, module, x, mask = _pair(
        act_dtype=torch.bfloat16, w_dtype=torch.float32
    )
    module.train()
    x = x.requires_grad_(True)
    y_ref = _eager(module, x, mask)
    y_ref.square().mean().backward()
    grads_ref = [
        x.grad.detach().clone(),
        module.layer_norm.weight.grad.detach().clone(),
        module.layer_norm.bias.grad.detach().clone(),
        module.swiglu.linear_a.weight.grad.detach().clone(),
        module.swiglu.linear_b.weight.grad.detach().clone(),
        module.linear_out.weight.grad.detach().clone(),
    ]
    leaves = [t.detach().requires_grad_(True) for t in (x, g, b, wa, wb, wo)]
    y = fused_swiglu_transition(*leaves, mask=mask)
    y.square().mean().backward()
    assert y.dtype == torch.bfloat16
    assert leaves[3].grad.dtype == torch.float32
    torch.testing.assert_close(y.float(), y_ref.float(), atol=5e-2, rtol=2e-2)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad.float(), ref.float(), atol=5e-2, rtol=2e-2)


def test_module_dispatch_matches_eager_when_disabled(monkeypatch):
    _set_tf32(False)
    torch.manual_seed(3)
    module = SwiGLUTransition(c_in=C, n=N_HIDDEN).cuda().eval()
    x = torch.randn(1, N_TOKEN, N_TOKEN, C, device="cuda") * 0.5
    mask = torch.ones(1, N_TOKEN, N_TOKEN, device="cuda")
    with torch.inference_mode():
        y_fused = module(x, mask=mask)
        monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0")
        y_eager = module(x, mask=mask)
    torch.testing.assert_close(y_fused, y_eager, atol=1e-5, rtol=1e-5)


def test_bf16_weights_are_ineligible():
    _set_tf32(False)
    g, b, wa, wb, wo, module, x, mask = _pair(act_dtype=torch.bfloat16)
    module.eval()
    assert not is_fused_swiglu_transition_eligible(x, g, b, wa, wb, wo)
    with torch.inference_mode():
        y = module(x, mask=mask.squeeze(-1))
        y_ref = _eager(module, x, mask)
    torch.testing.assert_close(y, y_ref)


def test_small_m_uses_module_eager():
    _set_tf32(False)
    g, b, wa, wb, wo, module, x, mask = _pair(n_token=32)
    module.eval()
    assert not is_fused_swiglu_transition_eligible(x, g, b, wa, wb, wo)
    with torch.inference_mode():
        y = module(x, mask=mask.squeeze(-1))
        y_ref = _eager(module, x, mask)
    compare_utils.assert_max_abs_diff_small(y, y_ref, 1e-5)


def test_training_forward_does_not_retain_ab():
    _set_tf32(False)
    g, b, wa, wb, wo, _module, x, mask = _pair(
        n_token=70, act_dtype=torch.bfloat16, w_dtype=torch.float32
    )
    x = x.requires_grad_(True)
    g, b, wa, wb, wo = [t.requires_grad_(True) for t in (g, b, wa, wb, wo)]
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
    assert (70 * 70, C) in saved_shapes
    assert (70 * 70, C * N_HIDDEN) not in saved_shapes


def test_weight_gradients_are_deterministic():
    _set_tf32(False)
    g, b, wa, wb, wo, _module, x, mask = _pair(
        n_token=70, act_dtype=torch.bfloat16, w_dtype=torch.float32
    )
    grad_out = torch.randn_like(x)

    def run():
        leaves = [t.detach().requires_grad_(True) for t in (x, g, b, wa, wb, wo)]
        fused_swiglu_transition(*leaves, mask=mask).backward(grad_out)
        return [t.grad.detach().clone() for t in leaves[1:]]

    assert all(torch.equal(a, b) for a, b in zip(run(), run()))


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_swiglu_transition import (
    _fused_swiglu_transition_bwd_dw_kernel,
    _fused_swiglu_transition_bwd_dx_kernel,
    _fused_swiglu_transition_fwd_kernel,
    _fused_swiglu_transition_ln_bwd_kernel,
    fused_swiglu_transition,
)
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)
def _device_caches(fn):
    if hasattr(fn, "device_caches"):
        return fn.device_caches
    return fn.fn.device_caches
def clear():
    for fn in (
        _fused_swiglu_transition_fwd_kernel,
        _fused_swiglu_transition_bwd_dx_kernel,
        _fused_swiglu_transition_bwd_dw_kernel,
        _fused_swiglu_transition_ln_bwd_kernel,
    ):
        _device_caches(fn).clear()
def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))
def run(n):
    x = torch.randn(1, n, n, c, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, n, n, 1, device="cuda", dtype=x.dtype)
    z = x.clone()
    fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z)
    xd = x.detach().requires_grad_(True)
    y = fused_swiglu_transition(
        xd,
        g.detach().requires_grad_(True),
        b.detach().requires_grad_(True),
        wa.detach().requires_grad_(True),
        wb.detach().requires_grad_(True),
        wo.detach().requires_grad_(True),
        mask=mask,
    )
    y.backward(torch.ones_like(y))
    clear()
def snapshot():
    return {
        "forward": count("_fused_swiglu_transition_fwd_kernel"),
        "bwd_dx": count("_fused_swiglu_transition_bwd_dx_kernel"),
        "bwd_dw": count("_fused_swiglu_transition_bwd_dw_kernel"),
        "ln_bwd": count("_fused_swiglu_transition_ln_bwd_kernel"),
    }
torch.backends.cuda.matmul.allow_tf32 = False
c, h = 128, 512
g = torch.randn(c, device="cuda")
b = torch.randn(c, device="cuda")
wa = torch.randn(h, c, device="cuda")
wb = torch.randn(h, c, device="cuda")
wo = torch.randn(c, h, device="cuda")
lengths = (64, 80, 96, 112, 128)
run(lengths[0])
after_first = snapshot()
for n in lengths[1:]:
    run(n)
after_all = snapshot()
assert after_first["forward"] >= 1, after_first
assert after_first["bwd_dx"] >= 1 and after_first["bwd_dw"] >= 1, after_first
assert after_first["ln_bwd"] >= 1, after_first
assert after_all == after_first, (after_first, after_all)
print(json.dumps({"after_first": after_first, "after_all": after_all}))
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "after_all" in result.stdout
