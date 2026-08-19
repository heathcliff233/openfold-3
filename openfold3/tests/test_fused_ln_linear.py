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

"""Parity, autograd, and compile-reuse tests for fused LN → Linear."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_ln_linear import (
    eager_ln_linear,
    fused_ln_linear,
    is_fused_ln_linear_eligible,
)
from openfold3.core.model.primitives.linear import Linear
from openfold3.core.model.primitives.normalization import LayerNorm

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

# Dense-concat fallback shape: c_z=128 + relpos 139.
C_IN, C_OUT, N_TOKEN = 267, 128, 64


def _weights(c_in=C_IN, c_out=C_OUT, dtype=torch.float32, ln_bias=False, lin_bias=False):
    torch.manual_seed(0)
    gamma = torch.randn(c_in, dtype=dtype, device="cuda")
    beta = torch.randn(c_in, dtype=dtype, device="cuda") if ln_bias else None
    weight = torch.randn(c_out, c_in, dtype=dtype, device="cuda") / c_in**0.5
    bias = torch.randn(c_out, dtype=dtype, device="cuda") if lin_bias else None
    return gamma, beta, weight, bias


def _pair(c_in=C_IN, c_out=C_OUT, n_token=N_TOKEN, act_dtype=torch.float32, w_dtype=None):
    w_dtype = act_dtype if w_dtype is None else w_dtype
    gamma, beta, weight, bias = _weights(c_in, c_out, w_dtype)
    x = torch.randn(1, n_token, n_token, c_in, dtype=act_dtype, device="cuda") * 0.5
    return gamma, beta, weight, bias, x


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_LN_LINEAR", "1")


def test_fused_matches_eager_ieee():
    _set_tf32(False)
    gamma, beta, weight, bias, x = _pair()
    with torch.inference_mode():
        y_fused = fused_ln_linear(x, gamma, beta, weight, bias)
        y_ref = eager_ln_linear(x, gamma, beta, weight, bias)
    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-5)


def test_fused_tf32_matches_eager_tf32():
    _set_tf32(True)
    gamma, beta, weight, bias, x = _pair()
    with torch.inference_mode():
        y_fused = fused_ln_linear(x, gamma, beta, weight, bias)
        y_ref = eager_ln_linear(x, gamma, beta, weight, bias)
    torch.testing.assert_close(y_fused, y_ref, atol=5e-3, rtol=2e-3)


def test_fused_bf16_mixed_matches_eager_mixed():
    _set_tf32(False)
    gamma, beta, weight, bias, x = _pair(act_dtype=torch.bfloat16, w_dtype=torch.float32)
    with torch.inference_mode():
        y_fused = fused_ln_linear(x, gamma, beta, weight, bias)
        y_ref = eager_ln_linear(x, gamma, beta, weight, bias)
    assert y_fused.dtype == torch.bfloat16
    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=5e-2, rtol=2e-2)


def test_fused_optional_biases_match_eager():
    _set_tf32(False)
    gamma, beta, weight, bias = _weights(ln_bias=True, lin_bias=True)
    x = torch.randn(1, N_TOKEN, N_TOKEN, C_IN, device="cuda") * 0.5
    with torch.inference_mode():
        y_fused = fused_ln_linear(x, gamma, beta, weight, bias)
        y_ref = eager_ln_linear(x, gamma, beta, weight, bias)
    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-5)


def test_autograd_matches_eager_tf32():
    _set_tf32(True)
    gamma, _, weight, _, x = _pair()
    leaves_e = [t.detach().requires_grad_(True) for t in (x, gamma, weight)]
    y_e = eager_ln_linear(leaves_e[0], leaves_e[1], None, leaves_e[2], None)
    y_e.square().mean().backward()
    grads_ref = [t.grad.detach().clone() for t in leaves_e]
    leaves = [t.detach().requires_grad_(True) for t in (x, gamma, weight)]
    y = fused_ln_linear(leaves[0], leaves[1], None, leaves[2], None)
    y.square().mean().backward()
    torch.testing.assert_close(y, y_e, atol=5e-3, rtol=2e-3)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad, ref, atol=5e-3, rtol=2e-3)
    assert y.data_ptr() != leaves[0].data_ptr()


def test_autograd_matches_eager_bf16_mixed():
    _set_tf32(False)
    gamma, _, weight, _, x = _pair(act_dtype=torch.bfloat16, w_dtype=torch.float32)
    leaves_e = [t.detach().requires_grad_(True) for t in (x, gamma, weight)]
    y_e = eager_ln_linear(leaves_e[0], leaves_e[1], None, leaves_e[2], None)
    y_e.square().mean().backward()
    grads_ref = [t.grad.detach().clone() for t in leaves_e]
    leaves = [t.detach().requires_grad_(True) for t in (x, gamma, weight)]
    y = fused_ln_linear(leaves[0], leaves[1], None, leaves[2], None)
    y.square().mean().backward()
    assert y.dtype == torch.bfloat16
    assert leaves[2].grad.dtype == torch.float32
    torch.testing.assert_close(y.float(), y_e.float(), atol=5e-2, rtol=2e-2)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad.float(), ref.float(), atol=5e-2, rtol=2e-2)


def test_bf16_weights_are_ineligible():
    _set_tf32(False)
    gamma, beta, weight, bias, x = _pair(act_dtype=torch.bfloat16)
    assert not is_fused_ln_linear_eligible(x, gamma, beta, weight, bias)


def test_small_m_is_ineligible():
    _set_tf32(False)
    gamma, beta, weight, bias, x = _pair(n_token=32)
    assert not is_fused_ln_linear_eligible(x, gamma, beta, weight, bias)


def test_layer_norm_linear_modules_match_fused():
    _set_tf32(False)
    gamma, _, weight, _, x = _pair()
    ln = LayerNorm(C_IN, create_offset=False).cuda()
    lin = Linear(C_IN, C_OUT, bias=False).cuda()
    with torch.no_grad():
        ln.weight.copy_(gamma)
        lin.weight.copy_(weight)
    with torch.inference_mode():
        y_mod = lin(ln(x))
        y_fused = fused_ln_linear(x, ln.weight, ln.bias, lin.weight, lin.bias, ln.eps)
    torch.testing.assert_close(y_fused, y_mod, atol=1e-5, rtol=1e-5)


def test_autograd_optional_biases_match_eager():
    _set_tf32(False)
    gamma, beta, weight, bias = _weights(ln_bias=True, lin_bias=True)
    x = torch.randn(1, N_TOKEN, N_TOKEN, C_IN, device="cuda") * 0.5
    tensors = (x, gamma, beta, weight, bias)
    leaves_e = [t.detach().requires_grad_(True) for t in tensors]
    y_e = eager_ln_linear(*leaves_e)
    y_e.square().mean().backward()
    grads_ref = [t.grad.detach().clone() for t in leaves_e]
    leaves = [t.detach().requires_grad_(True) for t in tensors]
    y = fused_ln_linear(*leaves)
    y.square().mean().backward()
    torch.testing.assert_close(y, y_e, atol=1e-5, rtol=1e-5)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad, ref, atol=5e-4, rtol=2e-4)


def test_weight_gradients_are_deterministic():
    _set_tf32(False)
    gamma, _, weight, _, x = _pair(
        n_token=70, act_dtype=torch.bfloat16, w_dtype=torch.float32
    )
    grad_out = torch.randn(1, 70, 70, C_OUT, dtype=torch.bfloat16, device="cuda")

    def run():
        leaves = [t.detach().requires_grad_(True) for t in (x, gamma, weight)]
        fused_ln_linear(leaves[0], leaves[1], None, leaves[2], None).backward(grad_out)
        return [t.grad.detach().clone() for t in leaves]

    assert all(torch.equal(a, b) for a, b in zip(run(), run()))


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_ln_linear import (
    _fused_ln_linear_bwd_dw_kernel,
    _fused_ln_linear_bwd_dx_kernel,
    _fused_ln_linear_fwd_kernel,
    fused_ln_linear,
)
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)
def _device_caches(fn):
    if hasattr(fn, "device_caches"):
        return fn.device_caches
    return fn.fn.device_caches
def clear():
    for fn in (
        _fused_ln_linear_fwd_kernel,
        _fused_ln_linear_bwd_dw_kernel,
        _fused_ln_linear_bwd_dx_kernel,
    ):
        _device_caches(fn).clear()
def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))
def run(n):
    x = torch.randn(1, n, n, c_in, device="cuda")
    fused_ln_linear(x, g, None, w, None)
    xd = x.detach().requires_grad_(True)
    y = fused_ln_linear(
        xd,
        g.detach().requires_grad_(True),
        None,
        w.detach().requires_grad_(True),
        None,
    )
    y.backward(torch.ones_like(y))
    clear()
def snapshot():
    return {
        "forward": count("_fused_ln_linear_fwd_kernel"),
        "bwd_dw": count("_fused_ln_linear_bwd_dw_kernel"),
        "bwd_dx": count("_fused_ln_linear_bwd_dx_kernel"),
    }
torch.backends.cuda.matmul.allow_tf32 = False
c_in, c_out = 267, 128
g = torch.randn(c_in, device="cuda")
w = torch.randn(c_out, c_in, device="cuda")
lengths = (64, 80, 96, 112, 128)
run(lengths[0])
after_first = snapshot()
for n in lengths[1:]:
    run(n)
after_all = snapshot()
assert after_first["forward"] >= 1, after_first
assert after_first["bwd_dw"] >= 1 and after_first["bwd_dx"] >= 1, after_first
assert after_all == after_first, (after_first, after_all)
print(json.dumps({"after_first": after_first, "after_all": after_all}))
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_LN_LINEAR"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "after_all" in result.stdout
