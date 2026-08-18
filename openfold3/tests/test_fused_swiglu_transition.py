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
    is_fused_swiglu_transition_enabled,
)
from openfold3.core.model.layers.transition import SwiGLUTransition

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

CASES = [(128, 4), (128, 2), (64, 2)]
N_VALUES = [70, 256]


def _weights(c, n, dtype, device, requires_grad=False):
    torch.manual_seed(0)
    H = n * c
    g = torch.randn(c, dtype=dtype, device=device, requires_grad=requires_grad)
    b = torch.randn(c, dtype=dtype, device=device, requires_grad=requires_grad)
    wa = (torch.randn(H, c, dtype=dtype, device=device) / c**0.5).requires_grad_(
        requires_grad
    )
    wb = (torch.randn(H, c, dtype=dtype, device=device) / c**0.5).requires_grad_(
        requires_grad
    )
    wo = (torch.randn(c, H, dtype=dtype, device=device) / H**0.5).requires_grad_(
        requires_grad
    )
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


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1")


@pytest.mark.parametrize("c,n", CASES)
@pytest.mark.parametrize("n_token", N_VALUES)
def test_fused_matches_eager_fp32_baseline(c, n, n_token):
    """fp32 IEEE fused vs module primitives."""
    assert is_fused_swiglu_transition_enabled()
    _set_tf32(False)
    dtype = torch.float32
    g, b, wa, wb, wo = _weights(c, n, dtype, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo).eval()
    x = torch.randn(1, n_token, n_token, c, dtype=dtype, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, 1, dtype=dtype, device="cuda")
    atol, rtol = 1e-5, 1e-5

    with torch.inference_mode():
        y_ref = _eager(module, x, mask)
        y_fused = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        z = x.clone()
        y_ip = fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z)
        y_ref_ip = _eager(module, x, mask, residual=x)

    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=atol, rtol=rtol)
    assert y_ip.data_ptr() == z.data_ptr()
    torch.testing.assert_close(y_ip.float(), y_ref_ip.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_token", [70, 256])
def test_fused_mixed_matches_eager(n_token):
    """bf16-mixed: fp32 masters, bf16 GEMM, no TF32, vs module primitives."""
    _set_tf32(False)
    g, b, wa, wb, wo = _weights(128, 4, torch.float32, "cuda")
    module = _bind_module(128, 4, g, b, wa, wb, wo).eval()
    x = torch.randn(1, n_token, n_token, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, 1, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        y_ref = _eager(module, x, mask)
        y_fused = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=5e-2, rtol=2e-2)


def test_bf16_weights_are_ineligible():
    """Triton requires fp32 masters; the module uses primitives."""
    _set_tf32(False)
    g, b, wa, wb, wo = _weights(128, 4, torch.bfloat16, "cuda")
    module = _bind_module(128, 4, g, b, wa, wb, wo).eval()
    x = torch.randn(1, 70, 70, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    mask = torch.ones(1, 70, 70, 1, dtype=torch.bfloat16, device="cuda")
    assert not is_fused_swiglu_transition_eligible(x, g, b, wa, wb, wo)
    with torch.inference_mode():
        y = module(x, mask=mask.squeeze(-1))
        y_ref = _eager(module, x, mask)
    torch.testing.assert_close(y, y_ref)


@pytest.mark.parametrize("n_token", [70, 256])
def test_fused_tf32_vs_default_fp32_baseline(n_token):
    """TF32 fused error is measured against default FP32 eager, not cuBLAS TF32."""
    c, n, dtype = 128, 4, torch.float32
    g, b, wa, wb, wo = _weights(c, n, dtype, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo).eval()
    x = torch.randn(1, n_token, n_token, c, dtype=dtype, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, 1, dtype=dtype, device="cuda")

    _set_tf32(False)
    with torch.inference_mode():
        y_fp32 = _eager(module, x, mask)

    _set_tf32(True)
    with torch.inference_mode():
        y_fused_tf32 = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        y_eager_tf32 = _eager(module, x, mask)

    # After round-nearest f32->tf32, fused TF32 matches eager TF32 vs FP32.
    torch.testing.assert_close(y_fused_tf32, y_fp32, atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(y_eager_tf32, y_fp32, atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(y_fused_tf32, y_eager_tf32, atol=5e-3, rtol=2e-3)


def test_small_m_uses_module_eager():
    _set_tf32(False)
    c, n, n_token, dtype = 128, 4, 32, torch.float32
    g, b, wa, wb, wo = _weights(c, n, dtype, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo).eval()
    x = torch.randn(1, n_token, n_token, c, dtype=dtype, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, 1, dtype=dtype, device="cuda")
    assert not is_fused_swiglu_transition_eligible(x, g, b, wa, wb, wo)
    with torch.inference_mode():
        y = module(x, mask=mask.squeeze(-1))
        y_ref = _eager(module, x, mask)
    compare_utils.assert_max_abs_diff_small(y, y_ref, 1e-5)


@pytest.mark.parametrize("n_token", [70, 128])
def test_module_fused_matches_eager_env_off(n_token, monkeypatch):
    torch.manual_seed(3)
    _set_tf32(False)
    module = SwiGLUTransition(c_in=128, n=4).cuda().eval()
    x = torch.randn(1, n_token, n_token, 128, device="cuda") * 0.5
    mask = torch.ones(1, n_token, n_token, device="cuda")
    with torch.inference_mode():
        y_fused = module(x, mask=mask)
        monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0")
        y_eager = module(x, mask=mask)
    torch.testing.assert_close(y_fused, y_eager, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_token", [70, 128])
def test_autograd_matches_eager_fp32(n_token):
    _set_tf32(False)
    torch.manual_seed(9)
    dtype = torch.float32
    c, n = 128, 4
    g, b, wa, wb, wo = _weights(c, n, dtype, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo).train()
    x = (
        torch.randn(1, n_token, n_token, c, dtype=dtype, device="cuda") * 0.5
    ).requires_grad_(True)
    mask = torch.ones(1, n_token, n_token, 1, dtype=dtype, device="cuda")

    y_ref = _eager(module, x, mask)
    y_ref.square().mean().backward()
    grads_ref = (
        x.grad.detach().clone(),
        module.layer_norm.weight.grad.detach().clone(),
        module.layer_norm.bias.grad.detach().clone(),
        module.swiglu.linear_a.weight.grad.detach().clone(),
        module.swiglu.linear_b.weight.grad.detach().clone(),
        module.linear_out.weight.grad.detach().clone(),
    )
    x.grad = None
    module.zero_grad(set_to_none=True)

    x_f = x.detach().requires_grad_(True)
    g_f = g.detach().requires_grad_(True)
    b_f = b.detach().requires_grad_(True)
    wa_f = wa.detach().requires_grad_(True)
    wb_f = wb.detach().requires_grad_(True)
    wo_f = wo.detach().requires_grad_(True)
    y = fused_swiglu_transition(x_f, g_f, b_f, wa_f, wb_f, wo_f, mask=mask)
    y.square().mean().backward()

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_f.grad, grads_ref[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g_f.grad, grads_ref[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b_f.grad, grads_ref[2], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(wa_f.grad, grads_ref[3], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(wb_f.grad, grads_ref[4], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(wo_f.grad, grads_ref[5], atol=1e-4, rtol=1e-4)
    assert y.data_ptr() != x_f.data_ptr()


@pytest.mark.parametrize("n_token", [70, 128, 256])
def test_autograd_matches_eager_bf16_mixed(n_token):
    """bf16 activations, fp32 masters vs module primitives."""
    _set_tf32(False)
    torch.manual_seed(9)
    c, n = 128, 4
    g, b, wa, wb, wo = _weights(c, n, torch.float32, "cuda")
    module = _bind_module(c, n, g, b, wa, wb, wo).train()
    x = (
        torch.randn(1, n_token, n_token, c, dtype=torch.bfloat16, device="cuda") * 0.5
    ).requires_grad_(True)
    mask = torch.ones(1, n_token, n_token, 1, dtype=torch.bfloat16, device="cuda")

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
    x.grad = None
    module.zero_grad(set_to_none=True)

    xs = [t.detach().requires_grad_(True) for t in (x, g, b, wa, wb, wo)]
    y = fused_swiglu_transition(*xs, mask=mask)
    y.square().mean().backward()

    assert y.dtype == torch.bfloat16
    assert xs[3].grad.dtype == torch.float32
    torch.testing.assert_close(y.float(), y_ref.float(), atol=5e-2, rtol=2e-2)
    for act, ref in zip(xs, grads_ref):
        torch.testing.assert_close(act.grad.float(), ref.float(), atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("n_token", [70, 128])
def test_module_bf16_mixed_matches_eager_env_off(n_token, monkeypatch):
    _set_tf32(False)
    torch.manual_seed(5)
    fused = SwiGLUTransition(c_in=128, n=4).cuda().train()
    eager = SwiGLUTransition(c_in=128, n=4).cuda().train()
    eager.load_state_dict(fused.state_dict())
    x = (
        torch.randn(1, n_token, n_token, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    ).requires_grad_(True)
    mask = torch.ones(1, n_token, n_token, device="cuda", dtype=torch.bfloat16)
    x_e = x.detach().clone().requires_grad_(True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_f = fused(x, mask=mask)
    monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_e = eager(x_e, mask=mask)

    torch.testing.assert_close(y_f.float(), y_e.float(), atol=5e-2, rtol=2e-2)
    y_f.float().square().mean().backward()
    y_e.float().square().mean().backward()
    torch.testing.assert_close(x.grad.float(), x_e.grad.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        fused.swiglu.linear_a.weight.grad,
        eager.swiglu.linear_a.weight.grad,
        atol=5e-2,
        rtol=2e-2,
    )


def test_compile_reuse_across_lengths():
    # Autotune may compile several tile configs for one precision/shape key;
    # length M is not in the key, so later Ns must not grow the filesystem cache.
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_swiglu_transition import (
    _fused_swiglu_transition_bwd_dw_kernel,
    _fused_swiglu_transition_bwd_dx_kernel,
    _fused_swiglu_transition_fwd_kernel,
    fused_swiglu_transition,
)
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)
def _device_caches(fn):
    # Autotuner wraps JITFunction (.fn); plain @triton.jit is the JITFunction.
    if hasattr(fn, "device_caches"):
        return fn.device_caches
    return fn.fn.device_caches
def clear():
    _device_caches(_fused_swiglu_transition_fwd_kernel).clear()
    _device_caches(_fused_swiglu_transition_bwd_dx_kernel).clear()
    _device_caches(_fused_swiglu_transition_bwd_dw_kernel).clear()
def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))
def run(n, x_dtype, w_dtype):
    x = torch.randn(1, n, n, c, device="cuda", dtype=x_dtype)
    mask = torch.ones(1, n, n, 1, device="cuda", dtype=x_dtype)
    gw = g.to(w_dtype)
    bw = b.to(w_dtype)
    waw = wa.to(w_dtype)
    wbw = wb.to(w_dtype)
    wow = wo.to(w_dtype)
    # Inference in-place (SAVE_ACTS=0) and training out-of-place (SAVE_ACTS=1).
    z = x.clone()
    fused_swiglu_transition(z, gw, bw, waw, wbw, wow, mask=mask, residual=z)
    xd = x.detach().requires_grad_(True)
    y = fused_swiglu_transition(
        xd,
        gw.detach().requires_grad_(True),
        bw.detach().requires_grad_(True),
        waw.detach().requires_grad_(True),
        wbw.detach().requires_grad_(True),
        wow.detach().requires_grad_(True),
        mask=mask,
    )
    y.backward(torch.ones_like(y))
    clear()
def snapshot():
    return {
        "forward": count("_fused_swiglu_transition_fwd_kernel"),
        "bwd_dx": count("_fused_swiglu_transition_bwd_dx_kernel"),
        "bwd_dw": count("_fused_swiglu_transition_bwd_dw_kernel"),
    }
torch.backends.cuda.matmul.allow_tf32 = False
c, h = 128, 512
g = torch.randn(c, device="cuda")
b = torch.randn(c, device="cuda")
wa = torch.randn(h, c, device="cuda")
wb = torch.randn(h, c, device="cuda")
wo = torch.randn(c, h, device="cuda")
lengths = (64, 80, 96, 112, 128)
results = {}
for x_dtype, w_dtype, tag in (
    (torch.float32, torch.float32, "fp32"),
    (torch.bfloat16, torch.float32, "bf16_mixed"),
):
    run(lengths[0], x_dtype, w_dtype)
    after_first = snapshot()
    for n in lengths[1:]:
        run(n, x_dtype, w_dtype)
    after_all = snapshot()
    results[tag] = {"after_first": after_first, "after_all": after_all}
    assert after_first["forward"] >= 1, (tag, after_first)
    assert after_first["bwd_dx"] >= 1 and after_first["bwd_dw"] >= 1, (tag, after_first)
    assert after_all == after_first, (tag, after_first, after_all)
print(json.dumps(results))
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
