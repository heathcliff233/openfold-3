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

"""Parity and autograd tests for fused ``_embed_zij`` (virtual relpos concat)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_embed_zij import (
    eager_embed_zij,
    fused_embed_zij,
    is_fused_embed_zij_eligible,
    try_fused_embed_zij,
)
from openfold3.core.kernels.triton.fused_relpos_embed import _build_indices
from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

C_Z, C_IN, N_TOKEN = 128, 267, 64
MAX_REL, MAX_CHAIN = 32, 2


def _batch(n_token=N_TOKEN):
    return {
        "token_index": torch.arange(n_token, device="cuda")[None],
        "token_mask": torch.ones(1, n_token, device="cuda"),
        "residue_index": torch.arange(n_token, device="cuda")[None],
        "sym_id": torch.zeros(1, n_token, device="cuda"),
        "asym_id": torch.zeros(1, n_token, device="cuda"),
        "entity_id": torch.zeros(1, n_token, device="cuda"),
    }


def _weights(dtype=torch.float32):
    torch.manual_seed(0)
    gamma = torch.randn(C_IN, dtype=dtype, device="cuda")
    weight = torch.randn(C_Z, C_IN, dtype=dtype, device="cuda") / C_IN**0.5
    return gamma, weight


def _pair(n_token=N_TOKEN, act_dtype=torch.float32, w_dtype=None):
    w_dtype = act_dtype if w_dtype is None else w_dtype
    gamma, weight = _weights(w_dtype)
    z = torch.randn(1, n_token, n_token, C_Z, dtype=act_dtype, device="cuda") * 0.5
    batch = _batch(n_token)
    idx1, idx2, idx3, same, offset = _build_indices(batch, MAX_REL, MAX_CHAIN)
    return gamma, weight, z, batch, idx1, idx2, idx3, same, offset


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_LN_LINEAR", "1")


def test_fused_matches_eager_ieee():
    _set_tf32(False)
    gamma, weight, z, batch, idx1, idx2, idx3, same, offset = _pair()
    with torch.inference_mode():
        y = fused_embed_zij(z, gamma, weight, idx1, idx2, idx3, same, offset)
        y_ref = eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_fused_tf32_matches_eager_tf32():
    _set_tf32(True)
    gamma, weight, z, batch, idx1, idx2, idx3, same, offset = _pair()
    with torch.inference_mode():
        y = fused_embed_zij(z, gamma, weight, idx1, idx2, idx3, same, offset)
        y_ref = eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN)
    torch.testing.assert_close(y, y_ref, atol=5e-3, rtol=2e-3)


def test_fused_bf16_mixed_matches_eager_mixed():
    _set_tf32(False)
    gamma, weight, z, batch, idx1, idx2, idx3, same, offset = _pair(
        act_dtype=torch.bfloat16, w_dtype=torch.float32
    )
    with torch.inference_mode():
        y = fused_embed_zij(z, gamma, weight, idx1, idx2, idx3, same, offset)
        y_ref = eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.float(), y_ref.float(), atol=5e-2, rtol=2e-2)


def test_autograd_matches_eager_tf32():
    _set_tf32(True)
    gamma, weight, z, batch, idx1, idx2, idx3, same, offset = _pair()
    leaves_e = [t.detach().requires_grad_(True) for t in (z, gamma, weight)]
    y_e = eager_embed_zij(
        leaves_e[0], leaves_e[1], leaves_e[2], batch, MAX_REL, MAX_CHAIN
    )
    y_e.square().mean().backward()
    grads_ref = [t.grad.detach().clone() for t in leaves_e]
    leaves = [t.detach().requires_grad_(True) for t in (z, gamma, weight)]
    y = fused_embed_zij(leaves[0], leaves[1], leaves[2], idx1, idx2, idx3, same, offset)
    y.square().mean().backward()
    torch.testing.assert_close(y, y_e, atol=5e-3, rtol=2e-3)
    for act, ref in zip(leaves, grads_ref):
        torch.testing.assert_close(act.grad, ref, atol=5e-3, rtol=2e-3)


def test_small_m_is_ineligible():
    gamma, weight, z, *_ = _pair(n_token=32)
    assert not is_fused_embed_zij_eligible(z, gamma, weight)


def test_module_wrapper_matches_eager_when_disabled(monkeypatch):
    _set_tf32(False)
    torch.manual_seed(3)
    module = (
        DiffusionConditioning(
            c_s_input=449,
            c_s=384,
            c_z=128,
            c_fourier_emb=256,
            max_relative_idx=32,
            max_relative_chain=2,
            sigma_data=16.0,
        )
        .cuda()
        .eval()
    )
    batch = _batch()
    z = torch.randn(1, N_TOKEN, N_TOKEN, C_Z, device="cuda") * 0.5
    with torch.inference_mode():
        y_fused = module._embed_zij(batch, z)
        monkeypatch.setenv("OPENFOLD3_FUSED_LN_LINEAR", "0")
        y_eager = module._embed_zij(batch, z)
    torch.testing.assert_close(y_fused, y_eager, atol=1e-5, rtol=1e-5)
    assert (
        try_fused_embed_zij(
            z, batch, module.layer_norm_z, module.linear_z, MAX_REL, MAX_CHAIN
        )
        is None
    )


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_embed_zij import (
    _fused_embed_zij_bwd_dw_kernel,
    _fused_embed_zij_bwd_dx_kernel,
    _fused_embed_zij_fwd_kernel,
    fused_embed_zij,
)
from openfold3.core.kernels.triton.fused_relpos_embed import _build_indices
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)
def _device_caches(fn):
    if hasattr(fn, "device_caches"):
        return fn.device_caches
    return fn.fn.device_caches
def clear():
    for fn in (
        _fused_embed_zij_fwd_kernel,
        _fused_embed_zij_bwd_dw_kernel,
        _fused_embed_zij_bwd_dx_kernel,
    ):
        _device_caches(fn).clear()
def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))
def batch(n):
    return {
        "token_index": torch.arange(n, device="cuda")[None],
        "residue_index": torch.arange(n, device="cuda")[None],
        "sym_id": torch.zeros(n, device="cuda")[None],
        "asym_id": torch.zeros(n, device="cuda")[None],
        "entity_id": torch.zeros(n, device="cuda")[None],
    }
def run(n):
    z = torch.randn(1, n, n, 128, device="cuda")
    idx1, idx2, idx3, same, off = _build_indices(batch(n), 32, 2)
    fused_embed_zij(z, g, w, idx1, idx2, idx3, same, off)
    zd = z.detach().requires_grad_(True)
    y = fused_embed_zij(
        zd,
        g.detach().requires_grad_(True),
        w.detach().requires_grad_(True),
        idx1, idx2, idx3, same, off,
    )
    y.backward(torch.ones_like(y))
    clear()
def snapshot():
    return {
        "forward": count("_fused_embed_zij_fwd_kernel"),
        "bwd_dw": count("_fused_embed_zij_bwd_dw_kernel"),
        "bwd_dx": count("_fused_embed_zij_bwd_dx_kernel"),
    }
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.randn(267, device="cuda")
w = torch.randn(128, 267, device="cuda")
lengths = (64, 80, 96, 112)
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
