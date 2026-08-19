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

"""Parity and compile-reuse tests for fused triangle multiplicative update."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_trimul import (
    eager_trimul,
    fused_trimul,
    fused_trimul_from_module,
    is_fused_trimul_eligible,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]

C_Z = 128
N_TOKEN = 64


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _randomize_observable(module):
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.linear_g.weight.normal_(0, 0.02)


def _module(outgoing=True, c_z=C_Z, act_dtype=torch.float32):
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    module = cls(c_z, c_z).cuda().eval()
    _randomize_observable(module)
    if act_dtype != torch.float32:
        # Keep fp32 Parameter masters; only activations change dtype.
        pass
    return module


def _weights(module):
    return (
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_z.weight,
        module.linear_g.weight,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
    )


def _pair(n=N_TOKEN, c_z=C_Z, dtype=torch.float32, use_mask=True):
    torch.manual_seed(0)
    z = torch.randn(1, n, n, c_z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype) if use_mask else None
    return z, mask


@pytest.fixture(autouse=True)
def _enable_fused(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_TRIMUL", "1")
    monkeypatch.delenv("OPENFOLD3_TRIMUL_CHUNK_CAP", raising=False)


def test_fused_matches_eager_ieee():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y_fused = fused_trimul(
            z, mask, *_weights(module), True, ln_in_eps=module.layer_norm_in.eps
        )
        y_ref = eager_trimul(
            z, mask, *_weights(module), True, ln_in_eps=module.layer_norm_in.eps
        )
    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-5)


def test_fused_tf32_matches_eager_tf32():
    _set_tf32(True)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y_fused = fused_trimul(z, mask, *_weights(module), True)
        y_ref = eager_trimul(z, mask, *_weights(module), True)
    torch.testing.assert_close(y_fused, y_ref, atol=5e-3, rtol=2e-3)


def test_fused_bf16_mixed_matches_eager_mixed():
    _set_tf32(False)
    module = _module()
    z, mask = _pair(dtype=torch.bfloat16)
    with torch.inference_mode():
        y_fused = fused_trimul(z, mask, *_weights(module), True)
        y_ref = eager_trimul(z, mask, *_weights(module), True)
    assert y_fused.dtype == torch.bfloat16
    torch.testing.assert_close(y_fused.float(), y_ref.float(), atol=8e-2, rtol=2e-2)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_fused_matches_module_ieee(outgoing, use_mask):
    _set_tf32(False)
    module = _module(outgoing=outgoing)
    z, mask = _pair(use_mask=use_mask)
    with torch.inference_mode():
        y_ref = module.forward(z.clone(), mask=mask, inplace_safe=False)
        y_fused = fused_trimul_from_module(module, z.clone(), mask)
    assert y_fused is not None
    torch.testing.assert_close(y_fused, y_ref, atol=8e-4, rtol=8e-4)


def test_incoming_tf32_matches_eager():
    _set_tf32(True)
    module = _module(outgoing=False)
    z, mask = _pair()
    with torch.inference_mode():
        y_fused = fused_trimul(z, mask, *_weights(module), False)
        y_ref = eager_trimul(z, mask, *_weights(module), False)
    torch.testing.assert_close(y_fused, y_ref, atol=5e-3, rtol=2e-3)


def test_with_add_inplace():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        upd = eager_trimul(z, mask, *_weights(module), True)
        ref = z + upd
        z_in = z.clone()
        fused = fused_trimul(
            z_in, mask, *_weights(module), True, residual=z_in
        )
    assert fused.data_ptr() == z_in.data_ptr()
    torch.testing.assert_close(fused, ref, atol=8e-4, rtol=8e-4)


def test_chunked_outgoing_matches_whole(monkeypatch):
    _set_tf32(False)
    module = _module()
    z, mask = _pair(n=80)
    with torch.inference_mode():
        whole = fused_trimul(z, mask, *_weights(module), True)
        chunked = fused_trimul(z, mask, *_weights(module), True, chunk_cap=32)
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


def test_chunked_incoming_matches_whole():
    _set_tf32(False)
    module = _module(outgoing=False)
    z, mask = _pair(n=80)
    with torch.inference_mode():
        whole = fused_trimul(z, mask, *_weights(module), False)
        chunked = fused_trimul(z, mask, *_weights(module), False, chunk_cap=32)
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


def test_bf16_weights_are_ineligible():
    _set_tf32(False)
    module = _module()
    z, _mask = _pair(dtype=torch.bfloat16)
    wa_p = module.linear_a_p.weight.to(torch.bfloat16)
    assert not is_fused_trimul_eligible(
        z,
        wa_p,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_z.weight,
        module.linear_g.weight,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
    )


def test_small_n_is_ineligible():
    _set_tf32(False)
    module = _module()
    z, _mask = _pair(n=32)
    assert not is_fused_trimul_eligible(z, *_weights(module))


def test_module_dispatch_uses_fused():
    _set_tf32(False)
    module = _module()
    z, mask = _pair()
    with torch.inference_mode():
        y_mod = module(z.clone(), mask=mask, inplace_safe=False)
        y_ref = eager_trimul(z, mask, *_weights(module), True)
    torch.testing.assert_close(y_mod, y_ref, atol=8e-4, rtol=8e-4)


def test_compile_reuse_across_lengths():
    script = r"""
import json
import os
from pathlib import Path
import torch
from openfold3.core.kernels.triton.fused_trimul import (
    _gated_dual_gemm_kernel,
    _gated_out_from_dm_kernel,
    _ln_stats_kernel,
    fused_trimul,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
)
cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

def snapshot():
    return {
        "dual": count("_gated_dual_gemm_kernel"),
        "out": count("_gated_out_from_dm_kernel"),
        "stats": count("_ln_stats_kernel"),
    }

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
m = TriangleMultiplicationOutgoing(128, 128).cuda().eval()
weights = (
    m.linear_a_p.weight,
    m.linear_a_g.weight,
    m.linear_b_p.weight,
    m.linear_b_g.weight,
    m.linear_z.weight,
    m.linear_g.weight,
    m.layer_norm_in.weight,
    m.layer_norm_in.bias,
    m.layer_norm_out.weight,
    m.layer_norm_out.bias,
)
lengths = (64, 80, 96, 112, 128)

def run(n):
    z = torch.randn(1, n, n, 128, device="cuda") * 0.5
    mask = torch.ones(1, n, n, device="cuda")
    with torch.inference_mode():
        fused_trimul(z, mask, *weights, True)

run(lengths[0])
after_first = snapshot()
for n in lengths[1:]:
    run(n)
after_all = snapshot()
assert after_first["dual"] >= 1, after_first
assert after_first["out"] >= 1, after_first
assert after_first["stats"] >= 1, after_first
assert after_all == after_first, (after_first, after_all)
print(json.dumps({"after_first": after_first, "after_all": after_all}))
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_TRIMUL"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "after_all" in result.stdout
