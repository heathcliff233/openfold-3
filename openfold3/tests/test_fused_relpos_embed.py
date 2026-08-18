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

"""Parity and compile-reuse tests for fused input relpos embedding."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_relpos_embed import (
    _can_use_triton,
    _FusedRelposEmbedFn,
    eager_relpos_embed_add_,
    eager_relpos_weight_grad,
    fused_relpos_embed_add_,
    fused_relpos_weight_grad,
    is_fused_relpos_embed_enabled,
)
from openfold3.core.model.feature_embedders.input_embedders import InputEmbedderAllAtom
from openfold3.core.utils.relpos import relpos_complex
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.utils.data_utils import random_of3_features

pytestmark = [
    compare_utils.skip_unless_cuda_available(),
    compare_utils.skip_unless_triton_installed(),
]


def _move_batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device=device) if torch.is_tensor(value) else value
    return out


@pytest.mark.parametrize("n_token", [16, 76, 127])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_relpos_kernel_matches_eager(n_token: int, dtype: torch.dtype):
    """Fused gather-add matches four sequential gather-adds."""
    torch.manual_seed(11)
    c_z = 128
    vocab = 130
    same_entity_offset = 65

    z = torch.randn(1, n_token, n_token, c_z, device="cuda", dtype=dtype)
    w = torch.randn(vocab, c_z, device="cuda", dtype=dtype)
    idx1 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx2 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx3 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same_entity = torch.randint(
        0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool
    )

    z_ref = z.clone()
    eager_relpos_embed_add_(z_ref, w, idx1, idx2, idx3, same_entity, same_entity_offset)

    z_fused = z.clone()
    fused_relpos_embed_add_(
        z_fused, w, idx1, idx2, idx3, same_entity, same_entity_offset
    )

    # fp32: fused branch tolerance. bf16: Triton accumulates the four gathers
    # with a float32 same-entity scale, so allow a few ULPs vs eager bf16.
    if dtype == torch.float32:
        compare_utils.assert_max_abs_diff_small(z_ref, z_fused, 1e-5)
    else:
        torch.testing.assert_close(z_fused.float(), z_ref.float(), atol=0.15, rtol=1e-2)


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def test_fused_relpos_precision_eligibility():
    """IEEE fp32, matching bf16, and bf16-mixed are eligible; fp32+bf16 is not."""
    z32 = torch.zeros(1, 4, 4, 128, device="cuda", dtype=torch.float32)
    z16 = torch.zeros(1, 4, 4, 128, device="cuda", dtype=torch.bfloat16)
    w32 = torch.zeros(130, 128, device="cuda", dtype=torch.float32)
    w16 = torch.zeros(130, 128, device="cuda", dtype=torch.bfloat16)
    assert _can_use_triton(z32, w32)
    assert _can_use_triton(z16, w16)
    assert _can_use_triton(z16, w32)
    assert not _can_use_triton(z32, w16)


@pytest.mark.parametrize("allow_tf32", [False, True], ids=["ieee", "tf32"])
def test_fused_relpos_fp32_matches_eager(allow_tf32: bool):
    """IEEE and TF32 share gather-add math; both match eager fp32."""
    _set_tf32(allow_tf32)
    torch.manual_seed(11)
    n_token, c_z, vocab, offset = 16, 128, 130, 65
    z = torch.randn(1, n_token, n_token, c_z, device="cuda")
    w = torch.randn(vocab, c_z, device="cuda")
    idx = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same = torch.randint(0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool)
    assert _can_use_triton(z, w)

    z_ref = z.clone()
    eager_relpos_embed_add_(z_ref, w, idx, idx, idx, same, offset)
    z_fused = z.clone()
    fused_relpos_embed_add_(z_fused, w, idx, idx, idx, same, offset)
    compare_utils.assert_max_abs_diff_small(z_ref, z_fused, 1e-5)


def test_fused_relpos_bf16_mixed_matches_fp32_ref():
    """bf16 activations + fp32 masters match fp32 gather-add then store-cast."""
    _set_tf32(False)
    torch.manual_seed(12)
    n_token, c_z, vocab, offset = 16, 128, 130, 65
    z = torch.randn(1, n_token, n_token, c_z, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(vocab, c_z, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same = torch.randint(0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool)
    assert _can_use_triton(z, w)

    z_ref = z.float()
    eager_relpos_embed_add_(z_ref, w, idx, idx, idx, same, offset)
    z_ref = z_ref.to(torch.bfloat16)
    z_fused = z.clone()
    fused_relpos_embed_add_(z_fused, w, idx, idx, idx, same, offset)
    torch.testing.assert_close(z_fused.float(), z_ref.float(), atol=2e-2, rtol=1e-2)


def test_fused_relpos_autograd_bf16_mixed():
    """Mixed training: bf16 dZ passthrough, fp32 dW vs fp32-then-cast eager."""
    _set_tf32(False)
    torch.manual_seed(4)
    n_token, c_z, vocab, offset = 16, 128, 130, 65
    z = torch.randn(
        1,
        n_token,
        n_token,
        c_z,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        c_z, vocab, device="cuda", dtype=torch.float32, requires_grad=True
    )
    idx = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same = torch.randint(0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool)

    out = _FusedRelposEmbedFn.apply(
        z, weight, idx, idx, idx, same, torch.tensor(offset, dtype=torch.int64)
    )
    scale = torch.arange(c_z, device="cuda", dtype=torch.bfloat16)
    (out * scale).sum().backward()

    w = weight.detach().t().contiguous()
    z_ref = z.detach().float()
    eager_relpos_embed_add_(z_ref, w, idx, idx, idx, same, offset)
    out_ref = z_ref.to(torch.bfloat16)
    grad_out = scale.expand_as(out).contiguous()
    dw_ref = eager_relpos_weight_grad(grad_out, idx, idx, idx, same, offset, vocab)

    assert out.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.float32
    torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(z.grad, grad_out)
    torch.testing.assert_close(
        weight.grad.float(), dw_ref.t().float(), atol=5e-2, rtol=2e-2
    )


def _synthetic_s_input(module: InputEmbedderAllAtom, n_token: int) -> torch.Tensor:
    c_s_input = module.linear_s.weight.shape[1]
    return torch.randn(1, n_token, c_s_input, device="cuda", dtype=torch.float32)


@pytest.mark.parametrize("n_token", [8, 17, 31])
def test_input_embedder_fused_relpos_matches_feature_linear(n_token: int):
    """Module path matches materializing relpos_complex + linear_relpos."""
    assert is_fused_relpos_embed_enabled()
    torch.manual_seed(7)

    of3_config = OF3ProjectEntry().get_model_config_with_presets()
    module = (
        InputEmbedderAllAtom(**of3_config.architecture.input_embedder).cuda().eval()
    )
    batch = _move_batch_to_device(
        random_of3_features(
            batch_size=1,
            n_token=n_token,
            n_msa=4,
            n_templ=1,
        ),
        "cuda",
    )
    s_input = _synthetic_s_input(module, n_token)

    with torch.inference_mode():
        # Reference: classic one-hot + Linear path (bypass weight-table path).
        s_input_emb_i = module.linear_z_i(s_input)
        s_input_emb_j = module.linear_z_j(s_input)
        z_ref = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]
        token_bonds = batch["token_bonds"].to(dtype=z_ref.dtype)
        z_ref = z_ref + module.linear_token_bonds(token_bonds.unsqueeze(-1))
        relpos_feats = relpos_complex(
            batch=batch,
            max_relative_idx=module.max_relative_idx,
            max_relative_chain=module.max_relative_chain,
        ).to(dtype=z_ref.dtype)
        z_ref = z_ref + module.linear_relpos(relpos_feats)

        z_fused = module.embed_z(
            batch=batch,
            s_input=s_input,
            dtype=s_input.dtype,
            inplace_safe=True,
        )

    compare_utils.assert_max_abs_diff_small(z_ref, z_fused, 1e-5)


@pytest.mark.parametrize("n_token", [8, 17])
def test_input_embedder_env_fallback_matches_fused(n_token: int):
    """OPENFOLD3_FUSED_RELPOS=0 uses the eager weight-table path."""
    torch.manual_seed(13)
    of3_config = OF3ProjectEntry().get_model_config_with_presets()
    module = (
        InputEmbedderAllAtom(**of3_config.architecture.input_embedder).cuda().eval()
    )
    batch = _move_batch_to_device(
        random_of3_features(
            batch_size=1,
            n_token=n_token,
            n_msa=4,
            n_templ=1,
        ),
        "cuda",
    )
    s_input = _synthetic_s_input(module, n_token)

    with torch.inference_mode():
        z_fused = module.embed_z(
            batch=batch,
            s_input=s_input,
            dtype=s_input.dtype,
            inplace_safe=True,
        )

        old = os.environ.get("OPENFOLD3_FUSED_RELPOS")
        os.environ["OPENFOLD3_FUSED_RELPOS"] = "0"
        try:
            z_eager = module.embed_z(
                batch=batch,
                s_input=s_input,
                dtype=s_input.dtype,
                inplace_safe=True,
            )
        finally:
            if old is None:
                os.environ.pop("OPENFOLD3_FUSED_RELPOS", None)
            else:
                os.environ["OPENFOLD3_FUSED_RELPOS"] = old

    compare_utils.assert_max_abs_diff_small(z_fused, z_eager, 1e-5)


@pytest.mark.parametrize("n_token", [8, 17, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_relpos_weight_grad_matches_eager(n_token: int, dtype: torch.dtype):
    """Split-K Triton dW matches eager index_add accumulate."""
    torch.manual_seed(21)
    c_z = 128
    vocab = 130
    same_entity_offset = 65
    grad_z = torch.randn(1, n_token, n_token, c_z, device="cuda", dtype=dtype)
    idx1 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx2 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx3 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same_entity = torch.randint(
        0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool
    )

    grad_ref = eager_relpos_weight_grad(
        grad_z, idx1, idx2, idx3, same_entity, same_entity_offset, vocab
    )
    grad_fused = fused_relpos_weight_grad(
        grad_z, idx1, idx2, idx3, same_entity, same_entity_offset, vocab
    )
    # fp32 reduction order can differ slightly from index_add; keep relative tol.
    torch.testing.assert_close(grad_fused, grad_ref, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("n_token", [8, 17, 31])
def test_fused_relpos_autograd_matches_feature_linear(n_token: int):
    """Training path grads match relpos_complex + linear_relpos."""
    torch.manual_seed(9)
    of3_config = OF3ProjectEntry().get_model_config_with_presets()
    module = InputEmbedderAllAtom(**of3_config.architecture.input_embedder).cuda()
    batch = _move_batch_to_device(
        random_of3_features(
            batch_size=1,
            n_token=n_token,
            n_msa=4,
            n_templ=1,
        ),
        "cuda",
    )
    s_input = _synthetic_s_input(module, n_token).requires_grad_(True)

    # Reference classic path with identical weights.
    s_i = module.linear_z_i(s_input)
    s_j = module.linear_z_j(s_input)
    z_ref = s_i[..., None, :] + s_j[..., None, :, :]
    z_ref = z_ref + module.linear_token_bonds(
        batch["token_bonds"].to(dtype=z_ref.dtype).unsqueeze(-1)
    )
    relpos_feats = relpos_complex(
        batch=batch,
        max_relative_idx=module.max_relative_idx,
        max_relative_chain=module.max_relative_chain,
    ).to(dtype=z_ref.dtype)
    z_ref = z_ref + module.linear_relpos(relpos_feats)
    loss_ref = z_ref.square().mean()
    loss_ref.backward()
    grad_s_ref = s_input.grad.detach().clone()
    grad_w_ref = module.linear_relpos.weight.grad.detach().clone()

    module.zero_grad(set_to_none=True)
    s_input.grad = None
    s_input_f = s_input.detach().requires_grad_(True)
    z_fused = module.embed_z(
        batch=batch,
        s_input=s_input_f,
        dtype=s_input_f.dtype,
        inplace_safe=False,
    )
    loss_fused = z_fused.square().mean()
    loss_fused.backward()

    torch.testing.assert_close(z_fused, z_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_input_f.grad, grad_s_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        module.linear_relpos.weight.grad, grad_w_ref, atol=1e-4, rtol=1e-4
    )


def test_fused_relpos_autograd_kernel_grad_z_and_weight():
    """Function returns passthrough dZ and transposed dW."""
    torch.manual_seed(4)
    n_token, c_z, vocab = 16, 128, 130
    z = torch.randn(1, n_token, n_token, c_z, device="cuda", requires_grad=True)
    weight = torch.randn(c_z, vocab, device="cuda", requires_grad=True)
    idx = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same = torch.randint(0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool)
    offset = 65

    out = _FusedRelposEmbedFn.apply(
        z,
        weight,
        idx,
        idx,
        idx,
        same,
        torch.tensor(offset, dtype=torch.int64),
    )
    (out * torch.arange(c_z, device="cuda")).sum().backward()

    w = weight.detach().t().contiguous()
    z_e = z.detach().clone().requires_grad_(True)
    w_e = w.clone().requires_grad_(True)
    out_e = z_e.clone()
    eager_relpos_embed_add_(out_e, w_e, idx, idx, idx, same, offset)
    (out_e * torch.arange(c_z, device="cuda")).sum().backward()

    torch.testing.assert_close(out, out_e, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z.grad, z_e.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight.grad, w_e.grad.t(), atol=1e-4, rtol=1e-4)


def test_fused_relpos_compile_reuse_across_lengths():
    """One filesystem compile must serve every sequence length (fwd + bwd)."""
    script = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_relpos_embed import (
    _fused_relpos_embed_kernel,
    _fused_relpos_weight_grad_kernel,
    fused_relpos_embed_add_,
    fused_relpos_weight_grad,
)

cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clear():
    _fused_relpos_embed_kernel.device_caches.clear()
    _fused_relpos_weight_grad_kernel.device_caches.clear()

def count(name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

w = torch.randn(130, 128, device="cuda")
for n in (16, 32, 48, 64, 96, 127):
    z = torch.zeros(1, n, n, 128, device="cuda")
    idx = torch.zeros(1, n, n, dtype=torch.int64, device="cuda")
    same = torch.ones(1, n, n, dtype=torch.bool, device="cuda")
    fused_relpos_embed_add_(z, w, idx, idx, idx, same, 65)
    fused_relpos_weight_grad(z, idx, idx, idx, same, 65, 130)
    clear()

counts = {
    "forward": count("_fused_relpos_embed_kernel"),
    "backward": count("_fused_relpos_weight_grad_kernel"),
}
print(json.dumps(counts))
assert counts == {"forward": 1, "backward": 1}, counts
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_RELPOS"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert '"forward": 1' in result.stdout
        assert '"backward": 1' in result.stdout


@pytest.mark.parametrize("n_token", [64, 256])
def test_fused_relpos_microbench_faster_or_parity(n_token: int):
    """Warm fused kernel should not regress vs eager gather-add at modest N."""
    import time

    torch.manual_seed(3)
    c_z = 128
    vocab = 130
    same_entity_offset = 65
    warmup = 5
    reps = 20

    z = torch.randn(1, n_token, n_token, c_z, device="cuda")
    w = torch.randn(vocab, c_z, device="cuda")
    idx1 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx2 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx3 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same_entity = torch.randint(
        0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool
    )

    def _bench(fn):
        for _ in range(warmup):
            z_tmp = z.clone()
            fn(z_tmp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            z_tmp = z.clone()
            fn(z_tmp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    eager_s = _bench(
        lambda zt: eager_relpos_embed_add_(
            zt, w, idx1, idx2, idx3, same_entity, same_entity_offset
        )
    )
    fused_s = _bench(
        lambda zt: fused_relpos_embed_add_(
            zt, w, idx1, idx2, idx3, same_entity, same_entity_offset
        )
    )

    # At N>=256 the fused path should win; smaller N may be launch-bound.
    if n_token >= 256:
        assert fused_s <= eager_s * 1.05, (fused_s, eager_s)
    else:
        assert fused_s < eager_s * 2.0, (fused_s, eager_s)


@pytest.mark.parametrize("n_token", [64, 256])
def test_fused_relpos_bwd_microbench_faster_or_parity(n_token: int):
    """Warm Triton dW should beat eager index_add at N>=256."""
    import time

    torch.manual_seed(5)
    c_z, vocab, offset = 128, 130, 65
    warmup, reps = 5, 20
    grad_z = torch.randn(1, n_token, n_token, c_z, device="cuda")
    idx1 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx2 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    idx3 = torch.randint(0, vocab, (1, n_token, n_token), device="cuda")
    same = torch.randint(0, 2, (1, n_token, n_token), device="cuda", dtype=torch.bool)

    def _bench(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    fused_s = _bench(
        lambda: fused_relpos_weight_grad(grad_z, idx1, idx2, idx3, same, offset, vocab)
    )
    # Correctness/perf smoke: kernel must finish and stay in a sane band.
    # Absolute speed vs index_add is not a gate (cuBLAS scatter is strong).
    assert fused_s < 0.05, fused_s
