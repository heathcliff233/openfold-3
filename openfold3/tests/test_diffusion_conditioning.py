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

import unittest
from unittest.mock import patch

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton.fused_swiglu_transition import (
    is_fused_swiglu_transition_eligible,
)
from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.core.model.layers.transition import SwiGLUTransition
from openfold3.core.utils.relpos import relpos_complex
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts


class TestDiffusionConditioning(unittest.TestCase):
    def _batch(self, batch_size: int, n_token: int) -> dict:
        return {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn(batch_size, device=si_trunk.device)
        )

        batch = self._batch(batch_size, n_token)

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (batch_size, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, n_token, n_token, c_z))

    def test_relpos_row_slice_matches_full(self):
        batch_size = 2
        n_token = 17
        batch = self._batch(batch_size, n_token)
        full = relpos_complex(batch, max_relative_idx=32, max_relative_chain=2)
        chunk = 5
        parts = [
            relpos_complex(
                batch,
                max_relative_idx=32,
                max_relative_chain=2,
                row_slice=slice(i, min(i + chunk, n_token)),
            )
            for i in range(0, n_token, chunk)
        ]
        rebuilt = torch.cat(parts, dim=-3)
        self.assertTrue(torch.equal(rebuilt, full))

    def test_chunked_pair_embed_matches_eager(self):
        batch_size = 1
        n_token = 37
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})
        dc = DiffusionConditioning(**diff_cond_config)
        dc.eval()

        torch.manual_seed(0)
        batch = self._batch(batch_size, n_token)
        zij_trunk = torch.randn(batch_size, n_token, n_token, c_z)

        with torch.no_grad():
            zij_chunked = dc._embed_zij_chunked(batch, zij_trunk)
        # Eager path is taken when gradients are enabled.
        with torch.enable_grad():
            zij_eager = dc._embed_zij(batch, zij_trunk.detach())

        self.assertTrue(
            torch.allclose(zij_chunked, zij_eager, atol=1e-5, rtol=1e-5),
            f"max abs diff={(zij_chunked - zij_eager).abs().max().item():.3e}",
        )

    def test_with_different_schedule(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        n_sample = 3

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn((batch_size, n_sample), device=si_trunk.device)
        )

        batch = {
            "token_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "token_mask": torch.ones((batch_size, 1, n_token)),
            "residue_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "sym_id": torch.zeros((batch_size, 1, n_token)),
            "asym_id": torch.zeros((batch_size, 1, n_token)),
            "entity_id": torch.zeros((batch_size, 1, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (batch_size, n_sample, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))

    def test_with_same_schedule(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn((1, 1), device=si_trunk.device)
        )

        batch = {
            "token_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "token_mask": torch.ones((batch_size, 1, n_token)),
            "residue_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "sym_id": torch.zeros((batch_size, 1, n_token)),
            "asym_id": torch.zeros((batch_size, 1, n_token)),
            "entity_id": torch.zeros((batch_size, 1, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (batch_size, 1, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))


def _cuda_conditioning_module(n_token: int = 64):
    proj_entry = OF3ProjectEntry()
    config = proj_entry.get_model_config_with_presets()
    c_s_input = consts.c_s + 65
    diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
    diff_cond_config.update(
        {"c_s": consts.c_s, "c_s_input": c_s_input, "c_z": consts.c_z}
    )
    return DiffusionConditioning(**diff_cond_config).cuda().eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@compare_utils.skip_unless_cuda_available()
@compare_utils.skip_unless_triton_installed()
def test_inference_pair_swiglu_inplace_matches_eager_add(dtype, monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1")
    n_token = 64
    dc = _cuda_conditioning_module(n_token)
    torch.manual_seed(0)
    si = torch.randn(1, n_token, consts.c_s, device="cuda", dtype=dtype)
    zij = torch.randn(1, n_token, n_token, consts.c_z, device="cuda", dtype=dtype)
    token_mask = torch.ones(1, n_token, device="cuda", dtype=dtype)
    weights = dc.transition_z[0]._fused_weight_args()
    if not is_fused_swiglu_transition_eligible(zij, *weights):
        pytest.skip("fused SwiGLU is not eligible for this shape")

    zij_ref = zij.clone()
    with torch.no_grad():
        monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0")
        _, zij_eager = dc._forward(si.clone(), zij_ref, token_mask)
        monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1")
        ptr_before = zij.data_ptr()
        calls = {"n": 0}
        real = SwiGLUTransition._transition_inplace

        def _wrapped(self, *args, **kwargs):
            calls["n"] += 1
            return real(self, *args, **kwargs)

        with patch.object(SwiGLUTransition, "_transition_inplace", _wrapped):
            _, zij_fused = dc._forward(si, zij, token_mask)
    assert calls["n"] == len(dc.transition_z)
    assert zij_fused.data_ptr() == ptr_before
    atol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(zij_fused, zij_eager, atol=atol, rtol=atol)


@compare_utils.skip_unless_cuda_available()
@compare_utils.skip_unless_triton_installed()
def test_training_pair_swiglu_stays_out_of_place(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "1")
    n_token = 64
    dc = _cuda_conditioning_module(n_token)
    torch.manual_seed(0)
    si = torch.randn(1, n_token, consts.c_s, device="cuda", requires_grad=True)
    zij = torch.randn(
        1, n_token, n_token, consts.c_z, device="cuda", requires_grad=True
    )
    token_mask = torch.ones(1, n_token, device="cuda")
    ptr_before = zij.data_ptr()
    with patch.object(
        SwiGLUTransition,
        "_transition_inplace",
        wraps=SwiGLUTransition._transition_inplace,
    ) as inplace:
        _, zij_out = dc._forward(si, zij, token_mask)
        zij_out.square().mean().backward()
    assert inplace.call_count == 0
    assert zij_out.data_ptr() != ptr_before
    assert zij.grad is not None


if __name__ == "__main__":
    unittest.main()
