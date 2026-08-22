#!/usr/bin/env python
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

"""Same-precision fused vs eager ``_embed_zij`` (TF32 and bf16-mixed)."""

from __future__ import annotations

import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.triton.fused_embed_zij import (  # noqa: E402
    eager_embed_zij,
    fused_embed_zij,
)
from openfold3.core.kernels.triton.fused_relpos_embed import (  # noqa: E402
    _build_indices,
)

C_Z = 128
C_IN = 267
MAX_REL, MAX_CHAIN = 32, 2
WARMUP, REPS = 5, 20
INFER_NS = (256, 384, 768)
PRECISIONS = ("tf32", "bf16-mixed")


def _batch(n):
    return {
        "token_index": torch.arange(n, device="cuda")[None],
        "residue_index": torch.arange(n, device="cuda")[None],
        "sym_id": torch.zeros(n, device="cuda")[None],
        "asym_id": torch.zeros(n, device="cuda")[None],
        "entity_id": torch.zeros(n, device="cuda")[None],
    }


def _set_precision(precision: str) -> torch.dtype:
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return torch.bfloat16 if precision == "bf16-mixed" else torch.float32


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1000


def _peak_U(fn, n):
    u = n * n * C_Z * 4
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / u


def _err(a, b):
    d = (a.float() - b.float()).abs()
    abs_err = d.max().item()
    return abs_err, abs_err / (b.float().abs().max().item() + 1e-12)


def bench(n: int, precision: str) -> dict:
    os.environ["OPENFOLD3_FUSED_LN_LINEAR"] = "1"
    dtype = _set_precision(precision)
    torch.manual_seed(0)
    gamma = torch.randn(C_IN, device="cuda")
    weight = torch.randn(C_Z, C_IN, device="cuda") / C_IN**0.5
    z = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    batch = _batch(n)
    idx1, idx2, idx3, same, offset = _build_indices(batch, MAX_REL, MAX_CHAIN)
    with torch.inference_mode():
        y_e = eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN)
        y_f = fused_embed_zij(z, gamma, weight, idx1, idx2, idx3, same, offset)
        abs_err, rel_err = _err(y_f, y_e)
        return {
            "precision": precision,
            "N": n,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "eager_ms": _time(
                lambda: eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN)
            ),
            "fused_ms": _time(
                lambda: fused_embed_zij(
                    z, gamma, weight, idx1, idx2, idx3, same, offset
                )
            ),
            "eager_U": _peak_U(
                lambda: eager_embed_zij(z, gamma, weight, batch, MAX_REL, MAX_CHAIN),
                n,
            ),
            "fused_U": _peak_U(
                lambda: fused_embed_zij(
                    z, gamma, weight, idx1, idx2, idx3, same, offset
                ),
                n,
            ),
        }


def main():
    print("INFERENCE _embed_zij (same-precision, no one-hot / concat)")
    print(
        f"{'prec':>11} | {'N':>4} | {'abs_err':>9} | {'rel_err':>9} | "
        f"{'eager_U':>8} | {'fused_U':>8} | {'eager':>7} | {'fused':>7} | {'e/f':>6}"
    )
    print("-" * 96)
    for n in INFER_NS:
        for precision in PRECISIONS:
            print(f"running {precision} N={n}...", flush=True)
            r = bench(n, precision)
            print(
                f"{r['precision']:>11} | {r['N']:>4} | {r['abs_err']:.2e} | "
                f"{r['rel_err']:.2e} | {r['eager_U']:7.2f}U | {r['fused_U']:7.2f}U | "
                f"{r['eager_ms']:6.2f} | {r['fused_ms']:6.2f} | "
                f"{r['eager_ms'] / r['fused_ms']:5.2f}x"
            )
    print()


if __name__ == "__main__":
    main()
