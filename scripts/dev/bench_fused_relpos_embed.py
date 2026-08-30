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

"""Microbenchmark for fused_relpos_embed_add_ kernel.

Compares the fused Triton kernel against the eager PyTorch path at multiple
sequence lengths, reporting correctness, peak memory, and wall time.
"""

from __future__ import annotations

import time
from functools import partial

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.triton.fused_relpos_embed import (  # noqa: E402
    eager_relpos_embed_add_,
    eager_relpos_weight_grad,
    fused_relpos_embed_add_,
    fused_relpos_weight_grad,
)

C = 128
VOCAB = 130
SAME_ENTITY_OFFSET = 65
WARMUP = 3
REPS = 10


def bench_one(N: int) -> dict:
    torch.cuda.empty_cache()
    z = torch.randn(1, N, N, C, device="cuda", dtype=torch.float32)
    w = torch.randn(VOCAB, C, device="cuda", dtype=torch.float32)
    idx1 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx2 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx3 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    same_entity = torch.randint(0, 2, (1, N, N), device="cuda", dtype=torch.bool)

    U_bytes = N * N * C * 4

    max_err = None
    if N <= 3000:
        z_ref = z.clone()
        eager_relpos_embed_add_(
            z_ref, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
        )

        z_test = z.clone()
        fused_relpos_embed_add_(
            z_test, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
        )
        torch.cuda.synchronize()
        max_err = (z_test - z_ref).abs().max().item()
        del z_ref, z_test

    eager_peak_U = None
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        z_tmp = z.clone()
        eager_relpos_embed_add_(
            z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
        )
        torch.cuda.synchronize()
        eager_peak_U = (torch.cuda.max_memory_allocated() - base) / U_bytes
        del z_tmp
    except torch.cuda.OutOfMemoryError:
        eager_peak_U = None
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    z_tmp = z.clone()
    fused_relpos_embed_add_(z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET)
    torch.cuda.synchronize()
    fused_peak_U = (torch.cuda.max_memory_allocated() - base) / U_bytes
    del z_tmp

    eager_ms = None
    try:
        for _ in range(WARMUP):
            z_tmp = z.clone()
            eager_relpos_embed_add_(
                z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            z_tmp = z.clone()
            eager_relpos_embed_add_(
                z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
            )
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / REPS * 1000
        del z_tmp
    except torch.cuda.OutOfMemoryError:
        eager_ms = None
        torch.cuda.empty_cache()

    for _ in range(WARMUP):
        z_tmp = z.clone()
        fused_relpos_embed_add_(
            z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        z_tmp = z.clone()
        fused_relpos_embed_add_(
            z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET
        )
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / REPS * 1000

    del z, w, idx1, idx2, idx3, same_entity
    torch.cuda.empty_cache()

    return {
        "N": N,
        "max_err": max_err,
        "eager_peak_U": eager_peak_U,
        "fused_peak_U": fused_peak_U,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
    }


def bench_bwd_one(N: int) -> dict:
    torch.cuda.empty_cache()
    grad_z = torch.randn(1, N, N, C, device="cuda", dtype=torch.float32)
    idx1 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx2 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx3 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    same_entity = torch.randint(0, 2, (1, N, N), device="cuda", dtype=torch.bool)
    U_bytes = N * N * C * 4

    max_err = None
    if N <= 3000:
        g_ref = eager_relpos_weight_grad(
            grad_z, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET, VOCAB
        )
        g_test = fused_relpos_weight_grad(
            grad_z, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET, VOCAB
        )
        torch.cuda.synchronize()
        max_err = (g_test - g_ref).abs().max().item()
        del g_ref, g_test

    def _peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / U_bytes

    def _time(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / REPS * 1000

    eager_fn = partial(
        eager_relpos_weight_grad,
        grad_z,
        idx1,
        idx2,
        idx3,
        same_entity,
        SAME_ENTITY_OFFSET,
        VOCAB,
    )
    fused_fn = partial(
        fused_relpos_weight_grad,
        grad_z,
        idx1,
        idx2,
        idx3,
        same_entity,
        SAME_ENTITY_OFFSET,
        VOCAB,
    )

    eager_peak_U = None
    eager_ms = None
    try:
        eager_peak_U = _peak(eager_fn)
        eager_ms = _time(eager_fn)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    fused_peak_U = _peak(fused_fn)
    fused_ms = _time(fused_fn)

    del grad_z, idx1, idx2, idx3, same_entity
    torch.cuda.empty_cache()
    return {
        "N": N,
        "max_err": max_err,
        "eager_peak_U": eager_peak_U,
        "fused_peak_U": fused_peak_U,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
    }


def _print_table(title: str, rows: list[dict]) -> None:
    print(title)
    print(
        f"{'N':>5} | {'err':>9} | {'eager_peak':>11} | {'fused_peak':>11} | "
        f"{'eager_ms':>9} | {'fused_ms':>9} | {'speedup':>7}"
    )
    print("-" * 80)
    for r in rows:
        err_str = f"{r['max_err']:.2e}" if r["max_err"] is not None else "skip"
        eager_peak_str = (
            f"{r['eager_peak_U']:.2f}U" if r["eager_peak_U"] is not None else "OOM"
        )
        eager_ms_str = f"{r['eager_ms']:.1f}ms" if r["eager_ms"] is not None else "OOM"
        speedup_str = (
            f"{r['eager_ms'] / r['fused_ms']:.2f}x"
            if r["eager_ms"] is not None
            else "N/A"
        )
        print(
            f"{r['N']:>5} | {err_str:>9} | {eager_peak_str:>11} | "
            f"{r['fused_peak_U']:.2f}U{'':>5} | {eager_ms_str:>9} | "
            f"{r['fused_ms']:.1f}ms{'':>4} | {speedup_str:>7}"
        )
    print()


def main():
    lengths = [256, 512, 1264, 2000]
    _print_table("Forward gather-add", [bench_one(N) for N in lengths])
    _print_table("Backward weight grad", [bench_bwd_one(N) for N in lengths])


if __name__ == "__main__":
    main()
