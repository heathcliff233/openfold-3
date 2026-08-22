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

"""Same-precision fused vs eager LN→Linear microbench (TF32 and bf16-mixed)."""

from __future__ import annotations

import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from openfold3.core.kernels.triton.fused_ln_linear import (  # noqa: E402
    eager_ln_linear,
    fused_ln_linear,
)

# DiffusionConditioning._embed_zij: cat(z, relpos) → c_z.
C_IN = 267
C_OUT = 128
C_Z = 128
WARMUP = 5
REPS = 20
INFER_NS = (256, 384, 768)
CKPT_NS = (384, 768)
PRECISIONS = ("tf32", "bf16-mixed")


def _weights():
    torch.manual_seed(0)
    gamma = torch.randn(C_IN, device="cuda")
    weight = torch.randn(C_OUT, C_IN, device="cuda") / C_IN**0.5
    return gamma, weight


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


def _peak_U(fn, N):
    u = N * N * C_Z * 4
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


def bench_inference(N: int, precision: str) -> dict:
    os.environ["OPENFOLD3_FUSED_LN_LINEAR"] = "1"
    dtype = _set_precision(precision)
    gamma, weight = _weights()
    x = torch.randn(1, N, N, C_IN, device="cuda", dtype=dtype) * 0.5
    with torch.inference_mode():
        y_e = eager_ln_linear(x, gamma, None, weight, None)
        y_f = fused_ln_linear(x, gamma, None, weight, None)
        abs_err, rel_err = _err(y_f, y_e)
        return {
            "precision": precision,
            "N": N,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "eager_ms": _time(lambda: eager_ln_linear(x, gamma, None, weight, None)),
            "fused_ms": _time(lambda: fused_ln_linear(x, gamma, None, weight, None)),
            "eager_U": _peak_U(
                lambda: eager_ln_linear(x, gamma, None, weight, None), N
            ),
            "fused_U": _peak_U(
                lambda: fused_ln_linear(x, gamma, None, weight, None), N
            ),
        }


def bench_checkpointed(N: int, precision: str, fused: bool) -> dict:
    os.environ["OPENFOLD3_FUSED_LN_LINEAR"] = "1"
    dtype = _set_precision(precision)
    gamma, weight = _weights()
    x0 = torch.randn(1, N, N, C_IN, device="cuda", dtype=dtype) * 0.5
    grad_out = torch.randn(1, N, N, C_OUT, device="cuda", dtype=dtype)
    u = N * N * C_Z * 4

    def forward_graph():
        x = x0.detach().requires_grad_(True)
        g = gamma.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        fn = fused_ln_linear if fused else eager_ln_linear
        return checkpoint(
            lambda *args: fn(args[0], args[1], None, args[2], None),
            x,
            g,
            w,
            use_reentrant=False,
        )

    def full_step():
        forward_graph().backward(grad_out)

    def peak_U():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        full_step()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / u

    fwd_ms = _time(forward_graph)
    full_ms = _time(full_step)
    return {
        "precision": precision,
        "path": "fused" if fused else "eager",
        "N": N,
        "peak_U": peak_U(),
        "fwd_ms": fwd_ms,
        "bwd_ms": full_ms - fwd_ms,
        "full_ms": full_ms,
    }


def _print_inference(rows):
    print("INFERENCE (same-precision fused vs eager LN→Linear, c_in=267→128)")
    print(
        f"{'prec':>11} | {'N':>4} | {'abs_err':>9} | {'rel_err':>9} | "
        f"{'eager_U':>8} | {'fused_U':>8} | {'eager':>7} | {'fused':>7} | {'e/f':>6}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['precision']:>11} | {r['N']:>4} | {r['abs_err']:.2e} | "
            f"{r['rel_err']:.2e} | {r['eager_U']:7.2f}U | {r['fused_U']:7.2f}U | "
            f"{r['eager_ms']:6.2f} | {r['fused_ms']:6.2f} | "
            f"{r['eager_ms'] / r['fused_ms']:5.2f}x"
        )
    print()


def _print_checkpointed(rows):
    print("CHECKPOINTED TRAINING (non-reentrant, same-precision LN→Linear)")
    print(
        f"{'prec':>11} | {'path':>6} | {'N':>4} | {'peak':>7} | "
        f"{'fwd':>7} | {'bwd':>7} | {'full':>7} | {'vs eager':>8}"
    )
    print("-" * 80)
    eager_ms = {
        (r["precision"], r["N"]): r["full_ms"] for r in rows if r["path"] == "eager"
    }
    for row in rows:
        ratio = row["full_ms"] / eager_ms[(row["precision"], row["N"])]
        print(
            f"{row['precision']:>11} | {row['path']:>6} | {row['N']:4d} | "
            f"{row['peak_U']:6.2f}U | {row['fwd_ms']:6.2f} | "
            f"{row['bwd_ms']:6.2f} | {row['full_ms']:6.2f} | {ratio:7.2f}x"
        )
    print()


def main():
    infer = []
    for N in INFER_NS:
        for precision in PRECISIONS:
            print(f"running inference {precision} N={N}...", flush=True)
            infer.append(bench_inference(N, precision))
    _print_inference(infer)

    ckpt = []
    for N in CKPT_NS:
        for precision in PRECISIONS:
            for fused in (False, True):
                label = "fused" if fused else "eager"
                print(f"running checkpointed {precision} {label} N={N}...", flush=True)
                ckpt.append(bench_checkpointed(N, precision, fused))
    _print_checkpointed(ckpt)


if __name__ == "__main__":
    main()
