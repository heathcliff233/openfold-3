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

"""Same-precision fused vs eager SwiGLU microbench (TF32 and bf16-mixed)."""

from __future__ import annotations

import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from openfold3.core.kernels.triton.fused_swiglu_transition import (  # noqa: E402
    fused_swiglu_transition,
)
from openfold3.core.model.layers.transition import SwiGLUTransition  # noqa: E402

C = 128
H = 512
WARMUP = 5
REPS = 20
INFER_NS = (256, 384, 768)
CKPT_NS = (384, 768)
PRECISIONS = ("tf32", "bf16-mixed")


def _weights():
    torch.manual_seed(0)
    g = torch.randn(C, device="cuda")
    b = torch.randn(C, device="cuda")
    wa = torch.randn(H, C, device="cuda") / C**0.5
    wb = torch.randn(H, C, device="cuda") / C**0.5
    wo = torch.randn(C, H, device="cuda") / H**0.5
    return g, b, wa, wb, wo


def _bind_module(g, b, wa, wb, wo):
    module = SwiGLUTransition(c_in=C, n=H // C).to(device=g.device, dtype=g.dtype)
    with torch.no_grad():
        module.layer_norm.weight.copy_(g)
        module.layer_norm.bias.copy_(b)
        module.swiglu.linear_a.weight.copy_(wa)
        module.swiglu.linear_b.weight.copy_(wb)
        module.linear_out.weight.copy_(wo)
    return module


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
    u = N * N * C * 4
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
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    dtype = _set_precision(precision)
    g, b, wa, wb, wo = _weights()
    module = _bind_module(g, b, wa, wb, wo).eval()
    x = torch.randn(1, N, N, C, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda", dtype=dtype)
    with torch.inference_mode():
        y_e = module._eager_transition(x, mask)
        y_f = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        abs_err, rel_err = _err(y_f, y_e)
        z = x.clone()
        return {
            "precision": precision,
            "N": N,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "eager_ms": _time(lambda: module._eager_transition(x, mask)),
            "fused_ms": _time(
                lambda: fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
            ),
            "ip_ms": _time(
                lambda z=z: fused_swiglu_transition(
                    z, g, b, wa, wb, wo, mask=mask, residual=z
                )
            ),
            "eager_U": _peak_U(lambda: module._eager_transition(x, mask), N),
            "fused_U": _peak_U(
                lambda: fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask), N
            ),
            "ip_U": _peak_U(
                lambda z=z: fused_swiglu_transition(
                    z, g, b, wa, wb, wo, mask=mask, residual=z
                ),
                N,
            ),
        }


def bench_checkpointed(N: int, precision: str, fused: bool) -> dict:
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    dtype = _set_precision(precision)
    g, b, wa, wb, wo = _weights()
    module = _bind_module(g, b, wa, wb, wo).train()
    x0 = torch.randn(1, N, N, C, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda", dtype=dtype)
    grad_out = torch.randn_like(x0)
    u = N * N * C * 4

    def forward_graph():
        x = x0.detach().requires_grad_(True)
        if not fused:
            module.zero_grad(set_to_none=True)
            return checkpoint(
                lambda value: module._eager_transition(value, mask),
                x,
                use_reentrant=False,
            )
        leaves = [
            x,
            g.detach().requires_grad_(True),
            b.detach().requires_grad_(True),
            wa.detach().requires_grad_(True),
            wb.detach().requires_grad_(True),
            wo.detach().requires_grad_(True),
        ]
        return checkpoint(
            lambda *args: fused_swiglu_transition(*args, mask=mask),
            *leaves,
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
    print("INFERENCE (same-precision fused vs eager)")
    print(
        f"{'prec':>11} | {'N':>4} | {'abs_err':>9} | {'rel_err':>9} | "
        f"{'eager_U':>8} | {'fused_U':>8} | {'ip_U':>5} | "
        f"{'eager':>7} | {'fused':>7} | {'ip':>7} | {'e/f':>6}"
    )
    print("-" * 112)
    for r in rows:
        print(
            f"{r['precision']:>11} | {r['N']:>4} | {r['abs_err']:.2e} | "
            f"{r['rel_err']:.2e} | {r['eager_U']:7.2f}U | {r['fused_U']:7.2f}U | "
            f"{r['ip_U']:4.2f}U | {r['eager_ms']:6.2f} | {r['fused_ms']:6.2f} | "
            f"{r['ip_ms']:6.2f} | {r['eager_ms'] / r['fused_ms']:5.2f}x"
        )
    print()


def _print_checkpointed(rows):
    print("CHECKPOINTED TRAINING (non-reentrant, one block, same precision)")
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
