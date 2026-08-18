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

"""Error / speed / memory for fused SwiGLU vs eager.

FP32 rows use IEEE eager as the baseline. The bf16-mixed row uses
bf16 activations with fp32 masters (Triton does not take bf16 weights).
"""

from __future__ import annotations

import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.triton.fused_swiglu_transition import (  # noqa: E402
    fused_swiglu_transition,
)
from openfold3.core.model.layers.transition import SwiGLUTransition  # noqa: E402

C = 128
H = 512
WARMUP = 5
REPS = 20


def _weights(dtype=torch.float32):
    torch.manual_seed(0)
    g = torch.randn(C, device="cuda", dtype=dtype)
    b = torch.randn(C, device="cuda", dtype=dtype)
    wa = torch.randn(H, C, device="cuda", dtype=dtype) / C**0.5
    wb = torch.randn(H, C, device="cuda", dtype=dtype) / C**0.5
    wo = torch.randn(C, H, device="cuda", dtype=dtype) / H**0.5
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


def _eager(module, x, mask):
    return module._eager_transition(x, mask)


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


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
    U = N * N * C * 4
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / U


def _err(a, b):
    d = (a.float() - b.float()).abs()
    abs_err = d.max().item()
    rel_err = abs_err / (b.float().abs().max().item() + 1e-12)
    return abs_err, rel_err


def _max_pair_err(pairs):
    """Max abs and max per-tensor rel across ``(actual, ref)`` pairs."""
    abs_err = 0.0
    rel_err = 0.0
    for a, b in pairs:
        ae, re = _err(a, b)
        abs_err = max(abs_err, ae)
        rel_err = max(rel_err, re)
    return abs_err, rel_err


def bench_fwd(N: int) -> list[dict]:
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    g, b, wa, wb, wo = _weights()
    module = _bind_module(g, b, wa, wb, wo).eval()
    x = torch.randn(1, N, N, C, device="cuda") * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda")

    # Fixed baseline: default FP32 eager (IEEE).
    _set_tf32(False)
    with torch.inference_mode():
        y_fp32 = _eager(module, x, mask)
        baseline_ms = _time(lambda: _eager(module, x, mask))
        baseline_U = _peak_U(lambda: _eager(module, x, mask), N)

    rows = []
    configs = [
        ("fused FP32", False, True),
        ("fused TF32", True, True),
        ("eager TF32", True, False),
    ]
    for label, tf32, use_fused in configs:
        _set_tf32(tf32)
        with torch.inference_mode():
            fn = (
                (lambda: fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask))
                if use_fused
                else (lambda: _eager(module, x, mask))
            )
            y = fn()
            abs_err, rel_err = _err(y, y_fp32)
            ms = _time(fn)
            peak_U = _peak_U(fn, N)
            ip_ms = ip_U = None
            if use_fused:
                z = x.clone()
                ip_ms = _time(
                    lambda: fused_swiglu_transition(
                        z, g, b, wa, wb, wo, mask=mask, residual=z
                    )
                )
                z = x.clone()
                ip_U = _peak_U(
                    lambda: fused_swiglu_transition(
                        z, g, b, wa, wb, wo, mask=mask, residual=z
                    ),
                    N,
                )
        rows.append(
            {
                "label": label,
                "N": N,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "baseline_ms": baseline_ms,
                "ms": ms,
                "ip_ms": ip_ms,
                "baseline_U": baseline_U,
                "U": peak_U,
                "ip_U": ip_U,
            }
        )
    return rows


def bench_bwd(N: int) -> list[dict]:
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    g, b, wa, wb, wo = _weights()
    module = _bind_module(g, b, wa, wb, wo).train()
    x0 = torch.randn(1, N, N, C, device="cuda") * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda")
    go = torch.randn_like(x0)

    def run(fused: bool):
        xd = x0.detach().requires_grad_(True)
        if fused:
            ts = [
                xd,
                g.detach().requires_grad_(True),
                b.detach().requires_grad_(True),
                wa.detach().requires_grad_(True),
                wb.detach().requires_grad_(True),
                wo.detach().requires_grad_(True),
            ]
            y = fused_swiglu_transition(*ts, mask=mask)
            y.backward(go)
            return [t.grad.detach().clone() for t in ts]
        module.zero_grad(set_to_none=True)
        y = _eager(module, xd, mask)
        y.backward(go)
        return [
            xd.grad.detach().clone(),
            module.layer_norm.weight.grad.detach().clone(),
            module.layer_norm.bias.grad.detach().clone(),
            module.swiglu.linear_a.weight.grad.detach().clone(),
            module.swiglu.linear_b.weight.grad.detach().clone(),
            module.linear_out.weight.grad.detach().clone(),
        ]

    _set_tf32(False)
    g_fp32 = run(False)
    baseline_ms = _time(lambda: run(False))
    baseline_U = _peak_U(lambda: run(False), N)

    rows = []
    configs = [
        ("fused FP32", False, True),
        ("fused TF32", True, True),
        ("eager TF32", True, False),
    ]
    for label, tf32, use_fused in configs:
        _set_tf32(tf32)
        grads = run(use_fused)
        abs_errs = [(a - b).abs().max().item() for a, b in zip(g_fp32, grads)]
        rel_errs = [
            ae / (a.abs().max().item() + 1e-12) for ae, a in zip(abs_errs, g_fp32)
        ]
        ms = _time(lambda: run(use_fused))
        peak_U = _peak_U(lambda: run(use_fused), N)
        rows.append(
            {
                "label": label,
                "N": N,
                "abs_err": max(abs_errs),
                "rel_err": max(rel_errs),
                "baseline_ms": baseline_ms,
                "ms": ms,
                "baseline_U": baseline_U,
                "U": peak_U,
            }
        )
    return rows


def _print_fwd(rows, title="FORWARD vs default FP32 eager (IEEE) baseline"):
    print(title)
    print(
        f"{'path':>18} | {'N':>4} | {'abs_err':>9} | {'rel_err':>9} | "
        f"{'base_U':>7} | {'path_U':>7} | {'ip_U':>5} | "
        f"{'base':>7} | {'path':>7} | {'ip':>7} | {'base/p':>6}"
    )
    print("-" * 120)
    for r in rows:
        ip_U = f"{r['ip_U']:4.2f}U" if r["ip_U"] is not None else "  n/a"
        ip_ms = f"{r['ip_ms']:6.2f}" if r["ip_ms"] is not None else "   n/a"
        print(
            f"{r['label']:>18} | {r['N']:>4} | {r['abs_err']:.2e} | "
            f"{r['rel_err']:.2e} | {r['baseline_U']:6.2f}U | {r['U']:6.2f}U | "
            f"{ip_U:>5} | {r['baseline_ms']:6.2f} | {r['ms']:6.2f} | "
            f"{ip_ms:>7} | {r['baseline_ms']/r['ms']:5.2f}x"
        )
    print()


def _print_bwd(rows, title="BACKWARD vs default FP32 eager (IEEE) baseline"):
    print(title)
    print(
        f"{'path':>18} | {'N':>4} | {'abs_err':>9} | {'rel_err':>9} | "
        f"{'base_U':>7} | {'path_U':>7} | {'base':>7} | {'path':>7} | {'base/p':>6}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['label']:>18} | {r['N']:>4} | {r['abs_err']:.2e} | "
            f"{r['rel_err']:.2e} | {r['baseline_U']:6.2f}U | {r['U']:6.2f}U | "
            f"{r['baseline_ms']:6.2f} | {r['ms']:6.2f} | "
            f"{r['baseline_ms']/r['ms']:5.2f}x"
        )
    print()


def bench_bwd_dtype(
    N: int,
    act_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    label: str,
):
    """Training fwd+bwd for one activation/weight dtype pair. No TF32."""
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    g, b, wa, wb, wo = _weights(weight_dtype)
    module = _bind_module(g, b, wa, wb, wo).train()
    x0 = torch.randn(1, N, N, C, device="cuda", dtype=act_dtype) * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda", dtype=act_dtype)
    go = torch.randn_like(x0)
    _set_tf32(False)

    def run(fused: bool):
        xd = x0.detach().requires_grad_(True)
        if fused:
            ts = [
                xd,
                g.detach().requires_grad_(True),
                b.detach().requires_grad_(True),
                wa.detach().requires_grad_(True),
                wb.detach().requires_grad_(True),
                wo.detach().requires_grad_(True),
            ]
            y = fused_swiglu_transition(*ts, mask=mask)
            y.backward(go)
            return y.detach(), [t.grad.detach().clone() for t in ts]
        module.zero_grad(set_to_none=True)
        y = _eager(module, xd, mask)
        y.backward(go)
        return y.detach(), [
            xd.grad.detach().clone(),
            module.layer_norm.weight.grad.detach().clone(),
            module.layer_norm.bias.grad.detach().clone(),
            module.swiglu.linear_a.weight.grad.detach().clone(),
            module.swiglu.linear_b.weight.grad.detach().clone(),
            module.linear_out.weight.grad.detach().clone(),
        ]

    y_e, g_e = run(False)
    y_f, g_f = run(True)
    abs_err, rel_err = _max_pair_err([(y_f, y_e), *zip(g_f, g_e)])
    return {
        "label": label,
        "N": N,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "baseline_ms": _time(lambda: run(False)),
        "ms": _time(lambda: run(True)),
        "baseline_U": _peak_U(lambda: run(False), N),
        "U": _peak_U(lambda: run(True), N),
    }


def bench_fwd_dtype(N: int, weight_dtype: torch.dtype, label: str) -> dict:
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1"
    g, b, wa, wb, wo = _weights(weight_dtype)
    module = _bind_module(g, b, wa, wb, wo).eval()
    x = torch.randn(1, N, N, C, device="cuda", dtype=torch.bfloat16) * 0.5
    mask = torch.ones(1, N, N, 1, device="cuda", dtype=torch.bfloat16)
    _set_tf32(False)
    with torch.inference_mode():
        y_e = _eager(module, x, mask)
        y_f = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        abs_err, rel_err = _err(y_f, y_e)
        z = x.clone()
        return {
            "label": label,
            "N": N,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "baseline_ms": _time(lambda: _eager(module, x, mask)),
            "ms": _time(
                lambda: fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
            ),
            "ip_ms": _time(
                lambda: fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z)
            ),
            "baseline_U": _peak_U(lambda: _eager(module, x, mask), N),
            "U": _peak_U(
                lambda: fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask), N
            ),
            "ip_U": _peak_U(
                lambda: fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z),
                N,
            ),
        }


def main():
    lengths = [128, 256, 384, 768]
    fwd, bwd, fwd16, bwd16 = [], [], [], []
    for N in lengths:
        print(f"running fwd N={N}...", flush=True)
        fwd.extend(bench_fwd(N))
        print(f"running bwd N={N}...", flush=True)
        bwd.extend(bench_bwd(N))
        print(f"running mixed N={N}...", flush=True)
        fwd16.append(bench_fwd_dtype(N, torch.float32, "fused bf16-mixed"))
        bwd16.append(
            bench_bwd_dtype(N, torch.bfloat16, torch.float32, "fused bf16-mixed")
        )
    _print_fwd(fwd)
    _print_bwd(bwd)
    _print_fwd(fwd16, "BF16-MIXED FORWARD (vs eager mixed)")
    _print_bwd(bwd16, "BF16-MIXED BACKWARD (vs eager mixed)")


if __name__ == "__main__":
    main()
