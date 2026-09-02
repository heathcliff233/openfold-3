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

"""Same-precision trimul microbench: eager, AMD Triton, cuEq, and fused.

Compares AF3 outgoing and incoming ``TriangleMultiplication*`` at ``c=128``.
Inference times all four backends. Checkpointed training times eager, fused,
and cuEq (AMD Triton is inference-only: in-place overwrite, no autograd).
cuEq requires matching activation/weight dtypes, so the bf16-mixed row uses a
downcast weight copy for that backend only. AMD Triton downcasts internally.
"""

from __future__ import annotations

import argparse
import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from openfold3.core.kernels.cueq_utils import is_cuequivariance_available  # noqa: E402
from openfold3.core.kernels.triton.fused_trimul import (  # noqa: E402
    eager_trimul,
    fused_trimul,
)

try:
    from cuequivariance_torch import triangle_multiplicative_update  # noqa: E402
except ImportError:  # pragma: no cover
    triangle_multiplicative_update = None
from openfold3.core.model.layers.triangular_multiplicative_update import (  # noqa: E402
    TRITON_AVAILABLE,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

C_Z = 128
WARMUP = 5
REPS = 20
INFER_NS = (256, 384, 768)
CKPT_NS = (384, 768)
PRECISIONS = ("tf32", "bf16-mixed")
DIRECTIONS = (True, False)
PATHS = ("eager", "amd-triton", "cueq", "fused")


def _dir_label(outgoing: bool) -> str:
    return "out" if outgoing else "in"


def _module(outgoing: bool = True):
    torch.manual_seed(0)
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    module = cls(C_Z, C_Z).cuda().eval()
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.linear_g.weight.normal_(0, 0.02)
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


def _set_precision(precision: str) -> torch.dtype:
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return torch.bfloat16 if precision == "bf16-mixed" else torch.float32


def _matching_module(module, dtype: torch.dtype):
    """Return a module whose parameters match ``dtype`` (needed by cuEq)."""
    if dtype == torch.float32:
        return module
    clone = type(module)(C_Z, C_Z).cuda().eval()
    clone.load_state_dict(module.state_dict())
    return clone.to(dtype=dtype)


def _run_path(path: str, module, z, mask, weights, outgoing: bool):
    if path == "eager":
        return eager_trimul(z, mask, *weights, outgoing)
    if path == "fused":
        os.environ["OPENFOLD3_FUSED_TRIMUL"] = "1"
        return fused_trimul(z, mask, *weights, outgoing)
    os.environ["OPENFOLD3_FUSED_TRIMUL"] = "0"
    if path == "amd-triton":
        # Inference Triton overwrites its input; clone outside the timed region.
        return module.forward(
            z,
            mask=mask,
            inplace_safe=True,
            use_triton_triangle_kernels=True,
            _add_with_inplace=False,
        )
    if path == "cueq":
        return module.forward(z, mask=mask, use_cueq_triangle_kernels=True)
    raise ValueError(path)


def _available(path: str) -> bool:
    if path == "amd-triton":
        return bool(TRITON_AVAILABLE)
    if path == "cueq":
        return is_cuequivariance_available()
    return True


def _prepare(path: str, z: torch.Tensor) -> torch.Tensor:
    return z.clone() if path == "amd-triton" else z


def _time(prepare, fn):
    for _ in range(WARMUP):
        fn(prepare())
    times = []
    for _ in range(REPS):
        work = prepare()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(work)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def _peak_U(prepare, fn, n):
    u = n * n * C_Z * 4
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    work = prepare()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(work)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / u


def _err(a, b):
    d = (a.float() - b.float()).abs()
    abs_err = d.max().item()
    return abs_err, abs_err / (b.float().abs().max().item() + 1e-12)


def bench_inference(n: int, precision: str, outgoing: bool) -> list[dict]:
    dtype = _set_precision(precision)
    module = _module(outgoing)
    weights = _weights(module)
    z = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
    cueq_module = _matching_module(module, dtype)
    direction = _dir_label(outgoing)
    rows = []
    y_ref = None
    with torch.inference_mode():
        for path in PATHS:
            if not _available(path):
                rows.append(
                    {
                        "precision": precision,
                        "dir": direction,
                        "N": n,
                        "path": path,
                        "skipped": True,
                    }
                )
                continue
            backend = cueq_module if path == "cueq" else module
            y = _run_path(path, backend, _prepare(path, z), mask, weights, outgoing)
            if path == "eager":
                y_ref = y
            abs_err, rel_err = (0.0, 0.0) if y_ref is None else _err(y, y_ref)

            def _call(work, p=path, m=backend):
                return _run_path(p, m, work, mask, weights, outgoing)

            def _prep(p=path):
                return _prepare(p, z)

            rows.append(
                {
                    "precision": precision,
                    "dir": direction,
                    "N": n,
                    "path": path,
                    "skipped": False,
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                    "ms": _time(_prep, _call),
                    "peak_U": _peak_U(_prep, _call, n),
                }
            )
    return rows


def _print_inference(rows):
    print("INFERENCE (c=128 out/in; abs_err vs matching-precision eager)")
    print("cuEq bf16-mixed uses matching bf16 weights; other paths keep fp32 masters.")
    print(
        f"{'prec':>11} | {'dir':>3} | {'N':>4} | {'path':>11} | "
        f"{'abs_err':>9} | {'rel_err':>9} | {'peak':>8} | {'ms':>7} | {'vs eager':>8}"
    )
    print("-" * 100)
    eager_ms = {
        (r["precision"], r["dir"], r["N"]): r["ms"]
        for r in rows
        if not r.get("skipped") and r["path"] == "eager"
    }
    for r in rows:
        if r.get("skipped"):
            print(
                f"{r['precision']:>11} | {r['dir']:>3} | {r['N']:>4} | "
                f"{r['path']:>11} | {'skipped':>9} | {'':>9} | {'':>8} | "
                f"{'':>7} | {'':>8}"
            )
            continue
        ratio = r["ms"] / eager_ms[(r["precision"], r["dir"], r["N"])]
        print(
            f"{r['precision']:>11} | {r['dir']:>3} | {r['N']:>4} | "
            f"{r['path']:>11} | {r['abs_err']:.2e} | {r['rel_err']:.2e} | "
            f"{r['peak_U']:7.2f}U | {r['ms']:6.2f} | {ratio:7.2f}x"
        )
    print()


def _cueq_trimul(
    z,
    mask,
    wa_p,
    wa_g,
    wb_p,
    wb_g,
    wz,
    wg,
    ln_in_w,
    ln_in_b,
    ln_out_w,
    ln_out_b,
    outgoing,
):
    if triangle_multiplicative_update is None:
        raise RuntimeError("cuequivariance_torch is not installed")
    return triangle_multiplicative_update(
        z,
        direction="outgoing" if outgoing else "incoming",
        mask=mask,
        norm_in_weight=ln_in_w,
        norm_in_bias=ln_in_b,
        p_in_weight=torch.cat([wa_p, wb_p], dim=0),
        g_in_weight=torch.cat([wa_g, wb_g], dim=0),
        norm_out_weight=ln_out_w,
        norm_out_bias=ln_out_b,
        p_out_weight=wz,
        g_out_weight=wg,
        eps=1e-5,
    )


def _train_fn(path: str):
    if path == "eager":
        return eager_trimul
    if path == "fused":
        return fused_trimul
    if path == "cueq":
        return _cueq_trimul
    raise ValueError(path)


def bench_checkpointed(n: int, precision: str, path: str, outgoing: bool) -> dict:
    os.environ["OPENFOLD3_FUSED_TRIMUL"] = "1"
    dtype = _set_precision(precision)
    module = _module(outgoing)
    weights = _weights(module)
    # cuEq requires matching activation/weight dtypes; other paths keep fp32 masters.
    if path == "cueq" and dtype != torch.float32:
        weights = tuple(t.to(dtype) for t in weights)
    z0 = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
    grad_out = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype)
    u = n * n * C_Z * 4
    fn = _train_fn(path)

    def forward_graph():
        leaves = [t.detach().requires_grad_(True) for t in (z0, *weights)]
        return checkpoint(
            lambda *args: fn(args[0], mask, *args[1:], outgoing),
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

    return {
        "precision": precision,
        "dir": _dir_label(outgoing),
        "path": path,
        "N": n,
        "peak_U": peak_U(),
        "fwd_ms": _time(lambda: z0, lambda _w: forward_graph()),
        "full_ms": _time(lambda: z0, lambda _w: full_step()),
    }


def _print_checkpointed(rows):
    print("CHECKPOINTED TRAINING (non-reentrant, same-precision trimul, out/in)")
    print("AMD Triton is inference-only (in-place overwrite, no autograd).")
    print("cuEq bf16-mixed uses matching bf16 weights; other paths keep fp32 masters.")
    print(
        f"{'prec':>11} | {'dir':>3} | {'path':>6} | {'N':>4} | {'peak':>7} | "
        f"{'fwd':>7} | {'bwd':>7} | {'full':>7} | {'vs eager':>8}"
    )
    print("-" * 88)
    eager_ms = {
        (r["precision"], r["dir"], r["N"]): r["full_ms"]
        for r in rows
        if r["path"] == "eager"
    }
    for row in rows:
        ratio = row["full_ms"] / eager_ms[(row["precision"], row["dir"], row["N"])]
        print(
            f"{row['precision']:>11} | {row['dir']:>3} | {row['path']:>6} | "
            f"{row['N']:4d} | {row['peak_U']:6.2f}U | {row['fwd_ms']:6.2f} | "
            f"{row['full_ms'] - row['fwd_ms']:6.2f} | {row['full_ms']:6.2f} | "
            f"{ratio:7.2f}x"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip inference and only run checkpointed training.",
    )
    args = parser.parse_args()

    if not args.train_only:
        infer = []
        for n in INFER_NS:
            for outgoing in DIRECTIONS:
                for precision in PRECISIONS:
                    print(
                        f"running inference {precision} {_dir_label(outgoing)} "
                        f"N={n}...",
                        flush=True,
                    )
                    infer.extend(bench_inference(n, precision, outgoing))
        _print_inference(infer)

    train_paths = ["eager", "fused"]
    if is_cuequivariance_available() and triangle_multiplicative_update is not None:
        train_paths.append("cueq")

    ckpt = []
    for n in CKPT_NS:
        for outgoing in DIRECTIONS:
            for precision in PRECISIONS:
                for path in train_paths:
                    print(
                        f"running checkpointed {precision} "
                        f"{_dir_label(outgoing)} {path} N={n}...",
                        flush=True,
                    )
                    ckpt.append(bench_checkpointed(n, precision, path, outgoing))
    _print_checkpointed(ckpt)


if __name__ == "__main__":
    main()
