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

"""Full-module trimul benchmark: eager, fused, AMD Triton, and cuEq.

Isolates ``TriangleMultiplication{Outgoing,Incoming}.forward`` at production
widths. Default lengths include the homo_1200 pair size (N=1264). cuEq first
call can include length-keyed setup; warm median is the speed number.

Inference times eager / fused / optional cuEq / AMD Triton for both directions.
Checkpointed training times eager vs fused only (AMD Triton is inference-only).
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from statistics import median, pstdev

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from openfold3.core.kernels.cueq_utils import is_cuequivariance_available  # noqa: E402
from openfold3.core.kernels.triton.fused_trimul import (  # noqa: E402
    fused_trimul_from_module,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (  # noqa: E402
    TRITON_AVAILABLE,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def _set_fused(enabled: bool) -> None:
    os.environ["OPENFOLD3_FUSED_TRIMUL"] = "1" if enabled else "0"


def _randomize_observable(module) -> None:
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.linear_g.weight.normal_(0, 0.02)


def _matching_module(module, dtype: torch.dtype):
    cls = type(module)
    clone = cls(module.c_z, module.c_hidden).cuda().eval()
    clone.load_state_dict(module.state_dict())
    return clone if dtype == torch.float32 else clone.to(dtype=dtype)


def _cuda_time_call(fn, work: torch.Tensor) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(work)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), out


def _bench(prepare, fn, *, reps: int, warmup: int):
    cell_t0 = time.perf_counter()
    with torch.inference_mode():
        work = prepare()
        torch.cuda.synchronize()
        first_ms, out = _cuda_time_call(fn, work)
        del out, work
        for _ in range(warmup):
            work = prepare()
            torch.cuda.synchronize()
            _, out = _cuda_time_call(fn, work)
            del out, work
        warm_times = []
        for _ in range(reps):
            work = prepare()
            torch.cuda.synchronize()
            elapsed_ms, out = _cuda_time_call(fn, work)
            warm_times.append(elapsed_ms)
            del out, work
        gc.collect()
        torch.cuda.empty_cache()
        work = prepare()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _, out = _cuda_time_call(fn, work)
        peak = torch.cuda.max_memory_allocated()
        del out, work
    return (
        median(sorted(warm_times)),
        pstdev(warm_times) if len(warm_times) > 1 else 0.0,
        first_ms,
        peak - base,
        time.perf_counter() - cell_t0,
    )


def _bench_train(prepare, fn, *, reps: int, warmup: int):
    """Checkpointed fwd+bwd. ``prepare`` returns a detached activation."""
    cell_t0 = time.perf_counter()
    work = prepare()
    torch.cuda.synchronize()
    first_ms, _ = _cuda_time_call(fn, work)
    del work
    for _ in range(warmup):
        work = prepare()
        torch.cuda.synchronize()
        _, _ = _cuda_time_call(fn, work)
        del work
    warm_times = []
    for _ in range(reps):
        work = prepare()
        torch.cuda.synchronize()
        elapsed_ms, _ = _cuda_time_call(fn, work)
        warm_times.append(elapsed_ms)
        del work
    gc.collect()
    torch.cuda.empty_cache()
    work = prepare()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _, _ = _cuda_time_call(fn, work)
    peak = torch.cuda.max_memory_allocated()
    del work
    return (
        median(sorted(warm_times)),
        pstdev(warm_times) if len(warm_times) > 1 else 0.0,
        first_ms,
        peak - base,
        time.perf_counter() - cell_t0,
    )


def _make_case(n: int, *, outgoing: bool, act_dtype: torch.dtype, c_z: int):
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    torch.manual_seed(2026 + n + int(outgoing))
    module = cls(c_z, c_z).cuda().eval()
    _randomize_observable(module)
    z = torch.randn(1, n, n, c_z, device="cuda", dtype=act_dtype) * 0.1
    mask = torch.ones(1, n, n, device="cuda", dtype=act_dtype)
    return module, z, mask


def _configure(kind: str, chunk_cap: int | None) -> None:
    _set_fused(kind in {"fused", "fused_inplace"})
    if chunk_cap is None:
        os.environ.pop("OPENFOLD3_TRIMUL_CHUNK_CAP", None)
    else:
        os.environ["OPENFOLD3_TRIMUL_CHUNK_CAP"] = str(chunk_cap)


def _run_variant(module, work, mask, *, kind: str, chunk_cap: int | None):
    if kind == "cueq":
        return module(
            work, mask=mask, inplace_safe=False, use_cueq_triangle_kernels=True
        )
    if kind == "legacy_triton":
        legacy_chunk = work.shape[-3] if chunk_cap is None else chunk_cap
        return module(
            work,
            mask=mask,
            inplace_safe=True,
            use_triton_triangle_kernels=True,
            _add_with_inplace=False,
            _inplace_chunk_size=legacy_chunk,
        )
    if kind == "fused_inplace":
        out = fused_trimul_from_module(
            module, work, mask, residual=work, chunk_cap=chunk_cap
        )
        if out is None:
            raise RuntimeError("fused inplace path was ineligible")
        return out
    return module(
        work,
        mask=mask,
        inplace_safe=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[256, 384, 768, 1264])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--include-cueq", action="store_true")
    parser.add_argument("--include-legacy-triton", action="store_true")
    parser.add_argument("--include-fused-inplace", action="store_true")
    parser.add_argument("--exclude-eager", action="store_true")
    parser.add_argument("--chunk-cap", type=int, nargs="+", default=None)
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also time checkpointed eager vs fused (both directions).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip inference and only run checkpointed training.",
    )
    parser.add_argument(
        "--train-n",
        type=int,
        nargs="+",
        default=[384, 768],
        help="Lengths for checkpointed training (default: 384 768).",
    )
    args = parser.parse_args()

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    act_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    variants = []
    if not args.exclude_eager:
        variants.append(("eager_default", "eager", None))
    variants.append(("fused", "fused", None))
    if args.include_fused_inplace:
        variants.append(("fused_inplace", "fused_inplace", None))
    if args.include_cueq and is_cuequivariance_available():
        variants.append(("cueq", "cueq", None))
    if args.include_legacy_triton and TRITON_AVAILABLE:
        variants.append(("legacy_triton", "legacy_triton", None))
    if args.chunk_cap:
        for cap in args.chunk_cap:
            variants.append((f"fused_chunk{cap}", "fused", cap))
            if args.include_fused_inplace:
                variants.append((f"fused_inplace_chunk{cap}", "fused_inplace", cap))
            if args.include_legacy_triton and TRITON_AVAILABLE:
                variants.append((f"legacy_triton_chunk{cap}", "legacy_triton", cap))

    header = (
        f"{'variant':<24}{'dir':>5}{'N':>6}{'median ms':>11}"
        f"{'std ms':>9}{'first ms':>10}{'peak U':>9}{'cell s':>8}"
    )
    try:
        if not args.train_only:
            print("INFERENCE (outgoing and incoming)")
            print(header)
            print("-" * len(header))
            for n in args.n:
                for outgoing in (True, False):
                    direction = "out" if outgoing else "in"
                    module, z, mask = _make_case(
                        n, outgoing=outgoing, act_dtype=act_dtype, c_z=args.c_z
                    )
                    cueq_module = (
                        _matching_module(module, act_dtype)
                        if any(k == "cueq" for _, k, _ in variants)
                        else module
                    )
                    u_bytes = n * n * args.c_z * z.element_size()
                    for label, kind, chunk_cap in variants:
                        _configure(kind, chunk_cap)
                        backend = cueq_module if kind == "cueq" else module

                        def prepare(z=z):
                            return z.clone()

                        def run(
                            work,
                            backend=backend,
                            mask=mask,
                            kind=kind,
                            chunk_cap=chunk_cap,
                        ):
                            return _run_variant(
                                backend, work, mask, kind=kind, chunk_cap=chunk_cap
                            )

                        try:
                            ms, std_ms, first_ms, peak, cell_s = _bench(
                                prepare, run, reps=args.reps, warmup=args.warmup
                            )
                            print(
                                f"{label:<24}{direction:>5}{n:>6}{ms:>11.3f}"
                                f"{std_ms:>9.3f}{first_ms:>10.3f}"
                                f"{peak / u_bytes:>9.2f}{cell_s:>8.1f}"
                            )
                        except (RuntimeError, TypeError) as exc:
                            print(f"{label:<24}{direction:>5}{n:>6}  ERROR: {exc}")
                            gc.collect()
                            torch.cuda.empty_cache()
                    print()

        if args.include_train or args.train_only:
            print("CHECKPOINTED TRAINING (non-reentrant; eager vs fused; out/in)")
            print(header)
            print("-" * len(header))
            train_kinds = (("eager_ckpt", False), ("fused_ckpt", True))
            for n in args.train_n:
                for outgoing in (True, False):
                    direction = "out" if outgoing else "in"
                    module, z, mask = _make_case(
                        n, outgoing=outgoing, act_dtype=act_dtype, c_z=args.c_z
                    )
                    u_bytes = n * n * args.c_z * 4
                    grad_out = torch.randn_like(z)
                    for label, fused in train_kinds:
                        _set_fused(fused)

                        def train_step(
                            work,
                            module=module,
                            mask=mask,
                            grad_out=grad_out,
                        ):
                            module.zero_grad(set_to_none=True)
                            z_leaf = work.detach().requires_grad_(True)
                            y = checkpoint(
                                lambda x: module(
                                    x,
                                    mask=mask,
                                    inplace_safe=False,
                                    use_cueq_triangle_kernels=False,
                                    use_triton_triangle_kernels=False,
                                ),
                                z_leaf,
                                use_reentrant=False,
                            )
                            y.backward(grad_out)
                            return y

                        try:
                            ms, std_ms, first_ms, peak, cell_s = _bench_train(
                                lambda z=z: z.detach().clone(),
                                train_step,
                                reps=args.reps,
                                warmup=args.warmup,
                            )
                            print(
                                f"{label:<24}{direction:>5}{n:>6}{ms:>11.3f}"
                                f"{std_ms:>9.3f}{first_ms:>10.3f}"
                                f"{peak / u_bytes:>9.2f}{cell_s:>8.1f}"
                            )
                        except (RuntimeError, TypeError) as exc:
                            print(f"{label:<24}{direction:>5}{n:>6}  ERROR: {exc}")
                            gc.collect()
                            torch.cuda.empty_cache()
                    print()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    main()
