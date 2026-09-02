#!/usr/bin/env python
# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-precision triangle-attention microbench: eager vs fused.

Compares AF3 starting-node (outgoing) and ending-node (incoming)
``TriangleAttention`` at production ``(c_z, c_hidden, H) = (128, 32, 4)``.
Reports warm wall time and peak-above-input U.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from openfold3.core.kernels.cueq_utils import is_cuequivariance_available  # noqa: E402
from openfold3.core.kernels.triton.fused_tri_attn import (  # noqa: E402
    eager_tri_attn,
    fused_tri_attn,
)
from openfold3.core.model.layers.triangular_attention import (  # noqa: E402
    TriangleAttention,
    TriangleAttentionEndingNode,
)

C_Z = 128
C_HIDDEN = 32
NO_HEADS = 4
WARMUP = 5
REPS = 20
INFER_NS = (256, 384, 768)
CKPT_NS = (256, 384)
PRECISIONS = ("tf32", "bf16-mixed")
DIRECTIONS = (True, False)


def _dir_label(starting: bool) -> str:
    return "out" if starting else "in"


def _u(n: int) -> int:
    return n * n * C_Z * 4


def _module(starting: bool = True):
    torch.manual_seed(0)
    cls = TriangleAttention if starting else TriangleAttentionEndingNode
    module = cls(C_Z, C_HIDDEN, NO_HEADS).cuda().eval()
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.mha.linear_o.weight.normal_(0, 0.02)
        module.mha.linear_g.weight.normal_(0, 0.02)
    return module


def _weights(module):
    return (
        module.layer_norm.weight,
        module.layer_norm.bias,
        module.linear_z.weight,
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
        module.mha.linear_g.weight,
        module.mha.linear_o.weight,
    )


def _kwargs(module):
    return dict(
        c_hidden=module.c_hidden,
        no_heads=module.no_heads,
        starting=module.starting,
        inf=module.inf,
        eps=module.layer_norm.eps,
    )


def _pair(n: int, dtype: torch.dtype):
    torch.manual_seed(0)
    z = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
    return z, mask


def _set_precision(name: str) -> torch.dtype:
    tf32 = name == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return torch.float32 if name == "tf32" else torch.bfloat16


def _sync():
    torch.cuda.synchronize()


def _bench(fn, warmup: int, reps: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    ms = (time.perf_counter() - t0) * 1000 / reps
    peak = torch.cuda.max_memory_allocated()
    return ms, peak


def _print_row(prec, direction, name, n, ms, u, eager_ms):
    ratio = ms / eager_ms if eager_ms else float("nan")
    print(
        f"{prec:<12} {direction:<4} {name:<8} {n:5d} {ms:8.2f} {u:7.2f} {ratio:7.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--n", type=int, nargs="*", default=None)
    parser.add_argument(
        "--cell",
        nargs=3,
        metavar=("DIR", "PREC", "N"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    os.environ.setdefault("OPENFOLD3_FUSED_TRI_ATTN_V1", "1")
    infer_ns = tuple(args.n) if args.n else INFER_NS
    header = (
        f"{'prec':<12} {'dir':<4} {'path':<8} {'N':>5} "
        f"{'ms':>8} {'U':>7} {'vs eager':>8}"
    )
    if not args.train_only:
        print("INFERENCE (same-precision TriangleAttention; out=starting, in=ending)")
        print(header)
        for starting in DIRECTIONS:
            module = _module(starting)
            weights = _weights(module)
            kwargs = _kwargs(module)
            direction = _dir_label(starting)
            for prec in PRECISIONS:
                dtype = _set_precision(prec)
                for n in infer_ns:
                    z, mask = _pair(n, dtype)
                    u = _u(n)
                    eager_ms = None

                    def eager(z=z, mask=mask, weights=weights, kwargs=kwargs):
                        return eager_tri_attn(z, mask, *weights, **kwargs)

                    def fused(z=z, mask=mask, weights=weights, kwargs=kwargs):
                        return fused_tri_attn(z, mask, *weights, **kwargs)

                    def fused_res(z=z, mask=mask, weights=weights, kwargs=kwargs):
                        zin = z.clone()
                        return fused_tri_attn(
                            zin, mask, *weights, residual=zin, **kwargs
                        )

                    paths = [("eager", eager), ("fused", fused), ("fused+r", fused_res)]
                    if is_cuequivariance_available():
                        z_c = z.clone()

                        def cueq(module=module, z_c=z_c, mask=mask):
                            return module.forward(
                                z_c,
                                mask=mask,
                                use_cueq_triangle_kernels=True,
                            )

                        paths.append(("cueq", cueq))
                    for name, fn in paths:
                        with torch.inference_mode():
                            os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = (
                                "0" if name in ("eager", "cueq") else "1"
                            )
                            ms, peak = _bench(fn, WARMUP, REPS)
                            os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1"
                        if name == "eager":
                            eager_ms = ms
                        _print_row(
                            prec,
                            direction,
                            name,
                            n,
                            ms,
                            (peak - z.numel() * z.element_size()) / u,
                            eager_ms,
                        )
        print()
    if args.infer_only:
        return
    ckpt_ns = tuple(args.n) if args.n and args.train_only else CKPT_NS
    if args.cell is None:
        print("CHECKPOINTED TRAINING (non-reentrant fwd+bwd; out=starting, in=ending)")
        print(header)
        script = os.path.abspath(__file__)
        for starting in DIRECTIONS:
            for prec in PRECISIONS:
                for n in ckpt_ns:
                    subprocess.run(
                        [
                            sys.executable,
                            script,
                            "--train-only",
                            "--cell",
                            _dir_label(starting),
                            prec,
                            str(n),
                        ],
                        check=True,
                    )
        return
    direction, prec, n_s = args.cell
    n = int(n_s)
    starting = direction == "out"
    _run_train_cell(starting, prec, n)


def _run_train_cell(starting: bool, prec: str, n: int) -> None:
    module = _module(starting)
    weights = _weights(module)
    kwargs = _kwargs(module)
    direction = _dir_label(starting)
    dtype = _set_precision(prec)
    z, mask = _pair(n, dtype)
    u = _u(n)

    def _ckpt(fn, zin, leaves):
        def run(fn=fn, zin=zin, leaves=leaves):
            y = checkpoint(
                lambda *a: fn(*a),
                zin,
                mask,
                *leaves,
                use_reentrant=False,
            )
            y.square().mean().backward()
            zin.grad = None
            for t in leaves:
                t.grad = None

        return run

    z_e = z.detach().requires_grad_(True)
    leaves_e = [t.detach().requires_grad_(True) for t in weights]
    z_f = z.detach().requires_grad_(True)
    leaves_f = [t.detach().requires_grad_(True) for t in weights]
    z_c = z.detach().requires_grad_(True)

    def _cueq_fn(x):
        return module.forward(x, mask=mask, use_cueq_triangle_kernels=True)

    def ckpt_cueq(zin=z_c):
        os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "0"
        y = checkpoint(_cueq_fn, zin, use_reentrant=False)
        y.square().mean().backward()
        zin.grad = None
        module.zero_grad(set_to_none=True)
        os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1"

    paths = [
        ("eager", _ckpt(lambda *a: eager_tri_attn(*a, **kwargs), z_e, leaves_e)),
        ("fused", _ckpt(lambda *a: fused_tri_attn(*a, **kwargs), z_f, leaves_f)),
    ]
    if is_cuequivariance_available():
        for p in module.parameters():
            p.requires_grad_(True)
        paths.append(("cueq", ckpt_cueq))
    eager_ms = None
    for name, fn in paths:
        torch.cuda.empty_cache()
        ms, peak = _bench(fn, 2, 5)
        if name == "eager":
            eager_ms = ms
        _print_row(prec, direction, name, n, ms, peak / u, eager_ms)


if __name__ == "__main__":
    main()
