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

"""Same-precision diffusion-attention microbench: eager vs fused.

Production token path: ``c_a=768``, ``c_hidden=48``, ``H=16``, ``c_z=128``.
Reports warm wall time and peak-above-input U. Pair-bias cache is a
separate opt-in (3U resident).
"""

from __future__ import annotations

import argparse
import os
import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.triton.fused_diffusion_attn import (  # noqa: E402
    eager_diffusion_attn,
    fused_diffusion_attn,
)
from openfold3.core.model.layers.attention_pair_bias import (  # noqa: E402
    AttentionPairBias,
)

C_HIDDEN = 48
NO_HEADS = 16
C_A = NO_HEADS * C_HIDDEN
C_S = 384
C_Z = 128
WARMUP = 5
REPS = 20


def _u(n: int) -> int:
    return n * n * 128 * 4


def _tensors(n: int, samples: int, dtype: torch.dtype):
    torch.manual_seed(0)
    q = torch.randn(1, samples, NO_HEADS, n, C_HIDDEN, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(1, samples, NO_HEADS, n, C_HIDDEN, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(1, samples, NO_HEADS, n, C_HIDDEN, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, samples, 1, 1, n, device="cuda", dtype=dtype)
    pair = torch.randn(1, 1, NO_HEADS, n, n, device="cuda", dtype=dtype) * 0.1
    scale = 1.0 / (C_HIDDEN**0.5)
    return q, k, v, mask, pair, scale


def _module_tensors(n: int, samples: int, dtype: torch.dtype):
    torch.manual_seed(0)
    a = torch.randn(1, samples, n, C_A, device="cuda", dtype=dtype) * 0.5
    s = torch.randn(1, samples, n, C_S, device="cuda", dtype=dtype) * 0.5
    z = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = torch.ones(1, 1, n, device="cuda", dtype=dtype)
    return a, s, z, mask


def _sync():
    torch.cuda.synchronize()


def _time(fn, reps: int) -> float:
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3 / reps


def _peak_u(fn, n: int) -> float:
    """Transient peak above live allocations (inputs / params already placed)."""
    _sync()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    _sync()
    return max(0.0, (torch.cuda.max_memory_allocated() - base) / _u(n))


def _input_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _leaves(*tensors: torch.Tensor):
    return [t.detach().requires_grad_(True) for t in tensors]


def _bench_train(args) -> None:
    header = (
        f"{'prec':<12}{'S':>4}{'N':>6}{'eager_ms':>10}{'fused_ms':>10}"
        f"{'speedup':>8}{'eager_U':>9}{'fused_U':>9}"
    )
    print("TRAINING SCORES PATH (fwd+bwd; S is the training sample dim, not inference 1/5)")
    print(header)
    for prec, dtype, tf32 in (
        ("tf32", torch.float32, True),
        ("bf16-mixed", torch.bfloat16, False),
    ):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        for samples in args.samples:
            for n in args.n:
                q0, k0, v0, mask, p0, scale = _tensors(n, samples, dtype)
                go = torch.randn_like(q0)

                def run(attn):
                    q, k, v, p = _leaves(q0, k0, v0, p0)
                    attn(q, k, v, mask, p, scale).backward(go)

                def eager():
                    run(eager_diffusion_attn)

                def fused():
                    run(fused_diffusion_attn)

                for _ in range(max(2, WARMUP // 2)):
                    eager()
                    fused()
                e_ms = _time(eager, args.reps)
                f_ms = _time(fused, args.reps)
                e_u = _peak_u(eager, n)
                f_u = _peak_u(fused, n)
                print(
                    f"{prec:<12}{samples:4d}{n:6d}{e_ms:10.3f}{f_ms:10.3f}"
                    f"{e_ms / f_ms:8.2f}x{e_u:9.3f}{f_u:9.3f}"
                )

    if args.scores_only:
        return

    print()
    print("TRAINING MODULE PATH (AttentionPairBias fwd+bwd)")
    print(header)
    module = (
        AttentionPairBias(
            c_q=C_A,
            c_k=C_A,
            c_v=C_A,
            c_s=C_S,
            c_z=C_Z,
            c_hidden=C_HIDDEN,
            no_heads=NO_HEADS,
            use_ada_layer_norm=True,
        )
        .cuda()
        .train()
    )
    for prec, dtype, tf32 in (
        ("tf32", torch.float32, True),
        ("bf16-mixed", torch.bfloat16, False),
    ):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        for samples in args.samples:
            for n in args.n:
                a0, s0, z0, mask = _module_tensors(n, samples, dtype)
                go = torch.randn_like(a0)

                def run(flag):
                    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = flag
                    a, z = a0.detach().requires_grad_(True), z0.detach().requires_grad_(True)
                    module(a, z, s=s0, mask=mask).backward(go)
                    module.zero_grad(set_to_none=True)

                def eager():
                    run("0")

                def fused():
                    run("1")

                for _ in range(max(2, WARMUP // 2)):
                    fused()
                    eager()
                e_ms = _time(eager, args.reps)
                f_ms = _time(fused, args.reps)
                e_u = _peak_u(eager, n)
                f_u = _peak_u(fused, n)
                print(
                    f"{prec:<12}{samples:4d}{n:6d}{e_ms:10.3f}{f_ms:10.3f}"
                    f"{e_ms / f_ms:8.2f}x{e_u:9.3f}{f_u:9.3f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=None)
    parser.add_argument("--samples", type=int, nargs="+", default=None)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument(
        "--train",
        action="store_true",
        help="fwd+bwd at training sample counts (default S=24,48 N=384,768)",
    )
    args = parser.parse_args()
    if args.train:
        args.n = args.n or [384, 768]
        args.samples = args.samples or [24, 48]
        _bench_train(args)
        return
    args.n = args.n or [256, 384, 768, 1024, 1264]
    args.samples = args.samples or [1, 5]
    os.environ.setdefault("OPENFOLD3_FUSED_DIFFUSION_ATTN", "1")
    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = "0"

    header = (
        f"{'prec':<12}{'S':>4}{'N':>6}{'eager_ms':>10}{'fused_ms':>10}"
        f"{'speedup':>8}{'eager_U':>9}{'fused_U':>9}"
    )
    print("SCORES PATH (precomputed QKV + pair bias)")
    print(header)
    for prec, dtype, tf32 in (
        ("tf32", torch.float32, True),
        ("bf16-mixed", torch.bfloat16, False),
    ):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        for samples in args.samples:
            for n in args.n:
                q, k, v, mask, pair, scale = _tensors(n, samples, dtype)
                baseline = _input_bytes(q, k, v, mask, pair)

                def eager():
                    return eager_diffusion_attn(q, k, v, mask, pair, scale)

                def fused():
                    return fused_diffusion_attn(q, k, v, mask, pair, scale)

                with torch.inference_mode():
                    for _ in range(WARMUP):
                        eager()
                        fused()
                    e_ms = _time(eager, args.reps)
                    f_ms = _time(fused, args.reps)
                    e_u = _peak_u(eager, n)
                    f_u = _peak_u(fused, n)
                print(
                    f"{prec:<12}{samples:4d}{n:6d}{e_ms:10.3f}{f_ms:10.3f}"
                    f"{e_ms / f_ms:8.2f}x{e_u:9.3f}{f_u:9.3f}"
                )

    if args.scores_only:
        return

    print()
    print("MODULE PATH (AttentionPairBias: AdaLN + pair LN/Linear + QKV + attn + gate + Wo)")
    print(header)
    for prec, dtype, tf32 in (
        ("tf32", torch.float32, True),
        ("bf16-mixed", torch.bfloat16, False),
    ):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        module = (
            AttentionPairBias(
                c_q=C_A,
                c_k=C_A,
                c_v=C_A,
                c_s=C_S,
                c_z=C_Z,
                c_hidden=C_HIDDEN,
                no_heads=NO_HEADS,
                use_ada_layer_norm=True,
            )
            .cuda()
            .to(dtype=torch.float32)
            .eval()
        )
        if dtype == torch.bfloat16:
            # fp32 Parameter masters; activations bf16
            pass
        for samples in args.samples:
            for n in args.n:
                a, s, z, mask = _module_tensors(n, samples, dtype)
                baseline = _input_bytes(a, s, z, mask) + sum(
                    p.numel() * p.element_size() for p in module.parameters()
                )

                def eager():
                    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "0"
                    return module(a, z, s=s, mask=mask)

                def fused():
                    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "1"
                    return module(a, z, s=s, mask=mask)

                with torch.inference_mode():
                    for _ in range(WARMUP):
                        fused()
                        eager()
                    e_ms = _time(eager, args.reps)
                    f_ms = _time(fused, args.reps)
                    e_u = _peak_u(eager, n)
                    f_u = _peak_u(fused, n)
                print(
                    f"{prec:<12}{samples:4d}{n:6d}{e_ms:10.3f}{f_ms:10.3f}"
                    f"{e_ms / f_ms:8.2f}x{e_u:9.3f}{f_u:9.3f}"
                )


if __name__ == "__main__":
    main()
