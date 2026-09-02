#!/usr/bin/env python
"""Breakdown of fused triangle-attention backward vs eager / cuEq."""

from __future__ import annotations

import os
import time
from collections import defaultdict

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.cueq_utils import is_cuequivariance_available  # noqa: E402
from openfold3.core.kernels.triton import fused_tri_attn as fta  # noqa: E402
from openfold3.core.kernels.triton.fused_tri_attn import (  # noqa: E402
    eager_tri_attn,
    fused_tri_attn,
)
from openfold3.core.model.layers.triangular_attention import (  # noqa: E402
    TriangleAttention,
    TriangleAttentionEndingNode,
)

C_Z, C_HIDDEN, NO_HEADS = 128, 32, 4
WARM = int(os.environ.get("PROF_WARM", "3"))
REPS = int(os.environ.get("PROF_REPS", "8"))


def _sync():
    torch.cuda.synchronize()


class _Timer:
    def __init__(self):
        self.ms = defaultdict(float)

    def add(self, name: str, ms: float):
        self.ms[name] += ms

    def report(self, reps: int) -> list[tuple[str, float]]:
        rows = [(k, self.ms[k] / reps) for k in self.ms]
        rows.sort(key=lambda kv: -kv[1])
        return rows


def _wrap(timer: _Timer, fn, name: str):
    def wrapped(*a, **k):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn(*a, **k)
        e.record()
        _sync()
        timer.add(name, s.elapsed_time(e))
        return out

    return wrapped


def _wrap_flash(timer: _Timer):
    def wrapped(*args, **kwargs):
        s_all, e_all = torch.cuda.Event(True), torch.cuda.Event(True)
        s_all.record()
        q, k, v, do, lse, delta, tb, mask, dtb_partial = args[:9]
        row_start, scale, mask_inf = args[9:12]
        accumulate_dtb = kwargs.get("accumulate_dtb", False)
        q = q if q.is_contiguous() else q.contiguous()
        k = k if k.is_contiguous() else k.contiguous()
        v = v if v.is_contiguous() else v.contiguous()
        do = do if do.is_contiguous() else do.contiguous()
        rows, j, heads, ch = q.shape
        split = min(fta._BWD_BIAS_SPLIT, max(rows, 1))
        dq_acc = torch.empty((rows, j, heads, ch), device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dummy = q
        mode = fta._gemm_mode(q)
        has_mask = mask is not None
        if j < fta._FLASH_WARM_J:
            fta._ensure_flash_bwd_autotune(q, heads, ch, has_mask, mode)
        shared = dict(
            I_dim=rows,
            J_dim=j,
            row_start=row_start,
            softmax_scale=float(scale),
            mask_inf=float(mask_inf),
            stride_q_i=q.stride(0),
            stride_q_j=q.stride(1),
            stride_q_h=q.stride(2),
            stride_q_c=q.stride(3),
            stride_tb_h=tb.stride(0),
            stride_tb_j=tb.stride(1),
            stride_tb_k=tb.stride(2),
            stride_mb_i=mask.stride(-2) if mask is not None else 0,
            stride_mb_j=mask.stride(-1) if mask is not None else 0,
            H=heads,
            CH=ch,
            HAS_MASK=has_mask,
            GEMM_MODE=mode,
        )

        def q_grid(meta):
            return (fta.triton.cdiv(j, meta["BLOCK_M"]), split * heads)

        def kv_grid(meta):
            return (fta.triton.cdiv(j, meta["BLOCK_N"]), rows * heads)

        mask_ptr = mask if mask is not None else dummy

        def _time(name, launch):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            launch()
            e.record()
            _sync()
            timer.add(name, s.elapsed_time(e))

        _time(
            "flash_dq_dbias",
            lambda: fta._flash_tri_attn_bwd_dq_kernel[q_grid](
                q,
                k,
                v,
                do,
                lse,
                delta,
                tb,
                mask_ptr,
                dq_acc,
                dtb_partial,
                SPLIT=split,
                ACCUM_DTB=accumulate_dtb,
                **shared,
            ),
        )
        _time(
            "flash_dkv",
            lambda: fta._flash_tri_attn_bwd_dkv_kernel[kv_grid](
                q,
                k,
                v,
                do,
                lse,
                delta,
                tb,
                mask_ptr,
                dk,
                dv,
                **shared,
            ),
        )
        dq = dq_acc if q.dtype == torch.float32 else dq_acc.to(dtype=q.dtype)
        e_all.record()
        _sync()
        timer.add("flash_bwd_total", s_all.elapsed_time(e_all))
        return dq, dk, dv

    return wrapped


def _install(timer: _Timer):
    fta._ln_apply = _wrap(timer, fta._ln_apply, "ln_apply")
    fta._project_qkvg = _wrap(timer, fta._project_qkvg, "remat_qkvg")
    fta._linear_dx = _wrap(timer, fta._linear_dx, "linear_dx")
    fta._linear_dx_soa = _wrap(timer, fta._linear_dx_soa, "linear_dx_soa")
    fta._linear_dx_gate_bwd = _wrap(timer, fta._linear_dx_gate_bwd, "dx_gate_bwd")
    fta._split_m_dw = _wrap(timer, fta._split_m_dw, "split_m_dw")
    fta._split_m_dw_soa = _wrap(timer, fta._split_m_dw_soa, "split_m_dw_soa")
    fta._ln_bwd = _wrap(timer, fta._ln_bwd, "ln_bwd")
    fta._flash_bwd = _wrap_flash(timer)


def _weights(m):
    return (
        m.layer_norm.weight,
        m.layer_norm.bias,
        m.linear_z.weight,
        m.mha.linear_q.weight,
        m.mha.linear_k.weight,
        m.mha.linear_v.weight,
        m.mha.linear_g.weight,
        m.mha.linear_o.weight,
    )


def _kwargs(m):
    return dict(
        c_hidden=m.c_hidden,
        no_heads=m.no_heads,
        starting=m.starting,
        inf=m.inf,
        eps=m.layer_norm.eps,
    )


def _bench_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000 / reps


def main():
    os.environ.setdefault("OPENFOLD3_FUSED_TRI_ATTN_V1", "1")
    starting = os.environ.get("PROF_DIR", "out") != "in"
    prec = os.environ.get("PROF_PREC", "tf32")
    n = int(os.environ.get("PROF_N", "384"))
    torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = prec == "tf32"
    dtype = torch.float32 if prec == "tf32" else torch.bfloat16
    fused_only = os.environ.get("PROF_FUSED_ONLY", "0") in {
        "1",
        "true",
        "yes",
    }
    cls = TriangleAttention if starting else TriangleAttentionEndingNode
    torch.manual_seed(0)
    m = cls(C_Z, C_HIDDEN, NO_HEADS).cuda().eval()
    w = _weights(m)
    kw = _kwargs(m)
    z = torch.randn(1, n, n, C_Z, device="cuda", dtype=dtype) * 0.5
    mask = (
        None
        if os.environ.get("PROF_NO_MASK", "0") in {"1", "true", "yes"}
        else torch.ones(1, n, n, device="cuda", dtype=dtype)
    )
    go = torch.randn_like(z)

    # Isolated totals
    zf = z.detach().requires_grad_(True)
    leaves_f = [t.detach().requires_grad_(True) for t in w]
    yf = fused_tri_attn(zf, mask, *leaves_f, **kw)

    def fused_bwd():
        yf.backward(go, retain_graph=True)
        zf.grad = None
        for t in leaves_f:
            t.grad = None

    if not fused_only:
        ze = z.detach().requires_grad_(True)
        leaves_e = [t.detach().requires_grad_(True) for t in w]
        ye = eager_tri_attn(ze, mask, *leaves_e, **kw)

        def eager_bwd():
            ye.backward(go, retain_graph=True)
            ze.grad = None
            for t in leaves_e:
                t.grad = None

    fused_fwd = _bench_ms(
        lambda: fused_tri_attn(
            z.detach().requires_grad_(True),
            mask,
            *[t.detach().requires_grad_(True) for t in w],
            **kw,
        ),
        WARM,
        REPS,
    )
    eager_fwd = (
        _bench_ms(
            lambda: eager_tri_attn(
                z.detach().requires_grad_(True),
                mask,
                *[t.detach().requires_grad_(True) for t in w],
                **kw,
            ),
            WARM,
            REPS,
        )
        if not fused_only
        else None
    )
    fused_b = _bench_ms(fused_bwd, WARM, REPS)
    eager_b = _bench_ms(eager_bwd, WARM, REPS) if not fused_only else None
    cueq_b = None
    if not fused_only and is_cuequivariance_available():
        os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "0"
        for p in m.parameters():
            p.requires_grad_(True)
        zc = z.detach().requires_grad_(True)
        yc = m.forward(zc, mask=mask, use_cueq_triangle_kernels=True)

        def cueq_bwd():
            yc.backward(go, retain_graph=True)
            zc.grad = None
            m.zero_grad(set_to_none=True)

        cueq_b = _bench_ms(cueq_bwd, WARM, REPS)
        os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1"

    timer = _Timer()
    _install(timer)
    # Rebind the saved graph after wrap: rebuild fused graph so bwd hits wrappers.
    zf = z.detach().requires_grad_(True)
    leaves_f = [t.detach().requires_grad_(True) for t in w]
    yf = fused_tri_attn(zf, mask, *leaves_f, **kw)
    for _ in range(WARM):
        yf.backward(go, retain_graph=True)
        zf.grad = None
        for t in leaves_f:
            t.grad = None
    timer.ms.clear()
    for _ in range(REPS):
        yf.backward(go, retain_graph=True)
        zf.grad = None
        for t in leaves_f:
            t.grad = None
    direction = "out" if starting else "in"
    print(f"CELL {prec} {direction} N={n}")
    totals = f"  totals  fused_fwd={fused_fwd:.2f} fused_bwd={fused_b:.2f}"
    if eager_fwd is not None and eager_b is not None:
        totals += f" eager_fwd={eager_fwd:.2f} eager_bwd={eager_b:.2f}"
    if cueq_b is not None:
        totals += f" cueq_bwd={cueq_b:.2f}"
    print(totals)
    print("  fused bwd breakdown (ms / step):")
    for name, ms in timer.report(REPS):
        pct = 100.0 * ms / fused_b if fused_b else 0.0
        print(f"    {name:<16} {ms:8.2f}  {pct:5.1f}%")
    print(
        "  flash configs:"
        f" dq={getattr(fta._flash_tri_attn_bwd_dq_kernel, 'best_config', None)}"
        f" dkv={getattr(fta._flash_tri_attn_bwd_dkv_kernel, 'best_config', None)}"
        f" split={fta._BWD_BIAS_SPLIT}"
    )


if __name__ == "__main__":
    main()
