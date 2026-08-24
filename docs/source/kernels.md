# cuEquivariance Kernels 

OF3 supports cuEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model.
Note: cuEquivariance acceleration can be used while DeepSpeed acceleration is enabled. 
      cuEquivariance would take precedence, and then would fall back to either DeepSpeed (if enabled) or PyTorch for the shapes it does not handle efficiently.
      Notably, it would fall back for shorter sequences (threshold controlled by `CUEQ_TRIATTN_FALLBACK_THRESHOLD` environment variable), and for shapes with hidden dimension > 128 (diffusion transformer shapes).

To enable cuequivariance with pixi, use the `openfold3-cuda12-pypi` or `openfold3-cuda13-pypi` environment. Below is a example inference command

```bash
pixi run -e openfold3-cuda12-pypi \
  run_openfold predict --query-json=query_ubiquitin.json  --runner-yaml=cuequivariance.yml
```

For other workflows, cuequivariance must first be installed with the cuequivariance optional dependency, e.g.

```bash
pip install openfold3[cuequivariance]
```

Then, to enable these kernels via the runner.yaml, add the following:

```yaml
# cuequivariance.yml
model_update:
  presets: 
    - "predict"
    - "low_mem"  # for lower memory systems
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: true  # set this to False to use cueq only
```

This runner.yml is specifically for inference, but similar settings can be used for training.

# Online template coordinate projection

OpenFold3 can replace resident N² template distogram / unit-vector features with compact O(N) coordinates and project pair features online in the template embedder. Enable it from dataset template settings:

```yaml
# Inference (`dataset_config_kwargs`) or training (`dataset_configs.*.*.config`)
template:
  use_coordinate_pair_features: true
```

When the batch contains `template_pseudo_beta_coords` / `template_frame_atom_coords`, the model uses the coordinate embedder automatically and streams templates on GPU (no template offload).

## Triton kernel

On CUDA with eligible shapes (`B=1`, output channels `64`, activation dtype `float32` or `bfloat16`), projection uses a length-generic Triton kernel (N and strides are not specialized). Training uses an autograd wrapper with a Triton split-K backward; inference may update the pair tensor in place. Geometry inputs stay fp32; under bf16 autocast, pair activations remain bf16 while the kernel accumulates in fp32.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_TEMPLATE_COORD=1

# Force the chunked eager reference path
OPENFOLD3_FUSED_TEMPLATE_COORD=0
```

TF32 for the Triton backward GEMM follows `torch.backends.cuda.matmul.allow_tf32`. Distogram settings must remain the defaults (`min_bin=3.25`, `max_bin=50.75`, `n_bins=39`).

# Fused input relative-position embedding

The input embedder can add relative-position pair features from compact index tables instead of materializing an N² one-hot `relpos_complex` tensor and running `linear_relpos` over it. When `linear_relpos` is bias-free (the default all-atom preset), OpenFold3 uses this weight-table path automatically for both inference and training.

Under inference / no-grad, the gather-add may update the pair tensor in place, and bias-free token-bonds are folded in with `addmm_` so a pair-sized linear output is not allocated. Under training, an autograd wrapper keeps a length-generic Triton split-K weight backward (deterministic per-split accumulate, fp32 reduction) and falls back to a differentiable eager gather-add when Triton is ineligible (for example CPU or non-contiguous inputs).

## Triton kernel

On CUDA with contiguous pair / weight tensors and activation dtype `float32` or `bfloat16`, the fused gather-add uses a length-generic Triton kernel (sequence length is only the launch grid, so one compile serves all N). Sequence-length specialization is intentionally avoided.

IEEE fp32 and TF32 share the same gather-add: there is no `tl.dot`, so `torch.backends.cuda.matmul.allow_tf32` does not change the math. bf16-mixed keeps fp32 Parameter masters; weight tiles are upcast on load, the add and `dW` accumulate in fp32, and pair activations stay bf16. Matching bf16 weights remain eligible. The eager fallback downcasts the table when the pair and weight dtypes do not match.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_RELPOS=1

# Force the eager weight-table path
OPENFOLD3_FUSED_RELPOS=0
```

If `linear_relpos` has a bias, the classic `relpos_complex` + `Linear` path is retained. Diffusion conditioning `_embed_zij` is a different op (`LN(cat(z, one-hot)) → Linear`); see **Fused LayerNorm-to-Linear** below.

# Fused pair SwiGLU transition

Pair `SwiGLUTransition` (LN → SwiGLU → Linear, optional mask and residual) can run as one length-generic Triton kernel instead of the module primitives. The dispatcher is `SwiGLUTransition`; ineligible shapes keep the eager path.

Under inference / no-grad, the kernel may write `residual + transition(x)` in place when `residual` is `x`. Under training, an autograd wrapper saves `x_hat` / `mean` / `rstd` and rematerializes the SwiGLU hidden activations; `dW` uses exclusive split-M tiles (deterministic, no atomics) and accumulates in fp32.

## Triton kernel

On CUDA with a contiguous activation, `float32` or `bfloat16` activations, fp32 Parameter masters, `c_in ≤ 256`, hidden width `≤ 512`, and at least 4096 rows, the fused path uses one `GEMM_MODE` for every `tl.dot`:

- `ieee` — fp32 activations and `torch.backends.cuda.matmul.allow_tf32` off
- `tf32` — fp32 activations and `allow_tf32` on
- `bf16` — bf16 activations with fp32 masters (tiles downcast for the GEMM, accumulate in fp32)

bf16 weights are ineligible and fall back to `SwiGLUTransition`. Sequence length is not a specialize key.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_SWIGLU_TRANSITION=1

# Force the SwiGLUTransition primitives
OPENFOLD3_FUSED_SWIGLU_TRANSITION=0
```

# Fused LayerNorm-to-Linear

`DiffusionConditioning` can fuse LayerNorm → Linear as one length-generic Triton kernel instead of `F.layer_norm` + `F.linear`. The shared backbone is `fused_ln_linear`. `_embed_zij` (`LN(cat(zij_trunk, relpos_complex)) → Linear`) uses a separate wrapper that synthesizes the concat row in registers from trunk `z` and the same compact relpos indices as the input-embedder path, so the 139-d one-hot and 267-d concat are never written. Sibling sites `linear_s` / `linear_n` stay eager (`c_in` or row count is ineligible).

Under training, an autograd wrapper saves `x` / `mean` / `rstd` (or `z` plus indices for `_embed_zij`) and rematerializes the LN output; `dW` uses exclusive split-M tiles (deterministic, no atomics) and accumulates in fp32. Ineligible shapes keep the eager primitives, and inference `_embed_zij` keeps the row-chunked `relpos_complex` fallback.

## Triton kernel

On CUDA with a contiguous activation, `float32` or `bfloat16` activations, fp32 Parameter masters, `c_in ≤ 512`, `c_out ≤ 512`, and at least 4096 rows, the fused path uses one `GEMM_MODE` for every `tl.dot`:

- `ieee` — fp32 activations and `torch.backends.cuda.matmul.allow_tf32` off
- `tf32` — fp32 activations and `allow_tf32` on
- `bf16` — bf16 activations with fp32 masters (tiles downcast for the GEMM, accumulate in fp32)

bf16 weights are ineligible. Sequence length is not a specialize key. `_embed_zij` matches the production site (no LN offset, no Linear bias).

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_LN_LINEAR=1

# Force eager LN + Linear (and the chunked _embed_zij fallback)
OPENFOLD3_FUSED_LN_LINEAR=0
```

# Fused triangle multiplicative update

`TriangleMultiplicationOutgoing` / `Incoming` can fuse LN_in → gated A/B projections → triangle einsum → LN_out → gated output [+ residual] as length-generic Triton kernels instead of the eager pair-stack primitives (or cuEq). Sequence length is not a specialize key. Under the 2a `OPENFOLD3_TRIMUL_CHUNK_CAP` memory profile the same kernels run an I-row / grouped-K schedule so the cap stays the memory bound.

In-place `z + update` writes are inference-only when `residual` is `z`. Training keeps A/B/X from the forward (non-reentrant checkpoint packs the first-forward copies so they do not stack across pair blocks). The output-gate backward fuses LN_out, the gated Linear, and exclusive split-M `dW` so `g` / `val` / `x_hat` are never stored; A-side and B-side `dW`/`dX` run sequentially. Ineligible shapes (bf16 weights, `c > 128`, too few rows, linear bias) keep the eager module.

## Triton kernel

On CUDA with a contiguous `[1, N, N, C]` activation, `float32` or `bfloat16` activations, fp32 Parameter masters, `C_z, C_hidden ≤ 128`, and at least 4096 pair rows, the fused path uses one `GEMM_MODE` for every `tl.dot`:

- `ieee` — fp32 activations and `torch.backends.cuda.matmul.allow_tf32` off
- `tf32` — fp32 activations and `allow_tf32` on
- `bf16` — bf16 activations with fp32 masters (downcast once before the MMA, accumulate in fp32)

Forward writes LN_in once, then dual-gemm and the output-gate GEMMs are clean MMAs (no in-loop LN or per-tile master convert). GEMM tiles autotune on ``GEMM_MODE`` only; LN tiles autotune on an empty key. Dual-gemm warms that cache on a large dummy grid so a small first ``M`` cannot lock the winner. Sequence length is not an autotune key.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_TRIMUL=1

# Force eager / cuEq / chunked inference trimul
OPENFOLD3_FUSED_TRIMUL=0
```

# Fused triangle attention

`TriangleAttention` (starting- and ending-node, including PairBlock's pre-transposed ending call) can fuse LN → pair-bias Linear → gated MHA → ``W_o`` [+ residual] as length-generic Triton kernels. Sequence length is not a specialize key. The pair-bias prologue decodes ``(i, j)`` from independent strides so a contig starting-node pair and a transposed ending-node view share one compile. Attention is flash / online-softmax: the ``[I, H, J, J]`` score matrix is never stored. Under the 2a ``OPENFOLD3_TRI_ATTN_CHUNK_CAP`` memory profile the same kernels run an I-row schedule so the cap stays the memory bound.

In-place ``z + update`` writes are inference-only when ``residual`` is ``z``. Training rematerializes QKV from saved ``z`` / LN stats and keeps ``LSE``, ungated ``O``, and the ``[H, I, J]`` triangle bias. Backward uses Triton LN / Linear / flash kernels and deterministic exclusive split-M ``dW`` (no atomics). Rematerialized QKVG and ``dQKVG`` stay in a ``[4, M, C]`` layout. ``W_o`` ``dX`` fuses with gate backward in one launch so ``d_attn`` is never stored; that fused kernel also emits flash ``delta`` while ``O`` / ``dO`` are live. QKVG ``dX`` overwrites each row block of ``d_z_hat``; pair-bias ``dX`` accumulates into the same buffer. Row-block ``dBias`` / ``dW`` partials persist until one final reduction. Pair-bias writes ``[H, I, J]`` directly. Row-block QKVG reuses the Phase 3a LN→Linear launch when eligible. GEMM tiles autotune on ``GEMM_MODE`` only; the first small-N launch warms a large dummy ``J`` / ``M`` so a test-sized call cannot lock the winner. Ineligible shapes (bf16 weights, ``c > 128``, too few rows, linear bias, missing gate) keep the eager module.

## Triton kernel

On CUDA with a ``[1, N, N, C]`` activation, ``float32`` or ``bfloat16`` activations, fp32 Parameter masters, ``C_z = H × c_hidden ≤ 128``, ``c_hidden ∈ {16, 32, 64, 128}``, and at least 4096 pair rows, the fused path uses one ``GEMM_MODE`` for every ``tl.dot``:

- `ieee` — fp32 activations and `torch.backends.cuda.matmul.allow_tf32` off
- `tf32` — fp32 activations and `allow_tf32` on
- `bf16` — bf16 activations with fp32 masters (downcast once before the MMA, accumulate in fp32)

GEMM tiles autotune on ``GEMM_MODE`` only. Sequence length is not an autotune key. The flash forward may group two pair-rows per program (``I_TILE``) so they share one triangle-bias load. Backward rematerializes QKVG and writes ``dQKVG`` as ``[4, M, C]``. One pair-row per program owns the flash backward; ``dBias`` folds into ``dQ`` with an exclusive I-split. Packed pair-bias ``dW`` / ``dX`` stay on the activation-major layout; QKVG ``dW`` / ``dX`` consume the SoA buffer. Deterministic ``dW`` remains tiled over 64 input channels.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_TRI_ATTN_V1=1

# Force eager / cuEq / AMD Triton-evo / chunked inference triangle attention
OPENFOLD3_FUSED_TRI_ATTN_V1=0
```

# Fused diffusion attention

Token-level `AttentionPairBias` in `DiffusionTransformer` (AdaLN, self-attention, `c_a=768`, `c_hidden=48`, `H=16`) can run as a length-generic Triton flash kernel instead of materializing `[B, S, H, N, N]` scores. Inference projects pair bias with `LN_z → Linear_z` from a packed view of `z` (no `z` row clone). The live pair tensor is `[H, N, N]` = 0.125U. Q/K/V keep their native strides (no `.contiguous()` clone). AdaLN, QKV, gate, `W_o`, and ada-out stay on the eager module. `N_Q` / `N_K` / `S` / strides are not specialize keys. One program owns a `(b, s, h)` Q-tile. Production token lengths are one-shot (`_PAIR_ROW_BLOCK=4096`); a Q-row loop exists only if that cap is patched below `N`.

Under training, an autograd wrapper saves `LSE` and rematerializes scores in two kernels (`dQ` + `dKV`). `dQ` loops `S` in one exclusive `(b, h)` program so `dBias` accumulates into `[B, S_pb, H, N_Q, N_K]` — no `[B, S, H, N, N]` scratch and no atomics. Pair-bias `LN_z → Linear_z` uses the `fused_ln_linear` dispatcher when eligible.

There is no production length cutoff: `S=1` and `S>1` both use the fused path when the other guards match. `OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS` remains as an optional override (default 0). Cross-attention (atom enc/dec) is not this kernel.

The optional pair-bias cache stores one `[*, H, N, N]` per token block (`N² × blocks × heads`, 3U at production `H=16`, 24 blocks) and reuses it across the 200-step rollout (`zij` does not depend on `t`). It is **off** by default; speed claims with the cache on must also report the resident U.

## Triton kernel

On CUDA with `[B, S, N, C]` query data, self-attention, two biases, `float32` or `bfloat16` activations, and `c_hidden ≤ 128`, the fused path uses one `GEMM_MODE` for every `tl.dot`:

- `ieee` — fp32 activations and `torch.backends.cuda.matmul.allow_tf32` off
- `tf32` — fp32 activations and `allow_tf32` on
- `bf16` — bf16 activations (native Tensor-Core `tl.dot`, accumulate in fp32)

Forward tiles are the measured per-precision table (bf16 `128×64`, fp32 `64×32`). Sequence length is not an autotune or specialize key. Inference pair bias is the 0.125U Linear output. `SampleDiffusion` drops triangle-kernel flags on the token diffusion path so cuEq / Triton-evo cannot steal an eligible launch.

```bash
# Default: use Triton when eligible (no S=1 length cutoff)
OPENFOLD3_FUSED_DIFFUSION_ATTN=1

# Optional override; default 0 (always on when eligible)
OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS=0

# Force the eager einsum + softmax scores path
OPENFOLD3_FUSED_DIFFUSION_ATTN=0

# Optional rollout pair-bias cache (off; +3U at production token shapes)
OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE=0
```
