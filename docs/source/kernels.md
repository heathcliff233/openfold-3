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
