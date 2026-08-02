# DSv3 NVFP4 port verification: `dfridman/deepseek-nvfp4-bringup` → `dfridman/ds-nvfp4`

**Date:** 2026-08-02
**Verified at:** `dfridman/ds-nvfp4` @ `cf1d8c44f` (= NVIDIA `main` @ `8685aa0d6` + 2 cherry-picks)
**Scope:** all commits of `dfridman/deepseek-nvfp4-bringup` starting with `f23eb634c`:

```
f23eb634c Support NVFP4 dense grouped MLP paths
d30a81864 Update NVFP4 dense grouped MLP paths
ff12ee114 Use grouped tensor GEMM for NVFP4 FC1 dgrad
d4ee2db4c Fix NVFP4 discrete input grouped GEMM layout
e40a8a118 Added BF16 Cublass GGEMM support
259141ec6 Made NVFP4 GGEMM CG compatible
42c0d7775 Fixed quantizer layout
de312c1ba Added support for MX-FP8 attention offloadin
```

**Method:** the combined diff of these 8 commits (~800 insertions across 6 files) was itemized into
individual functional changes; each item was located at HEAD (semantically, not textually) with
file:line evidence. Background: the bringup branch's `forward_grouped_mlp.py` /
`backward_grouped_mlp.py` no longer exist upstream — NVIDIA main merged them into the joint
`transformer_engine/pytorch/ops/fused/grouped_mlp.py` (PR #3117) and independently gained NVFP4
support there (PR #3133 and others), so most of this branch's work landed upstream in evolved form.

## Verdict

**All functionality from `f23eb634c` onward is present on `dfridman/ds-nvfp4`.**
Two commits are direct cherry-picks; the other six are covered (mostly as supersets) by the evolved
upstream implementations. Four behavioral differences worth knowing are listed below.

## Per-commit accounting

| Commit | Status on `dfridman/ds-nvfp4` |
|---|---|
| `f23eb634c` Support NVFP4 dense grouped MLP paths | Covered by `grouped_mlp.py` — group-quantize with one-group special case, FP4 scale-view permutes and per-group alpha formulas bit-for-bit identical, FC2 grouped-tensor GEMM incl. one-group `general_gemm` case. HEAD adds NVFP4 FC2 bias support and SReLU. |
| `d30a81864` Update NVFP4 dense grouped MLP paths | Covered; one narrow exception (behavioral difference #2). |
| `ff12ee114` NVFP4 FC1 dgrad via grouped tensor GEMM | Covered — multi-group grouped-tensor GEMM NN path and one-group `general_gemm` special case both exist (`grouped_mlp.py:2052-2083`). |
| `d4ee2db4c` Fix NVFP4 discrete-input GEMM layout | Covered — the C++ bug is structurally impossible at HEAD: dims always derive from the logical `shape()` (which un-transposes columnwise-only NVFP4, `common/common.h:386-406`) plus an explicit `storage_transposed` flag (`cublaslt_grouped_gemm.cu:601-650, 1756-1771`), so there is no swap to double-apply. C++ tests exercise NVFP4 discrete-A `transa=N` incl. CUDA-graph capture (`tests/cpp/operator/test_grouped_gemm.cu:1011-1014, 950-951, 962`). The Python workaround revert is also reflected (no concat workaround exists at HEAD). |
| `e40a8a118` BF16 cuBLAS GGEMM support | Covered as a superset — graph-safe BF16/FP16 grouped-tensor path (`grouped_linear.py:1300-1432, 1646-1865`) with bias fused into the GEMM epilogue and persistent cuBLAS workspaces (`cpp_extensions/gemm.py:457-478`; the old per-call `torch.empty` workspaces broke graph capture). FP32/pre-Hopper fall back to the legacy flow (they would have failed on the old path anyway). |
| `259141ec6` NVFP4 GGEMM CUDA-graph compatible | Covered, re-architected — per-group NVFP4 alpha (`amax_A*amax_B/(6²·448²)`) is computed on-device inside the GEMM setup kernel (`cublaslt_grouped_gemm.cu:1404-1421`), making the Python amax packing/caching obsolete; no `.tolist()`/`.item()` anywhere in `grouped_mlp.py`; offsets via `tex.splits_to_offsets_multi`, saved for backward. |
| `42c0d7775` Fixed quantizer layout | On the branch as `fb9266ac3` — hand-ported into main's role-based quantizer dispatch (`quantization.py`): when roles are absent, mode is forward, and `num_quantizers % 3 != 0`, the weight slot is `idx % 2 == 1`; the divisor-3 case falls through to the positional fallback (`idx % 3 == 1`), matching the original exactly. Explicit roles still win when provided. |
| `de312c1ba` MX-FP8 attention offloading | On the branch as `cf1d8c44f` — line-for-line identical to the original (`quantized_tensor.py` `prepare_for_saving`/`restore_from_saved`; the `cpu_offload_v1` module it depends on exists upstream). |

## Behavioral differences

1. **NVFP4 with `disable_rht=True` no longer fuses the grouped MLP.** The old branch fused anyway
   and silently forced RHT back on for activations/grads
   (`_enable_nvfp4_rht_for_group_quantize`); HEAD refuses to fuse (`grouped_mlp.py:734-736`) and
   falls back to unfused GroupedLinear — numerically faithful non-RHT, but unfused. With the
   default RHT-on NVFP4 recipe (the fused-MLP target config) behavior is equivalent.
2. **The `num_groups == 1` sync-avoidance is gone from GroupedLinear's legacy path.** HEAD's
   graph-safe path avoids CPU syncs entirely — at any group count — for NVFP4-with-RHT (discrete
   weights), MXFP8, FP8 current scaling, and FP8 block scaling (gate at
   `grouped_linear.py:794-854`). But recipes that fall back to the legacy split-quantize flow
   (FP8 delayed scaling, NVFP4 without RHT, NVFP4 with `single_grouped_weight`) now hit
   `split_sizes.tolist()` even for one group (`grouped_linear.py:1208, 1488`). Only matters when
   CUDA-graphing those specific configs.
3. **NVFP4 wgrad defaults to a different kernel** — the fused cuDNN CuTe wgrad path with per-group
   `global_scale = columnwise_amax/(448·6)` (`grouped_mlp.py:371-517`) instead of the old generic
   grouped GEMM NT fallback. The old-style fallback is still reachable via
   `NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP=1` for numerics A/B testing.
4. **Test-declared caveats at HEAD:** NVFP4+GeGLU+bias is skipped in tests ("TODO debug
   numerics"), and the NVFP4 fused MLP is tested only in the RHT variant
   (`tests/pytorch/test_grouped_mlp.py:760-806`). Same practical coverage as the old branch, but
   worth knowing if a config touches those corners.

Minor notes: the graph-safe GroupedLinear backward frees saved activations when autograd releases
the context rather than immediately after the wgrad GEMM (memory timing only, no functional
impact). HEAD also extends the fused path beyond the old branch: SReLU (squared-ReLU) grouped
MLPs, GeGLU with non-default params (cuDNN FE ≥ 1.24), and a fused GLU+Hadamard-amax kernel
(FE ≥ 1.26).

## Fused NVFP4 grouped-MLP support matrix (old branch vs HEAD)

| Config | Old branch (`de312c1ba`) | HEAD (`cf1d8c44f`) |
|---|---|---|
| MXFP8 recipe | fused (swiglu, geglu) | fused (swiglu, geglu, srelu) |
| NVFP4 recipe, RHT on (default) | fused fwd+bwd (swiglu, geglu alpha≈1.702 only) | fused fwd+bwd: ScaledSwiGLU, ScaledClampedQGeGLU (any params w/ FE ≥ 1.24), ScaledSReLU (FE ≥ 1.24; hadamard-amax fusion w/ FE ≥ 1.26) |
| NVFP4 recipe, `disable_rht=True` | fused, RHT silently forced on | not fused — falls back to basic GroupedLinear ops |
| FP8 delayed/current/block recipes | not fused | not fused |

## Key equivalence evidence (grouped MLP, at HEAD)

- NVFP4 group-quantize incl. one-group wrap + swizzle: `grouped_mlp.py:104-190`; plus a new
  amax-fed variant `:192-232`.
- FC1 forward NVFP4 (constants, dual scale permutes `(3,4,1,5,2,0)`/`(3,2,1,5,4,0)`, BF16 out,
  FP32 prob, per-group alpha, discrete-weight `b_major="k"`): `grouped_mlp.py:1131-1212, 1304`.
- FC2 forward NVFP4 (grouped-tensor GEMM TN; one-group `general_gemm` TN, split-accumulator off):
  `grouped_mlp.py:1334-1416`.
- FC2 dgrad NVFP4 (dy FP4 views, weight columnwise permutes `(1,2,0)`/`(3,4,2,5,1,0)`,
  alpha `sqrt(amax_dy·amax_wcol)/(448·6)` for GLU): `grouped_mlp.py:1694-1864`.
- FC1 dy requantization to grouped NVFP4 (swizzled-scales flag now genuinely tracked via
  `optimize_for_gemm`, `csrc/quantizer.cpp:2186`): `grouped_mlp.py:1868-1870, 1976-1991`.
- NVFP4 scale swizzling on logical-K view, fused into one C++ helper:
  `csrc/extensions/grouped_mlp_experimental.cpp` (doubles last FP4 data dim), called at
  `grouped_mlp.py:1292-1299, 1486-1493, 1852-1859, 2135-2142`.
- Fusion registration (env gate `NVTE_CUTEDSL_FUSED_GROUPED_MLP`, CC 10.x, FE ≥ 1.23, dims %64,
  runtime `m % 128 == 0`): `grouped_mlp.py:706-785, 829-846, 924-925, 2280-2315`.
