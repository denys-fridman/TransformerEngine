/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"
#include "../cast/nvfp4/quantize_transpose_nvfp4.cuh"

void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_silu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, stream);
}

void nvte_group_silu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_silu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, silu<fp32, fp32>>(input, output, nullptr,
                                                                       stream);
}

void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsilu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dsilu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_group_dsilu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dsilu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsilu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}

void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsilu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_group_quantize_dbias_dsilu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor activation_input,
                                     NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                                     NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsilu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

// Fused SwiGLU + NVFP4 quantize in a single TMA kernel pass.
// input:  [M, 2N] BF16 (gate || up concatenated along last dim)
// output: [M, N]  NVFP4 with rowwise (and optional columnwise) scale factors
// Computation: output[i,j] = silu(input[i,j]) * input[i,j+N]  then quantize to NVFP4.
void nvte_swiglu_nvfp4(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu_nvfp4);
  using namespace transformer_engine;
  const Tensor *input_t = convertNVTETensorCheck(input);
  Tensor *output_t = convertNVTETensorCheck(output);
  // Empty noop tensor (IS_GATED path skips the noop early-exit check)
  Tensor dummy_noop{};
  QuantizationConfig quant_config{};  // default: no SR, no RNG
  dispatch::nvfp4::quantize_transpose_gated</*use_2d=*/false, Empty, silu<fp32, fp32>>(
      *input_t, &dummy_noop, output_t, &quant_config, stream);
}

void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(grad, input, output, e, stream);
}

void nvte_clamped_swiglu(const NVTETensor input, NVTETensor output, float limit, float alpha,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_swiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha};
  gated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_clamped_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                          float limit, float alpha, cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_dswiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha};
  dgated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>, clamped_dsilu<fp32, fp32>>(
      grad, input, output, param, stream);
}
