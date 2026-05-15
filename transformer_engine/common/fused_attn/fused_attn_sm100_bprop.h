/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_sm100_bprop.h
 *  \brief Custom flash attention backward for SM100 (GB200/Blackwell).
 *
 *  Motivation: ncu roofline of cuDNN's sdpa_sm100_flash_bprop_f8_knob_31
 *  shows 44.3% SM utilization and 85.8% internal memory throughput — the
 *  kernel is L2-bandwidth bound, not compute-bound.  Using BQ=256 tiles
 *  (vs cuDNN's 128) reduces the number of L2 round-trips for the per-row
 *  softmax statistics (LSE) and dQ accumulator, targeting ~80% SM.
 *
 *  Algorithm: Flash Attention 2 backward (two-pass: dQ first, dK/dV second).
 *  Tile sizes: BQ=256 (query block), BK=128 (key/value block), D=128 (fixed).
 *  Thread config: 256 threads / 2 warp-groups (each handles BQ/2=128 rows).
 *  Compute: wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
 *  Memory:  cp.async (async shmem loads) with 2-stage double-buffering.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

/*!
 * \brief Check whether the custom SM100 bprop kernel supports this config.
 *
 * Conditions: SM100+, FP8 E4M3, no dropout, no ALiBi/relative-bias,
 *             head_dim==128, seqlen divisible by 256 (query block size).
 */
bool fused_attn_sm100_bprop_is_supported(int sm_arch,
                                          DType dtype,
                                          size_t head_dim,
                                          size_t max_seqlen_q,
                                          size_t max_seqlen_kv,
                                          float p_dropout,
                                          NVTE_Bias_Type bias_type,
                                          NVTE_Mask_Type mask_type);

/*!
 * \brief Run the custom SM100 flash attention backward.
 *
 * Inputs/outputs match the signature of fused_attn_fp8_bwd so the caller
 * can switch between the cuDNN and custom paths transparently.
 */
void fused_attn_sm100_bprop(size_t batch,
                              size_t num_attn_heads,
                              size_t max_seqlen_q,
                              size_t max_seqlen_kv,
                              size_t head_dim,
                              float attn_scale,
                              NVTE_QKV_Layout qkv_layout,
                              NVTE_Mask_Type mask_type,
                              const Tensor *input_Q,
                              const Tensor *input_K,
                              const Tensor *input_V,
                              const Tensor *input_O,
                              const Tensor *input_dO,
                              const Tensor *input_LSE,   // log-sum-exp [B, H, S]
                              const Tensor *output_dQ,
                              const Tensor *output_dK,
                              const Tensor *output_dV,
                              Tensor *workspace,
                              cudaStream_t stream);

}  // namespace transformer_engine
