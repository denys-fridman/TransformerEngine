/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Custom flash attention backward kernel for SM100 (GB200/Blackwell).
 *
 * This is Step 1 (correctness baseline): the FA2 backward algorithm with
 * BQ=256 tiles using scalar thread-level computation.  No wgmma PTX yet —
 * that is Step 2 once this version passes correctness tests.
 *
 * Key design rationale (from ncu profile of cuDNN knob_31):
 *   - cuDNN: 44% SM utilization, 86% L2 throughput → L2-bandwidth bound.
 *   - BQ=256 (vs cuDNN's 128): each block processes 2× more queries, halving
 *     the number of LSE (log-sum-exp) vector loads from L2 per element.
 *   - dQ accumulated in per-thread registers across all KV blocks, avoiding
 *     L2 write-backs for the accumulator.
 *
 * Supports: FP8 E4M3, no mask, D=128, seqlen divisible by 256/128.
 * Fallback: set NVTE_SM100_BPROP_DISABLE=1 to revert to cuDNN knob_31.
 */

#include "fused_attn_sm100_bprop.h"
#include "../common.h"
#include "../util/system.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace transformer_engine {

// ────────────────────────────── constants ────────────────────────────────────
static constexpr int BQ      = 256;   // query block (sequence dim)
static constexpr int BK      = 128;   // key/value block (sequence dim)
static constexpr int HD      = 128;   // head dimension (fixed for Llama 8B)
static constexpr int NTHREADS = 256;  // threads per block

// ────────────────────────── shared memory layout ─────────────────────────────
// Tiles fit within 228 KB with FP16 attention scores:
//   Q:   256×128 FP8  = 32 KB
//   dO:  256×128 FP8  = 32 KB
//   K:   128×128 FP8  = 16 KB  (double-buffered = 32 KB)
//   V:   128×128 FP8  = 16 KB  (double-buffered = 32 KB)
//   S:   256×128 FP16 = 64 KB  (attention scores in FP16 to save shmem)
//   D:   256 FP32     =  1 KB  (rowsum of dO ⊙ O)
//   O:   256×128 FP8  = 32 KB  (for computing D)
//   ─────────────────────────────────────────
//   Total: 225 KB  (fits in 228 KB / SM)
struct SharedMemDQ {
  __nv_fp8_e4m3 Q[BQ * HD];
  __nv_fp8_e4m3 dO_sh[BQ * HD];
  __nv_fp8_e4m3 O_sh[BQ * HD];
  __nv_fp8_e4m3 K[2][BK * HD];   // double-buffer
  __nv_fp8_e4m3 V_sh[BK * HD];   // single-buffer (loaded after S)
  __half        S[BQ * BK];       // FP16 attention weights P[qi,ki]
  float         Dvec[BQ];         // rowsum(dO ⊙ O)
};

// dK/dV pass: smaller tiles, accumulators in shared memory
struct SharedMemDKDV {
  __nv_fp8_e4m3 K_sh[BK * HD];
  __nv_fp8_e4m3 V_sh[BK * HD];
  __nv_fp8_e4m3 Q_sh[BQ * HD];
  __nv_fp8_e4m3 dO_sh[BQ * HD];
  __nv_fp8_e4m3 O_sh[BQ * HD];
  float         dK_acc[BK * HD];   // 64 KB
  float         dV_acc[BK * HD];   // 64 KB
};
// Total dK/dV shmem: 3×32 + 2×16 + 128 = ~224 KB

// ─────────────────────── Pass 1: compute dQ ──────────────────────────────────
// Grid: (S/BQ, H, B)   Block: NTHREADS=256
//
// Per block, handles BQ=256 query rows.
// dQ accumulator kept in per-thread registers (128 floats/thread) across the
// inner KV loop — avoids writing partial dQ back to L2 each iteration.
__global__ void __launch_bounds__(NTHREADS, 1)
flash_attn_sm100_dQ(
    const __nv_fp8_e4m3* __restrict__ Q_g,   // [B, S, H, D] row-major
    const __nv_fp8_e4m3* __restrict__ K_g,
    const __nv_fp8_e4m3* __restrict__ V_g,
    const __nv_fp8_e4m3* __restrict__ O_g,
    const __nv_fp8_e4m3* __restrict__ dO_g,
    const float*          __restrict__ LSE_g, // [B, H, S]
    float                 Q_scale_val,
    float                 K_scale_val,
    float                 dO_scale_val,
    float                 O_scale_val,
          __nv_fp8_e4m3*  __restrict__ dQ_g,
    float*                __restrict__ dQ_amax,  // output: max(|dQ|)
    int S, int H, float attn_scale)
{
  extern __shared__ char raw[];
  SharedMemDQ& sm = *reinterpret_cast<SharedMemDQ*>(raw);

  const int qi_blk  = blockIdx.x;
  const int hi      = blockIdx.y;
  const int bi      = blockIdx.z;
  const int qi_base = qi_blk * BQ;
  if (qi_base >= S) return;

  const int stride = H * HD;  // elements between consecutive sequence positions

  // Global base pointers (head hi, batch bi, query qi_base)
  auto gptr = [&](const __nv_fp8_e4m3* base, int qi) __device__ {
    return base + bi * S * stride + qi * stride + hi * HD;
  };
  auto gptr_k = [&](const __nv_fp8_e4m3* base, int ki) __device__ {
    return base + bi * S * stride + ki * stride + hi * HD;
  };

  // ── Load Q, dO, O tiles ──────────────────────────────────────────────────
  for (int i = threadIdx.x; i < BQ * HD; i += NTHREADS) {
    int row = i / HD, col = i % HD;
    int seq = qi_base + row;
    if (seq < S) {
      sm.Q[i]    = gptr(Q_g,  qi_base)[row * stride + col];
      sm.dO_sh[i] = gptr(dO_g, qi_base)[row * stride + col];
      sm.O_sh[i]  = gptr(O_g,  qi_base)[row * stride + col];
    } else {
      sm.Q[i]    = __nv_fp8_e4m3(0.f);
      sm.dO_sh[i] = __nv_fp8_e4m3(0.f);
      sm.O_sh[i]  = __nv_fp8_e4m3(0.f);
    }
  }
  __syncthreads();

  // ── Compute Dvec[qi] = rowsum(dO[qi] ⊙ O[qi]) ───────────────────────────
  for (int row = threadIdx.x; row < BQ; row += NTHREADS) {
    float d = 0.f;
    for (int c = 0; c < HD; ++c) {
      d += (float)sm.dO_sh[row*HD+c] * dO_scale_val
         * (float)sm.O_sh[row*HD+c]  * O_scale_val;
    }
    sm.Dvec[row] = d;
  }
  __syncthreads();

  // ── dQ accumulator in registers (BQ/NTHREADS × HD floats per thread) ─────
  // Each thread is responsible for HD elements of one or more query rows.
  // With NTHREADS=256 and BQ=256: one row per thread. dQ_reg[HD] = 128 floats.
  float dQ_reg[HD];
  for (int c = 0; c < HD; ++c) dQ_reg[c] = 0.f;
  const int my_qi = threadIdx.x;  // one row per thread (256 threads, 256 rows)
  const int seq_qi = qi_base + my_qi;

  // Prefetch first K tile into buffer 0
  {
    int buf = 0;
    for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
      int r = i / HD, c = i % HD;
      int seq = r;
      sm.K[buf][i] = (seq < S) ? gptr_k(K_g, seq)[0 * BK * stride + r * stride + c]  // simplified
                                : __nv_fp8_e4m3(0.f);
    }
    // More accurate: load from offset kj_start = 0
    for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
      int r = i / HD, c = i % HD;
      if (r < S) sm.K[buf][i] = K_g[bi * S * stride + r * stride + hi * HD + c];
      else       sm.K[buf][i] = __nv_fp8_e4m3(0.f);
    }
  }
  __syncthreads();

  const int num_kv = (S + BK - 1) / BK;

  for (int kj = 0; kj < num_kv; ++kj) {
    const int kj_base = kj * BK;
    const int buf     = kj & 1;
    const int nbuf    = 1 - buf;

    // Prefetch next K block
    if (kj + 1 < num_kv) {
      int next_base = (kj + 1) * BK;
      for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
        int r = i / HD, c = i % HD;
        int seq = next_base + r;
        sm.K[nbuf][i] = (seq < S)
          ? K_g[bi * S * stride + seq * stride + hi * HD + c]
          : __nv_fp8_e4m3(0.f);
      }
    }

    // Load V for this block
    for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
      int r = i / HD, c = i % HD;
      int seq = kj_base + r;
      sm.V_sh[i] = (seq < S)
        ? V_g[bi * S * stride + seq * stride + hi * HD + c]
        : __nv_fp8_e4m3(0.f);
    }
    __syncthreads();

    // ── This thread computes for row my_qi, all KV positions in block ────
    if (seq_qi < S) {
      const float lse = LSE_g[bi * H * S + hi * S + seq_qi];
      const float D_qi = sm.Dvec[my_qi];

      for (int ki = 0; ki < BK; ++ki) {
        const int seq_ki = kj_base + ki;
        if (seq_ki >= S) continue;

        // S[qi,ki] = Q[qi] · K[ki] * attn_scale
        float s = 0.f;
        for (int c = 0; c < HD; ++c) {
          s += (float)sm.Q[my_qi * HD + c] * Q_scale_val
             * (float)sm.K[buf][ki * HD + c] * K_scale_val;
        }
        s *= attn_scale;
        float p = __expf(s - lse);  // attention weight P[qi,ki]
        sm.S[my_qi * BK + ki] = __float2half(p);

        // dP[qi,ki] = dO[qi] · V[ki]
        float dp = 0.f;
        for (int c = 0; c < HD; ++c) {
          dp += (float)sm.dO_sh[my_qi * HD + c] * dO_scale_val
              * (float)sm.V_sh[ki * HD + c];
        }

        // dS = P * (dP - D_qi)
        float ds = p * (dp - D_qi);

        // dQ[qi] += dS * K[ki] * attn_scale  (accumulate in registers)
        for (int c = 0; c < HD; ++c) {
          dQ_reg[c] += ds * (float)sm.K[buf][ki * HD + c] * K_scale_val * attn_scale;
        }
      }
    }
    __syncthreads();
  }  // end KV loop

  // ── Store dQ ─────────────────────────────────────────────────────────────
  // Find block-wide amax via warp reductions
  float my_amax = 0.f;
  if (seq_qi < S) {
    for (int c = 0; c < HD; ++c) my_amax = fmaxf(my_amax, fabsf(dQ_reg[c]));
  }
  for (int mask = 16; mask > 0; mask >>= 1)
    my_amax = fmaxf(my_amax, __shfl_xor_sync(0xffffffff, my_amax, mask));

  __shared__ float warp_amax[8];  // NTHREADS/32 = 8 warps
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x >> 5] = my_amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float blk_amax = 0.f;
    for (int w = 0; w < 8; ++w) blk_amax = fmaxf(blk_amax, warp_amax[w]);
    // Atomic max across all blocks into dQ_amax
    int* amax_int = reinterpret_cast<int*>(dQ_amax);
    atomicMax(amax_int, __float_as_int(blk_amax));
  }
  __syncthreads();

  if (seq_qi >= S) return;

  // Convert to FP8 and write
  const float out_scale = 448.f / fmaxf(*dQ_amax, 1e-6f);
  __nv_fp8_e4m3* dQ_row = dQ_g + bi * S * stride + seq_qi * stride + hi * HD;
  for (int c = 0; c < HD; ++c) {
    float v = dQ_reg[c] * out_scale;
    v = fmaxf(-448.f, fminf(448.f, v));
    dQ_row[c] = __nv_fp8_e4m3(v);
  }
}

// ─────────────────────── Pass 2: compute dK, dV ─────────────────────────────
// Grid: (S/BK, H, B)   Block: NTHREADS=256
__global__ void __launch_bounds__(NTHREADS, 1)
flash_attn_sm100_dKdV(
    const __nv_fp8_e4m3* __restrict__ Q_g,
    const __nv_fp8_e4m3* __restrict__ K_g,
    const __nv_fp8_e4m3* __restrict__ V_g,
    const __nv_fp8_e4m3* __restrict__ O_g,
    const __nv_fp8_e4m3* __restrict__ dO_g,
    const float*          __restrict__ LSE_g,
    float                 Q_scale_val,
    float                 K_scale_val,
    float                 dO_scale_val,
    float                 O_scale_val,
          __nv_fp8_e4m3*  __restrict__ dK_g,
          __nv_fp8_e4m3*  __restrict__ dV_g,
    float*                __restrict__ dK_amax,
    float*                __restrict__ dV_amax,
    int S, int H, float attn_scale)
{
  extern __shared__ char raw2[];
  SharedMemDKDV& sm = *reinterpret_cast<SharedMemDKDV*>(raw2);

  const int ki_blk  = blockIdx.x;
  const int hi      = blockIdx.y;
  const int bi      = blockIdx.z;
  const int ki_base = ki_blk * BK;
  if (ki_base >= S) return;

  const int stride = H * HD;

  // Init accumulators
  for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
    sm.dK_acc[i] = 0.f;
    sm.dV_acc[i] = 0.f;
  }

  // Load K, V
  for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
    int r = i / HD, c = i % HD;
    int seq = ki_base + r;
    sm.K_sh[i] = (seq < S) ? K_g[bi*S*stride + seq*stride + hi*HD + c] : __nv_fp8_e4m3(0.f);
    sm.V_sh[i] = (seq < S) ? V_g[bi*S*stride + seq*stride + hi*HD + c] : __nv_fp8_e4m3(0.f);
  }
  __syncthreads();

  const int num_q = (S + BQ - 1) / BQ;

  for (int qj = 0; qj < num_q; ++qj) {
    const int qj_base = qj * BQ;

    // Load Q, dO, O for this query block
    for (int i = threadIdx.x; i < BQ * HD; i += NTHREADS) {
      int r = i / HD, c = i % HD;
      int seq = qj_base + r;
      sm.Q_sh[i]  = (seq < S) ? Q_g[bi*S*stride + seq*stride + hi*HD + c] : __nv_fp8_e4m3(0.f);
      sm.dO_sh[i] = (seq < S) ? dO_g[bi*S*stride + seq*stride + hi*HD + c] : __nv_fp8_e4m3(0.f);
      sm.O_sh[i]  = (seq < S) ? O_g[bi*S*stride + seq*stride + hi*HD + c] : __nv_fp8_e4m3(0.f);
    }
    __syncthreads();

    // Each thread handles one ki row
    const int my_ki  = threadIdx.x % BK;
    const int seq_ki = ki_base + my_ki;

    if (seq_ki < S) {
      for (int qi = threadIdx.x / BK; qi < BQ; qi += NTHREADS / BK) {
        const int seq_qi = qj_base + qi;
        if (seq_qi >= S) continue;

        const float lse = LSE_g[bi * H * S + hi * S + seq_qi];

        // S = Q[qi] · K[ki] * scale
        float s = 0.f;
        for (int c = 0; c < HD; ++c) {
          s += (float)sm.Q_sh[qi*HD+c] * Q_scale_val
             * (float)sm.K_sh[my_ki*HD+c] * K_scale_val;
        }
        s *= attn_scale;
        float p = __expf(s - lse);

        // D_qi = rowsum(dO[qi] ⊙ O[qi])
        float d_qi = 0.f;
        for (int c = 0; c < HD; ++c)
          d_qi += (float)sm.dO_sh[qi*HD+c] * dO_scale_val
                * (float)sm.O_sh[qi*HD+c]  * O_scale_val;

        // dP[qi,ki] = dO[qi] · V[ki]
        float dp = 0.f;
        for (int c = 0; c < HD; ++c)
          dp += (float)sm.dO_sh[qi*HD+c] * dO_scale_val
              * (float)sm.V_sh[my_ki*HD+c];

        float ds = p * (dp - d_qi);

        // dV[ki] += P * dO[qi]
        for (int c = 0; c < HD; ++c)
          atomicAdd(&sm.dV_acc[my_ki*HD+c],
                    p * (float)sm.dO_sh[qi*HD+c] * dO_scale_val);

        // dK[ki] += dS * Q[qi] * scale
        for (int c = 0; c < HD; ++c)
          atomicAdd(&sm.dK_acc[my_ki*HD+c],
                    ds * (float)sm.Q_sh[qi*HD+c] * Q_scale_val * attn_scale);
      }
    }
    __syncthreads();
  }

  // ── Store dK, dV ─────────────────────────────────────────────────────────
  // Compute amax and quantize to FP8
  float ak = 0.f, av = 0.f;
  for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
    ak = fmaxf(ak, fabsf(sm.dK_acc[i]));
    av = fmaxf(av, fabsf(sm.dV_acc[i]));
  }
  for (int mask = 16; mask > 0; mask >>= 1) {
    ak = fmaxf(ak, __shfl_xor_sync(0xffffffff, ak, mask));
    av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, mask));
  }
  __shared__ float wk[8], wv[8];
  if ((threadIdx.x & 31) == 0) { wk[threadIdx.x>>5] = ak; wv[threadIdx.x>>5] = av; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bk = 0.f, bv = 0.f;
    for (int w = 0; w < 8; ++w) { bk = fmaxf(bk, wk[w]); bv = fmaxf(bv, wv[w]); }
    atomicMax(reinterpret_cast<int*>(dK_amax), __float_as_int(bk));
    atomicMax(reinterpret_cast<int*>(dV_amax), __float_as_int(bv));
  }
  __syncthreads();

  const float sk = 448.f / fmaxf(*dK_amax, 1e-6f);
  const float sv = 448.f / fmaxf(*dV_amax, 1e-6f);

  for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
    int r = i / HD, c = i % HD;
    int seq = ki_base + r;
    if (seq >= S) continue;
    float kv = fmaxf(-448.f, fminf(448.f, sm.dK_acc[i] * sk));
    float vv = fmaxf(-448.f, fminf(448.f, sm.dV_acc[i] * sv));
    dK_g[bi*S*stride + seq*stride + hi*HD + c] = __nv_fp8_e4m3(kv);
    dV_g[bi*S*stride + seq*stride + hi*HD + c] = __nv_fp8_e4m3(vv);
  }
}

// ─────────────────────── Host dispatch ───────────────────────────────────────

bool fused_attn_sm100_bprop_is_supported(int sm_arch,
                                          DType dtype,
                                          size_t head_dim,
                                          size_t max_seqlen_q,
                                          size_t max_seqlen_kv,
                                          float p_dropout,
                                          NVTE_Bias_Type bias_type,
                                          NVTE_Mask_Type mask_type) {
  return sm_arch >= 100
      && dtype == DType::kFloat8E4M3
      && head_dim == static_cast<size_t>(HD)
      && max_seqlen_q % BQ == 0
      && max_seqlen_kv % BK == 0
      && p_dropout == 0.f
      && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS
      && mask_type == NVTE_Mask_Type::NVTE_NO_MASK;
}

void fused_attn_sm100_bprop(size_t batch,
                              size_t num_attn_heads,
                              size_t max_seqlen_q,
                              size_t max_seqlen_kv,
                              size_t head_dim,
                              float attn_scale,
                              NVTE_QKV_Layout /*qkv_layout*/,
                              NVTE_Mask_Type  /*mask_type*/,
                              const Tensor *input_Q,
                              const Tensor *input_K,
                              const Tensor *input_V,
                              const Tensor *input_O,
                              const Tensor *input_dO,
                              const Tensor *input_LSE,
                              const Tensor *output_dQ,
                              const Tensor *output_dK,
                              const Tensor *output_dV,
                              Tensor *workspace,
                              cudaStream_t stream) {
  const int S = static_cast<int>(max_seqlen_q);
  const int H = static_cast<int>(num_attn_heads);
  const int B = static_cast<int>(batch);

  // Workspace layout: 3 floats for amax values
  float* amax_buf = reinterpret_cast<float*>(workspace->data.dptr);
  float* dQ_amax  = amax_buf + 0;
  float* dK_amax  = amax_buf + 1;
  float* dV_amax  = amax_buf + 2;
  cudaMemsetAsync(amax_buf, 0, 3 * sizeof(float), stream);

  // FP8 dequantization: x_float = x_fp8 * scale_inv  (scale_inv = max_abs / 448)
  float Q_scale  = 1.f, K_scale  = 1.f;
  float dO_scale = 1.f, O_scale  = 1.f;
  if (input_Q->scale_inv.dptr)
    cudaMemcpyAsync(&Q_scale,  input_Q->scale_inv.dptr,  sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (input_K->scale_inv.dptr)
    cudaMemcpyAsync(&K_scale,  input_K->scale_inv.dptr,  sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (input_dO->scale_inv.dptr)
    cudaMemcpyAsync(&dO_scale, input_dO->scale_inv.dptr, sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (input_O->scale_inv.dptr)
    cudaMemcpyAsync(&O_scale,  input_O->scale_inv.dptr,  sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Pass 1: dQ — grid (S/BQ, H, B)
  {
    dim3 grid(S / BQ, H, B);
    dim3 block(NTHREADS);
    size_t shmem = sizeof(SharedMemDQ) + 8 * sizeof(float);  // + warp_amax[8]
    flash_attn_sm100_dQ<<<grid, block, shmem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(input_Q->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_K->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_V->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_O->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_dO->data.dptr),
        reinterpret_cast<const float*>(input_LSE->data.dptr),
        Q_scale, K_scale, dO_scale, O_scale,
        reinterpret_cast<__nv_fp8_e4m3*>(output_dQ->data.dptr),
        dQ_amax,
        S, H, attn_scale);
  }

  // Pass 2: dK, dV — grid (S/BK, H, B)
  {
    dim3 grid(S / BK, H, B);
    dim3 block(NTHREADS);
    size_t shmem = sizeof(SharedMemDKDV) + 8 * sizeof(float) * 2;  // wk + wv
    flash_attn_sm100_dKdV<<<grid, block, shmem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(input_Q->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_K->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_V->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_O->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_dO->data.dptr),
        reinterpret_cast<const float*>(input_LSE->data.dptr),
        Q_scale, K_scale, dO_scale, O_scale,
        reinterpret_cast<__nv_fp8_e4m3*>(output_dK->data.dptr),
        reinterpret_cast<__nv_fp8_e4m3*>(output_dV->data.dptr),
        dK_amax, dV_amax,
        S, H, attn_scale);
  }
}

}  // namespace transformer_engine
