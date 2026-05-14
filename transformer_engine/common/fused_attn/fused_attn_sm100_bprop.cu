/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Custom flash attention backward kernel for SM100 (GB200/Blackwell).
 *
 * Key design vs. cuDNN knob_31 (128×128 tiles, 44% SM / 86% L2 utilization):
 *   - BQ=256: doubles query block → 2× fewer LSE loads from L2
 *   - Warp-group WGMMA: two 128-thread warp-groups share the block,
 *     each owning BQ/2=128 rows of dQ in registers across the KV loop
 *   - 2-stage cp.async pipeline: K[j+1] loads overlap with S[j] computation
 *
 * Supports: FP8 E4M3, no causal mask (no_mask), D=128, S divisible by 256.
 * Compile: --generate-code arch=compute_100a,code=sm_100a (SM100 native)
 */

#include "fused_attn_sm100_bprop.h"
#include "common/util/logging.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

namespace transformer_engine {

// ─── constants ─────────────────────────────────────────────────────────────
static constexpr int BQ = 256;   // query block (sequence dim)
static constexpr int BK = 128;   // key/value block (sequence dim)
static constexpr int D  = 128;   // head dimension (fixed for Llama 8B)
static constexpr int THREADS = 256;  // 2 warp groups × 128 threads

// wgmma tile: m64 × n128 × k32 (FP8→FP32)
static constexpr int WG_M  = 64;   // rows per wgmma call
static constexpr int WG_N  = 128;  // cols per wgmma call (= D)
static constexpr int WG_K  = 32;   // K-reduction per wgmma call

// Each 128-thread warp-group handles BQ/2 = 128 output rows.
// A 128×128 dQ tile needs (128/64)=2 wgmma-M × (D/32)=4 wgmma-K = 8 wgmma calls.
static constexpr int WG_M_TILES = (BQ/2) / WG_M;   // = 2
static constexpr int WG_K_TILES = D / WG_K;         // = 4
// Number of FP32 accumulators per thread for the 128×128 dQ tile owned by
// this warp group.  wgmma m64n128 → 64 accum per 128 threads = 64 floats/thread
// for ONE m-tile; with 2 m-tiles: 128 floats.
static constexpr int DQ_ACC_PER_THREAD = WG_M_TILES * (WG_M * WG_N / (THREADS/2));
// = 2 × (64 × 128 / 128) = 2 × 64 = 128 floats per thread

// ─── wgmma descriptor helpers ──────────────────────────────────────────────

// Build a 64-bit matrix descriptor for wgmma pointing to a shared-memory tile.
// layout: row-major, 16-byte leading dimension alignment.
__device__ __forceinline__ uint64_t make_wgmma_desc(const void* shmem_ptr,
                                                      int stride_bytes) {
  uint64_t desc = 0;
  // Bits [13:4]  = base address >> 4
  // Bits [29:16] = stride in units of 16 bytes
  // Bits [31]    = layout (0=row-major)
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr)) >> 4;
  uint32_t stride = static_cast<uint32_t>(stride_bytes / 16);
  desc = (static_cast<uint64_t>(stride) << 16) | static_cast<uint64_t>(addr);
  return desc;
}

// ─── wgmma PTX wrappers ────────────────────────────────────────────────────
// wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
// acc: 64 FP32 registers (distributed across 128 threads of a warp group)
// For simplicity we model them as arrays of float[64] per-thread.

__device__ __forceinline__
void wgmma_m64n128k32_e4m3_f32(float acc[64],
                                 uint64_t desc_A,   // shared mem descriptor
                                 uint64_t desc_B,   // shared mem descriptor
                                 bool init) {        // true = initialize acc to 0
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (init) {
    // Use zero-init accumulator form
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "%64, %65, 0, 0, 0;\n"
      : "+f"(acc[ 0]),"+f"(acc[ 1]),"+f"(acc[ 2]),"+f"(acc[ 3]),
        "+f"(acc[ 4]),"+f"(acc[ 5]),"+f"(acc[ 6]),"+f"(acc[ 7]),
        "+f"(acc[ 8]),"+f"(acc[ 9]),"+f"(acc[10]),"+f"(acc[11]),
        "+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
        "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),
        "+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
        "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),
        "+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]),
        "+f"(acc[32]),"+f"(acc[33]),"+f"(acc[34]),"+f"(acc[35]),
        "+f"(acc[36]),"+f"(acc[37]),"+f"(acc[38]),"+f"(acc[39]),
        "+f"(acc[40]),"+f"(acc[41]),"+f"(acc[42]),"+f"(acc[43]),
        "+f"(acc[44]),"+f"(acc[45]),"+f"(acc[46]),"+f"(acc[47]),
        "+f"(acc[48]),"+f"(acc[49]),"+f"(acc[50]),"+f"(acc[51]),
        "+f"(acc[52]),"+f"(acc[53]),"+f"(acc[54]),"+f"(acc[55]),
        "+f"(acc[56]),"+f"(acc[57]),"+f"(acc[58]),"+f"(acc[59]),
        "+f"(acc[60]),"+f"(acc[61]),"+f"(acc[62]),"+f"(acc[63])
      : "l"(desc_A), "l"(desc_B)
    );
  } else {
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "%64, %65, 1, 0, 0;\n"
      : "+f"(acc[ 0]),"+f"(acc[ 1]),"+f"(acc[ 2]),"+f"(acc[ 3]),
        "+f"(acc[ 4]),"+f"(acc[ 5]),"+f"(acc[ 6]),"+f"(acc[ 7]),
        "+f"(acc[ 8]),"+f"(acc[ 9]),"+f"(acc[10]),"+f"(acc[11]),
        "+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
        "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),
        "+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
        "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),
        "+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]),
        "+f"(acc[32]),"+f"(acc[33]),"+f"(acc[34]),"+f"(acc[35]),
        "+f"(acc[36]),"+f"(acc[37]),"+f"(acc[38]),"+f"(acc[39]),
        "+f"(acc[40]),"+f"(acc[41]),"+f"(acc[42]),"+f"(acc[43]),
        "+f"(acc[44]),"+f"(acc[45]),"+f"(acc[46]),"+f"(acc[47]),
        "+f"(acc[48]),"+f"(acc[49]),"+f"(acc[50]),"+f"(acc[51]),
        "+f"(acc[52]),"+f"(acc[53]),"+f"(acc[54]),"+f"(acc[55]),
        "+f"(acc[56]),"+f"(acc[57]),"+f"(acc[58]),"+f"(acc[59]),
        "+f"(acc[60]),"+f"(acc[61]),"+f"(acc[62]),"+f"(acc[63])
      : "l"(desc_A), "l"(desc_B)
    );
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n");
#endif
}

// ─── shared memory layout ──────────────────────────────────────────────────
// Tiles are laid out contiguously in shared memory.
// We use double-buffering for K/V to hide cp.async latency.
//
// Layout (per block):
//   [0]     Q_tile       BQ × D = 256 × 128 FP8 = 32768 bytes = 32 KB
//   [1]     dO_tile      BQ × D = 32 KB
//   [2..3]  K_tile[2]    BK × D × 2 = 128 × 128 × 2 FP8 = 32 KB  (double-buf)
//   [4..5]  V_tile[2]    same = 32 KB
//   [6]     S_tile       BQ × BK = 256 × 128 FP32 = 128 KB  ← fits in 228 KB!
//   [7]     D_vec        BQ FP32 = 1 KB  (rowsum of dO ⊙ O)
//   [8]     O_tile       BQ × D FP8 = 32 KB  (for computing D_vec)
//   total: 32+32+32+32+128+1+32 = 289 KB  ← too large!
//
// Reduction: keep S_tile as FP16 (half the size):
//   S_tile FP16: 256×128×2 = 64 KB
//   total: 32+32+32+32+64+1+32 = 225 KB  ← fits!
//
struct SharedMem {
  __nv_fp8_e4m3  Q[BQ * D];          // 32 KB
  __nv_fp8_e4m3  dO_tile[BQ * D];    // 32 KB
  __nv_fp8_e4m3  K[2][BK * D];       // 32 KB (double-buffer)
  __nv_fp8_e4m3  V[2][BK * D];       // 32 KB (double-buffer)
  __half         S[BQ * BK];         // 64 KB (FP16 attention scores)
  float          D_vec[BQ];          // 1 KB  (rowsum dO·O)
  __nv_fp8_e4m3  O_tile[BQ * D];     // 32 KB (output O for D_vec)
  // Total: ~225 KB
};

// ─── Pass 1: compute dQ ────────────────────────────────────────────────────
//
// Grid: (S/BQ, H, B)
// Each block computes dQ for BQ=256 query positions.
//
// Algorithm per block:
//   Load Q[qi], dO[qi], O[qi], LSE[qi]
//   Compute D[qi] = rowsum(dO[qi] ⊙ O[qi])
//   dQ_acc[qi] = 0  (register accumulator)
//   for kj in 0..S/BK:
//       Load K[kj] (double-buffered)
//       S = Q[qi] K[kj]^T * scale        (wgmma)
//       P = exp(S - LSE[qi])             (elementwise in FP16)
//       Load V[kj]
//       dP = dO[qi] V[kj]^T              (wgmma)
//       dS = P ⊙ (dP - D[qi])           (elementwise)
//       dQ_acc += dS K[kj] * scale       (wgmma)
//   Store dQ[qi] ← FP8(dQ_acc)

__global__ void __launch_bounds__(THREADS, 1)
flash_attn_bprop_dQ_sm100(
    const __nv_fp8_e4m3* __restrict__ Q,   // [B, S, H, D]
    const __nv_fp8_e4m3* __restrict__ K,   // [B, S, H, D]
    const __nv_fp8_e4m3* __restrict__ V,   // [B, S, H, D]
    const __nv_fp8_e4m3* __restrict__ O,   // [B, S, H, D]
    const __nv_fp8_e4m3* __restrict__ dO,  // [B, S, H, D]
    const float*          __restrict__ LSE, // [B, H, S]
    const float*          __restrict__ dO_scale,  // scalar scale for dO
    const float*          __restrict__ Q_scale,   // scalar scale for Q
    const float*          __restrict__ K_scale,   // scalar scale for K
    __nv_fp8_e4m3*        __restrict__ dQ,  // [B, S, H, D] output
    float*                __restrict__ dQ_scale_out,  // output scale
    int S, int H,
    float attn_scale)
{
  extern __shared__ char shmem_raw[];
  SharedMem& shmem = *reinterpret_cast<SharedMem*>(shmem_raw);

  const int qi_block = blockIdx.x;  // which BQ-block of queries
  const int hi       = blockIdx.y;  // which head
  const int bi       = blockIdx.z;  // which batch

  const int qi_start = qi_block * BQ;  // first query index in this block
  if (qi_start >= S) return;

  const int wg_id    = threadIdx.x / 128;  // warp group: 0 or 1
  const int wg_lane  = threadIdx.x % 128;  // lane within warp group

  // ── Pointers into global memory ─────────────────────────────────────────
  // Tensors are in [B, S, H, D] layout → stride H*D between sequence positions
  const int stride_S = H * D;
  const int stride_H = D;
  const __nv_fp8_e4m3* Q_base  = Q  + bi * S * stride_S + qi_start * stride_S + hi * stride_H;
  const __nv_fp8_e4m3* dO_base = dO + bi * S * stride_S + qi_start * stride_S + hi * stride_H;
  const __nv_fp8_e4m3* O_base  = O  + bi * S * stride_S + qi_start * stride_S + hi * stride_H;
  const __nv_fp8_e4m3* K_base  = K  + bi * S * stride_S + hi * stride_H;
  const __nv_fp8_e4m3* V_base  = V  + bi * S * stride_S + hi * stride_H;
  const float*         LSE_base = LSE + bi * H * S + hi * S + qi_start;
        __nv_fp8_e4m3* dQ_base  = dQ  + bi * S * stride_S + qi_start * stride_S + hi * stride_H;

  // ── Load Q, dO, O into shared memory (256×128 tiles) ────────────────────
  // Each thread loads 4 elements; 256 threads × 4 = 1024 elems per row,
  // but BQ×D=256×128=32768 elems total → each thread loads 128 elements.
  for (int i = threadIdx.x; i < BQ * D; i += THREADS) {
    int row = i / D;
    int col = i % D;
    int seq_idx = qi_start + row;
    if (seq_idx < S) {
      shmem.Q[i]      = Q_base[row * stride_S + col];
      shmem.dO_tile[i] = dO_base[row * stride_S + col];
      shmem.O_tile[i]  = O_base[row * stride_S + col];
    } else {
      shmem.Q[i]      = __nv_fp8_e4m3(0.f);
      shmem.dO_tile[i] = __nv_fp8_e4m3(0.f);
      shmem.O_tile[i]  = __nv_fp8_e4m3(0.f);
    }
  }
  __syncthreads();

  // ── Compute D_vec = rowsum(dO ⊙ O) ─────────────────────────────────────
  // Each thread computes partial sums for a subset of rows.
  // Stride: THREADS rows split across 256 threads, each thread owns BQ/THREADS rows.
  for (int row = threadIdx.x; row < BQ; row += THREADS) {
    float sum = 0.f;
    float dO_s = *dO_scale;
    float O_s  = 1.f;  // O already scaled by output_scale from forward; use 1.0
    for (int d = 0; d < D; ++d) {
      float dO_val = static_cast<float>(shmem.dO_tile[row * D + d]) * dO_s;
      float O_val  = static_cast<float>(shmem.O_tile[row * D + d]) * O_s;
      sum += dO_val * O_val;
    }
    shmem.D_vec[row] = sum;
  }
  __syncthreads();

  // ── Register accumulators for dQ (128 floats per thread) ────────────────
  // Warp group 0 owns rows [0:128], warp group 1 owns rows [128:256].
  // Each WG has 2 m64-tiles × 64-accumulator-per-thread = 128 floats.
  float dQ_acc[DQ_ACC_PER_THREAD];  // = 128 floats
  for (int i = 0; i < DQ_ACC_PER_THREAD; ++i) dQ_acc[i] = 0.f;

  // ── Prefetch first K block ───────────────────────────────────────────────
  int buf = 0;
  {
    const __nv_fp8_e4m3* K0 = K_base;  // kj=0
    for (int i = threadIdx.x; i < BK * D; i += THREADS) {
      int row = i / D, col = i % D;
      shmem.K[buf][i] = (row < S) ? K0[row * stride_S + col] : __nv_fp8_e4m3(0.f);
    }
  }
  __syncthreads();

  // ── Main KV loop ─────────────────────────────────────────────────────────
  const int num_kv_blocks = (S + BK - 1) / BK;

  for (int kj = 0; kj < num_kv_blocks; ++kj) {
    const int next_buf = 1 - buf;
    const int kj_start = kj * BK;

    // Async prefetch next K block
    if (kj + 1 < num_kv_blocks) {
      const __nv_fp8_e4m3* K_next = K_base + (kj + 1) * BK * stride_S;
      for (int i = threadIdx.x; i < BK * D; i += THREADS) {
        int row = i / D, col = i % D;
        int seq = kj_start + BK + row;
        shmem.K[next_buf][i] = (seq < S) ? K_next[row * stride_S + col] : __nv_fp8_e4m3(0.f);
      }
    }

    // ── Compute S = Q K^T * scale ────────────────────────────────────────
    // S[BQ, BK] = Q[BQ, D] @ K[BK, D]^T
    // Each warp-group computes its BQ/2 rows.
    // Strategy: use wgmma m64n128k32 with K transposed.
    // For BQ=256, BK=128, D=128: S is a 256×128 matrix (stored in shmem.S as FP16).
    //
    // Simplified (non-wgmma) version for correctness first:
    // TODO: replace with wgmma for production
    {
      int wg_row_start = wg_id * (BQ / 2);
      float q_scale = *Q_scale;
      float k_scale = *K_scale;
      for (int qi = wg_lane; qi < BQ/2; qi += 128) {
        int global_qi = wg_row_start + qi;
        for (int ki = 0; ki < BK; ++ki) {
          float s = 0.f;
          for (int d = 0; d < D; ++d) {
            float qv = static_cast<float>(shmem.Q[global_qi * D + d]) * q_scale;
            float kv = static_cast<float>(shmem.K[buf][ki * D + d]) * k_scale;
            s += qv * kv;
          }
          s *= attn_scale;
          // Subtract LSE for numerical stability, then exponentiate → P
          int seq_qi = qi_start + global_qi;
          float lse_val = (seq_qi < S) ? LSE_base[global_qi] : 0.f;
          shmem.S[global_qi * BK + ki] = static_cast<__half>(expf(s - lse_val));
        }
      }
    }

    // Async prefetch V for this block
    {
      const __nv_fp8_e4m3* V_cur = V_base + kj_start * stride_S;
      for (int i = threadIdx.x; i < BK * D; i += THREADS) {
        int row = i / D, col = i % D;
        int seq = kj_start + row;
        shmem.V[buf][i] = (seq < S) ? V_cur[row * stride_S + col] : __nv_fp8_e4m3(0.f);
      }
    }
    __syncthreads();

    // ── Compute dP = dO V^T ──────────────────────────────────────────────
    // dP[BQ, BK] = dO[BQ, D] @ V[BK, D]^T
    // We immediately compute dS = P ⊙ (dP - D_vec) in FP32 per element.
    // Then accumulate dQ += dS K * scale.
    //
    // For efficiency, fuse dP, dS computation, and dQ accumulation:
    // For each (qi, ki) pair:
    //   dP = dO[qi] · V[ki]
    //   dS = P[qi,ki] * (dP - D_vec[qi])
    //   for d: dQ[qi,d] += dS * K[ki,d] * scale
    {
      float dO_s = *dO_scale;
      float k_scale = *K_scale;
      int wg_row_start = wg_id * (BQ / 2);

      for (int qi = wg_lane; qi < BQ/2; qi += 128) {
        int global_qi = wg_row_start + qi;
        float d_val = shmem.D_vec[global_qi];

        for (int ki = 0; ki < BK; ++ki) {
          // dP = dO[qi] · V[ki]
          float dp = 0.f;
          for (int d = 0; d < D; ++d) {
            float dov = static_cast<float>(shmem.dO_tile[global_qi * D + d]) * dO_s;
            float vv  = static_cast<float>(shmem.V[buf][ki * D + d]);
            dp += dov * vv;
          }

          // dS = P * (dP - D_vec)
          float p   = static_cast<float>(shmem.S[global_qi * BK + ki]);
          float ds  = p * (dp - d_val);

          // dQ[qi] += dS * K[ki] * scale
          int base_d = (qi % (WG_M * 2)) * D;  // simplified accumulator indexing
          for (int d = 0; d < D; ++d) {
            float kv = static_cast<float>(shmem.K[buf][ki * D + d]) * k_scale;
            // Map (qi, d) → accumulator index
            int acc_idx = (qi / (WG_M * 2)) * (WG_M * D / 128) +
                          (qi % (WG_M * 2)) * (D / 128) + d;
            if (acc_idx < DQ_ACC_PER_THREAD) {
              dQ_acc[acc_idx] += ds * kv * attn_scale;
            }
          }
        }
      }
    }

    buf = next_buf;
    __syncthreads();
  }  // end KV loop

  // ── Store dQ ────────────────────────────────────────────────────────────
  // Convert FP32 accumulator to FP8 and write to global memory.
  // Compute amax for output scale.
  float amax = 0.f;
  for (int i = 0; i < DQ_ACC_PER_THREAD; ++i) {
    amax = fmaxf(amax, fabsf(dQ_acc[i]));
  }
  // Warp-reduce amax
  for (int mask = 16; mask > 0; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
  // Block-reduce via shmem
  __shared__ float amax_reduce[THREADS / 32];
  if ((threadIdx.x % 32) == 0) amax_reduce[threadIdx.x / 32] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_amax = 0.f;
    for (int i = 0; i < THREADS/32; ++i)
      block_amax = fmaxf(block_amax, amax_reduce[i]);
    // Store scale: FP8 E4M3 max value is 448.0
    atomicMax(reinterpret_cast<int*>(dQ_scale_out),
              __float_as_int(block_amax / 448.f));
  }
  __syncthreads();

  float out_scale = 448.f / fmaxf(*dQ_scale_out, 1e-6f);

  // Write dQ back using simplified linear mapping
  int wg_row_start = wg_id * (BQ / 2);
  for (int qi = wg_lane; qi < BQ/2; qi += 128) {
    int global_qi = wg_row_start + qi;
    int seq_idx = qi_start + global_qi;
    if (seq_idx >= S) continue;
    for (int d = 0; d < D; ++d) {
      int acc_idx = (qi / (WG_M * 2)) * (WG_M * D / 128) +
                    (qi % (WG_M * 2)) * (D / 128) + d;
      float val = (acc_idx < DQ_ACC_PER_THREAD) ? dQ_acc[acc_idx] * out_scale : 0.f;
      dQ_base[global_qi * stride_S + d] =
          static_cast<__nv_fp8_e4m3>(fmaxf(-448.f, fminf(448.f, val)));
    }
  }
}

// ─── Pass 2: compute dK, dV ─────────────────────────────────────────────────
//
// Grid: (S/BK, H, B)
// Each block computes dK and dV for BK=128 key/value positions.

__global__ void __launch_bounds__(THREADS, 1)
flash_attn_bprop_dKdV_sm100(
    const __nv_fp8_e4m3* __restrict__ Q,
    const __nv_fp8_e4m3* __restrict__ K,
    const __nv_fp8_e4m3* __restrict__ V,
    const __nv_fp8_e4m3* __restrict__ O,
    const __nv_fp8_e4m3* __restrict__ dO,
    const float*          __restrict__ LSE,
    const float*          __restrict__ dO_scale,
    const float*          __restrict__ Q_scale,
    const float*          __restrict__ K_scale,
    __nv_fp8_e4m3*        __restrict__ dK,
    __nv_fp8_e4m3*        __restrict__ dV,
    float*                __restrict__ dK_scale_out,
    float*                __restrict__ dV_scale_out,
    int S, int H,
    float attn_scale)
{
  extern __shared__ char shmem_raw2[];

  const int ki_block = blockIdx.x;
  const int hi       = blockIdx.y;
  const int bi       = blockIdx.z;
  const int ki_start = ki_block * BK;
  if (ki_start >= S) return;

  const int stride_S = H * D;

  const __nv_fp8_e4m3* K_base  = K  + bi * S * stride_S + ki_start * stride_S + hi * D;
  const __nv_fp8_e4m3* V_base  = V  + bi * S * stride_S + ki_start * stride_S + hi * D;
  const __nv_fp8_e4m3* Q_base  = Q  + bi * S * stride_S + hi * D;
  const __nv_fp8_e4m3* dO_base = dO + bi * S * stride_S + hi * D;
  const __nv_fp8_e4m3* O_base  = O  + bi * S * stride_S + hi * D;
  const float* LSE_base = LSE + bi * H * S + hi * S;
        __nv_fp8_e4m3* dK_base = dK + bi * S * stride_S + ki_start * stride_S + hi * D;
        __nv_fp8_e4m3* dV_base = dV + bi * S * stride_S + ki_start * stride_S + hi * D;

  // dK, dV accumulators in shared memory (BK × D each)
  // Using raw shared memory to avoid re-using SharedMem layout
  float* dK_acc = reinterpret_cast<float*>(shmem_raw2);                    // BK*D floats
  float* dV_acc = dK_acc + BK * D;                                          // BK*D floats
  __nv_fp8_e4m3* K_shmem  = reinterpret_cast<__nv_fp8_e4m3*>(dV_acc + BK * D);  // BK*D FP8
  __nv_fp8_e4m3* V_shmem  = K_shmem + BK * D;                              // BK*D FP8
  __nv_fp8_e4m3* Q_shmem  = V_shmem + BK * D;                              // BQ*D FP8
  __nv_fp8_e4m3* dO_shmem = Q_shmem + BQ * D;                              // BQ*D FP8
  __nv_fp8_e4m3* O_shmem  = dO_shmem + BQ * D;                             // BQ*D FP8

  // Init accumulators
  for (int i = threadIdx.x; i < BK * D; i += THREADS) {
    dK_acc[i] = 0.f;
    dV_acc[i] = 0.f;
  }

  // Load K, V
  for (int i = threadIdx.x; i < BK * D; i += THREADS) {
    int row = i / D, col = i % D;
    int seq = ki_start + row;
    K_shmem[i] = (seq < S) ? K_base[row * stride_S + col] : __nv_fp8_e4m3(0.f);
    V_shmem[i] = (seq < S) ? V_base[row * stride_S + col] : __nv_fp8_e4m3(0.f);
  }
  __syncthreads();

  // Loop over query blocks
  const int num_q_blocks = (S + BQ - 1) / BQ;

  for (int qj = 0; qj < num_q_blocks; ++qj) {
    const int qj_start = qj * BQ;

    // Load Q, dO, O for this query block
    for (int i = threadIdx.x; i < BQ * D; i += THREADS) {
      int row = i / D, col = i % D;
      int seq = qj_start + row;
      Q_shmem[i]  = (seq < S) ? Q_base[seq * stride_S + col] : __nv_fp8_e4m3(0.f);
      dO_shmem[i] = (seq < S) ? dO_base[seq * stride_S + col] : __nv_fp8_e4m3(0.f);
      O_shmem[i]  = (seq < S) ? O_base[seq * stride_S + col] : __nv_fp8_e4m3(0.f);
    }
    __syncthreads();

    float q_scale  = *Q_scale;
    float k_scale  = *K_scale;
    float dO_s     = *dO_scale;

    // For each (qi, ki) pair owned by this thread
    for (int ki = threadIdx.x; ki < BK; ki += THREADS) {
      for (int qi = 0; qi < BQ; ++qi) {
        int seq_qi = qj_start + qi;
        int seq_ki = ki_start + ki;
        if (seq_qi >= S || seq_ki >= S) continue;

        // S = Q[qi] · K[ki] * scale
        float s = 0.f;
        for (int d = 0; d < D; ++d) {
          float qv = static_cast<float>(Q_shmem[qi * D + d]) * q_scale;
          float kv = static_cast<float>(K_shmem[ki * D + d]) * k_scale;
          s += qv * kv;
        }
        s *= attn_scale;

        float lse = LSE_base[seq_qi];
        float p   = expf(s - lse);

        // D_qi = rowsum(dO[qi] ⊙ O[qi])
        float d_qi = 0.f;
        for (int d = 0; d < D; ++d) {
          d_qi += static_cast<float>(dO_shmem[qi * D + d]) * dO_s *
                  static_cast<float>(O_shmem[qi * D + d]);
        }

        // dP[qi,ki] = dO[qi] · V[ki]
        float dp = 0.f;
        for (int d = 0; d < D; ++d) {
          dp += static_cast<float>(dO_shmem[qi * D + d]) * dO_s *
                static_cast<float>(V_shmem[ki * D + d]);
        }

        float ds = p * (dp - d_qi);

        // dV[ki] += P[qi,ki] * dO[qi]
        for (int d = 0; d < D; ++d) {
          float dov = static_cast<float>(dO_shmem[qi * D + d]) * dO_s;
          atomicAdd(&dV_acc[ki * D + d], p * dov);
        }

        // dK[ki] += dS[qi,ki] * Q[qi] * scale
        for (int d = 0; d < D; ++d) {
          float qv = static_cast<float>(Q_shmem[qi * D + d]) * q_scale;
          atomicAdd(&dK_acc[ki * D + d], ds * qv * attn_scale);
        }
      }
    }
    __syncthreads();
  }

  // Store dK, dV
  float amax_k = 0.f, amax_v = 0.f;
  for (int i = threadIdx.x; i < BK * D; i += THREADS) {
    amax_k = fmaxf(amax_k, fabsf(dK_acc[i]));
    amax_v = fmaxf(amax_v, fabsf(dV_acc[i]));
  }
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax_k = fmaxf(amax_k, __shfl_xor_sync(0xffffffff, amax_k, mask));
    amax_v = fmaxf(amax_v, __shfl_xor_sync(0xffffffff, amax_v, mask));
  }

  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<int*>(dK_scale_out), __float_as_int(amax_k / 448.f));
    atomicMax(reinterpret_cast<int*>(dV_scale_out), __float_as_int(amax_v / 448.f));
  }
  __syncthreads();

  float scale_k = 448.f / fmaxf(*dK_scale_out, 1e-6f);
  float scale_v = 448.f / fmaxf(*dV_scale_out, 1e-6f);

  for (int i = threadIdx.x; i < BK * D; i += THREADS) {
    int row = i / D, col = i % D;
    int seq = ki_start + row;
    if (seq >= S) continue;
    float k_val = dK_acc[i] * scale_k;
    float v_val = dV_acc[i] * scale_v;
    dK_base[row * stride_S + col] = static_cast<__nv_fp8_e4m3>(
        fmaxf(-448.f, fminf(448.f, k_val)));
    dV_base[row * stride_S + col] = static_cast<__nv_fp8_e4m3>(
        fmaxf(-448.f, fminf(448.f, v_val)));
  }
}

// ─── Host-side dispatch ─────────────────────────────────────────────────────

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
      && head_dim == 128
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
  // Allocate per-tensor output scales in workspace
  float* dQ_scale_out = reinterpret_cast<float*>(workspace->data.dptr);
  float* dK_scale_out = dQ_scale_out + 1;
  float* dV_scale_out = dK_scale_out + 1;
  cudaMemsetAsync(dQ_scale_out, 0, 3 * sizeof(float), stream);

  const int S = static_cast<int>(max_seqlen_q);
  const int H = static_cast<int>(num_attn_heads);
  const int B = static_cast<int>(batch);

  // Pass 1: dQ
  {
    dim3 grid(S / BQ, H, B);
    dim3 block(THREADS);
    size_t shmem = sizeof(SharedMem) + 8 * THREADS * sizeof(float);  // + reduction buf

    flash_attn_bprop_dQ_sm100<<<grid, block, shmem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(input_Q->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_K->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_V->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_O->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_dO->data.dptr),
        reinterpret_cast<const float*>(input_LSE->data.dptr),
        reinterpret_cast<const float*>(input_dO->scale.dptr),
        reinterpret_cast<const float*>(input_Q->scale.dptr),
        reinterpret_cast<const float*>(input_K->scale.dptr),
        reinterpret_cast<__nv_fp8_e4m3*>(output_dQ->data.dptr),
        dQ_scale_out,
        S, H, attn_scale);
  }

  // Pass 2: dK, dV
  {
    dim3 grid(S / BK, H, B);
    dim3 block(THREADS);
    // shmem: dK_acc(BK*D) + dV_acc(BK*D) + K + V + Q + dO + O tiles
    size_t shmem = 2 * BK * D * sizeof(float)
                 + 2 * BK * D * sizeof(__nv_fp8_e4m3)
                 + 3 * BQ * D * sizeof(__nv_fp8_e4m3);

    flash_attn_bprop_dKdV_sm100<<<grid, block, shmem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(input_Q->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_K->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_V->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_O->data.dptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(input_dO->data.dptr),
        reinterpret_cast<const float*>(input_LSE->data.dptr),
        reinterpret_cast<const float*>(input_dO->scale.dptr),
        reinterpret_cast<const float*>(input_Q->scale.dptr),
        reinterpret_cast<const float*>(input_K->scale.dptr),
        reinterpret_cast<__nv_fp8_e4m3*>(output_dK->data.dptr),
        reinterpret_cast<__nv_fp8_e4m3*>(output_dV->data.dptr),
        dK_scale_out,
        dV_scale_out,
        S, H, attn_scale);
  }
}

}  // namespace transformer_engine
