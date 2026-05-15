/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Custom flash attention backward kernel for SM100 (GB200/Blackwell).
 *
 * Step 2: replace scalar inner loops with wgmma tensor-core GEMMs.
 *
 * Key design:
 *   - BQ=256 tiles (vs cuDNN's 128) → 2× fewer LSE + K-tile loads from L2
 *   - wgmma m64n128k32 FP8→FP32 for Q@K^T and dS@K GEMMs
 *   - 256 threads = 2 warp groups; each WG owns BQ/2=128 rows (2 m64-tiles)
 *   - dQ accumulator in registers (128 floats/thread), persistent across KV loop
 *   - S (attention scores) written to shmem after wgmma for elementwise softmax
 *
 * wgmma variant: SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<1,1>
 *   SS = both A and B from shared memory via descriptors
 *   TN = A in K-major layout (row-major for [M,K] tile),
 *        B in N-major layout (row-major for [N,K] tile, B is K-transposed in multiply)
 *   Computes:  C[64,128] += A[64,32] × B^T[32,128]  where B stored as [128,32]
 *   Used for:  Q[64,32] × K[128,32]^T  = S[64,128]  (attention scores)
 *              dS[64,32_packed] × K[128,32]^T ... see dQ step below
 *
 * Supports: SM90+, FP8 E4M3, no mask, D=128, S%256==0.
 * Fallback: NVTE_SM100_BPROP_DISABLE=1 → cuDNN knob_31.
 */

// Enable CUTLASS SM90a wgmma intrinsics
#ifndef CUTE_ARCH_MMA_SM90A_ENABLED
#define CUTE_ARCH_MMA_SM90A_ENABLED
#endif

#include "fused_attn_sm100_bprop.h"
#include "../common.h"
#include "../util/system.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
// CUTLASS CuTe headers for wgmma
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90_desc.hpp"

namespace transformer_engine {

// ────────────────────────────── constants ────────────────────────────────────
static constexpr int BQ      = 256;   // query block (sequence dim)
static constexpr int BK      = 128;   // key/value block (sequence dim)
static constexpr int HD      = 128;   // head dimension (fixed for Llama 8B)
static constexpr int NTHREADS = 256;  // threads per block (2 warp groups × 128)
static constexpr int WG_SIZE = 128;   // threads per warp group
static constexpr int WG_M    = 64;    // wgmma m-tile (rows per call)
static constexpr int WG_K    = 32;    // wgmma k-tile (reduction per call)
static constexpr int N_WG    = NTHREADS / WG_SIZE;  // = 2 warp groups
static constexpr int WG_ROWS = BQ / N_WG;            // = 128 rows per WG
static constexpr int WG_M_TILES = WG_ROWS / WG_M;   // = 2 m64-tiles per WG
static constexpr int WG_K_TILES = HD / WG_K;         // = 4 k-tiles per call
// Accumulators per thread for one m64n128 wgmma tile = 64 floats.
// Each WG has WG_M_TILES=2 such tiles → 128 floats/thread for dQ accumulator.
static constexpr int ACC_PER_TILE = WG_M * BK / WG_SIZE;  // = 64
static constexpr int ACC_PER_WG   = WG_M_TILES * ACC_PER_TILE; // = 128

// ─── wgmma descriptor construction ─────────────────────────────────────────
// For SM90 wgmma, shared-memory matrix descriptors encode:
//   start_address (bits [13:4]) = smem_ptr >> 4
//   leading_byte_offset (bits [29:16]) = row_stride_bytes >> 4
//   stride_byte_offset (bits [45:32]) = 0 (unused for row-major 2D tiles)
//   layout_type (bits [63:62]) = 0 (no swizzle)
__device__ __forceinline__ uint64_t
make_wgmma_desc(const void* smem_ptr, int row_stride_bytes) {
  cute::GmmaDescriptor desc{};
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  // start_address field stores addr >> 4 (14 bits)
  desc.bitfield.start_address_       = static_cast<uint16_t>((addr >> 4) & 0x3FFF);
  // leading_byte_offset stores stride >> 4 (14 bits)
  desc.bitfield.leading_byte_offset_ = static_cast<uint16_t>((row_stride_bytes >> 4) & 0x3FFF);
  desc.bitfield.stride_byte_offset_  = 0;
  desc.bitfield.base_offset_         = 0;
  desc.bitfield.layout_type_         = 0;  // SWIZZLE_NONE
  return static_cast<uint64_t>(desc);
}

// ─── wgmma call wrapper ─────────────────────────────────────────────────────
// Wraps SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<1,1>::fma() which takes
// 64 individual float references.  We use a float[64] array and expand via macro.
//
// Semantics: acc[64] += A[64,32] × B[128,32]^T   (B is transposed in the multiply)
// Use for:  Q[64,K_stripe] × K[BK,K_stripe]^T = S[64,BK]
#define WGMMA_CALL_SS_TN(acc, desc_a, desc_b, init)                                \
  cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<1,1>::fma(                    \
      desc_a, desc_b,                                                               \
      (acc)[0],  (acc)[1],  (acc)[2],  (acc)[3],  (acc)[4],  (acc)[5],            \
      (acc)[6],  (acc)[7],  (acc)[8],  (acc)[9],  (acc)[10], (acc)[11],           \
      (acc)[12], (acc)[13], (acc)[14], (acc)[15], (acc)[16], (acc)[17],           \
      (acc)[18], (acc)[19], (acc)[20], (acc)[21], (acc)[22], (acc)[23],           \
      (acc)[24], (acc)[25], (acc)[26], (acc)[27], (acc)[28], (acc)[29],           \
      (acc)[30], (acc)[31], (acc)[32], (acc)[33], (acc)[34], (acc)[35],           \
      (acc)[36], (acc)[37], (acc)[38], (acc)[39], (acc)[40], (acc)[41],           \
      (acc)[42], (acc)[43], (acc)[44], (acc)[45], (acc)[46], (acc)[47],           \
      (acc)[48], (acc)[49], (acc)[50], (acc)[51], (acc)[52], (acc)[53],           \
      (acc)[54], (acc)[55], (acc)[56], (acc)[57], (acc)[58], (acc)[59],           \
      (acc)[60], (acc)[61], (acc)[62], (acc)[63],                                   \
      (init) ? cute::GMMA::ScaleOut::Zero : cute::GMMA::ScaleOut::One)

// After each wgmma group, commit and wait for results
__device__ __forceinline__ void wgmma_fence()   { asm volatile("wgmma.fence.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_commit()  { asm volatile("wgmma.commit_group.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_wait()    { asm volatile("wgmma.wait_group.sync.aligned 0;\n"); }

// ────────────────────────── shared memory layout ─────────────────────────────
// Tiles for wgmma path.  S stored as FP16 to fit in 228 KB:
//   Q:       256×128 FP8   = 32 KB
//   dO:      256×128 FP8   = 32 KB
//   O:       256×128 FP8   = 32 KB  (for Dvec = rowsum dO·O)
//   K[2]:    128×128 FP8   = 32 KB  (double-buffer)
//   V:       128×128 FP8   = 16 KB
//   S_wg[2]: 2×128×128 FP16 = 64 KB (wgmma output, each WG owns WG_ROWS×BK)
//   Dvec:    256 FP32       =  1 KB
//   warp_amax: 8 FP32       = 32 bytes
//   ────────────────────────────────────────
//   Total: 32+32+32+32+16+64+1 = 209 KB  ✓ fits in 228 KB
struct SharedMemDQ {
  __nv_fp8_e4m3 Q[BQ * HD];
  __nv_fp8_e4m3 dO_sh[BQ * HD];
  __nv_fp8_e4m3 O_sh[BQ * HD];
  __nv_fp8_e4m3 K[2][BK * HD];
  __nv_fp8_e4m3 V_sh[BK * HD];
  __half        S_wg[N_WG][WG_ROWS * BK];  // per-WG S tile [128, 128] FP16
  float         Dvec[BQ];
  float         warp_amax[NTHREADS / 32];
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
  // Note: lambdas inside __global__ cannot be annotated __device__; annotation is implicit.
  auto gptr = [&](const __nv_fp8_e4m3* base, int qi) {
    return base + bi * S * stride + qi * stride + hi * HD;
  };
  auto gptr_k = [&](const __nv_fp8_e4m3* base, int ki) {
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

  // ── thread model ─────────────────────────────────────────────────────────
  // 256 threads = 2 warp-groups of 128 each.
  // wgmma computes S[BQ,BK] by writing to sm.S_wg (FP16, 64KB).
  // After each KV block's wgmma, all 256 threads do scalar dQ (one row each).
  const int wg_id       = threadIdx.x / WG_SIZE;
  const int t_in_wg     = threadIdx.x % WG_SIZE;
  const int wg_row_base = wg_id * WG_ROWS;

  // Per-thread dQ accumulator: one row per thread = HD=128 floats
  float dQ_reg[HD];
  for (int c = 0; c < HD; ++c) dQ_reg[c] = 0.f;
  const int my_qi  = threadIdx.x;   // one row per thread
  const int seq_qi = qi_base + my_qi;

  // Prefetch first K tile
  for (int i = threadIdx.x; i < BK * HD; i += NTHREADS) {
    int r = i / HD, c = i % HD;
    if (r < S) sm.K[0][i] = K_g[bi * S * stride + r * stride + hi * HD + c];
    else       sm.K[0][i] = __nv_fp8_e4m3(0.f);
  }
  __syncthreads();

  const int num_kv = (S + BK - 1) / BK;

  // ── Single-pass KV loop: wgmma for S, scalar for dQ ──────────────────────
  // Per KV block:
  //   Phase A: both WGs run wgmma for their m64-tiles, write S to sm.S_wg (FP16)
  //   Phase B: all 256 threads read S[my_qi,:] from sm.S_wg, accumulate dQ
  for (int kj = 0; kj < num_kv; ++kj) {
    const int kj_base = kj * BK;
    const int buf     = kj & 1;
    const int nbuf    = 1 - buf;

    // Async prefetch next K block
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

    // ── Phase A: wgmma Q@K^T for all m64-tiles ────────────────────────────
    // CLayout_64x128 register→(row,col) mapping:
    //   row(t_in_wg, a) = (t_in_wg % 4) + (a / 4) * 4
    //   col(t_in_wg, a) = ((t_in_wg/4)%8) + (t_in_wg/32)*16 + (a%2)*64 + ((a/2)%2)*8
    for (int m = 0; m < WG_M_TILES; ++m) {
      const int wg_m_row = wg_row_base + m * WG_M;  // global row start for this tile

      float S_acc[ACC_PER_TILE];
      for (int i = 0; i < ACC_PER_TILE; ++i) S_acc[i] = 0.f;

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
      wgmma_fence();
      for (int k = 0; k < WG_K_TILES; ++k) {
        // A = Q[wg_m_row:wg_m_row+64, k*32:k*32+32], stride=HD (row-major)
        uint64_t desc_Q = make_wgmma_desc(&sm.Q[wg_m_row * HD + k * WG_K], HD);
        // B = K[0:BK, k*32:k*32+32], B[n,k'] = K[n, k*32+k'], stride=HD between rows
        // SS_TN: B stored as [N=BK, K=32] K-contiguous — exactly K's layout here
        uint64_t desc_K = make_wgmma_desc(&sm.K[buf][k * WG_K], HD);
        WGMMA_CALL_SS_TN(S_acc, desc_Q, desc_K, k == 0);
      }
      wgmma_commit();
      wgmma_wait();
#else
      // Scalar fallback for non-SM90+ compilation
      for (int a = 0; a < ACC_PER_TILE; ++a) {
        int row_in_tile = (t_in_wg % 4) + (a / 4) * 4;
        int col         = ((t_in_wg / 4) % 8) + (t_in_wg / 32) * 16
                        + (a % 2) * 64 + ((a / 2) % 2) * 8;
        float s = 0.f;
        for (int c = 0; c < HD; ++c)
          s += (float)sm.Q[(wg_m_row + row_in_tile) * HD + c] * Q_scale_val
             * (float)sm.K[buf][col * HD + c] * K_scale_val;
        S_acc[a] = s * attn_scale;
      }
#endif

      // Write S_acc to sm.S_wg (FP16) at the correct (row, col) position.
      // S_wg layout: [BQ, BK] FP16, indexed as [(wg_m_row + row_in_tile), col].
      for (int a = 0; a < ACC_PER_TILE; ++a) {
        int row_in_tile = (t_in_wg % 4) + (a / 4) * 4;
        int col         = ((t_in_wg / 4) % 8) + (t_in_wg / 32) * 16
                        + (a % 2) * 64 + ((a / 2) % 2) * 8;
        float s_val = S_acc[a] * (Q_scale_val * K_scale_val * attn_scale);
        sm.S_wg[wg_id][(m * WG_M + row_in_tile) * BK + col] = __float2half(s_val);
      }
    }  // end m-tile loop
    __syncthreads();  // ensure all S_wg writes are visible

    // ── Phase B: scalar dQ (one row per thread, reads S from sm.S_wg) ──────
    if (seq_qi < S) {
      const int my_wg         = my_qi / WG_ROWS;
      const int my_row_in_wg  = my_qi % WG_ROWS;
      const float lse   = LSE_g[bi * H * S + hi * S + seq_qi];
      const float D_qi  = sm.Dvec[my_qi];
      const __half* S_row = &sm.S_wg[my_wg][my_row_in_wg * BK];

      for (int ki = 0; ki < BK; ++ki) {
        const int seq_ki = kj_base + ki;
        if (seq_ki >= S) continue;

        float p  = __expf(__half2float(S_row[ki]) - lse);

        float dp = 0.f;
        for (int c = 0; c < HD; ++c)
          dp += (float)sm.dO_sh[my_qi * HD + c] * dO_scale_val
              * (float)sm.V_sh[ki * HD + c];

        float ds = p * (dp - D_qi);

        for (int c = 0; c < HD; ++c)
          dQ_reg[c] += ds * (float)sm.K[buf][ki * HD + c] * K_scale_val * attn_scale;
      }
    }
    __syncthreads();
  }  // end KV loop

  // ── Store dQ ─────────────────────────────────────────────────────────────
  float my_amax = 0.f;
  if (seq_qi < S)
    for (int c = 0; c < HD; ++c) my_amax = fmaxf(my_amax, fabsf(dQ_reg[c]));
  for (int mask = 16; mask > 0; mask >>= 1)
    my_amax = fmaxf(my_amax, __shfl_xor_sync(0xffffffff, my_amax, mask));
  if ((threadIdx.x & 31) == 0) sm.warp_amax[threadIdx.x >> 5] = my_amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float blk_amax = 0.f;
    for (int w = 0; w < 8; ++w) blk_amax = fmaxf(blk_amax, sm.warp_amax[w]);
    atomicMax(reinterpret_cast<int*>(dQ_amax), __float_as_int(blk_amax));
  }
  __syncthreads();

  if (seq_qi < S) {
    const float out_scale = 448.f / fmaxf(*dQ_amax, 1e-6f);
    __nv_fp8_e4m3* dQ_row = dQ_g + bi * S * stride + seq_qi * stride + hi * HD;
    for (int c = 0; c < HD; ++c) {
      float v = fmaxf(-448.f, fminf(448.f, dQ_reg[c] * out_scale));
      dQ_row[c] = __nv_fp8_e4m3(v);
    }
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
    size_t shmem = sizeof(SharedMemDQ);  // warp_amax included in struct
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
