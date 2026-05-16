/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Custom flash attention backward kernel.
 *
 * SM90 path: wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 (SS_TN)
 *   - 2 warp groups × 2 m64-tiles, accumulator in registers
 *
 * SM100 path: tcgen05.mma.cta_group::1.kind::f8f6f4 M=128,N=128,K=32 (FP8→FP32 UMMA)
 *   - One thread issues MMA; output to TMEM; warp-cooperative TMEM→shmem readback
 *   - Tile 0: Q[0:128,:]×K^T → TMEM at tmem_base (col 0)
 *   - Tile 1: Q[128:256,:]×K^T → TMEM at tmem_base+128 (col 128)
 *   - Readback: tcgen05.ld.sync.aligned.16x256b.x4; warp W covers DPs W*16..W*16+15
 *   - Register mapping: r=rep*4+f; dp=W*16+t1+(f>>1)*8; col=t0*2+(f&1)+rep*8+ci*32
 *
 * Common: BQ=256 tiles, FP8 E4M3, D=128, S%256==0.
 * Fallback: NVTE_SM100_BPROP_DISABLE=1 → cuDNN knob_31.
 */

#include "fused_attn_sm100_bprop.h"
#include "../common.h"
#include "../util/system.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
// No CUTLASS wgmma headers — use raw PTX to avoid synclog __device__-only conflicts.

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
// SM90 shared-memory matrix descriptor (64-bit):
//   bits [13:4]  = smem_ptr >> 4           (14-bit address)
//   bits [29:16] = row_stride_bytes >> 4   (14-bit leading dimension)
//   bits [63:62] = 0                        (SWIZZLE_NONE)
__device__ __forceinline__ uint64_t
make_wgmma_desc(const void* smem_ptr, int row_stride_bytes) {
  uint32_t addr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t start = static_cast<uint64_t>((addr >> 4) & 0x3FFFu);
  uint64_t stride = static_cast<uint64_t>((row_stride_bytes >> 4) & 0x3FFFu) << 16;
  return start | stride;
}

// ─── wgmma raw PTX wrapper ──────────────────────────────────────────────────
// wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
//   C[64,128] += A[64,32] × B[128,32]^T   (SS_TN: B in [N=128,K=32] layout)
// init=true  → zero-initialize accumulators (scale_D=0)
// init=false → accumulate into existing values (scale_D=1)
#define WGMMA_SS_TN_m64n128k32_f32_e4m3(acc, desc_a, desc_b, init)                 \
  do {                                                                               \
    asm volatile(                                                                    \
      "{\n"                                                                         \
      ".reg .pred p;\n"                                                             \
      "setp.ne.b32 p, %66, 0;\n"                                                   \
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                    \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"            \
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"            \
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"           \
      "%64, %65, p, 1, 1;\n"                                                        \
      "}\n"                                                                         \
      : "+f"((acc)[ 0]),"+f"((acc)[ 1]),"+f"((acc)[ 2]),"+f"((acc)[ 3]),          \
        "+f"((acc)[ 4]),"+f"((acc)[ 5]),"+f"((acc)[ 6]),"+f"((acc)[ 7]),          \
        "+f"((acc)[ 8]),"+f"((acc)[ 9]),"+f"((acc)[10]),"+f"((acc)[11]),           \
        "+f"((acc)[12]),"+f"((acc)[13]),"+f"((acc)[14]),"+f"((acc)[15]),           \
        "+f"((acc)[16]),"+f"((acc)[17]),"+f"((acc)[18]),"+f"((acc)[19]),           \
        "+f"((acc)[20]),"+f"((acc)[21]),"+f"((acc)[22]),"+f"((acc)[23]),           \
        "+f"((acc)[24]),"+f"((acc)[25]),"+f"((acc)[26]),"+f"((acc)[27]),           \
        "+f"((acc)[28]),"+f"((acc)[29]),"+f"((acc)[30]),"+f"((acc)[31]),           \
        "+f"((acc)[32]),"+f"((acc)[33]),"+f"((acc)[34]),"+f"((acc)[35]),           \
        "+f"((acc)[36]),"+f"((acc)[37]),"+f"((acc)[38]),"+f"((acc)[39]),           \
        "+f"((acc)[40]),"+f"((acc)[41]),"+f"((acc)[42]),"+f"((acc)[43]),           \
        "+f"((acc)[44]),"+f"((acc)[45]),"+f"((acc)[46]),"+f"((acc)[47]),           \
        "+f"((acc)[48]),"+f"((acc)[49]),"+f"((acc)[50]),"+f"((acc)[51]),           \
        "+f"((acc)[52]),"+f"((acc)[53]),"+f"((acc)[54]),"+f"((acc)[55]),           \
        "+f"((acc)[56]),"+f"((acc)[57]),"+f"((acc)[58]),"+f"((acc)[59]),           \
        "+f"((acc)[60]),"+f"((acc)[61]),"+f"((acc)[62]),"+f"((acc)[63])            \
      : "l"(desc_a), "l"(desc_b), "r"((int)(!(init))));                            \
  } while(0)

// SM90 (Hopper) wgmma barrier instructions.
// SM100 (Blackwell) uses UMMA instead — wgmma not available on sm_100a.
// We gate our wgmma path to SM90 only; SM100 uses the scalar fallback.
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 905
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n"); }
__device__ __forceinline__ void wgmma_wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;\n"); }
#endif

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

#if __CUDA_ARCH__ >= 1000
  // Allocate TMEM for 2 M=128×N=128 FP32 tiles (256 columns = 128 KB).
  // One warp allocates; __syncthreads distributes the result to all threads.
  __shared__ uint32_t tmem_base_sh;
  if (threadIdx.x < 32) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(&tmem_base_sh);
    asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(dst), "r"(256u));
  }
  __syncthreads();
  const uint32_t tmem_base = tmem_base_sh;
#endif

  // ── Single-pass KV loop: GEMM for S, scalar for dQ ───────────────────────
  // Per KV block:
  //   Phase A: SM90 wgmma / SM100 UMMA → sm.S_wg (FP16, [N_WG][WG_ROWS*BK])
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

    // ── Phase A: Q@K^T → sm.S_wg[N_WG][WG_ROWS*BK] (FP16) ───────────────────
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 905
    // SM90: wgmma m64n128k32, 2 WGs × 2 m64-tiles each
    for (int m = 0; m < WG_M_TILES; ++m) {
      const int wg_m_row = wg_row_base + m * WG_M;
      float S_acc[ACC_PER_TILE];
      for (int i = 0; i < ACC_PER_TILE; ++i) S_acc[i] = 0.f;
      wgmma_fence();
      for (int k = 0; k < WG_K_TILES; ++k) {
        uint64_t desc_Q = make_wgmma_desc(&sm.Q[wg_m_row * HD + k * WG_K], HD);
        uint64_t desc_K = make_wgmma_desc(&sm.K[buf][k * WG_K], HD);
        WGMMA_SS_TN_m64n128k32_f32_e4m3(S_acc, desc_Q, desc_K, k == 0);
      }
      wgmma_commit();
      wgmma_wait();
      for (int a = 0; a < ACC_PER_TILE; ++a) {
        int row_in_tile = (t_in_wg % 4) + (a / 4) * 4;
        int col = ((t_in_wg / 4) % 8) + (t_in_wg / 32) * 16
                + (a % 2) * 64 + ((a / 2) % 2) * 8;
        float s_val = S_acc[a] * (Q_scale_val * K_scale_val * attn_scale);
        sm.S_wg[wg_id][(m * WG_M + row_in_tile) * BK + col] = __float2half(s_val);
      }
    }

#elif __CUDA_ARCH__ >= 1000
    // SM100: tcgen05.mma.cta_group::1.kind::f8f6f4 (FP8→FP32 UMMA)
    // InstrDescriptor: E4M3 A, E4M3 B, F32 C, M=128, N=128, K-major A+B.
    //   c_format=F32(1)→bits[5:4]=0x10; n_dim=N/8=16→16<<17=0x200000;
    //   m_dim=M/16=8→8<<24=0x8000000  ⟹  desc=0x08200010
    static constexpr uint32_t UMMA_IDESC = 0x08200010u;
    // Issue 2 M=128 tiles over WG_K_TILES=4 K-tiles (K=32 each)
    for (int k = 0; k < WG_K_TILES; ++k) {
      uint64_t desc_Q0   = make_wgmma_desc(&sm.Q[0   * HD + k * WG_K], HD);
      uint64_t desc_Q128 = make_wgmma_desc(&sm.Q[128 * HD + k * WG_K], HD);
      uint64_t desc_K_   = make_wgmma_desc(&sm.K[buf][k * WG_K], HD);
      uint32_t accum = (k != 0) ? 1u : 0u;  // 0=init/clear C, 1=accumulate
      if (threadIdx.x == 0) {
        // Tile 0: TMEM rows 0-127 at tmem_base
        asm volatile(
          "{\n\t .reg .pred p;\n\t setp.ne.b32 p,%4,0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],%1,%2,%3,{0,0,0,0},p;\n\t}\n"
          :: "r"(tmem_base),"l"(desc_Q0),"l"(desc_K_),
             "r"(UMMA_IDESC),"r"(accum) : "memory");
        // Tile 1: TMEM rows 0-127 at tmem_base+128 (col offset 128)
        asm volatile(
          "{\n\t .reg .pred p;\n\t setp.ne.b32 p,%4,0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],%1,%2,%3,{0,0,0,0},p;\n\t}\n"
          :: "r"(tmem_base + 128u),"l"(desc_Q128),"l"(desc_K_),
             "r"(UMMA_IDESC),"r"(accum) : "memory");
      }
    }
    // CTA-wide commit + implicit barrier
    asm volatile("tcgen05.commit.cta_group::1.sync.aligned;\n" ::: "memory");
    // TMEM→shmem: warp W covers DPs [W*16, W*16+16), 4 rounds × 32 cols = 128 cols.
    // Thread mapping for tcgen05.ld.sync.aligned.16x256b.x4.b32:
    //   lane=T%32, t0=lane%4, t1=lane/4; register r=rep*4+f, rep∈[0,4), f∈[0,4)
    //   dp  = W*16 + t1 + (f>>1)*8    (→ S row in [0,128))
    //   col = t0*2 + (f&1) + rep*8 + ci*32  (→ S col in [0,128))
    {
      const int warp_id = threadIdx.x / 32;
      const int lane    = threadIdx.x % 32;
      const int t0 = lane % 4, t1 = lane / 4;
      const float qk_scale = Q_scale_val * K_scale_val * attn_scale;
      const uint32_t wdp = (uint32_t)(warp_id * 16) << 16;
      for (int ci = 0; ci < 4; ++ci) {
        const uint32_t coff = (uint32_t)(ci * 32);
        uint32_t s0[16], s1[16];
        asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x4.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];\n"
          : "=r"(s0[0]),"=r"(s0[1]),"=r"(s0[2]),"=r"(s0[3]),
            "=r"(s0[4]),"=r"(s0[5]),"=r"(s0[6]),"=r"(s0[7]),
            "=r"(s0[8]),"=r"(s0[9]),"=r"(s0[10]),"=r"(s0[11]),
            "=r"(s0[12]),"=r"(s0[13]),"=r"(s0[14]),"=r"(s0[15])
          : "r"(tmem_base + wdp + coff));
        asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x4.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];\n"
          : "=r"(s1[0]),"=r"(s1[1]),"=r"(s1[2]),"=r"(s1[3]),
            "=r"(s1[4]),"=r"(s1[5]),"=r"(s1[6]),"=r"(s1[7]),
            "=r"(s1[8]),"=r"(s1[9]),"=r"(s1[10]),"=r"(s1[11]),
            "=r"(s1[12]),"=r"(s1[13]),"=r"(s1[14]),"=r"(s1[15])
          : "r"(tmem_base + 128u + wdp + coff));
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        for (int rep = 0; rep < 4; ++rep) {
          for (int f = 0; f < 4; ++f) {
            const int r   = rep * 4 + f;
            const int dp  = warp_id * 16 + t1 + (f >> 1) * 8;
            const int col = t0 * 2 + (f & 1) + rep * 8 + ci * 32;
            sm.S_wg[0][dp * BK + col] = __float2half(__uint_as_float(s0[r]) * qk_scale);
            sm.S_wg[1][dp * BK + col] = __float2half(__uint_as_float(s1[r]) * qk_scale);
          }
        }
      }
    }

#else
    // Generic scalar fallback (for unsupported architectures; correctness only)
    for (int m = 0; m < WG_M_TILES; ++m) {
      const int wg_m_row = wg_row_base + m * WG_M;
      float S_acc[ACC_PER_TILE];
      for (int a = 0; a < ACC_PER_TILE; ++a) {
        int row_in_tile = (t_in_wg % 4) + (a / 4) * 4;
        int col = ((t_in_wg / 4) % 8) + (t_in_wg / 32) * 16
                + (a % 2) * 64 + ((a / 2) % 2) * 8;
        float s = 0.f;
        if (col < BK)
          for (int c = 0; c < HD; ++c)
            s += (float)sm.Q[(wg_m_row + row_in_tile) * HD + c]
               * (float)sm.K[buf][col * HD + c];
        S_acc[a] = s * attn_scale;
      }
      for (int a = 0; a < ACC_PER_TILE; ++a) {
        int row_in_tile = (t_in_wg % 4) + (a / 4) * 4;
        int col = ((t_in_wg / 4) % 8) + (t_in_wg / 32) * 16
                + (a % 2) * 64 + ((a / 2) % 2) * 8;
        float s_val = S_acc[a] * (Q_scale_val * K_scale_val * attn_scale);
        sm.S_wg[wg_id][(m * WG_M + row_in_tile) * BK + col] = __float2half(s_val);
      }
    }
#endif
    __syncthreads();  // ensure all S_wg writes visible before Phase B

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

#if __CUDA_ARCH__ >= 1000
  // Release TMEM allocation (warp 0)
  if (threadIdx.x < 32) {
    asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
      :: "r"(tmem_base), "r"(256u));
  }
#endif

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
