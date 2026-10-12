/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/normalization.h>

#include <algorithm>
#include <cstdint>

#include "../../cast/mxfp8/swizzle.cuh"
#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"

namespace transformer_engine {
namespace rmsnorm_bwd_fused {

// One CTA handles kRows consecutive rows and the kCluster CTAs of a thread block cluster cover one
// 32-row block, the MXFP8 columnwise scaling block. The CTA's rows of dz and x are loaded once by
// bulk copies into shared memory and stay there: sweep 1 reduces sum(dz * gamma * x) per row,
// sweep 2 writes dx (plus the residual gradient), rounded to the input type, over dz in shared
// memory and accumulates the per-column partials of dgamma (and of dbias and the columnwise amax)
// in registers, which the cluster reduces through distributed shared memory into one fp32 partial
// row per 32-row block. Sweep 3 stores dx, or quantizes it to MXFP8 with rowwise and columnwise
// scales (storing it as well when a high-precision copy is requested). The partial rows are
// reduced in the kernel by the last-arriving CTAs of a three-level tree (ticket counters in global
// memory), every level a fixed-order sum, so the result is deterministic and no separate reduction
// kernel is needed.
constexpr int kThreads = 384;  // one thread per 8 columns
constexpr int kWarps = kThreads / 32;
constexpr int kRows = 8;
constexpr int kCluster = 32 / kRows;
constexpr int kMaxCols = kThreads * 8;
constexpr int kMaxGroups = 64;  // level-1 groups of 32-row blocks
constexpr int kFan2 = 8;        // level-1 groups per level-2 group
constexpr int kMaxGroups2 = kMaxGroups / kFan2;
constexpr int kTicketSlots = 1 + kMaxGroups + kMaxGroups2;
constexpr int kMinGroupSize = 16;
constexpr int kMiscBytes = 16 /*mbarrier*/ + kWarps * kRows * 4 /*row sums*/ + kRows * 4 /*k*/ +
                           kRows * 4 /*rsigma*/ + 16 /*flag*/;

struct Params {
  const void *dz;       // [M, N]
  const void *x;        // [M, N]
  const void *add;      // [M, N] or nullptr
  const float *rsigma;  // [M]
  const void *gamma;    // [N]
  void *dx;             // [M, N] in the input type; with MXFP8 output the optional copy
  uint8_t *rowwise_data;
  uint8_t *columnwise_data;
  e8m0_t *rowwise_scales;
  e8m0_t *columnwise_scales;
  float *part;       // [nb32, pstride] column partials (dgamma | dbias) of each 32-row block
  float *part2;      // [ngroups, pstride] level-1 group partials
  float *part3;      // [ceil(ngroups / kFan2), pstride] level-2 group partials
  int32_t *tickets;  // [kTicketSlots], zero at rest
  void *dgamma;      // [N]
  void *dbias;       // [N] or nullptr
  int M, N, pstride, nb32, group, ngroups;
  int rowwise_scale_stride;     // columns of the rowwise scale matrix
  int columnwise_scale_stride;  // columns of the columnwise scale matrix
  int columnwise_scale_rows;    // rows of the columnwise scale matrix
  float fp8_max_rcp;
  bool zero_centered_gamma;
  bool e5m2;
  bool swizzled_scales;
  bool grads_fp32;  // dgamma and dbias in fp32 instead of the input type
};

// Shared memory: [dz slab, kRows x N][x region][misc]. The x region holds the x slab and the
// cluster inboxes the peers push into: the columnwise amax [kCluster][N] (MXFP8) and the dgamma
// (and dbias) slices [kCluster][1 or 2][N / kCluster] fp32. With MXFP8 output the inboxes alias
// the x slab, which is dead once sweep 2 is done; otherwise they follow it.
__host__ __device__ constexpr int inbox_bytes(int N, bool mx, bool dbias) {
  return (mx ? kCluster * N * 2 : 0) + kCluster * (dbias ? 2 : 1) * (N / kCluster) * 4;
}
__host__ __device__ constexpr int x_region_bytes(int N, bool mx, bool dbias) {
  const int slab = kRows * N * 2;
  const int inbox = inbox_bytes(N, mx, dbias);
  return mx ? (slab > inbox ? slab : inbox) : slab + inbox;
}
__host__ __device__ constexpr size_t smem_bytes(int N, bool mx, bool dbias) {
  return static_cast<size_t>(kRows * N * 2) + x_region_bytes(N, mx, dbias) + kMiscBytes;
}

template <typename T>
struct Pack2;

template <>
struct Pack2<bf16> {
  static __device__ __forceinline__ float elem(uint32_t w, int i) {
    return __uint_as_float(i ? (w & 0xFFFF0000u) : (w << 16));
  }
  static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
  }
  static __device__ __forceinline__ float from_bits(uint32_t h) { return __uint_as_float(h << 16); }
  // Exact for a value that is already a bf16
  static __device__ __forceinline__ uint32_t to_bits(float f) { return __float_as_uint(f) >> 16; }
};

template <>
struct Pack2<fp16> {
  static __device__ __forceinline__ float elem(uint32_t w, int i) {
    return __half2float(
        __ushort_as_half(static_cast<unsigned short>(i ? (w >> 16) : (w & 0xFFFFu))));
  }
  static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
  }
  static __device__ __forceinline__ float from_bits(uint32_t h) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
  }
  static __device__ __forceinline__ uint32_t to_bits(float f) {
    return __half_as_ushort(__float2half_rn(f));
  }
};

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Global -> shared bulk copy of `bytes` (a multiple of 16), completing on the mbarrier.
__device__ __forceinline__ void bulk_g2s(void *dst, const void *src, uint32_t bytes,
                                         uint64_t *mbar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(mbar))
      : "memory");
}

__device__ __forceinline__ void prefetch_l2(const void *ptr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(__cvta_generic_to_global(ptr)));
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Address of the same shared memory location in the CTA of the given cluster rank.
__device__ __forceinline__ uint32_t mapa(uint32_t smem_addr, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_cluster(uint32_t addr, float4 v) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(v.x), "f"(v.y),
               "f"(v.z), "f"(v.w)
               : "memory");
}
__device__ __forceinline__ void st_cluster(uint32_t addr, uint4 v) {
  asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}

// Two floats to two FP8 bytes (round to nearest, saturate), element 0 in the low byte.
__device__ __forceinline__ uint16_t cvt_fp8x2(float lo, float hi, bool e5m2) {
  uint16_t r;
  if (e5m2) {
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  } else {
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  }
  return r;
}

__device__ __forceinline__ size_t scale_idx(int i, int j, int stride, int num_tiles_x,
                                            bool swizzled) {
  using dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  return swizzled ? gemm_swizzled_scale_idx(i, j, num_tiles_x)
                  : static_cast<size_t>(i) * stride + j;
}

__device__ __forceinline__ float4 ld_cg(const float *p) {
  return __ldcg(reinterpret_cast<const float4 *>(p));
}

// Sum of `nrows` fp32 rows of stride `stride`, in row order, written by the whole CTA.
__device__ __forceinline__ void sum_rows(const float *src, int nrows, int stride, float *dst) {
  for (int i = threadIdx.x; i < (stride >> 2); i += kThreads) {
    const int c = i * 4;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int j = 0;
    for (; j + 4 <= nrows; j += 4) {
      const float4 v0 = ld_cg(src + static_cast<size_t>(j) * stride + c);
      const float4 v1 = ld_cg(src + static_cast<size_t>(j + 1) * stride + c);
      const float4 v2 = ld_cg(src + static_cast<size_t>(j + 2) * stride + c);
      const float4 v3 = ld_cg(src + static_cast<size_t>(j + 3) * stride + c);
      s.x += v0.x, s.y += v0.y, s.z += v0.z, s.w += v0.w;
      s.x += v1.x, s.y += v1.y, s.z += v1.z, s.w += v1.w;
      s.x += v2.x, s.y += v2.y, s.z += v2.z, s.w += v2.w;
      s.x += v3.x, s.y += v3.y, s.z += v3.z, s.w += v3.w;
    }
    for (; j < nrows; ++j) {
      const float4 v = ld_cg(src + static_cast<size_t>(j) * stride + c);
      s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
    }
    *reinterpret_cast<float4 *>(dst + c) = s;
  }
}

// The CTA takes a ticket after fencing its stores; true in every thread when it was the last of
// `count` arrivals, which also re-arms the counter.
__device__ __forceinline__ bool last_arrival(int32_t *counter, int count, int32_t *s_flag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const bool last = atomicAdd(counter, 1) == count - 1;
    if (last) {
      *counter = 0;
      __threadfence();
    }
    *s_flag = last;
  }
  __syncthreads();
  return *s_flag != 0;
}

template <typename IType>
__device__ __forceinline__ void store_grad(void *ptr, int i, float v, bool fp32) {
  if (fp32) {
    reinterpret_cast<float *>(ptr)[i] = v;
  } else {
    reinterpret_cast<IType *>(ptr)[i] = static_cast<IType>(v);
  }
}
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

template <typename IType, bool HAS_ADD, bool MX, bool HAS_DBIAS>
__global__ void __launch_bounds__(kThreads, 2) rmsnorm_bwd_fused_kernel(const Params p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  using P = Pack2<IType>;
  constexpr int NARR = HAS_DBIAS ? 2 : 1;  // dgamma (| dbias) slices in the inbox
  extern __shared__ __align__(128) uint8_t smem[];

  const int N = p.N;
  const int NV = N >> 3;
  const int cslice = N / kCluster;  // columns of the partial row each CTA of the cluster reduces
  const int slab = kRows * N * 2;
  uint16_t *s_dz = reinterpret_cast<uint16_t *>(smem);
  uint16_t *s_x = reinterpret_cast<uint16_t *>(smem + slab);
  uint16_t *s_amax_in = s_x;  // [kCluster][N]
  float *s_grad_in =          // [kCluster][NARR][cslice]
      reinterpret_cast<float *>(smem + slab + (MX ? kCluster * N * 2 : slab));
  uint8_t *misc = smem + slab + x_region_bytes(N, MX, HAS_DBIAS);
  uint64_t *mbar = reinterpret_cast<uint64_t *>(misc);
  float *s_red = reinterpret_cast<float *>(misc + 16);
  float *s_k = s_red + kWarps * kRows;
  float *s_rsigma = s_k + kRows;
  int32_t *s_flag = reinterpret_cast<int32_t *>(s_rsigma + kRows);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int blk32 = blockIdx.x / kCluster;
  const int row0 = blockIdx.x * kRows;
  const int rank = static_cast<int>(cluster_ctarank());
  const bool owner = tid < NV;  // of columns c0 .. c0 + 7
  const int c0 = tid * 8;
  const int PS = p.pstride;
  const IType *dz_g = reinterpret_cast<const IType *>(p.dz);
  const IType *x_g = reinterpret_cast<const IType *>(p.x);

  // Bulk loads of the dz and x slabs; the residual gradient and gamma are loaded meanwhile.
  if (tid == 0) {
    ptx::mbarrier_init(mbar, 1);
    fence_mbarrier_init_cluster();
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (!MX) cluster_arrive();  // the dedicated inbox may be pushed to once every peer has started
  if (tid == 0) {
    ptx::mbarrier_arrive_expect_tx(mbar, static_cast<uint32_t>(2 * slab));
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      bulk_g2s(s_dz + r * N, dz_g + static_cast<size_t>(row0 + r) * N, N * 2, mbar);
      bulk_g2s(s_x + r * N, x_g + static_cast<size_t>(row0 + r) * N, N * 2, mbar);
    }
  }
  const IType *add_g = reinterpret_cast<const IType *>(p.add) + static_cast<size_t>(row0) * N + c0;
  if (HAS_ADD && owner) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) prefetch_l2(add_g + r * N);
  }
  if (tid < kRows) s_rsigma[tid] = p.rsigma[row0 + tid];
  float g[8];
  {
    uint4 gv = make_uint4(0u, 0u, 0u, 0u);
    if (owner) {
      gv = __ldg(reinterpret_cast<const uint4 *>(reinterpret_cast<const IType *>(p.gamma) + c0));
    }
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    const float shift = p.zero_centered_gamma ? 1.0f : 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) g[c] = P::elem(gw[c >> 1], c & 1) + shift;
  }
  __syncthreads();
  ptx::mbarrier_wait_parity(mbar, 0);

  // Sweep 1: k = rsigma^3 / N * sum(dz * gamma * x) per row
  {
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
    if (owner) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
        const uint4 b = *reinterpret_cast<const uint4 *>(s_x + r * N + c0);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c)
          sum += P::elem(aw[c >> 1], c & 1) * g[c] * P::elem(bw[c >> 1], c & 1);
        acc[r] = sum;
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) s_red[warp * kRows + r] = acc[r];
    }
    __syncthreads();
    if (tid < kRows) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) sum += s_red[w * kRows + tid];
      const float rs = s_rsigma[tid];
      s_k[tid] = sum * rs * rs * rs / static_cast<float>(N);
    }
    __syncthreads();
  }

  // Sweep 2: dx = rsigma * dz * gamma - x * k (+ add), rounded, over dz in shared memory; the
  // per-column dgamma partial sum(dz * x * rsigma) of the CTA's rows in registers
  float dg[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) dg[c] = 0.0f;
  if (owner) {
    uint4 add_v[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      add_v[r] = HAS_ADD ? __ldg(reinterpret_cast<const uint4 *>(add_g + r * N))
                         : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint4 a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
      const uint4 b = *reinterpret_cast<const uint4 *>(s_x + r * N + c0);
      const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
      const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
      const uint32_t dw[4] = {add_v[r].x, add_v[r].y, add_v[r].z, add_v[r].w};
      const float rs = s_rsigma[r];
      const float k = s_k[r];
      uint32_t ow[4];
#pragma unroll
      for (int c = 0; c < 8; c += 2) {
        float v2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float dz = P::elem(aw[c >> 1], e);
          const float x = P::elem(bw[c >> 1], e);
          float v = rs * dz * g[c + e] - x * k;
          if (HAS_ADD) v += P::elem(dw[c >> 1], e);
          v2[e] = v;
          dg[c + e] += dz * x * rs;
        }
        ow[c >> 1] = P::pack(v2[0], v2[1]);
      }
      *reinterpret_cast<uint4 *>(s_dz + r * N + c0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  }
  __syncthreads();           // the x slab is dead
  if (MX) cluster_arrive();  // so the peers may push into it

  // Per-column dbias and amax of the rounded dx over the CTA's rows
  float db[8], cm[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) db[c] = cm[c] = 0.0f;
  if ((MX || HAS_DBIAS) && owner) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint4 a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
      const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float v = P::elem(aw[c >> 1], c & 1);
        if (HAS_DBIAS) db[c] += v;
        if (MX) cm[c] = fmaxf(cm[c], fabsf(v));
      }
    }
  }

  // Push the partials to the peer that reduces these columns (and the amax to every peer)
  cluster_wait();
  if (owner) {
    const int q = c0 / cslice;
    const int cc = c0 - q * cslice;
    const uint32_t dst = mapa(smem_u32(s_grad_in + (rank * NARR) * cslice + cc), q);
    st_cluster(dst, make_float4(dg[0], dg[1], dg[2], dg[3]));
    st_cluster(dst + 16u, make_float4(dg[4], dg[5], dg[6], dg[7]));
    if (HAS_DBIAS) {
      const uint32_t dstb = mapa(smem_u32(s_grad_in + (rank * NARR + 1) * cslice + cc), q);
      st_cluster(dstb, make_float4(db[0], db[1], db[2], db[3]));
      st_cluster(dstb + 16u, make_float4(db[4], db[5], db[6], db[7]));
    }
    if (MX) {
      uint32_t cmw[4];
#pragma unroll
      for (int c = 0; c < 8; c += 2)
        cmw[c >> 1] = P::to_bits(cm[c]) | (P::to_bits(cm[c + 1]) << 16);
      const uint32_t local = smem_u32(s_amax_in + rank * N + c0);
#pragma unroll
      for (int peer = 0; peer < kCluster; ++peer) {
        st_cluster(mapa(local, peer), make_uint4(cmw[0], cmw[1], cmw[2], cmw[3]));
      }
    }
  }
  cluster_arrive();
  cluster_wait();  // every push has landed; only local shared memory is read from here on

  // The 32-row block's partial row, this CTA's column slice
  {
    float *part_row = p.part + static_cast<size_t>(blk32) * PS;
    for (int i = tid; i < (NARR * cslice) >> 2; i += kThreads) {
      const int arr = (i * 4) / cslice;
      const int cc = i * 4 - arr * cslice;
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int q = 0; q < kCluster; ++q) {
        const float4 v =
            *reinterpret_cast<const float4 *>(s_grad_in + (q * NARR + arr) * cslice + cc);
        sum.x += v.x, sum.y += v.y, sum.z += v.z, sum.w += v.w;
      }
      *reinterpret_cast<float4 *>(part_row + arr * N + rank * cslice + cc) = sum;
    }
  }
  __syncthreads();
  // Level-1 ticket, taken now so that its latency overlaps sweep 3
  const int g1 = blk32 / p.group;
  const int gsize1 = min(p.group, p.nb32 - g1 * p.group);
  if (tid == 0) {
    __threadfence();
    const bool last = atomicAdd(p.tickets + 1 + g1, 1) == gsize1 * kCluster - 1;
    if (last) p.tickets[1 + g1] = 0;
    *s_flag = last;
  }

  // Sweep 3: dx to global memory, or its MXFP8 rowwise and columnwise quantization
  float col_rcp[8];
  if (MX && owner) {
    e8m0_t ec[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t bits = 0u;  // the magnitudes compare as their bits
#pragma unroll
      for (int q = 0; q < kCluster; ++q)
        bits = max(bits, static_cast<uint32_t>(s_amax_in[q * N + c0 + c]));
      ec[c] = ptx::float_to_e8m0(P::from_bits(bits) * p.fp8_max_rcp);
      col_rcp[c] = ptx::exp2f_rcp<float>(ec[c]);
    }
    if (c0 / cslice == rank) {  // one writer per column scale
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        p.columnwise_scales[scale_idx(c0 + c, blk32, p.columnwise_scale_stride,
                                      p.columnwise_scale_rows / 4, p.swizzled_scales)] = ec[c];
      }
    }
  }
  if (MX && rank == 0) {  // zero the padding columns of the block's scales
    for (int i = N + tid; i < p.columnwise_scale_stride; i += kThreads) {
      p.columnwise_scales[scale_idx(i, blk32, p.columnwise_scale_stride,
                                    p.columnwise_scale_rows / 4, p.swizzled_scales)] = 0;
    }
  }
  const int npad_r = p.rowwise_scale_stride - (N >> 5);
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int row = row0 + r;
    uint4 a = make_uint4(0u, 0u, 0u, 0u);
    if (owner) a = *reinterpret_cast<const uint4 *>(s_dz + r * N + c0);
    if (!MX || p.dx != nullptr) {
      if (owner) {
        *reinterpret_cast<uint4 *>(reinterpret_cast<IType *>(p.dx) + static_cast<size_t>(row) * N +
                                   c0) = a;
      }
    }
    if (MX) {
      const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
      float v[8];
      float amax = 0.0f;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        v[c] = P::elem(aw[c >> 1], c & 1);
        amax = fmaxf(amax, fabsf(v[c]));
      }
      // A 32-column rowwise block is 4 consecutive lanes
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      const e8m0_t er = ptx::float_to_e8m0(amax * p.fp8_max_rcp);
      const float row_rcp = ptx::exp2f_rcp<float>(er);
      if (owner) {
        uint16_t qr[4], qc[4];
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
          qr[c >> 1] = cvt_fp8x2(v[c] * row_rcp, v[c + 1] * row_rcp, p.e5m2);
          qc[c >> 1] = cvt_fp8x2(v[c] * col_rcp[c], v[c + 1] * col_rcp[c + 1], p.e5m2);
        }
        const size_t off = static_cast<size_t>(row) * N + c0;
        *reinterpret_cast<uint2 *>(p.rowwise_data + off) =
            make_uint2(qr[0] | (static_cast<uint32_t>(qr[1]) << 16),
                       qr[2] | (static_cast<uint32_t>(qr[3]) << 16));
        *reinterpret_cast<uint2 *>(p.columnwise_data + off) =
            make_uint2(qc[0] | (static_cast<uint32_t>(qc[1]) << 16),
                       qc[2] | (static_cast<uint32_t>(qc[3]) << 16));
        if ((tid & 3) == 0) {
          p.rowwise_scales[scale_idx(row, tid >> 2, p.rowwise_scale_stride,
                                     p.rowwise_scale_stride / 4, p.swizzled_scales)] = er;
        }
      }
      if (tid < npad_r) {  // zero the padding blocks of the row's scales
        p.rowwise_scales[scale_idx(row, (N >> 5) + tid, p.rowwise_scale_stride,
                                   p.rowwise_scale_stride / 4, p.swizzled_scales)] = 0;
      }
    }
  }

  // Three-level reduction of the partial rows by the last-arriving CTA of each group
  __syncthreads();
  if (*s_flag == 0) return;
  __threadfence();
  sum_rows(p.part + static_cast<size_t>(g1) * p.group * PS, gsize1, PS,
           p.part2 + static_cast<size_t>(g1) * PS);
  const int g2 = g1 / kFan2;
  const int gsize2 = min(kFan2, p.ngroups - g2 * kFan2);
  if (!last_arrival(p.tickets + 1 + kMaxGroups + g2, gsize2, s_flag)) return;
  sum_rows(p.part2 + static_cast<size_t>(g2) * kFan2 * PS, gsize2, PS,
           p.part3 + static_cast<size_t>(g2) * PS);
  const int ngroups2 = (p.ngroups + kFan2 - 1) / kFan2;
  if (!last_arrival(p.tickets, ngroups2, s_flag)) return;
  for (int i = tid; i < (PS >> 2); i += kThreads) {
    const int c = i * 4;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int gg = 0; gg < ngroups2; ++gg) {
      const float4 v = ld_cg(p.part3 + static_cast<size_t>(gg) * PS + c);
      sum.x += v.x, sum.y += v.y, sum.z += v.z, sum.w += v.w;
    }
    const float sv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int cc = c + e;
      if (cc < N) {
        store_grad<IType>(p.dgamma, cc, sv[e], p.grads_fp32);
      } else {
        store_grad<IType>(p.dbias, cc - N, sv[e], p.grads_fp32);
      }
    }
  }
#else
  NVTE_DEVICE_ERROR("rmsnorm_bwd_fused is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <typename IType, bool HAS_ADD, bool MX, bool HAS_DBIAS>
void launch(const Params &p, cudaStream_t stream) {
  auto kernel = rmsnorm_bwd_fused_kernel<IType, HAS_ADD, MX, HAS_DBIAS>;
  const size_t smem = smem_bytes(p.N, MX, HAS_DBIAS);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(p.M / kRows);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kCluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  config.attrs = attr;
  config.numAttrs = 1;
  NVTE_CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, p));
}

bool is_aligned(const void *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; }

}  // namespace rmsnorm_bwd_fused
}  // namespace transformer_engine

void nvte_rmsnorm_bwd_fused(const NVTETensor dz, const NVTETensor x, const NVTETensor rsigma,
                            const NVTETensor gamma, const NVTETensor add, NVTETensor dx,
                            NVTETensor dx_high_precision, NVTETensor dgamma, NVTETensor dbias,
                            NVTETensor workspace, NVTETensor counter,
                            const bool zero_centered_gamma, cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_bwd_fused);
  using namespace transformer_engine;
  namespace rb = transformer_engine::rmsnorm_bwd_fused;

  const Tensor &dz_cpp = *convertNVTETensorCheck(dz);
  const Tensor &x_cpp = *convertNVTETensorCheck(x);
  const Tensor &rsigma_cpp = *convertNVTETensorCheck(rsigma);
  const Tensor &gamma_cpp = *convertNVTETensorCheck(gamma);
  const Tensor *add_cpp = add != nullptr ? convertNVTETensor(add) : nullptr;
  const bool has_add = add_cpp != nullptr && add_cpp->has_data();
  Tensor *dx_cpp = convertNVTETensorCheck(dx);
  Tensor *dx_hp_cpp = dx_high_precision != nullptr ? convertNVTETensor(dx_high_precision) : nullptr;
  const bool has_dx_hp = dx_hp_cpp != nullptr && dx_hp_cpp->has_data();
  Tensor *dgamma_cpp = convertNVTETensorCheck(dgamma);
  Tensor *dbias_cpp = dbias != nullptr ? convertNVTETensor(dbias) : nullptr;
  const bool has_dbias = dbias_cpp != nullptr && dbias_cpp->has_data();
  Tensor *workspace_cpp = convertNVTETensorCheck(workspace);

  NVTE_CHECK(cuda::sm_arch() >= 90, "rmsnorm_bwd_fused requires compute capability 9.0+.");
  NVTE_CHECK(!x_cpp.data.shape.empty(), "x must have at least one dimension.");
  const auto [rows, cols] = x_cpp.flat_2d_dims();
  const DType itype = x_cpp.dtype();
  NVTE_CHECK(itype == DType::kBFloat16 || itype == DType::kFloat16,
             "x must be a BF16 or FP16 tensor, but got dtype=", to_string(itype), ".");
  NVTE_CHECK(rows > 0 && rows % 32 == 0 && cols > 0 && cols % 32 == 0 && cols <= rb::kMaxCols,
             "x must have a multiple of 32 rows and a multiple of 32 columns up to ", rb::kMaxCols,
             ", but got shape=", x_cpp.shape(), ".");
  NVTE_CHECK(dz_cpp.data.shape == x_cpp.data.shape && dz_cpp.dtype() == itype,
             "dz must have the shape and dtype of x.");
  NVTE_CHECK(rsigma_cpp.dtype() == DType::kFloat32 && rsigma_cpp.numel() == rows,
             "rsigma must be an FP32 tensor of ", rows, " elements.");
  NVTE_CHECK(gamma_cpp.dtype() == itype && gamma_cpp.numel() == cols,
             "gamma must have the dtype of x and ", cols, " elements.");
  if (has_add) {
    NVTE_CHECK(add_cpp->data.shape == x_cpp.data.shape && add_cpp->dtype() == itype,
               "add must have the shape and dtype of x.");
  }
  const bool mx = is_mxfp8_scaling(dx_cpp->scaling_mode);
  if (mx) {
    NVTE_CHECK(dx_cpp->dtype() == DType::kFloat8E4M3 || dx_cpp->dtype() == DType::kFloat8E5M2,
               "An MXFP8 dx must be E4M3 or E5M2.");
    NVTE_CHECK(dx_cpp->has_data() && dx_cpp->has_columnwise_data() &&
                   dx_cpp->data.numel() == rows * cols &&
                   dx_cpp->columnwise_data.numel() == rows * cols,
               "An MXFP8 dx must have rowwise and columnwise data of ", rows * cols, " elements.");
    NVTE_CHECK(dx_cpp->scale_inv.shape.size() == 2 && dx_cpp->scale_inv.shape[0] >= rows &&
                   dx_cpp->scale_inv.shape[1] >= cols / 32 && dx_cpp->scale_inv.shape[1] % 4 == 0,
               "The rowwise scales of dx must be a 2D tensor of at least ", rows, " x ", cols / 32,
               " elements with a multiple of 4 columns, but got shape=", dx_cpp->scale_inv.shape,
               ".");
    NVTE_CHECK(dx_cpp->columnwise_scale_inv.shape.size() == 2 &&
                   dx_cpp->columnwise_scale_inv.shape[0] >= rows / 32 &&
                   dx_cpp->columnwise_scale_inv.shape[0] % 4 == 0 &&
                   dx_cpp->columnwise_scale_inv.shape[1] >= cols,
               "The columnwise scales of dx must be a 2D tensor of at least ", rows / 32, " x ",
               cols, " elements with a multiple of 4 rows, but got shape=",
               dx_cpp->columnwise_scale_inv.shape, ".");
    NVTE_CHECK(rb::is_aligned(dx_cpp->data.dptr) && rb::is_aligned(dx_cpp->columnwise_data.dptr),
               "The MXFP8 data of dx must be 16-byte aligned.");
    if (has_dx_hp) {
      NVTE_CHECK(dx_hp_cpp->flat_2d_dims() == x_cpp.flat_2d_dims() && dx_hp_cpp->dtype() == itype,
                 "dx_high_precision must have the shape and dtype of x.");
      NVTE_CHECK(rb::is_aligned(dx_hp_cpp->data.dptr),
                 "dx_high_precision must be 16-byte aligned.");
    }
  } else {
    NVTE_CHECK(dx_cpp->flat_2d_dims() == x_cpp.flat_2d_dims() && dx_cpp->dtype() == itype,
               "dx must have the shape and dtype of x.");
    NVTE_CHECK(rb::is_aligned(dx_cpp->data.dptr), "dx must be 16-byte aligned.");
    NVTE_CHECK(!has_dx_hp, "dx_high_precision applies to an MXFP8 dx.");
  }
  const DType gtype = dgamma_cpp->dtype();
  NVTE_CHECK((gtype == itype || gtype == DType::kFloat32) && dgamma_cpp->numel() == cols,
             "dgamma must have the dtype of gamma or FP32 and ", cols, " elements.");
  if (has_dbias) {
    NVTE_CHECK(dbias_cpp->dtype() == gtype && dbias_cpp->numel() == cols,
               "dbias must have the dtype of dgamma and ", cols, " elements.");
  }
  NVTE_CHECK(rb::is_aligned(dz_cpp.data.dptr) && rb::is_aligned(x_cpp.data.dptr) &&
                 rb::is_aligned(gamma_cpp.data.dptr) &&
                 (!has_add || rb::is_aligned(add_cpp->data.dptr)),
             "dz, x, gamma and add must be 16-byte aligned.");

  const int nb32 = static_cast<int>(rows / 32);
  const int group = std::max(rb::kMinGroupSize, (nb32 + rb::kMaxGroups - 1) / rb::kMaxGroups);
  const int ngroups = (nb32 + group - 1) / group;
  const int ngroups2 = (ngroups + rb::kFan2 - 1) / rb::kFan2;
  const int pstride = static_cast<int>(cols) * (has_dbias ? 2 : 1);
  const size_t workspace_bytes = static_cast<size_t>(nb32 + ngroups + ngroups2) * pstride * 4;
  if (workspace_cpp->data.numel() == 0) {
    workspace_cpp->data.shape = {workspace_bytes};
    workspace_cpp->data.dtype = DType::kByte;
    return;
  }
  NVTE_CHECK(
      workspace_cpp->data.dtype == DType::kByte && workspace_cpp->data.numel() >= workspace_bytes &&
          rb::is_aligned(workspace_cpp->data.dptr),
      "workspace must be a 16-byte aligned byte tensor of at least ", workspace_bytes, " bytes.");
  const Tensor &counter_cpp = *convertNVTETensorCheck(counter);
  NVTE_CHECK(counter_cpp.dtype() == DType::kInt32 && counter_cpp.numel() >= rb::kTicketSlots,
             "counter must be an int32 tensor of at least ", rb::kTicketSlots, " elements.");

  rb::Params p{};
  p.dz = dz_cpp.data.dptr;
  p.x = x_cpp.data.dptr;
  p.add = has_add ? add_cpp->data.dptr : nullptr;
  p.rsigma = reinterpret_cast<const float *>(rsigma_cpp.data.dptr);
  p.gamma = gamma_cpp.data.dptr;
  if (mx) {
    p.rowwise_data = reinterpret_cast<uint8_t *>(dx_cpp->data.dptr);
    p.columnwise_data = reinterpret_cast<uint8_t *>(dx_cpp->columnwise_data.dptr);
    p.rowwise_scales = reinterpret_cast<e8m0_t *>(dx_cpp->scale_inv.dptr);
    p.columnwise_scales = reinterpret_cast<e8m0_t *>(dx_cpp->columnwise_scale_inv.dptr);
    p.rowwise_scale_stride = static_cast<int>(dx_cpp->scale_inv.shape[1]);
    p.columnwise_scale_rows = static_cast<int>(dx_cpp->columnwise_scale_inv.shape[0]);
    p.columnwise_scale_stride = static_cast<int>(dx_cpp->columnwise_scale_inv.shape[1]);
    p.e5m2 = dx_cpp->dtype() == DType::kFloat8E5M2;
    p.fp8_max_rcp =
        p.e5m2 ? Quantized_Limits<fp8e5m2>::max_norm_rcp : Quantized_Limits<fp8e4m3>::max_norm_rcp;
    p.swizzled_scales = dx_cpp->with_gemm_swizzled_scales;
    p.dx = has_dx_hp ? dx_hp_cpp->data.dptr : nullptr;
  } else {
    p.dx = dx_cpp->data.dptr;
  }
  float *ws = reinterpret_cast<float *>(workspace_cpp->data.dptr);
  p.part = ws;
  p.part2 = p.part + static_cast<size_t>(nb32) * pstride;
  p.part3 = p.part2 + static_cast<size_t>(ngroups) * pstride;
  p.tickets = reinterpret_cast<int32_t *>(counter_cpp.data.dptr);
  p.dgamma = dgamma_cpp->data.dptr;
  p.dbias = has_dbias ? dbias_cpp->data.dptr : nullptr;
  p.M = static_cast<int>(rows);
  p.N = static_cast<int>(cols);
  p.pstride = pstride;
  p.nb32 = nb32;
  p.group = group;
  p.ngroups = ngroups;
  p.zero_centered_gamma = zero_centered_gamma;
  p.grads_fp32 = gtype == DType::kFloat32;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      itype, IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          has_add, HAS_ADD,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              mx, MX,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  has_dbias, HAS_DBIAS, rb::launch<IType, HAS_ADD, MX, HAS_DBIAS>(p, stream);))));
}
