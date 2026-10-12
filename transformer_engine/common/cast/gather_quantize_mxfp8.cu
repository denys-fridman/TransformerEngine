/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <algorithm>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/ptx_arch_spec.cuh"
#include "../utils.cuh"
#include "mxfp8/swizzle.cuh"

namespace transformer_engine {
namespace gather_quantize_mxfp8 {

// A tile is 32 gathered rows x 128 columns: one columnwise MXFP8 block along the rows and one
// 128-column swizzled-scale tile along the columns. Four tiles stacked along the rows form a unit
// (128 rows x 128 columns), whose scales fill one 512-byte swizzled scale tile in each orientation.
// The rows of a tile are staged in shared memory by 16-byte cp.async copies (zero-filled for
// padding rows) through a kStages-deep ring, and persistent CTAs take units from a global counter
// so the work drains over whatever SMs are free.
constexpr int kTileRows = 32;
constexpr int kTileCols = 128;
constexpr int kTilesPerUnit = 4;
constexpr int kUnitRows = kTileRows * kTilesPerUnit;
constexpr int kThreads = 128;  // 16 column chunks of 8 elements x 8 groups of 4 rows
constexpr int kRowBytes = kTileCols * 2;
constexpr int kTileBytes = kTileRows * kRowBytes;
constexpr int kStages = 3;
constexpr int kMinBlocksPerSM = 7;
constexpr int kScaleTileBytes = 512;

// Dynamic shared memory: the tile ring, the barriers, then the per-tile exchange buffers and the
// group table (num_groups sizes and starts), see smem_bytes().
constexpr int kBarsOffset = kStages * kTileBytes;
constexpr int kXchgOffset = kBarsOffset + 128;  // per-warp columnwise amax partials, 16-bit
constexpr int kColScaleOffset = kXchgOffset + 4 * kTileCols * 2;  // columnwise scale_inv, fp32
constexpr int kRowwiseScalesOffset = kColScaleOffset + kTileCols * 4;
constexpr int kColumnwiseScalesOffset = kRowwiseScalesOffset + kScaleTileBytes;
constexpr int kMiscOffset = kColumnwiseScalesOffset + kScaleTileBytes;
constexpr int kGroupsOffset = kMiscOffset + 128;

constexpr size_t smem_bytes(int num_groups) {
  return kGroupsOffset + 2 * sizeof(int32_t) * num_groups;
}

struct Params {
  const void *src;              // [num_tokens, hidden] BF16/FP16
  const int32_t *row_to_token;  // [num_rows], negative = padding row
  const int64_t *split_sizes;   // [num_groups], multiples of kUnitRows
  uint8_t *rowwise_data;        // [num_rows, hidden] E4M3
  e8m0_t *rowwise_scales;       // swizzled, one tile grid over (num_rows, hidden_padded)
  uint8_t *columnwise_data;     // [num_rows, hidden] E4M3
  e8m0_t *columnwise_scales;    // swizzled, one tile grid per group
  void *high_precision;         // [num_rows, hidden] gathered rows, or nullptr
  int32_t *counter;             // {next unit, finished CTAs}, zero before and after the launch
  int num_tokens;
  int hidden;
  int num_rows;
  int num_groups;
  int num_col_tiles;  // hidden_padded / kTileCols
  int hidden_padded;  // hidden rounded up to kTileCols
};

template <typename IType>
struct Packed2;

template <>
struct Packed2<bf16> {
  static __device__ __forceinline__ uint32_t abs(uint32_t a) {
    const auto r = __habs2(*reinterpret_cast<const __nv_bfloat162 *>(&a));
    return *reinterpret_cast<const uint32_t *>(&r);
  }
  static __device__ __forceinline__ uint32_t max(uint32_t a, uint32_t b) {
    const auto r = __hmax2(*reinterpret_cast<const __nv_bfloat162 *>(&a),
                           *reinterpret_cast<const __nv_bfloat162 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);
  }
  static __device__ __forceinline__ float2 to_float2(uint32_t a) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&a));
  }
};

template <>
struct Packed2<fp16> {
  static __device__ __forceinline__ uint32_t abs(uint32_t a) {
    const auto r = __habs2(*reinterpret_cast<const __half2 *>(&a));
    return *reinterpret_cast<const uint32_t *>(&r);
  }
  static __device__ __forceinline__ uint32_t max(uint32_t a, uint32_t b) {
    const auto r =
        __hmax2(*reinterpret_cast<const __half2 *>(&a), *reinterpret_cast<const __half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);
  }
  static __device__ __forceinline__ float2 to_float2(uint32_t a) {
    return __half22float2(*reinterpret_cast<const __half2 *>(&a));
  }
};

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte global -> shared copy; src_bytes 0 zero-fills the destination.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void *src, uint32_t src_bytes) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes)
               : "memory");
#endif
}

// Arrive on the mbarrier once the thread's cp.async copies have landed.
__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(mbar))
               : "memory");
#endif
}

// Four floats to four E4M3 bytes, element i in byte i (round to nearest, saturate).
__device__ __forceinline__ uint32_t cvt_e4m3x4(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

struct Unit {
  int row0;
  int col_tile;
};

// Units are ordered by 256-row chunk, column tile minor, so that the two 128-row halves of a chunk
// at one column tile are consecutive and neighbouring CTAs read the same gathered rows' other
// column tiles at about the same time.
__device__ __forceinline__ Unit decode_unit(int u, int num_col_tiles, int num_full_chunks) {
  const int per_chunk = 2 * num_col_tiles;
  if (u < num_full_chunks * per_chunk) {
    const int chunk = u / per_chunk;
    const int rem = u - chunk * per_chunk;
    return {chunk * 2 * kUnitRows + (rem & 1) * kUnitRows, rem >> 1};
  }
  return {num_full_chunks * 2 * kUnitRows, u - num_full_chunks * per_chunk};
}

// Every CTA counts itself out; the last one re-arms the counter for the next launch.
__device__ __forceinline__ void finish(int32_t *counter) {
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(counter + 1, 1) == static_cast<int>(gridDim.x) - 1) {
      atomicExch(counter, 0);
      atomicExch(counter + 1, 0);
      __threadfence();
    }
  }
}

template <typename IType, bool kWriteHighPrecision>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSM)
    gather_quantize_kernel(const Params p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t *bars = reinterpret_cast<uint64_t *>(smem + kBarsOffset);
  uint16_t *xchg = reinterpret_cast<uint16_t *>(smem + kXchgOffset);
  float *col_scale_rcp = reinterpret_cast<float *>(smem + kColScaleOffset);
  uint8_t *rowwise_scales_tile = smem + kRowwiseScalesOffset;
  uint8_t *columnwise_scales_tile = smem + kColumnwiseScalesOffset;
  int32_t *misc = reinterpret_cast<int32_t *>(smem + kMiscOffset);
  int32_t *sizes = reinterpret_cast<int32_t *>(smem + kGroupsOffset);
  int32_t *starts = sizes + p.num_groups;
  volatile int32_t *units_v = misc + 8;  // ring of the CTA's grabbed units
  volatile int32_t *end_v = misc + 12;   // the CTA's tile count once known
  constexpr int kEndUnknown = 0x7FFFFFFF;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int c = tid & 15;   // 8-column chunk of the tile row
  const int rg = tid >> 4;  // rows 4rg .. 4rg+3 of the tile
  const int hidden = p.hidden;
  const int num_col_tiles = p.num_col_tiles;
  const float fp8_max_rcp = Quantized_Limits<fp8e4m3>::max_norm_rcp;

  for (int g = tid; g < p.num_groups; g += kThreads) {
    sizes[g] = static_cast<int32_t>(p.split_sizes[g]);
  }
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      ptx::mbarrier_init(&bars[s], kThreads);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int rows = 0;
    for (int g = 0; g < p.num_groups; ++g) {
      starts[g] = rows;
      rows += sizes[g];
    }
    misc[0] = rows;
  }
  __syncthreads();
  const int rows_used = misc[0];
  const int num_full_chunks = rows_used / (2 * kUnitRows);
  const int num_units = (rows_used / kUnitRows) * num_col_tiles;

  // Thread 0 grabs the CTA's k-th unit; the first grab past the end fixes the CTA's tile count.
  auto grab_unit = [&](int k) {
    const int u = atomicAdd(p.counter, 1);
    units_v[k & 3] = u;
    if (u >= num_units && end_v[0] == kEndUnknown) {
      end_v[0] = k * kTilesPerUnit;
    }
  };
  if (tid == 0) {
    end_v[0] = kEndUnknown;
    grab_unit(0);
  }
  __syncthreads();
  if (end_v[0] <= 0) {
    finish(p.counter);
    return;
  }
  auto unit_of_tile = [&](int it) {
    return decode_unit(units_v[(it / kTilesPerUnit) & 3], num_col_tiles, num_full_chunks);
  };

  const uint32_t data0 = smem_u32(smem);
  const IType *src = reinterpret_cast<const IType *>(p.src);

  // Stage the 32 gathered row segments of tile `it` into ring slot `stage`.
  auto issue = [&](int it, int stage) {
    const Unit unit = unit_of_tile(it);
    const int row0 = unit.row0 + (it % kTilesPerUnit) * kTileRows;
    const int col0 = unit.col_tile * kTileCols;
    const bool chunk_in_bounds = col0 + c * 8 < hidden;
    const uint32_t dst = data0 + stage * kTileBytes + 16 * c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = 4 * rg + i;
      const int tok = p.row_to_token[row0 + r];
      const bool valid = tok >= 0 && tok < p.num_tokens && chunk_in_bounds;
      const IType *row = valid ? src + static_cast<size_t>(tok) * hidden + col0 + 8 * c : src;
      cp_async_16(dst + r * kRowBytes, row, valid ? 16u : 0u);
    }
    cp_async_mbarrier_arrive(&bars[stage]);
  };

#pragma unroll 1
  for (int s = 0; s < kStages; ++s) {
    issue(s, s);  // a unit has kTilesPerUnit >= kStages tiles
  }

  int g = 0;  // group of the current unit; units arrive in increasing row order
#pragma unroll 1
  for (int it = 0; it < end_v[0]; ++it) {
    const int stage = it % kStages;
    const uint32_t parity = static_cast<uint32_t>((it / kStages) & 1);
    const Unit unit = unit_of_tile(it);
    const int m = it % kTilesPerUnit;  // 32-row block within the unit
    const int row0 = unit.row0 + m * kTileRows;
    const int col0 = unit.col_tile * kTileCols;
    const bool chunk_in_bounds = col0 + c * 8 < hidden;
    // The next unit is grabbed at the tile whose refill (after the last barrier below) issues
    // that unit's first tile.
    if (tid == 0 && (it + kStages) % kTilesPerUnit == 0 && end_v[0] == kEndUnknown) {
      grab_unit((it + kStages) / kTilesPerUnit);
    }
    if (m == 0) {
      // The two 128-row halves of a 256-row chunk alternate, so the walk also steps back.
      while (g > 0 && row0 < starts[g]) --g;
      while (row0 >= starts[g] + sizes[g]) ++g;  // skips empty groups
    }

    ptx::mbarrier_wait_parity(&bars[stage], parity);

    // The thread's 4 rows x 8 elements (zeros in padding rows and past the hidden size).
    uint32_t x[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = *reinterpret_cast<const uint4 *>(smem + stage * kTileBytes +
                                                       (4 * rg + i) * kRowBytes + 16 * c);
      x[i][0] = v.x;
      x[i][1] = v.y;
      x[i][2] = v.z;
      x[i][3] = v.w;
    }

    // Rowwise amax of each 32-column block (4 lanes) and the thread's columnwise partial.
    uint32_t col_max[4];
    uint32_t row_max[2];
    {
      uint32_t rm[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t a[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          a[k] = Packed2<IType>::abs(x[i][k]);
          col_max[k] = (i == 0) ? a[k] : Packed2<IType>::max(col_max[k], a[k]);
        }
        uint32_t mm =
            Packed2<IType>::max(Packed2<IType>::max(a[0], a[1]), Packed2<IType>::max(a[2], a[3]));
        mm = Packed2<IType>::max(mm, mm >> 16);
        rm[i] = mm & 0xFFFFu;
      }
      row_max[0] = rm[0] | (rm[1] << 16);
      row_max[1] = rm[2] | (rm[3] << 16);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        row_max[j] = Packed2<IType>::max(row_max[j], __shfl_xor_sync(0xffffffffu, row_max[j], 1));
        row_max[j] = Packed2<IType>::max(row_max[j], __shfl_xor_sync(0xffffffffu, row_max[j], 2));
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        col_max[k] = Packed2<IType>::max(col_max[k], __shfl_xor_sync(0xffffffffu, col_max[k], 16));
      }
    }
    if (lane < 16) {
      *reinterpret_cast<uint4 *>(smem + kXchgOffset + warp * (kTileCols * 2) + 16 * c) =
          make_uint4(col_max[0], col_max[1], col_max[2], col_max[3]);
    }

    // Rowwise scales (the same in the block's 4 lanes).
    float row_scale_rcp[4];
    e8m0_t row_exp[4];
    {
      const float2 am01 = Packed2<IType>::to_float2(row_max[0]);
      const float2 am23 = Packed2<IType>::to_float2(row_max[1]);
      row_exp[0] = ptx::float_to_e8m0(am01.x * fp8_max_rcp);
      row_exp[1] = ptx::float_to_e8m0(am01.y * fp8_max_rcp);
      row_exp[2] = ptx::float_to_e8m0(am23.x * fp8_max_rcp);
      row_exp[3] = ptx::float_to_e8m0(am23.y * fp8_max_rcp);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        row_scale_rcp[i] = ptx::exp2f_rcp<float>(row_exp[i]);
      }
    }
    __syncthreads();  // columnwise partials visible

    // Columnwise scale of column 8c + rg from the 4 warp partials, shared through shared memory.
    {
      const int col = 8 * c + rg;
      uint32_t a = xchg[0 * kTileCols + col];
      a = Packed2<IType>::max(a, xchg[1 * kTileCols + col]);
      a = Packed2<IType>::max(a, xchg[2 * kTileCols + col]);
      a = Packed2<IType>::max(a, xchg[3 * kTileCols + col]);
      const float amax = Packed2<IType>::to_float2(a & 0xFFFFu).x;
      const e8m0_t e = ptx::float_to_e8m0(amax * fp8_max_rcp);
      col_scale_rcp[col] = ptx::exp2f_rcp<float>(e);
      // Swizzled tile entry of (column, row block m); zero for the tile's padding columns.
      columnwise_scales_tile[(col & 31) * 16 + ((col >> 5) & 3) * 4 + m] =
          (col0 + col < hidden) ? e : static_cast<e8m0_t>(0);
    }
    if ((c & 3) == 0) {
      const int j = c >> 2;  // 32-column block within the tile
      const bool block_in_bounds = col0 + j * 32 < hidden;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        rowwise_scales_tile[(4 * rg + i) * 16 + m * 4 + j] =
            block_in_bounds ? row_exp[i] : static_cast<e8m0_t>(0);
      }
    }
    __syncthreads();  // columnwise scale_inv visible

    // Quantize and store.
    float cs[8];
    {
      const float4 s0 = *reinterpret_cast<const float4 *>(col_scale_rcp + 8 * c);
      const float4 s1 = *reinterpret_cast<const float4 *>(col_scale_rcp + 8 * c + 4);
      cs[0] = s0.x;
      cs[1] = s0.y;
      cs[2] = s0.z;
      cs[3] = s0.w;
      cs[4] = s1.x;
      cs[5] = s1.y;
      cs[6] = s1.z;
      cs[7] = s1.w;
    }
    if (chunk_in_bounds) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const size_t off =
            static_cast<size_t>(row0 + 4 * rg + i) * hidden + static_cast<size_t>(col0 + 8 * c);
        float f[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 v = Packed2<IType>::to_float2(x[i][k]);
          f[2 * k] = v.x;
          f[2 * k + 1] = v.y;
        }
        const float rs = row_scale_rcp[i];
        uint2 qr, qc;
        qr.x = cvt_e4m3x4(f[0] * rs, f[1] * rs, f[2] * rs, f[3] * rs);
        qr.y = cvt_e4m3x4(f[4] * rs, f[5] * rs, f[6] * rs, f[7] * rs);
        qc.x = cvt_e4m3x4(f[0] * cs[0], f[1] * cs[1], f[2] * cs[2], f[3] * cs[3]);
        qc.y = cvt_e4m3x4(f[4] * cs[4], f[5] * cs[5], f[6] * cs[6], f[7] * cs[7]);
        *reinterpret_cast<uint2 *>(p.rowwise_data + off) = qr;
        *reinterpret_cast<uint2 *>(p.columnwise_data + off) = qc;
        if constexpr (kWriteHighPrecision) {
          *reinterpret_cast<uint4 *>(reinterpret_cast<IType *>(p.high_precision) + off) =
              make_uint4(x[i][0], x[i][1], x[i][2], x[i][3]);
        }
      }
    }
    __syncthreads();  // ring slot free for the refill, scale tiles complete, next unit published

    if (m == kTilesPerUnit - 1) {
      // The unit's 128 rows are one swizzled 512-byte scale tile in each orientation.
      if (tid < 32) {
        const uint4 v = *reinterpret_cast<const uint4 *>(rowwise_scales_tile + 16 * tid);
        const size_t tile =
            static_cast<size_t>(unit.row0 / kUnitRows) * num_col_tiles + unit.col_tile;
        *reinterpret_cast<uint4 *>(p.rowwise_scales + tile * kScaleTileBytes + 16 * tid) = v;
      } else if (tid < 64) {
        const int t = tid - 32;
        const uint4 v = *reinterpret_cast<const uint4 *>(columnwise_scales_tile + 16 * t);
        const size_t group_base = static_cast<size_t>(starts[g] / 32) * p.hidden_padded;
        const size_t tile = static_cast<size_t>(unit.col_tile) * (sizes[g] / kUnitRows) +
                            (unit.row0 - starts[g]) / kUnitRows;
        *reinterpret_cast<uint4 *>(p.columnwise_scales + group_base + tile * kScaleTileBytes +
                                   16 * t) = v;
      }
    }
    if (it + kStages < end_v[0]) {
      issue(it + kStages, stage);
    }
  }
  finish(p.counter);
#else
  NVTE_DEVICE_ERROR("gather_quantize_mxfp8 is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <typename IType, bool kWriteHighPrecision>
void launch(const Params &p, cudaStream_t stream) {
  auto kernel = gather_quantize_kernel<IType, kWriteHighPrecision>;
  const size_t smem = smem_bytes(p.num_groups);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int blocks_per_sm = 0;
  NVTE_CHECK_CUDA(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, smem));
  NVTE_CHECK(blocks_per_sm > 0, "gather_quantize_mxfp8 needs ", smem,
             " bytes of shared memory per block, more than the device allows.");
  // An upper bound: units past the groups' rows are skipped by the kernel.
  const int64_t max_units = static_cast<int64_t>(p.num_rows / kUnitRows) * p.num_col_tiles;
  const int64_t grid =
      std::min<int64_t>(static_cast<int64_t>(cuda::sm_count()) * blocks_per_sm, max_units);
  kernel<<<static_cast<unsigned>(grid), kThreads, smem, stream>>>(p);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace gather_quantize_mxfp8
}  // namespace transformer_engine

void nvte_gather_quantize_mxfp8(const NVTETensor input, const NVTETensor row_to_token,
                                const NVTETensor split_sizes, NVTETensor output,
                                NVTETensor high_precision_output, NVTETensor counter,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_gather_quantize_mxfp8);
  using namespace transformer_engine;
  namespace gq = transformer_engine::gather_quantize_mxfp8;

  const Tensor &input_cpp = *convertNVTETensorCheck(input);
  const Tensor &row_to_token_cpp = *convertNVTETensorCheck(row_to_token);
  const Tensor &split_sizes_cpp = *convertNVTETensorCheck(split_sizes);
  const Tensor &output_cpp = *convertNVTETensorCheck(output);
  const Tensor &counter_cpp = *convertNVTETensorCheck(counter);
  const Tensor *high_precision_cpp =
      high_precision_output != nullptr ? convertNVTETensor(high_precision_output) : nullptr;
  const bool write_high_precision = high_precision_cpp != nullptr && high_precision_cpp->has_data();

  NVTE_CHECK(cuda::sm_arch() >= 90, "gather_quantize_mxfp8 requires compute capability 9.0+.");
  NVTE_CHECK(input_cpp.dim() == 2 &&
                 (input_cpp.dtype() == DType::kBFloat16 || input_cpp.dtype() == DType::kFloat16),
             "input must be a 2D BF16 or FP16 tensor, but got dtype=", to_string(input_cpp.dtype()),
             " and shape=", input_cpp.shape(), ".");
  const size_t num_tokens = input_cpp.shape()[0];
  const size_t hidden = input_cpp.shape()[1];
  NVTE_CHECK(num_tokens > 0 && hidden > 0 && hidden % 32 == 0,
             "input must have at least one row and a hidden size that is a multiple of 32, but got"
             " shape=",
             input_cpp.shape(), ".");
  NVTE_CHECK(row_to_token_cpp.dim() == 1 && row_to_token_cpp.dtype() == DType::kInt32,
             "row_to_token must be a 1D int32 tensor.");
  const size_t num_rows = row_to_token_cpp.numel();
  NVTE_CHECK(num_rows % gq::kUnitRows == 0, "The number of rows must be a multiple of ",
             gq::kUnitRows, ", but got ", num_rows, ".");
  NVTE_CHECK(split_sizes_cpp.dim() == 1 && split_sizes_cpp.dtype() == DType::kInt64 &&
                 split_sizes_cpp.numel() > 0,
             "split_sizes must be a non-empty 1D int64 tensor.");
  NVTE_CHECK(
      output_cpp.scaling_mode == NVTE_MXFP8_1D_SCALING && output_cpp.dtype() == DType::kFloat8E4M3,
      "output must be an E4M3 MXFP8 tensor.");
  NVTE_CHECK(output_cpp.has_data() && output_cpp.has_columnwise_data(),
             "output must have rowwise and columnwise data.");
  NVTE_CHECK(output_cpp.data.numel() == num_rows * hidden &&
                 output_cpp.columnwise_data.numel() == num_rows * hidden,
             "output data must have ", num_rows * hidden, " elements.");
  const size_t hidden_padded = (hidden + gq::kTileCols - 1) / gq::kTileCols * gq::kTileCols;
  const size_t num_scales =
      (num_rows / gq::kUnitRows) * (hidden_padded / gq::kTileCols) * gq::kScaleTileBytes;
  NVTE_CHECK(output_cpp.scale_inv.numel() >= num_scales &&
                 output_cpp.columnwise_scale_inv.numel() >= num_scales,
             "output scales must have at least ", num_scales, " entries.");
  if (write_high_precision) {
    NVTE_CHECK(high_precision_cpp->dtype() == input_cpp.dtype() &&
                   high_precision_cpp->numel() == num_rows * hidden,
               "high_precision_output must have the input's dtype and ", num_rows * hidden,
               " elements.");
  }
  NVTE_CHECK(counter_cpp.dtype() == DType::kInt32 && counter_cpp.numel() >= 2,
             "counter must be an int32 tensor of at least 2 elements.");
  if (num_rows == 0) {
    return;
  }

  gq::Params p{};
  p.src = input_cpp.data.dptr;
  p.row_to_token = reinterpret_cast<const int32_t *>(row_to_token_cpp.data.dptr);
  p.split_sizes = reinterpret_cast<const int64_t *>(split_sizes_cpp.data.dptr);
  p.rowwise_data = reinterpret_cast<uint8_t *>(output_cpp.data.dptr);
  p.rowwise_scales = reinterpret_cast<e8m0_t *>(output_cpp.scale_inv.dptr);
  p.columnwise_data = reinterpret_cast<uint8_t *>(output_cpp.columnwise_data.dptr);
  p.columnwise_scales = reinterpret_cast<e8m0_t *>(output_cpp.columnwise_scale_inv.dptr);
  p.high_precision = write_high_precision ? high_precision_cpp->data.dptr : nullptr;
  p.counter = reinterpret_cast<int32_t *>(counter_cpp.data.dptr);
  p.num_tokens = static_cast<int>(num_tokens);
  p.hidden = static_cast<int>(hidden);
  p.num_rows = static_cast<int>(num_rows);
  p.num_groups = static_cast<int>(split_sizes_cpp.numel());
  p.num_col_tiles = static_cast<int>(hidden_padded / gq::kTileCols);
  p.hidden_padded = static_cast<int>(hidden_padded);

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      input_cpp.dtype(), IType,
      if (write_high_precision) { gq::launch<IType, true>(p, stream); } else {
        gq::launch<IType, false>(p, stream);
      });
}
