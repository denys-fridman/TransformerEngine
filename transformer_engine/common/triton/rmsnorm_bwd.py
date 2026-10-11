# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for the RMSNorm backward pass."""

import triton
import triton.language as tl


@triton.jit
def _e8m0_from_amax(amax, FP8_MAX_RCP: tl.constexpr):
    """Biased exponent of the smallest power of two >= amax / fp8_max (ptx::float_to_e8m0)."""
    v = amax * FP8_MAX_RCP
    bits = v.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    round_up = (mant > 0) & (exp != 0xFE) & ((exp != 0) | (mant > 0x400000))
    e = exp + round_up.to(tl.int32)
    e = tl.where(exp == 0xFF, tl.where(mant > 0, 0xFF, 0xFE), e)  # NaN stays NaN, inf saturates
    return tl.where(v == 0.0, 0, e)


@triton.jit
def _e8m0_scale_rcp(e):
    """Multiplier 2^(127 - e) that quantizes a value whose block exponent is e (ptx::exp2f_rcp)."""
    r = ((254 - e) << 23).to(tl.float32, bitcast=True)
    r = tl.where(e == 254, 5.877471754111438e-39, r)  # 2^-127 is subnormal
    return tl.where(e == 255, float("nan"), r)


@triton.jit
def _swizzled_scale_idx(i, j, num_tiles_x):
    """Offset of scale (i, j) in the GEMM layout of 128x4 tiles (mxfp8::swizzle::gemm_swizzled_scale_idx)."""
    i_in_tile = i % 128
    tile = (i // 128) * num_tiles_x + j // 4
    return tile * 512 + (i_in_tile % 32) * 16 + (i_in_tile // 32) * 4 + j % 4


@triton.jit
def rmsnorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    w_ptr,
    add_ptr,
    dx_ptr,
    dw_part_ptr,
    rowwise_data_ptr,
    rowwise_scale_ptr,
    columnwise_data_ptr,
    columnwise_scale_ptr,
    M,
    N,
    rowwise_scale_stride,
    columnwise_scale_rows,
    columnwise_scale_stride,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    HAS_ADD: tl.constexpr,
    STORE_DX: tl.constexpr,
    ROWWISE: tl.constexpr,
    COLUMNWISE: tl.constexpr,
    SWIZZLED_SCALES: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP8_MAX_RCP: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """RMSNorm backward for one block of ROWS rows of dy, x [M, N] and rstd [M].

    dx = rstd * dy * w - x * rstd^3 / N * sum(dy * w * x) (+ add), rounded to OUT_DTYPE, is stored
    to dx (STORE_DX) and/or quantized to MXFP8 with rowwise (32 columns) and columnwise (the
    block's 32 rows) e8m0 scales in the compact or GEMM-swizzled layout. The block's fp32 partial
    sums of dw = sum(dy * x * rstd) go to dw_part[program, :].
    """
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

    acc = tl.zeros([ROWS], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED_GAMMA:
            w += 1.0
        acc += tl.sum(dy * w[None, :] * x, axis=1)
    c = acc * rstd * rstd * rstd / N

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED_GAMMA:
            w += 1.0
        dx = rstd[:, None] * dy * w[None, :] - x * c[:, None]
        if HAS_ADD:
            dx += tl.load(add_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dx = dx.to(OUT_DTYPE)
        if STORE_DX:
            tl.store(dx_ptr + offs, dx, mask=mask)
        dw_part = tl.sum(dy * x * rstd[:, None], axis=0)
        tl.store(dw_part_ptr + pid.to(tl.int64) * N + cols, dw_part, mask=col_mask)
        if ROWWISE or COLUMNWISE:
            dx = dx.to(tl.float32)  # quantize the rounded gradient, as a separate cast would
        if ROWWISE:
            blocks = tl.reshape(tl.abs(dx), [ROWS, BLOCK_N // 32, 32])
            e = _e8m0_from_amax(tl.max(blocks, axis=2), FP8_MAX_RCP)
            scale = tl.expand_dims(_e8m0_scale_rcp(e), 2)
            scale = tl.reshape(tl.broadcast_to(scale, [ROWS, BLOCK_N // 32, 32]), [ROWS, BLOCK_N])
            data = (dx * scale).to(rowwise_data_ptr.dtype.element_ty)
            tl.store(rowwise_data_ptr + offs, data, mask=mask)
            blk = start // 32 + tl.arange(0, BLOCK_N // 32)
            blk_mask = row_mask[:, None] & (blk < N // 32)[None, :]
            if SWIZZLED_SCALES:
                idx = _swizzled_scale_idx(rows[:, None], blk[None, :], rowwise_scale_stride // 4)
            else:
                idx = rows[:, None] * rowwise_scale_stride + blk[None, :]
            tl.store(rowwise_scale_ptr + idx, e.to(tl.uint8), mask=blk_mask)
        if COLUMNWISE:
            e = _e8m0_from_amax(tl.max(tl.abs(dx), axis=0), FP8_MAX_RCP)
            data = (dx * _e8m0_scale_rcp(e)[None, :]).to(columnwise_data_ptr.dtype.element_ty)
            tl.store(columnwise_data_ptr + offs, data, mask=mask)
            if SWIZZLED_SCALES:
                idx = _swizzled_scale_idx(cols, pid, columnwise_scale_rows // 4)
            else:
                idx = pid.to(tl.int64) * columnwise_scale_stride + cols
            tl.store(columnwise_scale_ptr + idx, e.to(tl.uint8), mask=col_mask)


@triton.jit
def column_sum_kernel(
    inp_ptr, out_ptr, n_rows, n_cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr
):
    """out[c] = sum over rows of the row-major fp32 matrix inp [n_rows, n_cols], in out's dtype."""
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, n_rows, BLOCK_R):
        rows = (start + tl.arange(0, BLOCK_R)).to(tl.int64)
        mask = (rows < n_rows)[:, None] & col_mask[None, :]
        acc += tl.sum(
            tl.load(inp_ptr + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0), axis=0
        )
    tl.store(out_ptr + cols, acc.to(out_ptr.dtype.element_ty), mask=col_mask)
