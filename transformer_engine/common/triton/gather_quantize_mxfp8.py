# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel that gathers rows and quantizes them to grouped MXFP8 storage."""

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
def _gather_quantize_mxfp8_kernel(
    src_ptr,
    row_to_token_ptr,
    split_sizes_ptr,
    rowwise_data_ptr,
    rowwise_scale_ptr,
    columnwise_data_ptr,
    columnwise_scale_ptr,
    high_precision_ptr,
    hidden,
    hidden_padded,
    num_scale_tiles_x,
    num_groups,
    FP8_MAX_RCP: tl.constexpr,
    NUM_GROUPS_POW2: tl.constexpr,
    WRITE_HIGH_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantize BLOCK_M gathered rows x BLOCK_K columns to MXFP8, rowwise and columnwise.

    Output row r is ``src[row_to_token[r]]``, a zero row when ``row_to_token[r] < 0``; rows at or past
    the sum of ``split_sizes`` are left unwritten. Scales are written in the layouts of
    ``tex.group_quantize`` with GEMM-swizzled scales: one 128x4 tile grid over the whole
    (rows, hidden) tensor for the rowwise scales, and for the columnwise scales the tile grid of
    each group's (rows, hidden) slice, groups stacked at ``(group start / 32) * hidden_padded``.
    Scale entries of the tiles' padding columns are zero, as the group quantize writes them.
    Group sizes and BLOCK_M divide 128, so a program never straddles two groups.
    """
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    groups = tl.arange(0, NUM_GROUPS_POW2)
    sizes = tl.load(split_sizes_ptr + groups, mask=groups < num_groups, other=0)
    row0 = pid_m * BLOCK_M
    if row0 < tl.sum(sizes, axis=0):
        rows = row0 + tl.arange(0, BLOCK_M)
        rows64 = rows.to(tl.int64)
        tokens = tl.load(row_to_token_ptr + rows)
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        col_valid = cols < hidden
        src = tl.load(
            src_ptr + tl.maximum(tokens, 0).to(tl.int64)[:, None] * hidden + cols[None, :],
            mask=(tokens >= 0)[:, None] & col_valid[None, :],
            other=0.0,
        )
        out_offsets = rows64[:, None] * hidden + cols[None, :]
        if WRITE_HIGH_PRECISION:
            tl.store(high_precision_ptr + out_offsets, src, mask=col_valid[None, :])
        x = src.to(tl.float32)
        ax = tl.abs(x)

        # Rowwise: blocks of 32 columns, one scale grid over the whole tensor.
        NUM_ROW_BLOCKS: tl.constexpr = BLOCK_K // 32
        amax_r = tl.max(tl.reshape(ax, (BLOCK_M, NUM_ROW_BLOCKS, 32)), axis=2)
        e_r = _e8m0_from_amax(amax_r, FP8_MAX_RCP)
        scale_r = tl.reshape(
            tl.broadcast_to(_e8m0_scale_rcp(e_r)[:, :, None], (BLOCK_M, NUM_ROW_BLOCKS, 32)),
            (BLOCK_M, BLOCK_K),
        )
        tl.store(
            rowwise_data_ptr + out_offsets,
            (x * scale_r).to(rowwise_data_ptr.dtype.element_ty),
            mask=col_valid[None, :],
        )
        col_blocks = pid_k * NUM_ROW_BLOCKS + tl.arange(0, NUM_ROW_BLOCKS)
        e_r = tl.where((col_blocks * 32 < hidden)[None, :], e_r, 0)
        tl.store(
            rowwise_scale_ptr
            + _swizzled_scale_idx(rows[:, None], col_blocks[None, :], num_scale_tiles_x),
            e_r.to(rowwise_scale_ptr.dtype.element_ty),
            mask=(col_blocks < num_scale_tiles_x * 4)[None, :],
        )

        # Columnwise: blocks of 32 rows, one scale grid per group.
        NUM_COL_BLOCKS: tl.constexpr = BLOCK_M // 32
        amax_c = tl.max(tl.reshape(ax, (NUM_COL_BLOCKS, 32, BLOCK_K)), axis=1)
        e_c = _e8m0_from_amax(amax_c, FP8_MAX_RCP)
        scale_c = tl.reshape(
            tl.broadcast_to(_e8m0_scale_rcp(e_c)[:, None, :], (NUM_COL_BLOCKS, 32, BLOCK_K)),
            (BLOCK_M, BLOCK_K),
        )
        tl.store(
            columnwise_data_ptr + out_offsets,
            (x * scale_c).to(columnwise_data_ptr.dtype.element_ty),
            mask=col_valid[None, :],
        )
        starts = tl.cumsum(sizes, axis=0) - sizes
        group = tl.sum(((starts <= row0) & (groups < num_groups)).to(tl.int32), axis=0) - 1
        group_rows = tl.sum(tl.where(groups == group, sizes, 0), axis=0)
        group_start = tl.sum(tl.where(groups == group, starts, 0), axis=0)
        row_blocks = (row0 - group_start) // 32 + tl.arange(0, NUM_COL_BLOCKS)
        e_c = tl.where(col_valid[None, :], e_c, 0)
        tl.store(
            columnwise_scale_ptr
            + (group_start // 32) * hidden_padded
            + _swizzled_scale_idx(cols[None, :], row_blocks[:, None], group_rows // 128),
            e_c.to(columnwise_scale_ptr.dtype.element_ty),
            mask=(cols < hidden_padded)[None, :],
        )
