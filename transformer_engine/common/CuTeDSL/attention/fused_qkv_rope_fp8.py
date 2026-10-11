# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused QKV RoPE with FP8 quantization at the attention boundary (CuTe DSL).

``rope_fwd``: one pass over the ``[s, b, h*(dq+dk+dv)]`` bf16 output of the QKV linear adds its
bias, applies RoPE to the q and k heads, writes q | k | v contiguously in ``sbhd`` order into one
flat bf16 buffer and reduces the per-tensor amax, from which the last CTA derives the
Float8CurrentScaling scale (``compute_scale_from_amax`` semantics). A TE cast of the flat buffer
with that scale then gives the FP8 q/k/v that DotProductAttention takes as they are.

``rope_bwd``: one pass over the bf16 dq/dk/dv applies the RoPE backward, packs d(qkv) and
quantizes it to MXFP8, rowwise and columnwise with compact or GEMM-swizzled E8M0 scales, together
with per-32-token partial column sums that ``dbias_reduce`` turns into the bias gradient.

Both kernels take head dim 64 (= rotary dim), non-interleaved RoPE, ``s*b % 128 == 0`` and
``h*(dq+dk+dv) % 128 == 0`` (no MXFP8 scale padding); the backward tiles 4 q heads per CTA. The
PyTorch frontend (``transformer_engine/pytorch/attention/fused_qkv_rope_quantize.py``) builds the
cute tensors and launches the compiled kernels.
"""

import cuda.bindings.driver as cuda  # pylint: disable=no-name-in-module
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T

VEC = 8  # bf16 elements per 16-byte vector
HEAD_DIM = 64
NVEC_D = HEAD_DIM // VEC  # vectors per head
HALF = NVEC_D // 2  # the RoPE partner of vector i is vector i +- HALF
FLT_MAX = 3.4028234663852886e38
F32_INF = float("inf")
F32_MIN_NORMAL = 1.1754943508222875e-38

FWD_TOKENS_PER_BLOCK = 2
FWD_LANES_PER_SLOT = 4  # 4 lanes x 2 vectors (the RoPE pair) cover one 64-wide head
BWD_ROWS = 32  # tokens per CTA tile (one MXFP8 columnwise block)
BWD_THREADS = 256
BWD_Q_HEADS_PER_CTA = (
    4  # q CTA tile: 32 tokens x 4 heads; kv CTA tile: 32 tokens x (k head, v head)
)
RED_THREADS = 256
RED_COLS = 32


def _fma(a, b, c):
    """a*b + c with one rounding, on 8-wide fp32 vectors."""
    r = mlir_math.fma(a.ir_value(), b.ir_value(), c.ir_value())
    return type(a)(r, a.shape, a.dtype)


def _rope(x, cos_v, rot, sin_v):
    """x*cos + rot*sin contracted as fma(x, cos, rot*sin), like TE's fused_rope kernel."""
    return _fma(x, cos_v, rot * sin_v)


def _f32_bits(v: Float32) -> Int32:
    return Int32(llvm.bitcast(T.i32(), v.ir_value()))


def _bits_f32(b: Int32) -> Float32:
    return Float32(llvm.bitcast(T.f32(), b.ir_value()))


def _block_scale(amax: Float32, max_norm_rcp: cutlass.Constexpr):
    """MXFP8 block scale from the block amax (TE cast_mxfp8 semantics):
    e8m0 = float_to_e8m0(amax / fp8_max); returns (scale as Float8E8M0FNU, 1/scale in fp32)."""
    sf = (amax * Float32(max_norm_rcp)).to(cutlass.Float8E8M0FNU).to(Float32)
    # 1/2^k is exact; an all-zero block (e8m0 0, scale 0) quantizes to zeros for any scale_inv
    sinv = Float32(1.0) / cute.arch.fmax(sf, Float32(F32_MIN_NORMAL))
    return sf.to(cutlass.Float8E8M0FNU), sinv


def _swizzled_scale_idx(i: Int32, j: Int32, num_tiles_x: Int32) -> Int32:
    """TE gemm_swizzled_scale_idx: the cuBLAS block-scaling layout in 128 x 4 tiles."""
    tile_y = i // 128
    tile_x = j // 4
    iy = i % 128
    ix = j % 4
    return (tile_y * num_tiles_x + tile_x) * 512 + (iy % 32) * 16 + (iy // 32) * 4 + ix


# ---------------------------------------------------------------------------------------------
# Forward: bias + RoPE(q, k), relocation into one flat q | k | v buffer, per-tensor amax and scale
# ---------------------------------------------------------------------------------------------
@cute.kernel
def _rope_fwd_kernel(
    gX: cute.Tensor,
    gBias: cute.Tensor,
    gOut: cute.Tensor,
    gCos: cute.Tensor,
    gSin: cute.Tensor,
    gScale: cute.Tensor,
    gAmax: cute.Tensor,
    gWsAmax: cute.Tensor,
    gWsCount: cute.Tensor,
    b: Int32,
    m_tokens: Int32,
    nblocks: Int32,
    ng: cutlass.Constexpr,
    nqpg: cutlass.Constexpr,
    nslots: cutlass.Constexpr,
    nwarps: cutlass.Constexpr,
    nq_elems: cutlass.Constexpr,
    nk_elems: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    fp8_max: cutlass.Constexpr,
    amax_epsilon: cutlass.Constexpr,
    pow_2_scales: cutlass.Constexpr,
):
    """thread <-> (token, head slot, quarter): the thread owns the 16-byte vectors ``quarter`` and
    ``quarter + HALF`` of one head (the RoPE pair), so the 4 lanes of a slot read 64 contiguous
    bytes and a warp's 8 slots read 8 full 128-byte lines."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    hidden = ng * (nqpg + 2) * HEAD_DIM
    gsz = (nqpg + 2) * HEAD_DIM
    nqh = ng * nqpg
    tpt = nslots * FWD_LANES_PER_SLOT  # threads per token

    smem_ptr = cute.arch.alloc_smem(Float32, nwarps)
    smem = cute.make_tensor(smem_ptr, cute.make_layout((nwarps,)))

    tok_in_blk = tidx // tpt
    rem = tidx % tpt
    slot = rem // FWD_LANES_PER_SLOT
    quarter = rem % FWD_LANES_PER_SLOT
    t = bidx * FWD_TOKENS_PER_BLOCK + tok_in_blk
    g = slot // (nqpg + 2)
    r = slot % (nqpg + 2)
    kv_r = cutlass.max(r - nqpg, -1) + 1  # 0: q head, 1: k, 2: v
    is_kv = cutlass.min(kv_r, 1)
    col_vec = (g * gsz + r * HEAD_DIM) // VEC + quarter
    in_vec = (t * hidden) // VEC + col_vec
    out_q = (t * nqh + g * nqpg + r) * HEAD_DIM
    out_kv = nq_elems + (kv_r - 1) * nk_elems + (t * ng + g) * HEAD_DIM
    out_vec = ((1 - is_kv) * out_q + is_kv * out_kv) // VEC + quarter
    cs_vec = (t // b) * NVEC_D + quarter

    local_amax = Float32(0.0)
    if t < m_tokens:
        x_lo = gX[(None, in_vec)].load().to(Float32)
        x_hi = gX[(None, in_vec + HALF)].load().to(Float32)
        if cutlass.const_expr(has_bias):
            x_lo = x_lo + gBias[(None, col_vec)].load().to(Float32)
            x_hi = x_hi + gBias[(None, col_vec + HALF)].load().to(Float32)
        out_lo = x_lo
        out_hi = x_hi
        if kv_r < 2:
            cos_lo = gCos[(None, cs_vec)].load()
            cos_hi = gCos[(None, cs_vec + HALF)].load()
            sin_lo = gSin[(None, cs_vec)].load()
            sin_hi = gSin[(None, cs_vec + HALF)].load()
            # non-interleaved: rotate = -x[d + d2/2] for d < d2/2, x[d - d2/2] otherwise
            out_lo = _rope(x_lo, cos_lo, x_hi * Float32(-1.0), sin_lo)
            out_hi = _rope(x_hi, cos_hi, x_lo, sin_hi)
        o_lo = out_lo.to(BFloat16)
        o_hi = out_hi.to(BFloat16)
        gOut[(None, out_vec)] = o_lo
        gOut[(None, out_vec + HALF)] = o_hi
        a_lo = cute.math.absf(o_lo.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        a_hi = cute.math.absf(o_hi.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        local_amax = cute.arch.fmax(a_lo, a_hi)

    # block amax -> one atomic per CTA; the last CTA derives the scale and resets the workspace
    warp_id = tidx // 32
    lane_id = tidx % 32
    local_amax = cute.arch.warp_reduction_max(local_amax)
    if lane_id == 0:
        smem[warp_id] = local_amax
    cute.arch.sync_threads()
    if tidx == 0:
        blk = smem[0]
        for w in cutlass.range_constexpr(1, nwarps):
            blk = cute.arch.fmax(blk, smem[w])
        cute.arch.atomic_fmax(gWsAmax.iterator, blk, sign_bit=False)
        cute.arch.fence_acq_rel_gpu()
        old = Int32(cute.arch.atomic_add(gWsCount.iterator.llvm_ptr, Int32(1)))
        if old == nblocks - 1:
            cute.arch.fence_acq_rel_gpu()
            amax = gWsAmax[0]
            gWsAmax[0] = Float32(0.0)
            gWsCount[0] = Int32(0)
            gAmax[0] = amax
            # compute_scale_from_amax: 1 for amax == 0, inf or nan, else fp8_max / amax
            # (fp32 max if that overflows), optionally truncated to a power of two
            if amax < Float32(amax_epsilon):
                amax = Float32(amax_epsilon)
            scale = Float32(1.0)
            if amax > Float32(0.0):
                if amax < Float32(F32_INF):
                    scale = Float32(fp8_max) / amax
            if scale == Float32(F32_INF):
                scale = Float32(FLT_MAX)
            if cutlass.const_expr(pow_2_scales):
                scale = _bits_f32(_f32_bits(scale) & 0xFF800000)
            gScale[0] = scale


@cute.jit
def rope_fwd(
    mX: cute.Tensor,
    mBias: cute.Tensor,
    mOut: cute.Tensor,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    mScale: cute.Tensor,
    mAmax: cute.Tensor,
    mWsAmax: cute.Tensor,
    mWsCount: cute.Tensor,
    b: Int32,
    m_tokens: Int32,
    nblocks: Int32,
    ng: cutlass.Constexpr,
    nqpg: cutlass.Constexpr,
    nq_elems: cutlass.Constexpr,
    nk_elems: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    fp8_max: cutlass.Constexpr,
    amax_epsilon: cutlass.Constexpr,
    pow_2_scales: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """x [m, ng*(nqpg+2)*64] bf16 (+ bias [ng*(nqpg+2)*64]) -> out [m*ng*(nqpg+2)*64] bf16 as
    q_rot | k_rot | v in sbhd order, amax[1], scale[1]; cos/sin [s, 64] fp32; ws_* zeroed once."""
    nslots = ng * (nqpg + 2)
    nthreads = FWD_TOKENS_PER_BLOCK * nslots * FWD_LANES_PER_SLOT
    nwarps = (nthreads + 31) // 32
    gX = cute.zipped_divide(mX, (VEC,))
    gBias = cute.zipped_divide(mBias, (VEC,))
    gOut = cute.zipped_divide(mOut, (VEC,))
    gCos = cute.zipped_divide(mCos, (VEC,))
    gSin = cute.zipped_divide(mSin, (VEC,))
    _rope_fwd_kernel(
        gX,
        gBias,
        gOut,
        gCos,
        gSin,
        mScale,
        mAmax,
        mWsAmax,
        mWsCount,
        b,
        m_tokens,
        nblocks,
        ng,
        nqpg,
        nslots,
        nwarps,
        nq_elems,
        nk_elems,
        has_bias,
        fp8_max,
        amax_epsilon,
        pow_2_scales,
    ).launch(grid=(nblocks, 1, 1), block=(nthreads, 1, 1), stream=stream)


# ---------------------------------------------------------------------------------------------
# Backward: RoPE backward + packing into d(qkv), MXFP8 rowwise and columnwise quantization,
# per-32-token column sums for the bias gradient
# ---------------------------------------------------------------------------------------------
@cute.kernel
def _rope_bwd_kernel(
    gSrcA: cute.Tensor,
    gSrcB: cute.Tensor,
    gCos: cute.Tensor,
    gSin: cute.Tensor,
    gRowData: cute.Tensor,
    gRowScale: cute.Tensor,
    gColData: cute.Tensor,
    gColScale: cute.Tensor,
    gPartial: cute.Tensor,
    b: Int32,
    m_tokens: Int32,
    ng: cutlass.Constexpr,
    nqpg: cutlass.Constexpr,
    is_kv: cutlass.Constexpr,
    npass: cutlass.Constexpr,
    swizzled: cutlass.Constexpr,
    max_norm_rcp: cutlass.Constexpr,
):
    """grid = (m/32, ng, nz). q kernel (is_kv False): z selects BWD_Q_HEADS_PER_CTA q heads of
    group g, npass 4, source gSrcA = dq (sbhd, ng*nqpg heads); kv kernel (is_kv True): npass 2,
    pass 0 = k (gSrcA, RoPE backward), pass 1 = v (gSrcB, copy)."""
    tidx, _, _ = cute.arch.thread_idx()
    bx, g, bz = cute.arch.block_idx()
    hidden = ng * (nqpg + 2) * HEAD_DIM
    gsz = (nqpg + 2) * HEAD_DIM
    nqh = ng * nqpg
    width = npass * HEAD_DIM  # tile width in columns
    fp8_t = gRowData.element_type

    s_bf = cute.arch.alloc_smem(BFloat16, BWD_ROWS * width)
    smem_bf = cute.make_tensor(s_bf, cute.make_layout((BWD_ROWS * width,)))
    smem_bf8 = cute.zipped_divide(smem_bf, (VEC,))
    s_f8 = cute.arch.alloc_smem(fp8_t, BWD_ROWS * width)
    smem_f8 = cute.make_tensor(s_f8, cute.make_layout((BWD_ROWS * width,)))
    smem_f88 = cute.zipped_divide(smem_f8, (VEC,))

    row = tidx // NVEC_D  # 0..31
    c = tidx % NVEC_D  # vector (8 dims) within the head
    cp = c ^ HALF  # partner vector (dims +- 32)
    sgn = Float32(1 - 2 * (c // HALF))  # +sin[d+32] for d < 32, -sin[d-32] for d >= 32
    t = bx * BWD_ROWS + row
    cs_vec = (t // b) * NVEC_D
    t_row_base = t * hidden

    # phase A, row-mapped: RoPE backward, pack, rowwise MXFP8, stage the bf16 tile
    for p in cutlass.range_constexpr(npass):
        if cutlass.const_expr(is_kv):
            head_col = (nqpg + p) * HEAD_DIM  # k: nqpg*D, v: (nqpg+1)*D within the group
            src_vec = ((t * ng + g) * HEAD_DIM) // VEC
        else:
            hq = g * nqpg + bz * BWD_Q_HEADS_PER_CTA + p
            head_col = (bz * BWD_Q_HEADS_PER_CTA + p) * HEAD_DIM
            src_vec = ((t * nqh + hq) * HEAD_DIM) // VEC
        col0 = g * gsz + head_col  # packed column of dim 0 of this head
        if cutlass.const_expr(is_kv and p == 1):
            f = gSrcB[(None, src_vec + c)].load().to(Float32)
        else:
            x = gSrcA[(None, src_vec + c)].load().to(Float32)
            xp = gSrcA[(None, src_vec + cp)].load().to(Float32)
            cos_v = gCos[(None, cs_vec + c)].load()
            sin_p = gSin[(None, cs_vec + cp)].load() * sgn
            # the bf16 d(qkv) TE's RoPE backward would store is what gets quantized
            f = _rope(x, cos_v, xp, sin_p).to(BFloat16).to(Float32)
        smem_bf8[(None, row * (width // VEC) + p * NVEC_D + c)] = f.to(BFloat16)
        # rowwise MXFP8 block = 32 columns = 4 lanes
        amax = cute.math.absf(f).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
        e8, sinv = _block_scale(amax, max_norm_rcp)
        gRowData[(None, (t_row_base + col0) // VEC + c)] = (f * sinv).to(fp8_t)
        if c % 4 == 0:
            cb = (col0 // 32) + c // 4
            if cutlass.const_expr(swizzled):
                sidx = _swizzled_scale_idx(t, cb, hidden // 128)
            else:
                sidx = t * (hidden // 32) + cb
            gRowScale[sidx] = e8
    cute.arch.sync_threads()

    # phase B, column-mapped: columnwise MXFP8 (32 tokens per block) and the column sums
    for j in cutlass.range_constexpr((width + BWD_THREADS - 1) // BWD_THREADS):
        col = tidx + j * BWD_THREADS
        if col < width:
            camax = Float32(0.0)
            csum = Float32(0.0)
            for rr in cutlass.range_constexpr(BWD_ROWS):
                v = smem_bf[rr * width + col].to(Float32)
                camax = cute.arch.fmax(camax, cute.arch.fmax(v, Float32(0.0) - v))
                csum = csum + v
            e8, sinv = _block_scale(camax, max_norm_rcp)
            for rr in cutlass.range_constexpr(BWD_ROWS):
                v = smem_bf[rr * width + col].to(Float32)
                smem_f8[rr * width + col] = (v * sinv).to(fp8_t)
            if cutlass.const_expr(is_kv):
                gcol = g * gsz + nqpg * HEAD_DIM + col
            else:
                gcol = g * gsz + bz * BWD_Q_HEADS_PER_CTA * HEAD_DIM + col
            if cutlass.const_expr(swizzled):
                sidx = _swizzled_scale_idx(gcol, bx, m_tokens // 128)
            else:
                sidx = bx * hidden + gcol
            gColScale[sidx] = e8
            gPartial[bx * hidden + gcol] = csum
    cute.arch.sync_threads()

    # phase C, row-mapped: copy the columnwise fp8 tile out in 8-byte vectors
    for p in cutlass.range_constexpr(npass):
        if cutlass.const_expr(is_kv):
            head_col = (nqpg + p) * HEAD_DIM
        else:
            head_col = (bz * BWD_Q_HEADS_PER_CTA + p) * HEAD_DIM
        col0 = g * gsz + head_col
        v8 = smem_f88[(None, row * (width // VEC) + p * NVEC_D + c)].load()
        gColData[(None, (t_row_base + col0) // VEC + c)] = v8


@cute.jit
def rope_bwd(
    mSrcA: cute.Tensor,
    mSrcB: cute.Tensor,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    mRowData: cute.Tensor,
    mRowScale: cute.Tensor,
    mColData: cute.Tensor,
    mColScale: cute.Tensor,
    mPartial: cute.Tensor,
    b: Int32,
    m_tokens: Int32,
    nrow_blk: Int32,
    ng: cutlass.Constexpr,
    nqpg: cutlass.Constexpr,
    is_kv: cutlass.Constexpr,
    npass: cutlass.Constexpr,
    nz: cutlass.Constexpr,
    swizzled: cutlass.Constexpr,
    max_norm_rcp: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """dq [m*ng*nqpg*64] (q kernel) or dk, dv [m*ng*64] (kv kernel) bf16 -> MXFP8 d(qkv):
    rowwise/columnwise data [m*N] fp8, rowwise scales [m*N/32] and columnwise scales [m/32*N]
    e8m0, partial column sums [m/32, N] fp32, N = ng*(nqpg+2)*64."""
    gA = cute.zipped_divide(mSrcA, (VEC,))
    gB = cute.zipped_divide(mSrcB, (VEC,))
    gCos = cute.zipped_divide(mCos, (VEC,))
    gSin = cute.zipped_divide(mSin, (VEC,))
    gRow = cute.zipped_divide(mRowData, (VEC,))
    gCol = cute.zipped_divide(mColData, (VEC,))
    _rope_bwd_kernel(
        gA,
        gB,
        gCos,
        gSin,
        gRow,
        mRowScale,
        gCol,
        mColScale,
        mPartial,
        b,
        m_tokens,
        ng,
        nqpg,
        is_kv,
        npass,
        swizzled,
        max_norm_rcp,
    ).launch(grid=(nrow_blk, ng, nz), block=(BWD_THREADS, 1, 1), stream=stream)


# ---------------------------------------------------------------------------------------------
# Bias gradient: [R, N] fp32 partial sums -> [N]
# ---------------------------------------------------------------------------------------------
@cute.kernel
def _dbias_reduce_kernel(gP: cute.Tensor, gOut: cute.Tensor, nrows: Int32, ncols: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    nwarps = RED_THREADS // 32
    smem_ptr = cute.arch.alloc_smem(Float32, RED_THREADS)
    smem = cute.make_tensor(smem_ptr, cute.make_layout((RED_THREADS,)))
    warp_id = tidx // 32
    lane_id = tidx % 32
    col = bidx * RED_COLS + lane_id
    acc = Float32(0.0)
    if col < ncols:
        for r in cutlass.range(warp_id, nrows, nwarps):
            acc = acc + gP[r * ncols + col]
    smem[tidx] = acc
    cute.arch.sync_threads()
    if tidx < RED_COLS:
        tot = Float32(0.0)
        for w in cutlass.range_constexpr(nwarps):
            tot = tot + smem[w * 32 + tidx]
        if bidx * RED_COLS + tidx < ncols:
            gOut[bidx * RED_COLS + tidx] = tot.to(gOut.element_type)


@cute.jit
def dbias_reduce(
    mP: cute.Tensor,
    mOut: cute.Tensor,
    nrows: Int32,
    ncols: Int32,
    nblocks: Int32,
    stream: cuda.CUstream,
):
    """partial [nrows*ncols] fp32 -> out [ncols] (out's dtype), nblocks = ceil(ncols / 32)."""
    _dbias_reduce_kernel(mP, mOut, nrows, ncols).launch(
        grid=(nblocks, 1, 1), block=(RED_THREADS, 1, 1), stream=stream
    )
