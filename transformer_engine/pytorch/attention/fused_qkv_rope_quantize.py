# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused QKV RoPE with FP8 quantization: eligibility check and PyTorch launch glue.

:func:`~transformer_engine.pytorch.attention.rope.apply_fused_qkv_rotary_pos_emb` routes a call
with ``bias``, ``qkv_quantizer`` or ``grad_qkv_quantizer`` here. The forward returns the rotated
q/k/v as Float8Tensors sharing one current-scaling scale, which DotProductAttention takes as they
are; the backward returns the gradient of qkv quantized to MXFP8 for the backward of the linear
that produced qkv, plus the bias gradient. The kernels live in
``transformer_engine/common/CuTeDSL/attention/fused_qkv_rope_fp8.py``.
"""

from __future__ import annotations

import functools
import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import DType, TE_DType_To_Torch
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch.utils import get_device_compute_capability

_HEAD_DIM = 64
_FP8_DTYPES = (DType.kFloat8E4M3, DType.kFloat8E5M2)
_compiled: Dict[tuple, Any] = {}
_workspaces: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


@functools.lru_cache(maxsize=None)
def _runtime() -> Optional[SimpleNamespace]:
    """The CuTeDSL runtime and the kernel module, or None when nvidia-cutlass-dsl is missing."""
    try:
        # pylint: disable=import-outside-toplevel
        import cuda.bindings.driver as cuda_driver  # pylint: disable=no-name-in-module
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import Int32

        from transformer_engine.common.CuTeDSL.attention import fused_qkv_rope_fp8 as kernel
    except ImportError as e:
        logging.getLogger(__name__).warning(
            "The fused QKV RoPE + quantize kernel cannot be imported (%s)", e
        )
        return None
    return SimpleNamespace(
        cuda=cuda_driver,
        cute=cute,
        from_dlpack=from_dlpack,
        kernel=kernel,
        Int32=Int32,
        E8M0=cutlass.Float8E8M0FNU,
        fp8_dtypes={DType.kFloat8E4M3: cutlass.Float8E4M3FN, DType.kFloat8E5M2: cutlass.Float8E5M2},
    )


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a is b or (
        a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()
    )


def fused_qkv_rope_quantize_supported(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    qkv_quantizer: Optional[Quantizer],
    grad_qkv_quantizer: Optional[Quantizer],
    bias: Optional[torch.Tensor] = None,
    *,
    tensor_format: str = "sbhd",
    start_positions: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    cp_size: int = 1,
) -> bool:
    """Whether :class:`FusedQKVRoPEQuantizeFunc` takes this call: contiguous bf16 ``qkv`` of shape
    ``[s, b, h * (dq + dk + dv)]`` in ``sbhd`` with head dim 64 and a multiple of 4 q heads per
    group, ``s * b`` and ``h * (dq + dk + dv)`` multiples of 128, the same non-interleaved freqs
    of rotary dim 64 for q and k, no start positions or context parallelism, a bias of
    ``qkv``'s dtype if any, a Float8CurrentScalingQuantizer without amax reduction for q/k/v
    and a 1D MXFP8Quantizer for the qkv gradient, on compute capability 10.x with
    nvidia-cutlass-dsl installed."""
    if tensor_format != "sbhd" or start_positions is not None or interleaved or cp_size != 1:
        return False
    if qkv.dim() != 3 or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return False
    s, b, hidden = qkv.shape
    dq, dk, dv = (int(x) for x in qkv_split_arg_list)
    if dk != _HEAD_DIM or dv != _HEAD_DIM or dq % (4 * _HEAD_DIM) != 0:
        return False
    if hidden % (dq + dk + dv) != 0 or (s * b) % 128 != 0 or hidden % 128 != 0:
        return False
    num_groups = hidden // (dq + dk + dv)
    if num_groups * (dq // _HEAD_DIM + 2) > 128:  # forward: 8 threads per head, 1024 per block
        return False
    if q_freqs.shape[-1] != _HEAD_DIM or q_freqs.shape[0] < s or not _same_tensor(q_freqs, k_freqs):
        return False
    if bias is not None and (bias.dtype != qkv.dtype or bias.shape != (hidden,)):
        return False
    if not isinstance(qkv_quantizer, Float8CurrentScalingQuantizer):
        return False
    if qkv_quantizer.with_amax_reduction or qkv_quantizer.dtype not in _FP8_DTYPES:
        return False
    if not isinstance(grad_qkv_quantizer, MXFP8Quantizer):
        return False
    if grad_qkv_quantizer.with_2d_quantization or grad_qkv_quantizer.dtype not in _FP8_DTYPES:
        return False
    if qkv.device.type != "cuda" or get_device_compute_capability()[0] != 10:
        return False
    return _runtime() is not None


def _vec(rt: SimpleNamespace, t: torch.Tensor, align: int = 16):
    return rt.from_dlpack(t.detach().reshape(-1), assumed_align=align)


def _vec_as(rt: SimpleNamespace, t: torch.Tensor, cute_dtype: Any, align: int = 16):
    """A buffer viewed with a CuTe element type torch cannot carry (E8M0 scales, FP8 data)."""
    ct = rt.from_dlpack(t.detach().reshape(-1), assumed_align=align)
    ct.element_type = cute_dtype
    return ct


def _stream(rt: SimpleNamespace):
    return rt.cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _workspace(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-device amax and CTA counter of the forward kernel; its last CTA zeroes them again."""
    key = str(device)
    ws = _workspaces.get(key)
    if ws is None:
        ws = (
            torch.zeros(1, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _workspaces[key] = ws
    return ws


def _rope_fwd(
    rt: SimpleNamespace,
    qkv: torch.Tensor,
    bias: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_groups: int,
    q_heads_per_group: int,
    quantizer: Float8CurrentScalingQuantizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (flat q | k | v bf16 [s*b*hidden], amax [1], scale [1])."""
    s, b, hidden = qkv.shape
    m = s * b
    device = qkv.device
    flat = torch.empty(m * hidden, dtype=qkv.dtype, device=device)
    amax = torch.empty(1, dtype=torch.float32, device=device)
    scale = torch.empty(1, dtype=torch.float32, device=device)
    ws_amax, ws_count = _workspace(device)
    nblocks = m // rt.kernel.FWD_TOKENS_PER_BLOCK
    stream = _stream(rt)
    args = (
        _vec(rt, qkv),
        _vec(rt, qkv if bias is None else bias),
        _vec(rt, flat),
        _vec(rt, cos),
        _vec(rt, sin),
        _vec(rt, scale, 4),
        _vec(rt, amax, 4),
        _vec(rt, ws_amax, 4),
        _vec(rt, ws_count, 4),
        rt.Int32(b),
        rt.Int32(m),
        rt.Int32(nblocks),
    )
    # Constexpr parameters are baked in at compile time: cute.compile takes them, the compiled
    # function takes the runtime arguments and the stream alone
    consts = (
        num_groups,
        q_heads_per_group,
        m * num_groups * q_heads_per_group * _HEAD_DIM,
        m * num_groups * _HEAD_DIM,
        bias is not None,
        float(torch.finfo(TE_DType_To_Torch[quantizer.dtype]).max),
        float(quantizer.amax_epsilon),
        bool(quantizer.force_pow_2_scales),
    )
    key = ("fwd", s, b, *consts, str(device))
    fn = _compiled.get(key)
    if fn is None:
        fn = rt.cute.compile(rt.kernel.rope_fwd, *args, *consts, stream)
        _compiled[key] = fn
    fn(*args, stream)
    return flat, amax, scale


def _rope_bwd(
    rt: SimpleNamespace,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_groups: int,
    q_heads_per_group: int,
    quantizer: MXFP8Quantizer,
) -> Tuple[torch.Tensor, ...]:
    """-> MXFP8 d(qkv) as (rowwise data [m, N] u8, rowwise scales [m, N/32] u8, columnwise data
    [m, N] u8, columnwise scales [m/32, N] u8, partial column sums [m/32, N] fp32)."""
    s, b = dq.shape[:2]
    m = s * b
    hidden = num_groups * (q_heads_per_group + 2) * _HEAD_DIM
    device = dq.device
    fp8_dtype = TE_DType_To_Torch[quantizer.dtype]
    row_data = torch.empty(m, hidden, dtype=torch.uint8, device=device)
    col_data = torch.empty(m, hidden, dtype=torch.uint8, device=device)
    row_scale = torch.empty(m, hidden // 32, dtype=torch.uint8, device=device)
    col_scale = torch.empty(m // 32, hidden, dtype=torch.uint8, device=device)
    partial = torch.empty(m // 32, hidden, dtype=torch.float32, device=device)
    swizzled = bool(quantizer.optimize_for_gemm)
    # TE's 1.0f / max_norm, rounded to fp32
    max_norm_rcp = torch.tensor(1.0 / torch.finfo(fp8_dtype).max, dtype=torch.float32).item()
    stream = _stream(rt)
    common = (
        _vec(rt, cos),
        _vec(rt, sin),
        _vec_as(rt, row_data.view(fp8_dtype), rt.fp8_dtypes[quantizer.dtype]),
        _vec_as(rt, row_scale, rt.E8M0, 1),
        _vec_as(rt, col_data.view(fp8_dtype), rt.fp8_dtypes[quantizer.dtype]),
        _vec_as(rt, col_scale, rt.E8M0, 1),
        _vec(rt, partial, 4),
        rt.Int32(b),
        rt.Int32(m),
        rt.Int32(m // rt.kernel.BWD_ROWS),
    )
    q_heads_per_cta = rt.kernel.BWD_Q_HEADS_PER_CTA
    for name, args, consts in (
        (
            "bwd_q",
            (_vec(rt, dq), _vec(rt, dq), *common),
            (
                num_groups,
                q_heads_per_group,
                False,
                q_heads_per_cta,
                q_heads_per_group // q_heads_per_cta,
            ),
        ),
        (
            "bwd_kv",
            (_vec(rt, dk), _vec(rt, dv), *common),
            (num_groups, q_heads_per_group, True, 2, 1),
        ),
    ):
        consts = (*consts, swizzled, max_norm_rcp)
        key = (name, s, b, *consts, str(device))
        fn = _compiled.get(key)
        if fn is None:
            fn = rt.cute.compile(rt.kernel.rope_bwd, *args, *consts, stream)
            _compiled[key] = fn
        fn(*args, stream)
    return row_data, row_scale, col_data, col_scale, partial


def _dbias_reduce(rt: SimpleNamespace, partial: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    nrows, ncols = partial.shape
    out = torch.empty(ncols, dtype=dtype, device=partial.device)
    nblocks = (ncols + rt.kernel.RED_COLS - 1) // rt.kernel.RED_COLS
    args = (
        _vec(rt, partial, 4),
        _vec(rt, out, 2),
        rt.Int32(nrows),
        rt.Int32(ncols),
        rt.Int32(nblocks),
        _stream(rt),
    )
    key = ("dbias", nrows, ncols, str(dtype), str(partial.device))
    fn = _compiled.get(key)
    if fn is None:
        fn = rt.cute.compile(rt.kernel.dbias_reduce, *args)
        _compiled[key] = fn
    fn(*args)
    return out


class FusedQKVRoPEQuantizeFunc(torch.autograd.Function):
    """Fused QKV RoPE returning FP8 q/k/v for DotProductAttention and the MXFP8 gradient of qkv
    for the linear that produced it. Call it through ``apply_fused_qkv_rotary_pos_emb`` after
    checking :func:`fused_qkv_rope_quantize_supported`."""

    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        bias: Optional[torch.Tensor],
        freqs: torch.Tensor,
        qkv_split_arg_list: List[int],
        qkv_quantizer: Float8CurrentScalingQuantizer,
        grad_qkv_quantizer: MXFP8Quantizer,
    ) -> Tuple[Float8Tensor, Float8Tensor, Float8Tensor]:
        """qkv [s, b, h*(dq+dk+dv)] bf16 (+ bias [h*(dq+dk+dv)]) -> q [s, b, h*dq/64, 64],
        k and v [s, b, h, 64] as Float8Tensors (views of one buffer, one scale)."""
        rt = _runtime()
        s, b, hidden = qkv.shape
        dq, dk, dv = (int(x) for x in qkv_split_arg_list)
        num_groups = hidden // (dq + dk + dv)
        q_heads_per_group = dq // _HEAD_DIM
        freqs = freqs[:s].reshape(s, _HEAD_DIM).to(torch.float32)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        flat, amax, scale = _rope_fwd(
            rt, qkv, bias, cos, sin, num_groups, q_heads_per_group, qkv_quantizer
        )
        # TE's per-tensor cast with the scale from the RoPE pass: data = sat(x * scale),
        # scale_inv = 1 / scale
        out = Float8Tensor(
            shape=flat.shape,
            dtype=qkv.dtype,
            data=torch.empty_like(flat, dtype=torch.uint8),
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=qkv.device),
            fp8_dtype=qkv_quantizer.dtype,
            quantizer=qkv_quantizer,
            requires_grad=False,
            device=qkv.device,
        )
        tex.quantize(
            flat,
            Float8Quantizer(scale, amax, qkv_quantizer.dtype, rowwise=True, columnwise=False),
            out,
            None,
        )
        shapes = (
            (s, b, num_groups * q_heads_per_group, _HEAD_DIM),
            (s, b, num_groups, _HEAD_DIM),
            (s, b, num_groups, _HEAD_DIM),
        )
        outs, offset = [], 0
        for shape in shapes:
            numel = math.prod(shape)
            data = out._data[offset : offset + numel].view(shape)
            outs.append(Float8Tensor.make_like(out, data=data, dtype=qkv.dtype))
            offset += numel
        q, k, v = outs
        ctx.save_for_backward(cos, sin)
        ctx.qkv_shape = (s, b, hidden)
        ctx.num_groups = num_groups
        ctx.q_heads_per_group = q_heads_per_group
        ctx.has_bias = bias is not None
        ctx.dtype = qkv.dtype
        ctx.grad_qkv_quantizer = grad_qkv_quantizer
        return q, k, v

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor):
        """dq/dk/dv (bf16 from attention) -> (MXFP8Tensor d(qkv) [s, b, h*(dq+dk+dv)], d(bias))."""
        rt = _runtime()
        cos, sin = ctx.saved_tensors
        grads = []
        for grad in (dq, dk, dv):
            if isinstance(grad, QuantizedTensorStorage):
                grad = grad.dequantize(dtype=ctx.dtype)
            grads.append(grad.contiguous())
        quantizer = ctx.grad_qkv_quantizer
        row_data, row_scale, col_data, col_scale, partial = _rope_bwd(
            rt, *grads, cos, sin, ctx.num_groups, ctx.q_heads_per_group, quantizer
        )
        grad_qkv = MXFP8Tensor(
            shape=ctx.qkv_shape,
            dtype=ctx.dtype,
            rowwise_data=row_data.view(ctx.qkv_shape),
            rowwise_scale_inv=row_scale,
            columnwise_data=col_data.view(ctx.qkv_shape),
            columnwise_scale_inv=col_scale,
            fp8_dtype=quantizer.dtype,
            quantizer=quantizer,
            with_gemm_swizzled_scales=bool(quantizer.optimize_for_gemm),
            requires_grad=False,
            device=dq.device,
        )
        grad_bias = _dbias_reduce(rt, partial, ctx.dtype) if ctx.has_bias else None
        return grad_qkv, grad_bias, None, None, None, None
