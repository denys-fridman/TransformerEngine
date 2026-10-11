# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the Triton RMSNorm backward kernel.

``NVTE_RMSNORM_BWD_USE_TRITON=1`` routes the RMSNorm backward calls that :func:`is_supported`
accepts to one streaming kernel that computes the input gradient, adds a residual gradient to it,
emits it as an MXFP8 tensor when the consuming operation takes one, and reduces the weight
gradient from per-block partial sums; every other call stays on the C++ extension.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from transformer_engine.common.triton.rmsnorm_bwd import column_sum_kernel, rmsnorm_bwd_kernel
from ..constants import MXFP8_BLOCK_SCALING_SIZE, TE_DType_To_Torch
from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer

_BLOCK_N = 256
_NUM_WARPS = 4
_TL_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def is_enabled() -> bool:
    """Whether ``NVTE_RMSNORM_BWD_USE_TRITON=1`` selects the Triton kernel."""
    return os.environ.get("NVTE_RMSNORM_BWD_USE_TRITON", "0") == "1"


def is_supported(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    *,
    add: Optional[torch.Tensor] = None,
    quantizer: Optional[Quantizer] = None,
) -> bool:
    """Whether :func:`rmsnorm_bwd` handles this call.

    The kernel is enabled, the tensors are contiguous CUDA tensors of one FP32, FP16 or BF16
    dtype with x, dy and add of shape [M, N], weight of N elements and rstd of M FP32 elements,
    and the output quantizer, if any, is a 1D MXFP8 quantizer with M and N multiples of 32.
    """
    if not is_enabled():
        return False
    if x.dim() != 2 or x.dtype not in _TL_DTYPE or not x.is_cuda:
        return False
    tensors = (dy, x, weight) if add is None else (dy, x, weight, add)
    for t in tensors:
        if isinstance(t, QuantizedTensorStorage) or t.dtype != x.dtype or not t.is_contiguous():
            return False
    M, N = x.shape
    if dy.shape != x.shape or (add is not None and add.shape != x.shape) or weight.numel() != N:
        return False
    if rstd.dtype != torch.float32 or rstd.numel() != M or not rstd.is_contiguous():
        return False
    if quantizer is not None:
        if not isinstance(quantizer, MXFP8Quantizer) or quantizer.with_2d_quantization:
            return False
        if M % MXFP8_BLOCK_SCALING_SIZE != 0 or N % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
    return True


def rmsnorm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    *,
    zero_centered_gamma: bool,
    add: Optional[torch.Tensor] = None,
    quantizer: Optional[MXFP8Quantizer] = None,
    out_shape: Optional[torch.Size] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm backward pass in one streaming kernel.

    Returns the input gradient (plus ``add``) with shape ``out_shape`` (default: x's), either in
    x's dtype or, with ``quantizer``, as the MXFP8 tensor it builds, with both the rowwise and
    the columnwise copies, and the weight gradient in the weight's dtype.
    """
    M, N = x.shape
    if out_shape is None:
        out_shape = x.shape
    rows = triton.cdiv(M, MXFP8_BLOCK_SCALING_SIZE)
    dw_part = torch.empty((rows, N), dtype=torch.float32, device=x.device)

    if quantizer is None:
        dx = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        fp8_max_rcp = 0.0
        rowwise_data = rowwise_scale = columnwise_data = columnwise_scale = dw_part
        swizzled = False
    else:
        # The consumer chooses the copies it needs only in its own backward pass
        quantizer = quantizer.copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        dx = quantizer.make_empty(out_shape, dtype=x.dtype, device=x.device)
        fp8_dtype = TE_DType_To_Torch[quantizer.dtype]
        fp8_max_rcp = 1.0 / torch.finfo(fp8_dtype).max
        rowwise_data = dx._rowwise_data.view(fp8_dtype)
        columnwise_data = dx._columnwise_data.view(fp8_dtype)
        rowwise_scale = dx._rowwise_scale_inv
        columnwise_scale = dx._columnwise_scale_inv
        swizzled = dx._with_gemm_swizzled_scales
        if swizzled and (M % 128 != 0 or N % 128 != 0):
            # Padding is interleaved with the scales in the GEMM layout
            rowwise_scale.zero_()
            columnwise_scale.zero_()

    rmsnorm_bwd_kernel[(rows,)](
        dy,
        x,
        rstd,
        weight,
        dy if add is None else add,
        dx if quantizer is None else dw_part,
        dw_part,
        rowwise_data,
        rowwise_scale,
        columnwise_data,
        columnwise_scale,
        M,
        N,
        rowwise_scale.shape[1] if quantizer is not None else 0,
        columnwise_scale.shape[0] if quantizer is not None else 0,
        columnwise_scale.shape[1] if quantizer is not None else 0,
        ZERO_CENTERED_GAMMA=zero_centered_gamma,
        HAS_ADD=add is not None,
        STORE_DX=quantizer is None,
        ROWWISE=quantizer is not None,
        COLUMNWISE=quantizer is not None,
        SWIZZLED_SCALES=swizzled,
        OUT_DTYPE=_TL_DTYPE[x.dtype],
        FP8_MAX_RCP=fp8_max_rcp,
        ROWS=MXFP8_BLOCK_SCALING_SIZE,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    dw = torch.empty(N, dtype=weight.dtype, device=x.device)
    column_sum_kernel[(triton.cdiv(N, 32),)](
        dw_part, dw, rows, N, BLOCK_R=32, BLOCK_C=32, num_warps=4
    )
    return dx, dw
