# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the fused RMSNorm backward kernels.

``NVTE_RMSNORM_BWD_USE_TRITON=1`` routes the RMSNorm backward calls that :func:`is_supported`
accepts to one streaming kernel that computes the input gradient, adds a residual gradient to it,
emits it as an MXFP8 tensor when the consuming operation takes one, and reduces the weight
gradient; every other call stays on the C++ extension. On Blackwell and later the kernel is the
CUDA one (``tex.rmsnorm_bwd_fused``, which reads its inputs once and reduces the weight gradient
in the kernel), elsewhere the Triton one; ``NVTE_RMSNORM_BWD_BACKEND=cuda|triton`` forces either.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

import transformer_engine_torch as tex
from transformer_engine.common.triton.rmsnorm_bwd import column_sum_kernel, rmsnorm_bwd_kernel
from ..constants import MXFP8_BLOCK_SCALING_SIZE, TE_DType_To_Torch
from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer

_BLOCK_N = 256
_NUM_WARPS = 4
_TL_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}

_CUDA_MAX_N = 3072
_CUDA_COUNTER_SIZE = 73  # of nvte_rmsnorm_bwd_fused
_counters: Dict[Tuple[int, int], torch.Tensor] = {}


def is_enabled() -> bool:
    """Whether ``NVTE_RMSNORM_BWD_USE_TRITON=1`` selects the fused kernel."""
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


def _use_cuda_kernel(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, add: Optional[torch.Tensor]
) -> bool:
    """CUDA kernel on Blackwell and later, Triton elsewhere, unless the backend is forced.

    The CUDA kernel takes 16-byte aligned FP16 or BF16 tensors with M and N multiples of 32 and
    N up to 3072.
    """
    backend = os.getenv("NVTE_RMSNORM_BWD_BACKEND", "auto").lower()
    if backend not in ("auto", "cuda", "triton"):
        raise ValueError(f"NVTE_RMSNORM_BWD_BACKEND must be auto, cuda or triton, got {backend!r}.")
    if backend == "triton":
        return False
    if backend == "auto" and torch.cuda.get_device_capability(x.device)[0] < 10:
        return False
    M, N = x.shape
    if x.dtype == torch.float32 or M % 32 != 0 or N % 32 != 0 or N > _CUDA_MAX_N:
        return False
    tensors = (dy, x, weight) if add is None else (dy, x, weight, add)
    return all(t.data_ptr() % 16 == 0 for t in tensors)


def _counter(device: torch.device) -> torch.Tensor:
    """The CUDA kernel's ticket counter, one per stream so that stream order keeps it zero
    between launches."""
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    counter = _counters.get(key)
    if counter is None:
        counter = torch.zeros(_CUDA_COUNTER_SIZE, dtype=torch.int32, device=device)
        _counters[key] = counter
    return counter


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
    the columnwise copies and the values in x's dtype kept as its high-precision data, and the
    weight gradient in the weight's dtype.
    """
    M, N = x.shape
    if out_shape is None:
        out_shape = x.shape
    dx_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if quantizer is None:
        dx = dx_out
    else:
        # The consumer chooses the copies it needs only in its own backward pass
        quantizer = quantizer.copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        dx = quantizer.make_empty(out_shape, dtype=x.dtype, device=x.device)
        # For consumers that need the values before quantization (bias gradients,
        # accumulation with other gradients)
        dx._high_precision_data = dx_out
    dw = torch.empty(N, dtype=weight.dtype, device=x.device)

    # The CUDA kernel writes dx in one format only
    if quantizer is None and _use_cuda_kernel(dy, x, weight, add):
        tex.rmsnorm_bwd_fused(
            dy,
            x,
            rstd,
            weight,
            add,
            dx,
            None,
            dw,
            None,
            _counter(x.device),
            zero_centered_gamma,
        )
        return dx, dw

    rows = triton.cdiv(M, MXFP8_BLOCK_SCALING_SIZE)
    dw_part = torch.empty((rows, N), dtype=torch.float32, device=x.device)
    if quantizer is None:
        fp8_max_rcp = 0.0
        rowwise_data = rowwise_scale = columnwise_data = columnwise_scale = dw_part
        swizzled = False
    else:
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
        dx_out,
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
        ROWWISE=quantizer is not None,
        COLUMNWISE=quantizer is not None,
        SWIZZLED_SCALES=swizzled,
        OUT_DTYPE=_TL_DTYPE[x.dtype],
        FP8_MAX_RCP=fp8_max_rcp,
        ROWS=MXFP8_BLOCK_SCALING_SIZE,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    column_sum_kernel[(triton.cdiv(N, 32),)](
        dw_part, dw, rows, N, BLOCK_R=32, BLOCK_C=32, num_warps=4
    )
    return dx, dw
