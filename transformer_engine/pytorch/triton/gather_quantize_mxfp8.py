# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the gather + MXFP8 group quantize Triton kernel."""

import math

import torch
import triton

import transformer_engine_torch as tex
from transformer_engine.common.triton.gather_quantize_mxfp8 import _gather_quantize_mxfp8_kernel

from ..constants import DType, MXFP8_BLOCK_SCALING_SIZE
from ..tensor.grouped_tensor import GroupedTensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..utils import round_up_to_nearest_multiple

_BLOCK_M = 32
_BLOCK_K = 256
_NUM_WARPS = 4


def gather_quantize_mxfp8(
    src: torch.Tensor,
    row_to_token: torch.Tensor,
    split_sizes: torch.Tensor,
    quantizer: MXFP8Quantizer,
    *,
    keep_high_precision: bool = False,
) -> GroupedTensor:
    """Gather rows of ``src`` and quantize them to an MXFP8 ``GroupedTensor`` in one pass.

    Returns what ``tex.group_quantize(src[row_to_token], quantizer, num_groups, split_sizes)``
    returns, bit for bit, without materializing the gathered rows: rowwise and columnwise E4M3
    data with GEMM-swizzled E8M0 scales, ready for the grouped GEMMs.

    Parameters
    ----------
    src : torch.Tensor
        Rows to gather from, ``(num_tokens, hidden)`` in BF16 or FP16, ``hidden % 32 == 0``.
    row_to_token : torch.Tensor
        Source row of every output row, int32 ``(num_rows,)`` with ``num_rows % 128 == 0``; a
        negative entry is a padding row and quantizes to zeros.
    split_sizes : torch.Tensor
        Rows of every group, int64 on the device, each a multiple of 128 and summing to at most
        ``num_rows``. Rows past the sum are left unwritten.
    quantizer : MXFP8Quantizer
        E4M3, 1D block scaling, rowwise and columnwise usage, ``optimize_for_gemm``.
    keep_high_precision : bool, default = False
        Also store the gathered rows as ``high_precision_data``, for consumers that need them
        besides the quantized data (bias gradients).
    """
    if not isinstance(quantizer, MXFP8Quantizer) or quantizer.dtype != DType.kFloat8E4M3:
        raise ValueError("gather_quantize_mxfp8 requires an E4M3 MXFP8Quantizer.")
    if quantizer.with_2d_quantization or not quantizer.optimize_for_gemm:
        raise ValueError(
            "gather_quantize_mxfp8 requires 1D block scaling with GEMM-swizzled scales."
        )
    if not (quantizer.rowwise_usage and quantizer.columnwise_usage):
        raise ValueError("gather_quantize_mxfp8 produces rowwise and columnwise data.")
    if (
        src.dim() != 2
        or src.dtype not in (torch.bfloat16, torch.float16)
        or not src.is_contiguous()
    ):
        raise ValueError("src must be a contiguous 2D BF16 or FP16 tensor.")
    if (
        row_to_token.dim() != 1
        or row_to_token.dtype != torch.int32
        or not row_to_token.is_contiguous()
    ):
        raise ValueError("row_to_token must be a contiguous int32 vector.")
    if split_sizes.dim() != 1 or split_sizes.dtype != torch.int64 or not split_sizes.is_cuda:
        raise ValueError("split_sizes must be an int64 device vector.")
    num_rows = row_to_token.numel()
    hidden = src.size(1)
    if num_rows % 128 != 0 or hidden % MXFP8_BLOCK_SCALING_SIZE != 0:
        raise ValueError(
            "gather_quantize_mxfp8 requires num_rows % 128 == 0 and hidden % 32 == 0, got"
            f" ({num_rows}, {hidden})."
        )

    num_groups = split_sizes.numel()
    shape = (num_rows, hidden)
    device = src.device
    rowwise_data = torch.empty(num_rows * hidden, dtype=torch.uint8, device=device)
    columnwise_data = torch.empty(num_rows * hidden, dtype=torch.uint8, device=device)
    rowwise_scale_inv = torch.empty(
        math.prod(quantizer.get_scale_shape(shape, False)), dtype=torch.uint8, device=device
    )
    columnwise_scale_inv = torch.empty(
        math.prod(quantizer.get_scale_shape(shape, True)), dtype=torch.uint8, device=device
    )
    high_precision_data = (
        torch.empty(num_rows * hidden, dtype=src.dtype, device=device)
        if keep_high_precision
        else None
    )
    if num_rows > 0:
        hidden_padded = round_up_to_nearest_multiple(hidden, 128)
        _gather_quantize_mxfp8_kernel[(triton.cdiv(hidden, _BLOCK_K), num_rows // _BLOCK_M)](
            src,
            row_to_token,
            split_sizes,
            rowwise_data.view(torch.float8_e4m3fn),
            rowwise_scale_inv,
            columnwise_data.view(torch.float8_e4m3fn),
            columnwise_scale_inv,
            high_precision_data if keep_high_precision else src,
            hidden,
            hidden_padded,
            hidden_padded // 128,
            num_groups,
            FP8_MAX_RCP=1.0 / 448.0,
            NUM_GROUPS_POW2=triton.next_power_of_2(num_groups),
            WRITE_HIGH_PRECISION=keep_high_precision,
            BLOCK_M=_BLOCK_M,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )
    return GroupedTensor(
        shape=shape,
        dtype=src.dtype,
        quantizer=quantizer,
        num_tensors=num_groups,
        data=rowwise_data,
        columnwise_data=columnwise_data,
        scale_inv=rowwise_scale_inv,
        columnwise_scale_inv=columnwise_scale_inv,
        first_dims=split_sizes,
        tensor_offsets=tex.splits_to_offsets(split_sizes, hidden),
        with_gemm_swizzled_scales=True,
        high_precision_data=high_precision_data,
    )
