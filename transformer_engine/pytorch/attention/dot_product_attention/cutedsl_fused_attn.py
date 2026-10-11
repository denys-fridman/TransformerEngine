# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL fused FP8 attention backward: eligibility check and PyTorch launch glue.

``NVTE_CUTEDSL_FP8_ATTN_BWD=1`` routes the eligible FP8 current-scaling backward calls of
DotProductAttention (see :func:`fp8_bwd_supported`) to the kernel in
``transformer_engine/common/CuTeDSL/attention/fused_attn_bwd_fp8.py``; every other call stays
on cuDNN.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from transformer_engine.pytorch.constants import DType, TE_DType_To_Torch
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
from transformer_engine.pytorch.utils import get_device_compute_capability

_HEAD_DIM = 64
_FP8_DTYPES = (DType.kFloat8E4M3, DType.kFloat8E5M2)
_compiled: Dict[tuple, Any] = {}
_workspaces: Dict[tuple, torch.Tensor] = {}


@functools.lru_cache(maxsize=None)
def _runtime() -> Optional[SimpleNamespace]:
    """The CuTeDSL runtime and the kernel module, or None when nvidia-cutlass-dsl is missing."""
    try:
        # pylint: disable=import-outside-toplevel
        import cuda.bindings.driver as cuda_driver  # pylint: disable=no-name-in-module
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import BFloat16, Float32, Float8E4M3FN, Float8E5M2, Int32

        from transformer_engine.common.CuTeDSL.attention import fused_attn_bwd_fp8 as kernel
    except ImportError as e:
        logging.getLogger("DotProductAttention").warning(
            "NVTE_CUTEDSL_FP8_ATTN_BWD=1 but the CuTeDSL attention kernel cannot be imported"
            " (%s); using cuDNN",
            e,
        )
        return None
    return SimpleNamespace(
        cuda=cuda_driver,
        cute=cute,
        from_dlpack=from_dlpack,
        kernel=kernel,
        fp8_dtypes={DType.kFloat8E4M3: Float8E4M3FN, DType.kFloat8E5M2: Float8E5M2},
        BFloat16=BFloat16,
        Float32=Float32,
        Int32=Int32,
    )


def fp8_bwd_supported(args: Any, out: torch.Tensor, d_out: Any) -> bool:
    """Whether the FP8 fused attention backward described by ``args`` (a ``FusedAttnBwdArgs``)
    runs on the CuTeDSL kernel: ``NVTE_CUTEDSL_FP8_ATTN_BWD=1``, a compute capability 10.x device,
    e4m3 Q/K/V with per-tensor scales and a bf16 O (Float8CurrentScaling with
    ``NVTE_DPA_FP8CS_O_in_F16=1``), delayed-scaling S/dP quantizers, head_dim 64, same-length
    ``sbhd`` Q/K/V, a causal or causal + sliding-window mask, no bias, no dropout, a vanilla or
    learnable softmax and non-deterministic mode (dQ is accumulated with atomics)."""
    if os.environ.get("NVTE_CUTEDSL_FP8_ATTN_BWD", "0") != "1":
        return False
    if FusedAttnBackend.cast(args.fused_attention_backend) != FusedAttnBackend["FP8"]:
        return False
    q, k, v = args.q_fp8, args.k_fp8, args.v_fp8
    if not all(isinstance(t, Float8TensorStorage) for t in (q, k, v, d_out)):
        return False
    if any(t._fp8_dtype != DType.kFloat8E4M3 for t in (q, k, v)):
        return False
    if d_out._fp8_dtype not in _FP8_DTYPES:
        return False
    if isinstance(out, QuantizedTensorStorage) or out.dtype != torch.bfloat16:
        return False
    if not isinstance(args.S_quantizer, Float8Quantizer):
        return False
    if args.S_quantizer.dtype != DType.kFloat8E4M3:
        return False
    if not isinstance(args.dP_quantizer, Float8Quantizer):
        return False
    if args.dP_quantizer.dtype not in _FP8_DTYPES:
        return False
    # the kernel writes bf16 dQ/dK/dV, as cuDNN does for a current-scaling dQKV quantizer
    if not isinstance(args.dQKV_quantizer, Float8CurrentScalingQuantizer):
        return False
    if args.qkv_layout != "sbhd_sbhd_sbhd" or args.dqkv_layout != "sbhd_sbhd_sbhd":
        return False
    if args.o_format != "sbhd" or args.qkv_scale_inv_format is not None:
        return False
    if args.cu_seqlens_q_padded is not None or args.cu_seqlens_kv_padded is not None:
        return False
    if args.attn_bias_type != "no_bias" or args.dropout_p != 0.0 or args.deterministic:
        return False
    window_left, window_right = args.window_size
    if args.attn_mask_type != "causal" or window_right not in (-1, 0) or window_left < -1:
        return False
    if args.softmax_type not in ("vanilla", "learnable"):
        return False
    s, b, h, d = q.shape
    s_k, b_k, h_k, d_k = k.shape
    if (d, d_k) != (_HEAD_DIM, _HEAD_DIM) or v.shape != k.shape:
        return False
    if (s, b) != (s_k, b_k) or h % h_k != 0:
        return False
    if args.max_seqlen_q != s or args.max_seqlen_kv != s:
        return False
    if get_device_compute_capability()[0] != 10:
        return False
    return _runtime() is not None


def _fp8_view(data: torch.Tensor, fp8_dtype: DType) -> torch.Tensor:
    torch_dtype = TE_DType_To_Torch[fp8_dtype]
    return data if data.dtype == torch_dtype else data.view(torch_dtype)


def _q_like(rt: SimpleNamespace, t: torch.Tensor, h_r: int, h_k: int, cute_dtype: Any):
    """[s, b, h, d] (h = h_k*h_r, head h = kv_head*h_r + r) -> cute (s, d, h_r, h_k, b)."""
    s, b, _, d = t.shape
    view = t.detach().view(s, b, h_k, h_r, d).permute(0, 4, 3, 2, 1)
    ct = rt.from_dlpack(view, assumed_align=16)
    ct.element_type = cute_dtype
    return ct.mark_layout_dynamic(leading_dim=1)


def _kv_like(rt: SimpleNamespace, t: torch.Tensor, cute_dtype: Any):
    """[s, b, h_k, d] -> cute (s, d, 1, h_k, b)."""
    s, b, h_k, d = t.shape
    view = t.detach().view(s, b, h_k, 1, d).permute(0, 4, 3, 2, 1)
    ct = rt.from_dlpack(view, assumed_align=16)
    ct.element_type = cute_dtype
    return ct.mark_layout_dynamic(leading_dim=1)


def _lse(rt: SimpleNamespace, stats: torch.Tensor, h_r: int, h_k: int):
    """[b, h, s, 1] fp32 softmax stats -> cute (s, h_r, h_k, b)."""
    b, _, s, _ = stats.shape
    view = stats.detach().view(b, h_k, h_r, s).permute(3, 2, 1, 0)
    return rt.from_dlpack(view, assumed_align=4).mark_layout_dynamic(leading_dim=0)


def _vec(rt: SimpleNamespace, t: torch.Tensor, align: int = 4):
    return rt.from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic()


def _workspace(rt: SimpleNamespace, s: int, h: int, b: int, d: int, device: torch.device):
    key = (s, h, b, d, str(device))
    ws = _workspaces.get(key)
    if ws is None:
        nbytes = rt.kernel.FusedSdpaBwdFp8._get_workspace_size(s, s, d, h, b, rt.Float32)
        # the dQ accumulator region starts zeroed; the kernel re-zeroes it after reading
        ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        _workspaces[key] = ws
    return ws


def fp8_bwd(
    args: Any, out: torch.Tensor, d_out: Any, softmax_stats: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the CuTeDSL FP8 attention backward for a call that passed :func:`fp8_bwd_supported`.

    Returns ``(dq, dk, dv, d_softmax_offset)`` (bf16, bf16, bf16, fp32 ``[1, h, 1, 1]`` or None)
    and writes amax(dS*attn_scale) into the dP quantizer, as the cuDNN path does."""
    rt = _runtime()
    q, k, v = args.q_fp8, args.k_fp8, args.v_fp8
    s, b, h, d = q.shape
    h_k = k.shape[2]
    h_r = h // h_k
    attn_scale = args.attn_scale if args.attn_scale is not None else 1.0 / math.sqrt(d)
    window_left = int(args.window_size[0])
    has_sink = args.softmax_type != "vanilla"
    do_dtype, ds_dtype = d_out._fp8_dtype, args.dP_quantizer.dtype
    device = q.device

    dq = torch.empty((s, b, h, d), dtype=torch.bfloat16, device=device)
    dk = torch.empty(k.shape, dtype=torch.bfloat16, device=device)
    dv = torch.empty(v.shape, dtype=torch.bfloat16, device=device)
    amax = torch.zeros(1, dtype=torch.float32, device=device)
    dsink = torch.zeros(h, dtype=torch.float32, device=device) if has_sink else None
    sink = None
    if has_sink:
        sink = args.aux_softmax_offset.detach().reshape(h).to(torch.float32).contiguous()
    scale_s = args.S_quantizer.scale.reshape(1).to(torch.float32)
    scale_dp = args.dP_quantizer.scale.reshape(1).to(torch.float32)
    # scales[6] = log2(scale_s) is folded into the pre-pass LSE and scales[7] =
    # scale_dP*attn_scale/scale_s into the pre-pass -D and the dP descale (FusedSdpaBwdFp8)
    scales = torch.cat(
        [
            q._scale_inv.reshape(1),
            k._scale_inv.reshape(1),
            v._scale_inv.reshape(1),
            d_out._scale_inv.reshape(1),
            scale_s,
            scale_dp,
            torch.log2(scale_s),
            scale_dp * (attn_scale / scale_s),
        ]
    ).to(torch.float32)

    # the K=64 FP8 MMA exists on sm_107 only (cutlass.utils.rubin_helpers); other tcgen05 parts
    # take the K=32 instruction through cutlass.utils.blackwell_helpers
    rubin = get_device_compute_capability() == (10, 7) and rt.kernel.sm107_utils is not None
    mma_k = 64 if rubin else 32
    stream = rt.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    e4m3 = rt.fp8_dtypes[DType.kFloat8E4M3]
    kernel_args = (
        (s, s, d, ((h_r, h_k), b)),
        _q_like(rt, _fp8_view(q._data, DType.kFloat8E4M3), h_r, h_k, e4m3),
        _kv_like(rt, _fp8_view(k._data, DType.kFloat8E4M3), e4m3),
        _kv_like(rt, _fp8_view(v._data, DType.kFloat8E4M3), e4m3),
        _q_like(rt, out, h_r, h_k, rt.BFloat16),
        _q_like(rt, dq, h_r, h_k, rt.BFloat16),
        _kv_like(rt, dk, rt.BFloat16),
        _kv_like(rt, dv, rt.BFloat16),
        _q_like(rt, _fp8_view(d_out._data, do_dtype), h_r, h_k, rt.fp8_dtypes[do_dtype]),
        _lse(rt, softmax_stats, h_r, h_k),
        _vec(rt, scales),
        _vec(rt, sink) if has_sink else None,
        _vec(rt, dsink) if has_sink else None,
        _vec(rt, amax),
        None,
        None,
        rt.Float32(attn_scale),
        rt.Int32(window_left) if window_left >= 0 else None,
        rt.Int32(0),  # causal: no keys to the right of the diagonal
        _vec(rt, _workspace(rt, s, h, b, d, device), align=16),
        stream,
    )
    key = (s, b, h, h_k, d, window_left, has_sink, do_dtype, ds_dtype, mma_k)
    fn = _compiled.get(key)
    if fn is None:
        kernel = rt.kernel.FusedSdpaBwdFp8(
            e4m3,
            rt.fp8_dtypes[do_dtype],
            e4m3,
            rt.fp8_dtypes[ds_dtype],
            rt.BFloat16,
            rt.Float32,
            (128, 128, d),
            False,
            rt.kernel.fmha_utils.MaskEnum.WINDOW_MASK_BWD,
            has_sink,
            mma_k=mma_k,
        )
        fn = rt.cute.compile(kernel, *kernel_args)
        _compiled[key] = fn
    fn(*kernel_args)
    args.dP_quantizer.amax.view(-1)[:1].copy_(amax)
    return dq, dk, dv, dsink.view(1, h, 1, 1) if has_sink else None
