# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the gather + MXFP8 group quantize Triton kernel and its use as a pre-quantized
input / grad output of the fused grouped MLP."""

from __future__ import annotations

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.triton.gather_quantize_mxfp8 import gather_quantize_mxfp8

from utils import make_recipe, reset_rng_states

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)


@pytest.fixture(autouse=True)
def _reset_rng_states():
    reset_rng_states()


class _InjectGrad(torch.autograd.Function):
    """Replace the gradient flowing into ``x`` with ``grad``, as a dispatcher's backward would."""

    @staticmethod
    def forward(ctx, x, grad):  # pylint: disable=arguments-differ
        ctx.injected_grad = grad
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return ctx.injected_grad, None


def _make_quantizer() -> MXFP8Quantizer:
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = True
    return quantizer


def _make_gather(num_tokens, split_sizes, num_rows, pad_fraction=0.1):
    """Random source rows for every output row (-1 marks padding rows)."""
    row_to_token = torch.randint(0, num_tokens, (num_rows,), dtype=torch.int32, device="cuda")
    row_to_token[torch.rand(num_rows, device="cuda") < pad_fraction] = -1
    split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device="cuda")
    return row_to_token, split_sizes


def _gathered_rows(src, row_to_token):
    rows = src[row_to_token.clamp(min=0).long()]
    return torch.where((row_to_token >= 0)[:, None], rows, torch.zeros_like(rows))


def _swizzled_idx(i, j, num_tiles_x):
    i_in_tile = i % 128
    return (
        ((i // 128) * num_tiles_x + j // 4) * 512
        + (i_in_tile % 32) * 16
        + (i_in_tile // 32) * 4
        + j % 4
    )


def _written_scale_entries(split_sizes, hidden):
    """Flat indices of the rowwise and columnwise scales of the rows in use."""
    device = split_sizes.device
    num_used = int(split_sizes.sum().item())
    hidden_padded = (hidden + 127) // 128 * 128
    num_tiles_x = hidden_padded // 128
    rows = torch.arange(num_used, device=device)[:, None]
    blocks = torch.arange(num_tiles_x * 4, device=device)[None, :]
    rowwise = _swizzled_idx(rows, blocks, num_tiles_x).reshape(-1)
    columnwise = []
    start = 0
    for group_rows in split_sizes.tolist():
        if group_rows > 0:
            cols = torch.arange(hidden_padded, device=device)[:, None]
            row_blocks = torch.arange(group_rows // 32, device=device)[None, :]
            columnwise.append(
                (start // 32) * hidden_padded
                + _swizzled_idx(cols, row_blocks, group_rows // 128).reshape(-1)
            )
        start += group_rows
    return rowwise, torch.cat(columnwise)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden", (128, 2880))
def test_matches_group_quantize(dtype: torch.dtype, hidden: int) -> None:
    """The gather + quantize equals tex.group_quantize of the gathered rows, bit for bit."""
    num_tokens, split_sizes, num_rows = 300, [256, 0, 512, 128, 384], 1536
    src = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda") * 3
    row_to_token, split_sizes = _make_gather(num_tokens, split_sizes, num_rows)
    num_used = int(split_sizes.sum().item())

    grouped = gather_quantize_mxfp8(
        src, row_to_token, split_sizes, _make_quantizer(), keep_high_precision=True
    )
    rows = _gathered_rows(src, row_to_token)
    ref = tex.group_quantize(
        rows,
        _make_quantizer(),
        split_sizes.numel(),
        split_sizes,
        tensor_offsets=tex.splits_to_offsets(split_sizes, hidden),
    )

    assert grouped._with_gemm_swizzled_scales and ref._with_gemm_swizzled_scales
    used = num_used * hidden
    torch.testing.assert_close(grouped.rowwise_data[:used], ref.rowwise_data[:used], rtol=0, atol=0)
    torch.testing.assert_close(
        grouped.columnwise_data[:used], ref.columnwise_data[:used], rtol=0, atol=0
    )
    rowwise_idx, columnwise_idx = _written_scale_entries(split_sizes, hidden)
    assert grouped.scale_inv.numel() == ref.scale_inv.numel()
    assert grouped.columnwise_scale_inv.numel() == ref.columnwise_scale_inv.numel()
    torch.testing.assert_close(
        grouped.scale_inv[rowwise_idx], ref.scale_inv[rowwise_idx], rtol=0, atol=0
    )
    torch.testing.assert_close(
        grouped.columnwise_scale_inv[columnwise_idx],
        ref.columnwise_scale_inv[columnwise_idx],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        grouped.high_precision_data.view(num_rows, hidden)[:num_used],
        rows[:num_used],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("bias", (False, True))
def test_fused_grouped_mlp_prequantized(bias: bool) -> None:
    """The fused grouped MLP fed the gathered rows and fed their gather + quantize agree bit for
    bit, in the forward (input) and in the backward (grad output with high-precision rows)."""
    fused_cls = te.ops.fused.GroupedMLP_CuTeGEMMGLU
    if not fused_cls.is_supported():
        pytest.skip("CuTe DSL fused grouped MLP is not supported on this system")
    dtype = torch.bfloat16
    group_size, hidden, num_tokens = 4, 256, 200
    split_sizes = [256 * i for i in range(group_size)]
    num_rows = sum(split_sizes)
    row_to_token, split_sizes = _make_gather(num_tokens, split_sizes, num_rows)
    recipe = make_recipe("mxfp8")

    x_src = torch.rand(num_tokens, hidden, dtype=dtype, device="cuda") - 0.5
    dy_src = torch.rand(num_tokens, hidden, dtype=dtype, device="cuda") - 0.5
    probs = torch.rand(num_rows, dtype=dtype, device="cuda")
    x_rows = _gathered_rows(x_src, row_to_token)
    dy_rows = _gathered_rows(dy_src, row_to_token)
    x_grouped = gather_quantize_mxfp8(x_src, row_to_token, split_sizes, _make_quantizer())
    dy_grouped = gather_quantize_mxfp8(
        dy_src, row_to_token, split_sizes, _make_quantizer(), keep_high_precision=True
    )

    with te.quantized_model_init(enabled=True, recipe=recipe):
        fc1 = te.ops.GroupedLinear(group_size, hidden, 2 * hidden, bias=bias, dtype=dtype)
        fc2 = te.ops.GroupedLinear(
            group_size, hidden, hidden, bias=bias, dtype=dtype, scale_bias=bias
        )
    module = te.ops.Sequential(fc1, te.ops.ScaledSwiGLU(glu_interleave_size=32), fc2)

    def run(x, dy):
        plain_input = isinstance(x, torch.Tensor) and not hasattr(x, "rowwise_data")
        x_in = x.detach().clone().requires_grad_() if plain_input else x
        probs_in = probs.detach().clone().requires_grad_()
        fc2_extra = (split_sizes, probs_in) if bias else (split_sizes,)
        with te.autocast(enabled=True, recipe=recipe):
            y = module(x_in, split_sizes, probs_in, *fc2_extra)
        _InjectGrad.apply(y, dy).backward(torch.ones_like(y))
        grads = [probs_in.grad] + [p.grad.detach().clone() for p in module.parameters()]
        for p in module.parameters():
            p.grad = None
        return y.detach(), x_in.grad if plain_input else None, grads

    y_ref, dx_ref, grads_ref = run(x_rows, dy_rows)
    assert isinstance(module._module_groups[0]._forward_ops[0][0], fused_cls)
    y_in, _, grads_in = run(x_grouped, dy_rows)
    y_grad, dx_grad, grads_grad = run(x_rows, dy_grouped)

    torch.testing.assert_close(y_in, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(y_grad, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx_grad, dx_ref, rtol=0, atol=0)
    for grad_in, grad_grad, grad_ref in zip(grads_in, grads_grad, grads_ref):
        torch.testing.assert_close(grad_in, grad_ref, rtol=0, atol=0)
        torch.testing.assert_close(grad_grad, grad_ref, rtol=0, atol=0)
