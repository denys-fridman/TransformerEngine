# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the fused RMSNorm backward kernels (NVTE_RMSNORM_BWD_USE_TRITON=1), CUDA and
Triton."""

from __future__ import annotations

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch.triton import rmsnorm_bwd as triton_rmsnorm_bwd

from utils import dtype_tols, make_recipe, quantization_tols, reset_rng_states

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
cuda_kernel_available = torch.cuda.get_device_capability()[0] >= 9

_dtypes = [torch.float32, torch.float16, torch.bfloat16]
# (45, 70): row and column tails are masked
_shapes = [(32, 2880), (160, 256), (45, 70)]


@pytest.fixture(autouse=True)
def _use_triton_kernel(monkeypatch):
    monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", "1")
    reset_rng_states()


@pytest.fixture(params=("cuda", "triton"))
def backend(request, monkeypatch):
    """Force one of the two fused kernels; the CUDA one needs compute capability 9.0+."""
    if request.param == "cuda" and not cuda_kernel_available:
        pytest.skip("CUDA kernel requires compute capability 9.0+")
    monkeypatch.setenv("NVTE_RMSNORM_BWD_BACKEND", request.param)
    return request.param


@pytest.fixture
def cuda_kernel_calls(monkeypatch):
    """Count of the calls to the CUDA kernel's extension."""
    calls = []
    run = tex.rmsnorm_bwd_fused

    def spy(*args, **kwargs):
        calls.append(args)
        return run(*args, **kwargs)

    monkeypatch.setattr(tex, "rmsnorm_bwd_fused", spy)
    return calls


@pytest.fixture
def kernel_calls(monkeypatch):
    """Keyword arguments of every call to the Triton kernel wrapper."""
    calls = []
    run = triton_rmsnorm_bwd.rmsnorm_bwd

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return run(*args, **kwargs)

    monkeypatch.setattr(triton_rmsnorm_bwd, "rmsnorm_bwd", spy)
    return calls


def _make_inputs(shape, dtype, with_add, eps=1e-5):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    dy = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.rand(shape[1], dtype=dtype, device="cuda") + 0.5
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + eps)
    add = torch.randn(shape, dtype=dtype, device="cuda") if with_add else None
    return dy, x, rstd, w, add


def _reference(dy, x, rstd, w, add, zero_centered_gamma):
    """RMSNorm backward in FP64 from the forward pass's rstd."""
    dy, x, w, rstd = dy.double(), x.double(), w.double(), rstd.double()[:, None]
    if zero_centered_gamma:
        w = w + 1
    dyw = dy * w
    dx = rstd * dyw - x * rstd**3 * (dyw * x).sum(dim=1, keepdim=True) / x.shape[1]
    if add is not None:
        dx = dx + add.double()
    return dx, (dy * x * rstd).sum(dim=0)


def test_is_supported(monkeypatch):
    dy, x, rstd, w, _ = _make_inputs((64, 256), torch.bfloat16, with_add=False)
    assert triton_rmsnorm_bwd.is_supported(dy, x, rstd, w)
    assert not triton_rmsnorm_bwd.is_supported(dy.t().contiguous().t(), x, rstd, w)
    assert not triton_rmsnorm_bwd.is_supported(dy, x, rstd, w.float())
    assert not triton_rmsnorm_bwd.is_supported(dy, x, rstd.half(), w)
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    assert triton_rmsnorm_bwd.is_supported(dy, x, rstd, w, quantizer=quantizer)
    quantizer_2d = MXFP8Quantizer(tex.DType.kFloat8E4M3, with_2d_quantization=True)
    assert not triton_rmsnorm_bwd.is_supported(dy, x, rstd, w, quantizer=quantizer_2d)
    dy, x, rstd, w, _ = _make_inputs((45, 70), torch.bfloat16, with_add=False)
    assert triton_rmsnorm_bwd.is_supported(dy, x, rstd, w)
    assert not triton_rmsnorm_bwd.is_supported(dy, x, rstd, w, quantizer=quantizer)
    monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", "0")
    assert not triton_rmsnorm_bwd.is_supported(dy, x, rstd, w)


@pytest.mark.parametrize("shape", _shapes)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("zero_centered_gamma", (False, True))
@pytest.mark.parametrize("with_add", (False, True))
def test_kernel(backend, cuda_kernel_calls, shape, dtype, zero_centered_gamma, with_add):
    """Kernel against an FP64 reference and the C++ kernels."""
    dy, x, rstd, w, add = _make_inputs(shape, dtype, with_add)
    assert triton_rmsnorm_bwd.is_supported(dy, x, rstd, w, add=add)
    dx, dw = triton_rmsnorm_bwd.rmsnorm_bwd(
        dy, x, rstd, w, zero_centered_gamma=zero_centered_gamma, add=add
    )
    # The CUDA kernel takes 16-bit tensors with M and N multiples of 32
    M, N = shape
    cuda_kernel_applies = dtype != torch.float32 and M % 32 == 0 and N % 32 == 0
    assert len(cuda_kernel_calls) == int(backend == "cuda" and cuda_kernel_applies)
    assert dx.dtype == dtype and dx.shape == x.shape
    assert dw.dtype == dtype and dw.shape == w.shape
    dx_ref, dw_ref = _reference(dy, x, rstd, w, add, zero_centered_gamma)
    tols = dtype_tols(dtype)
    torch.testing.assert_close(dx.double(), dx_ref, **tols)
    torch.testing.assert_close(dw.double(), dw_ref, **tols)
    if add is None:
        dx_cpp, dw_cpp = tex.rmsnorm_bwd(dy, x, rstd, w, 0, zero_centered_gamma)
    else:
        dx_cpp, dw_cpp = tex.rmsnorm_bwd_add(dy, x, add, rstd, w, 0, zero_centered_gamma)
    torch.testing.assert_close(dx, dx_cpp, **tols)
    torch.testing.assert_close(dw, dw_cpp, **tols)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("shape", ((128, 2880), (96, 256)))  # with and without padded scales
@pytest.mark.parametrize("fp8_dtype", (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2))
@pytest.mark.parametrize("optimize_for_gemm", (False, True))
def test_mxfp8_output(backend, cuda_kernel_calls, shape, fp8_dtype, optimize_for_gemm):
    """Quantized output bitwise equal to TE's quantizer applied to the kernel's own gradient."""
    dy, x, rstd, w, add = _make_inputs(shape, torch.bfloat16, with_add=True)
    dx, dw = triton_rmsnorm_bwd.rmsnorm_bwd(dy, x, rstd, w, zero_centered_gamma=False, add=add)
    quantizer = MXFP8Quantizer(fp8_dtype)
    quantizer.optimize_for_gemm = optimize_for_gemm
    assert triton_rmsnorm_bwd.is_supported(dy, x, rstd, w, add=add, quantizer=quantizer)
    dx_q, dw_q = triton_rmsnorm_bwd.rmsnorm_bwd(
        dy, x, rstd, w, zero_centered_gamma=False, add=add, quantizer=quantizer
    )
    ref = quantizer(dx)
    assert dx_q._fp8_dtype == fp8_dtype
    assert dx_q._with_gemm_swizzled_scales == optimize_for_gemm
    assert torch.equal(dx_q._rowwise_data, ref._rowwise_data)
    assert torch.equal(dx_q._columnwise_data, ref._columnwise_data)
    M, N = shape
    if optimize_for_gemm:
        # TE zeroes the padding of the GEMM scale layout
        assert torch.equal(dx_q._rowwise_scale_inv, ref._rowwise_scale_inv)
        assert torch.equal(dx_q._columnwise_scale_inv, ref._columnwise_scale_inv)
    else:
        assert torch.equal(
            dx_q._rowwise_scale_inv[:M, : N // 32], ref._rowwise_scale_inv[:M, : N // 32]
        )
        assert torch.equal(
            dx_q._columnwise_scale_inv[: M // 32, :N], ref._columnwise_scale_inv[: M // 32, :N]
        )
    assert torch.equal(dw_q, dw)
    # The gradient before quantization is kept and is what dequantize returns
    assert torch.equal(dx_q._high_precision_data, dx)
    assert torch.equal(dx_q.dequantize(), dx)
    assert torch.equal(dx_q.view(-1, N).dequantize(), dx.view(-1, N))
    assert len(cuda_kernel_calls) == (2 if backend == "cuda" else 0)


@pytest.mark.skipif(
    not cuda_kernel_available, reason="CUDA kernel requires compute capability 9.0+"
)
@pytest.mark.parametrize("shape", ((32, 2880), (160, 256), (96, 3072)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("zero_centered_gamma", (False, True))
@pytest.mark.parametrize("with_add", (False, True))
def test_backends_match(monkeypatch, shape, dtype, zero_centered_gamma, with_add):
    """The CUDA and Triton kernels agree within the dtype's tolerance, and the CUDA kernel's
    in-kernel reduction is deterministic."""
    dy, x, rstd, w, add = _make_inputs(shape, dtype, with_add)
    grads = {}
    for backend in ("cuda", "triton", "cuda"):
        monkeypatch.setenv("NVTE_RMSNORM_BWD_BACKEND", backend)
        dx, dw = triton_rmsnorm_bwd.rmsnorm_bwd(
            dy, x, rstd, w, zero_centered_gamma=zero_centered_gamma, add=add
        )
        if backend in grads:
            assert torch.equal(dx, grads[backend][0]) and torch.equal(dw, grads[backend][1])
        grads[backend] = (dx, dw)
    tols = dtype_tols(dtype)
    torch.testing.assert_close(grads["cuda"][0], grads["triton"][0], **tols)
    torch.testing.assert_close(grads["cuda"][1], grads["triton"][1], **tols)


@pytest.mark.skipif(
    not cuda_kernel_available, reason="CUDA kernel requires compute capability 9.0+"
)
@pytest.mark.parametrize("grads_dtype", (torch.bfloat16, torch.float32))
def test_cuda_kernel_dbias(grads_dtype):
    """The CUDA kernel's optional bias gradient (column sum of dx) and FP32 gradients, with
    and without MXFP8 output."""
    shape = (1024, 2880)
    dy, x, rstd, w, add = _make_inputs(shape, torch.bfloat16, with_add=True)
    counter = torch.zeros(73, dtype=torch.int32, device="cuda")
    dx = torch.empty_like(x)
    dgamma = torch.empty(shape[1], dtype=grads_dtype, device="cuda")
    dbias = torch.empty(shape[1], dtype=grads_dtype, device="cuda")
    tex.rmsnorm_bwd_fused(dy, x, rstd, w, add, dx, None, None, dgamma, dbias, counter, False)
    assert torch.all(counter == 0)
    dx_ref, dw_ref = _reference(dy, x, rstd, w, add, False)
    torch.testing.assert_close(dx.double(), dx_ref, **dtype_tols(torch.bfloat16))
    # FP32 sums of 1024 rows against the FP64 reference
    tols = dict(rtol=1e-4, atol=1e-3) if grads_dtype == torch.float32 else dtype_tols(grads_dtype)
    torch.testing.assert_close(dgamma.double(), dw_ref, **tols)
    torch.testing.assert_close(dbias.double(), dx.double().sum(dim=0), **tols)
    if mxfp8_available:
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        dx_q = quantizer.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        dx_hp = torch.empty_like(x)
        dgamma_q = torch.empty_like(dgamma)
        dbias_q = torch.empty_like(dbias)
        tex.rmsnorm_bwd_fused(
            dy, x, rstd, w, add, dx_q, quantizer, dx_hp, dgamma_q, dbias_q, counter, False
        )
        assert torch.equal(dgamma_q, dgamma) and torch.equal(dbias_q, dbias)
        assert torch.equal(dx_hp, dx)
        ref = quantizer(dx)
        assert torch.equal(dx_q._rowwise_data, ref._rowwise_data)
        assert torch.equal(dx_q._columnwise_data, ref._columnwise_data)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("zero_centered_gamma", (False, True))
@pytest.mark.parametrize("with_extra_grad", (None, False, True))
def test_ops_rmsnorm(
    backend, dtype, zero_centered_gamma, with_extra_grad, monkeypatch, kernel_calls
):
    """RMSNorm op, alone (with_extra_grad=None) or after MakeExtraOutput, with the kernel on and off."""
    shape, N = (6, 16, 256), 256
    x = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.rand(N, dtype=dtype, device="cuda") + 0.5
    dy1 = torch.randn(shape, dtype=dtype, device="cuda")
    dy2 = torch.randn(shape, dtype=dtype, device="cuda")
    grads = []
    for enabled in ("1", "0"):
        monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", enabled)
        norm = te_ops.RMSNorm(
            N, eps=0.3, device="cuda", dtype=dtype, zero_centered_gamma=zero_centered_gamma
        )
        with torch.no_grad():
            norm.weight.copy_(w)
        x_test = x.clone().requires_grad_()
        if with_extra_grad is None:
            loss = (norm(x_test) * dy1).sum()
        else:
            y1, y2 = te_ops.Sequential(te_ops.MakeExtraOutput(), norm)(x_test)
            loss = (y1 * dy1).sum()
            if with_extra_grad:
                loss = loss + (y2 * dy2).sum()
        loss.backward()
        grads.append((x_test.grad, norm.weight.grad))
    assert len(kernel_calls) == 1
    assert (kernel_calls[0]["add"] is not None) == bool(with_extra_grad)
    tols = dtype_tols(dtype)
    for grad_triton, grad_cpp in zip(*grads):
        torch.testing.assert_close(grad_triton, grad_cpp, **tols)


@pytest.mark.parametrize("dtype", _dtypes)
def test_layernorm_linear(backend, dtype, monkeypatch, kernel_calls):
    """LayerNormLinear with RMSNorm, kernel on and off."""
    x = torch.randn(96, 256, dtype=dtype, device="cuda")
    dy = torch.randn(96, 128, dtype=dtype, device="cuda")
    grads = []
    for enabled in ("1", "0"):
        monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", enabled)
        reset_rng_states()
        module = te.LayerNormLinear(
            256, 128, normalization="RMSNorm", params_dtype=dtype, device="cuda"
        )
        x_test = x.clone().requires_grad_()
        module(x_test).backward(dy)
        grads.append((x_test.grad, module.layer_norm_weight.grad, module.weight.grad))
    assert len(kernel_calls) == 1
    tols = dtype_tols(dtype)
    for grad_triton, grad_cpp in zip(*grads):
        torch.testing.assert_close(grad_triton, grad_cpp, **tols)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("with_extra_grad", (False, True))
def test_ops_linear_add_rmsnorm_mxfp8(backend, with_extra_grad, monkeypatch, kernel_calls):
    """The fused backward hands the preceding Linear its MXFP8 grad output directly."""
    shape, N = (64, 256), 256
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device="cuda")
    dy1 = torch.randn(shape, dtype=dtype, device="cuda")
    dy2 = torch.randn(shape, dtype=dtype, device="cuda")
    recipe = make_recipe("mxfp8")
    grads = []
    for enabled in ("1", "0"):
        monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", enabled)
        reset_rng_states()
        model = te_ops.Sequential(
            te_ops.Linear(N, N, bias=False, device="cuda", dtype=dtype),
            te_ops.MakeExtraOutput(),
            te_ops.RMSNorm(N, device="cuda", dtype=dtype),
        )
        x_test = x.clone().requires_grad_()
        with te.autocast(enabled=True, recipe=recipe):
            y1, y2 = model(x_test)
        loss = (y1 * dy1).sum()
        if with_extra_grad:
            loss = loss + (y2 * dy2).sum()
        loss.backward()
        grads.append((x_test.grad, model[0].weight.grad, model[2].weight.grad))
    assert len(kernel_calls) == 1
    assert isinstance(kernel_calls[0]["quantizer"], MXFP8Quantizer)
    tols = quantization_tols("mxfp8")
    for grad_triton, grad_cpp in zip(*grads):
        torch.testing.assert_close(grad_triton, grad_cpp, **tols)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("with_extra_grad", (False, True))
def test_ops_quantize_add_rmsnorm_mxfp8(with_extra_grad, monkeypatch, kernel_calls):
    """A Quantize op quantizing in the backward pass gets the MXFP8 grad from the fused
    backward (or from the quantizer on the C++ path) with the gradient before quantization
    kept, so that the input's other gradient accumulates with it exactly."""
    shape, N = (64, 256), 256
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device="cuda")
    dy1 = torch.randn(shape, dtype=dtype, device="cuda")
    dy2 = torch.randn(shape, dtype=dtype, device="cuda")
    dy3 = torch.randn(shape, dtype=dtype, device="cuda")
    recipe = make_recipe("mxfp8")
    grads = {}
    for quantize in (True, False):
        for enabled in ("1", "0"):
            monkeypatch.setenv("NVTE_RMSNORM_BWD_USE_TRITON", enabled)
            reset_rng_states()
            ops = [te_ops.MakeExtraOutput(), te_ops.RMSNorm(N, device="cuda", dtype=dtype)]
            if quantize:
                ops.insert(0, te_ops.Quantize(forward=False, backward=True))
            model = te_ops.Sequential(*ops)
            x_test = x.clone().requires_grad_()
            with te.autocast(enabled=True, recipe=recipe):
                y1, y2 = model(x_test)
            # x_test gets a second gradient that autograd accumulates with the Sequential's
            loss = (y1 * dy1).sum() + (x_test * dy3).sum()
            if with_extra_grad:
                loss = loss + (y2 * dy2).sum()
            loss.backward()
            grads[quantize, enabled] = x_test.grad
    assert len(kernel_calls) == 2
    assert isinstance(kernel_calls[0]["quantizer"], MXFP8Quantizer)
    assert kernel_calls[1]["quantizer"] is None
    for enabled in ("1", "0"):
        assert not isinstance(grads[True, enabled], MXFP8Tensor)
        assert torch.equal(grads[True, enabled], grads[False, enabled])
    torch.testing.assert_close(grads[True, "1"], grads[True, "0"], **dtype_tols(dtype))
