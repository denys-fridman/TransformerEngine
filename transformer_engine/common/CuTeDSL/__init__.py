# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

To expose CuTeDSL kernels to C++, they should be registered via `register_cutedsl_backends()`.
They should provide a string function name which can be used to retrieve the corresponding CuTeDSL kernel function via TVM-FFI.
Kernels launched from a framework frontend (``attention``) do not go through TVM-FFI.
"""


def register_cutedsl_backends():
    """Register all available CuTeDSL backends for on-demand compilation via TVM-FFI.
    The C++ dispatcher retrieves them by the names defined here.
    """
    import tvm_ffi  # pylint: disable=import-outside-toplevel

    from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (  # pylint: disable=import-outside-toplevel
        get_mxfp8_quantization_function,
    )

    tvm_ffi.register_global_func(
        "get_mxfp8_quantization_function", get_mxfp8_quantization_function, override=True
    )
