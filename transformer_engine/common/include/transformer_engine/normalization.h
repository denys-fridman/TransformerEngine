/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file normalization.h
 *  \brief LayerNorm and RMSNorm functions.
 */

#ifndef TRANSFORMER_ENGINE_NORMALIZATION_H_
#define TRANSFORMER_ENGINE_NORMALIZATION_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute LayerNorm on the input.
 *
 * The formula used:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}} \gamma + \beta
 * @f]
 *
 * Calling this function with workspace set to an empty tensor will not perform the operation,
 * but instead set the shape and type of the workspace tensor to the required values.
 *
 *  N is the product of the input dimensions preceding H. Inputs and outputs
 *  retain their logical shape; normalization is over the last dimension.
 *
 *  \param[in]     x                   Input tensor of shape [..., H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     beta                Beta tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [..., H].
 *  \param[out]    mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[out]    rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_layernorm_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta,
                        const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                        NVTETensor workspace, const int multiprocessorCount,
                        const bool zero_centered_gamma, cudaStream_t stream);

/*! \brief Compute backward of LayerNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}\gamma + \beta
 * @f]
 * else
 * with respect to \f$x\f$, \f$\gamma\f$ and \f$\beta\f$.
 *
 * Calling this function with workspace set to an empty tensor will not perform the operation,
 * but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [..., H].
 *  \param[in]     x                   Forward input tensor of shape [..., H].
 *  \param[in]     mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[in]     rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [..., H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    dbeta               Gradient for beta tensor of shape [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_layernorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor mu,
                        const NVTETensor rsigma, const NVTETensor gamma, NVTETensor dx,
                        NVTETensor dgamma, NVTETensor dbeta, NVTETensor workspace,
                        const int multiprocessorCount, const bool zero_centered_gamma,
                        cudaStream_t stream);

/*! \brief Compute RMSNorm.
 *
 * The formula used:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 *
 * Calling this function with workspace set to an empty tensor will not perform the operation,
 * but instead set the shape and type of the workspace tensor to the required values.
 *
 *  N is the product of the input dimensions preceding H. Inputs and outputs
 *  retain their logical shape; normalization is over the last dimension.
 *
 *  \param[in]     x                   Input tensor of shape [..., H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [..., H].
 *  \param[out]    rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_fwd(const NVTETensor x, const NVTETensor gamma, const float epsilon, NVTETensor z,
                      NVTETensor rsigma, NVTETensor workspace, const int multiprocessorCount,
                      const bool zero_centered_gamma, cudaStream_t stream);

/*! \brief Compute backward of RMSNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 * with respect to \f$x\f$ and \f$gamma\f$.
 *
 * Calling this function with workspace set to an empty tensor will not perform the operation,
 * but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [..., H].
 *  \param[in]     x                   Forward input tensor of shape [..., H].
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [..., H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor rsigma,
                      const NVTETensor gamma, NVTETensor dx, NVTETensor dgamma,
                      NVTETensor workspace, const int multiprocessorCount,
                      const bool zero_centered_gamma, cudaStream_t stream);

/*! \brief Compute backward of RMSNorm and add additional tensor to output gradient
 *
 * Calling this function with workspace set to an empty tensor will not perform the operation,
 * but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [..., H].
 *  \param[in]     x                   Forward input tensor of shape [..., H].
 *  \param[in]     add                 Additional tensor to add to output gradient [..., H].
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [..., H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_bwd_add(const NVTETensor dz, const NVTETensor x, const NVTETensor add,
                          const NVTETensor rsigma, const NVTETensor gamma, NVTETensor dx,
                          NVTETensor dgamma, NVTETensor workspace, const int multiprocessorCount,
                          const bool zero_centered_gamma, cudaStream_t stream);

/*! \brief Compute backward of RMSNorm in one pass over dz and x, with optional fusions.
 *
 *  dx = dz * gamma * rsigma - x * rsigma^3 / H * sum(dz * gamma * x) (+ add), rounded to the
 *  input type and either stored, or quantized to MXFP8 with rowwise (1x32) and columnwise (32x1)
 *  scales when dx is an MXFP8 tensor (compact or GEMM-swizzled scales; the padding of the scale
 *  matrices along H is written as zero, the padding along N is left untouched). dgamma, and the
 *  column sum dbias of the rounded dx if requested, are reduced in the kernel in a fixed order,
 *  so they are deterministic. Requires compute capability 9.0+.
 *
 *  Calling this function with workspace set to an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [..., H], BF16 or FP16.
 *  \param[in]     x                   Forward input tensor of shape [..., H], the dtype of dz.
 *                                     N, the product of the leading dimensions, and H are
 *                                     multiples of 32; H is at most 3072.
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H], the dtype of x.
 *  \param[in]     add                 Optional tensor of shape [..., H] added to dx; NULL or
 *                                     empty to skip.
 *  \param[out]    dx                  Output gradient of shape [..., H] in the dtype of x, or
 *                                     an MXFP8 tensor with rowwise and columnwise data.
 *  \param[out]    dx_high_precision   Optional tensor of shape [..., H] in the dtype of x that
 *                                     receives the rounded dx when dx is MXFP8; NULL or empty
 *                                     to skip.
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H], in the dtype of
 *                                     gamma or FP32.
 *  \param[out]    dbias               Optional column sum of dx, shape [H], the dtype of dgamma;
 *                                     NULL or empty to skip.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in,out] counter             Int32 tensor of at least 73 elements, zero before the
 *                                     call, used to order the reduction and left zero afterwards.
 *  \param[in]     zero_centered_gamma Multiply normalized values by @f$ \gamma+1 @f$ instead of @f$ \gamma @f$
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_bwd_fused(const NVTETensor dz, const NVTETensor x, const NVTETensor rsigma,
                            const NVTETensor gamma, const NVTETensor add, NVTETensor dx,
                            NVTETensor dx_high_precision, NVTETensor dgamma, NVTETensor dbias,
                            NVTETensor workspace, NVTETensor counter,
                            const bool zero_centered_gamma, cudaStream_t stream);

/*! \brief Set whether to enable cuDNN backend for normalization forward.
 *
 *  \param[in]  enable  Whether to enable cuDNN backend.
 */
void nvte_enable_cudnn_norm_fwd(bool enable);

/*! \brief Set whether to enable cuDNN backend for normalization backward.
 *
 *  \param[in]  enable  Whether to enable cuDNN backend.
 */
void nvte_enable_cudnn_norm_bwd(bool enable);

/*! \brief Control whether norm computes `gamma += 1.0` for zero-centered gamma
 *  in weight dtype. If set to false, it will compute in compute dtype.
 *
 *  Currently this only applies to the CuDNN backend. If CuDNN is not used,
 *  this setting has no effect.
 *
 *  \param[in]  enable  Whether to enable zero-centered gamma.
 */
void nvte_enable_zero_centered_gamma_in_weight_dtype(bool enable);

#ifdef __cplusplus
/*! \brief Normalization function type */
enum class NVTE_Norm_Type { LayerNorm, RMSNorm };
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_NORMALIZATION_H_
