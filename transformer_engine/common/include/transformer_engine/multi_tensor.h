/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file multi_tensor.h
 *  \brief Functions handling multi tensor kernels.
 */

#ifndef TRANSFORMER_ENGINE_MULTI_TENSOR_H_
#define TRANSFORMER_ENGINE_MULTI_TENSOR_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!  \brief Computes L2 norm for a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]     noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in]     tensor_lists            2D array of input tensors.
 *  \param[in]     num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]     num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]     output                  Scratch space. Required size grows with number of inputs.
 *  \param[in]     output_per_tensor       Fixed size auxilliary scratch space.
 *  \param[out]    ret                     L2 norm of all inputs.
 *  \param[out]    ret_per_tensor          L2 norm for each tensor.
 *  \param[in]     per_tensor              Whether to calculate per tensor or cumulative norm.
 *  \param[in]     max_chunks_per_tensor   Maximum number of chunks in any input tensor.
 *  \param[in]     stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_l2norm_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                   const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                   NVTETensor output, NVTETensor output_per_tensor, NVTETensor ret,
                                   NVTETensor ret_per_tensor, int per_tensor,
                                   int max_chunks_per_tensor, cudaStream_t stream);

/*!  \brief Computes L2 norm for a list of tensors after unscaling.
 *
 * Unscaling is only done for computing the L2 norm. The tensors themselves are not updated.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]     noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in]     tensor_lists            2D array of input tensors.
 *  \param[in]     num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]     num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]     output                  Scratch space. Required size grows with number of inputs.
 *  \param[in]     output_per_tensor       Fixed size auxilliary scratch space.
 *  \param[out]    ret                     L2 norm of all inputs.
 *  \param[out]    ret_per_tensor          L2 norm for each tensor.
 *  \param[in]     inv_scale               Scalar for the unscaling operation.
 *  \param[in]     per_tensor              Whether to calculate per tensor or cumulative norm.
 *  \param[in]     max_chunks_per_tensor   Maximum number of chunks in any input tensor.
 *  \param[in]     stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_unscale_l2norm_cuda(int chunk_size, NVTETensor noop_flag,
                                           NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                           const size_t num_tensors_per_list, NVTETensor output,
                                           NVTETensor output_per_tensor, NVTETensor ret,
                                           NVTETensor ret_per_tensor, NVTETensor inv_scale,
                                           int per_tensor, int max_chunks_per_tensor,
                                           cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                 const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                 const float lr, const float beta1, const float beta2,
                                 const float epsilon, const int step, const int mode,
                                 const int bias_correction, const float weight_decay,
                                 cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          where the master parameters only store the remainder bits.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_param_remainder_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          when model parameters are in Float8 precision.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      fp8_dtype               FP8 data type for model parameters.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_fp8_cuda(int chunk_size, NVTETensor noop_flag,
                                     NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                     const size_t num_tensors_per_list, const float lr,
                                     const float beta1, const float beta2, const float epsilon,
                                     const int step, const int mode, const int bias_correction,
                                     const float weight_decay, const NVTEDType fp8_dtype,
                                     cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          with CUDA graph support and LR scheduling.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      inv_scale               Scalar for the unscaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_capturable_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          with CUDA graph support, LR scheduling, and FP32 master weights.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      inv_scale               Scalar for the unscaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_capturable_master_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream);

/*!  \brief Compute and apply the Adam update with CUDA graph support, LR scheduling and FP32
 *          master weights to one contiguous buffer, e.g. a distributed optimizer's shard of a
 *          parameter bucket.
 *
 *  The same update as nvte_multi_tensor_adam_capturable_master_cuda, bitwise, in one launch over
 *  flat buffers that share an element index. A sorted segment table assigns elements to param
 *  groups: elements [seg_starts[i], seg_starts[i + 1]) belong to group seg_groups[i], or are
 *  padding and left untouched when seg_groups[i] is negative. Per-group arguments are arrays of
 *  num_groups entries.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in,out]  grad                    Gradient buffer, scaled by inv_scale in place when
 *                                          store_scaled_grad is set.
 *  \param[out]     param                   Parameter buffer.
 *  \param[in,out]  exp_avg                 First moment buffer.
 *  \param[in,out]  exp_avg_sq              Second moment buffer.
 *  \param[in,out]  master_param            FP32 master parameter buffer.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in]      seg_starts              Int64 element offsets of the segments, num_segments + 1 entries ending with the buffer size.
 *  \param[in]      seg_groups              Int32 param group of each segment, negative for padding.
 *  \param[in]      lr                      Learning rate of each param group (single element tensors).
 *  \param[in]      beta1                   Coefficient for first moment of gradient of each param group.
 *  \param[in]      beta2                   Coefficient for second moment of gradient of each param group.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability of each param group.
 *  \param[in]      step                    Iteration counter of each param group (single element Int32 tensors).
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates, per param group.
 *  \param[in]      weight_decay            L2 penalty for weight decay of each param group.
 *  \param[in]      num_groups              Number of param groups (at most 8).
 *  \param[in]      inv_scale               Scalar for the unscaling operation.
 *  \param[in]      store_scaled_grad       Whether to write the unscaled gradient back to grad.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_flat_adam_capturable_master_cuda(
    NVTETensor grad, NVTETensor param, NVTETensor exp_avg, NVTETensor exp_avg_sq,
    NVTETensor master_param, NVTETensor noop_flag, NVTETensor seg_starts, NVTETensor seg_groups,
    const NVTETensor *lr, const float *beta1, const float *beta2, const float *epsilon,
    const NVTETensor *step, const int mode, const int *bias_correction, const float *weight_decay,
    const size_t num_groups, NVTETensor inv_scale, const int store_scaled_grad,
    cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for SGD optimizer.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      wd                      Weight decay (L2 penalty).
 *  \param[in]      momentum                Momentum factor.
 *  \param[in]      dampening               Dampening factor.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      nesterov                Whether or not to enable nesterov momentum.
 *  \param[in]      first_run               Whether momentum buffers have been initialized.
 *  \param[in]      wd_after_momentum       Whether to applied weight decay after momentum update.
 *  \param[in]      scale                   Scalar for the scaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_sgd_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                float wd, float momentum, float dampening, float lr, int nesterov,
                                int first_run, int wd_after_momentum, float scale,
                                cudaStream_t stream);

/*!  \brief Check overflow and scale a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[out]     is_infinite             Whether the kernel detected a non-finite input value.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      scale                   Scalar for the scaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_scale_cuda(int chunk_size, NVTETensor is_infinite, NVTETensor **tensor_lists,
                                  const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                  float scale, cudaStream_t stream);

/*!  \brief Check overflow and scale a list of tensors. scale is tensor input.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[out]     is_infinite             Whether the kernel detected a non-finite input value.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      scale                   Tensor for the scaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_scale_tensor_cuda(int chunk_size, NVTETensor is_infinite,
                                         NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                         const size_t num_tensors_per_list, NVTETensor scale,
                                         cudaStream_t stream);

/*!  \brief Check overflow and scale a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      max_fp8                 Maximum representible value in underlying FP8 format.
 *  \param[in]      force_pow_2_scales      Ensure scaling factors are a power of 2.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_compute_scale_and_scale_inv_cuda(int chunk_size, NVTETensor noop_flag,
                                                        NVTETensor **tensor_lists,
                                                        const size_t num_tensor_lists,
                                                        const size_t num_tensors_per_list,
                                                        float max_fp8, int force_pow_2_scales,
                                                        float epsilon, cudaStream_t stream);

/*!  \brief Compute E8M0 scale_inv for a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_compute_scale_inv_e8m0_cuda(int chunk_size, NVTETensor **tensor_lists,
                                                   const size_t num_tensor_lists,
                                                   const size_t num_tensors_per_list,
                                                   cudaStream_t stream);

/*! \brief Split a tensor along dimension 0 and compute the amax for each split.
 *
 *  This function is experimental and the API is not stable.
 *
 *  For each tensor split, compute the maximum absolute value (amax)
 *  and populate the amax of the corresponding output tensor.
 *
 *  \param[in]      input            Input tensor.
 *  \param[in,out]  outputs          Array of output tensors. Only the amax is updated.
 *  \param[in]      split_sections   Size of each tensor split along dimension 0.
 *  \param[in]      num_tensors      Number of tensor splits.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_group_amax(const NVTETensor input, NVTETensor *outputs, const size_t *split_sections,
                     size_t num_tensors, cudaStream_t stream);

/*! \brief Grouped-tensor amax without doing hadamard transform.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input            NVTEGroupedTensor Input tensor.
 *  \param[in,out]  output           NVTEGroupedTensor Output tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_group_amax_graph_safe(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_MULTI_TENSOR_H_
