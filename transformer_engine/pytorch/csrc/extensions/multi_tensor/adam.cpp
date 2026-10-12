/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                              num_tensors, lr, beta1, beta2, epsilon, step, mode, bias_correction,
                              weight_decay, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_param_remainder_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors, lr, beta1,
      beta2, epsilon, step, mode, bias_correction, weight_decay, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_fp8_cuda(int chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_fp8_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(),
                                  num_lists, num_tensors, lr, beta1, beta2, epsilon, step, mode,
                                  bias_correction, weight_decay, static_cast<NVTEDType>(fp8_dtype),
                                  at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists,
                                       at::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, at::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists,
                                              at::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, at::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_master_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

void flat_adam_capturable_master_cuda(at::Tensor grad, at::Tensor param, at::Tensor exp_avg,
                                      at::Tensor exp_avg_sq, at::Tensor master_param,
                                      at::Tensor noop_flag, at::Tensor seg_starts,
                                      at::Tensor seg_groups, std::vector<at::Tensor> lr,
                                      std::vector<float> beta1, std::vector<float> beta2,
                                      std::vector<float> epsilon, std::vector<at::Tensor> step,
                                      const int mode, std::vector<int> bias_correction,
                                      std::vector<float> weight_decay, at::Tensor inv_scale,
                                      bool store_scaled_grad) {
  auto grad_cu = makeTransformerEngineTensor(grad);
  auto param_cu = makeTransformerEngineTensor(param);
  auto exp_avg_cu = makeTransformerEngineTensor(exp_avg);
  auto exp_avg_sq_cu = makeTransformerEngineTensor(exp_avg_sq);
  auto master_param_cu = makeTransformerEngineTensor(master_param);
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto seg_starts_cu = makeTransformerEngineTensor(seg_starts);
  auto seg_groups_cu = makeTransformerEngineTensor(seg_groups);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);
  const size_t num_groups = lr.size();
  NVTE_CHECK(step.size() == num_groups && beta1.size() == num_groups &&
                 beta2.size() == num_groups && epsilon.size() == num_groups &&
                 bias_correction.size() == num_groups && weight_decay.size() == num_groups,
             "Expected one lr, beta1, beta2, epsilon, step, bias_correction and weight_decay per "
             "param group");
  std::vector<TensorWrapper> lr_cu, step_cu;
  std::vector<NVTETensor> lr_ptr, step_ptr;
  for (size_t i = 0; i < num_groups; ++i) {
    lr_cu.push_back(makeTransformerEngineTensor(lr[i]));
    step_cu.push_back(makeTransformerEngineTensor(step[i]));
    lr_ptr.push_back(lr_cu.back().data());
    step_ptr.push_back(step_cu.back().data());
  }

  nvte_flat_adam_capturable_master_cuda(
      grad_cu.data(), param_cu.data(), exp_avg_cu.data(), exp_avg_sq_cu.data(),
      master_param_cu.data(), noop_flag_cu.data(), seg_starts_cu.data(), seg_groups_cu.data(),
      lr_ptr.data(), beta1.data(), beta2.data(), epsilon.data(), step_ptr.data(), mode,
      bias_correction.data(), weight_decay.data(), num_groups, inv_scale_cu.data(),
      store_scaled_grad ? 1 : 0, at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
