// Implements CUDA Kernel for Checking S active.

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


__global__ void check_s_active_cuda_kernel(
  torch::PackedTensorAccessor<bool,2,torch::RestrictPtrTraits,size_t> s,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> s_active,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> s_num_active,
  long batch_size, long vec_size)
{
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_idx<batch_size)
  {
    for (int v_idx=0; v_idx < vec_size; v_idx++)
    {
      if (v_idx == 0)
        s_num_active[b_idx] = 0;
      if (s[b_idx][v_idx])
        s_active[b_idx][s_num_active[b_idx]++]=v_idx;
    }
  }
}

void check_s_active_cuda(torch::Tensor s,
                         torch::Tensor s_active,
                         torch::Tensor s_num_active)
{
  const auto batch_size = s.size(0);
  const auto vec_size = s.size(1);

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  check_s_active_cuda_kernel<<<blocks, threads>>>(
    s.packed_accessor<bool,2,torch::RestrictPtrTraits,size_t>(),
    s_active.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
    s_num_active.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
    batch_size, vec_size);
}
