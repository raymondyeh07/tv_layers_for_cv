// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


__global__ void check_inactive_cuda_kernel(
  torch::PackedTensorAccessor<bool,2,torch::RestrictPtrTraits,size_t> cond,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  long batch_size, long vec_size)
{
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_idx<batch_size)
  {
    for (int v_idx=0; v_idx < vec_size; v_idx++)
    {
      if (v_idx == 0)
        num_inactive[b_idx] = 0;
      if (cond[b_idx][v_idx])
        inactive[b_idx][num_inactive[b_idx]++]=v_idx;
    }
  }
}

void check_inactive_cuda(torch::Tensor cond,
                         torch::Tensor inactive,
                         torch::Tensor num_inactive)
{
  const auto batch_size = cond.size(0);
  const auto vec_size = cond.size(1);

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  check_inactive_cuda_kernel<<<blocks, threads>>>(
    cond.packed_accessor<bool,2,torch::RestrictPtrTraits,size_t>(),
    inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
    num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
    batch_size, vec_size);
}
