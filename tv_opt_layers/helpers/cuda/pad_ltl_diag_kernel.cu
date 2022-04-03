// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void pad_ltl_diag_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits,size_t> ltl,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> s_num_active,
  long batch_size, long diag_size)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Row index
  const int d_idx = blockIdx.y;
  // Check within range
  if(b_idx < batch_size)
  {
    const auto max_s = s_num_active[b_idx];
    if (d_idx >= max_s && d_idx < diag_size)
      ltl[b_idx][d_idx][d_idx] = 1;
  }
}


void pad_ltl_diag_cuda(torch::Tensor ltl,
                       torch::Tensor s_num_active)
{
  const auto batch_size = ltl.size(0);
  const auto diag_size = ltl.size(1);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, diag_size);

  AT_DISPATCH_FLOATING_TYPES(ltl.type(), "pad_ltl_diag_cuda", ([&]
  {
    pad_ltl_diag_cuda_kernel<scalar_t><<<blocks, threads>>>(
      ltl.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      s_num_active.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, diag_size);
  }));
}
