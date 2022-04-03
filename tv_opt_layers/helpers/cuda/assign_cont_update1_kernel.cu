// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void assign_cont_update1_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> src,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dst,
  torch::PackedTensorAccessor<bool, 1,torch::RestrictPtrTraits,size_t> cont_update,
  long batch_size, long vec_size)
{
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_idx = blockIdx.y;
  // Check within range
  if(b_idx < batch_size && cont_update[b_idx] && v_idx < vec_size)
  {
    dst[b_idx][v_idx] = src[b_idx][v_idx];
  }
}


void assign_cont_update1_cuda(torch::Tensor src, torch::Tensor dst,
                              torch::Tensor cont_update
                             )
{
  const auto batch_size = src.size(0);
  const auto vec_size = src.size(1);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);

  AT_DISPATCH_FLOATING_TYPES(src.type(), "assign_cont_update1_cuda", ([&]
  {
    assign_cont_update1_cuda_kernel<scalar_t><<<blocks, threads>>>(
      src.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      dst.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size);
  }));
}
