// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void setup_sign_all_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits,size_t> sign_all,
  torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits,size_t> sign_z,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> s_active,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> s_num_active,
  long batch_size, long vec_size)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Row index
  const int v_idx = blockIdx.y;
  // Check within range
  if(b_idx < batch_size)
  {
    const auto max_s = s_num_active[b_idx];
    if (v_idx < max_s)
    {
      sign_all[b_idx][v_idx] = -1*sign_z[b_idx][s_active[b_idx][v_idx]];
    }
  }
}


void setup_sign_all_cuda(torch::Tensor sign_all,
                         torch::Tensor sign_z,
                         torch::Tensor s_active,
                         torch::Tensor s_num_active)
{
  const auto batch_size = sign_all.size(0);
  const auto vec_size = sign_all.size(1);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);;


  AT_DISPATCH_FLOATING_TYPES(sign_all.type(),  "setup_sign_all_cuda", ([&]
  {
    setup_sign_all_cuda_kernel<scalar_t><<<blocks, threads>>>(
      sign_all.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      sign_z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      s_active.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      s_num_active.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size);
  }));
}
