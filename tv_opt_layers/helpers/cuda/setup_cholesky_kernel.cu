// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void setup_cholesky_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> g,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  torch::PackedTensorAccessor<bool, 1,torch::RestrictPtrTraits,size_t> cont_update,
  long batch_size, long vec_size)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Column index
  const int v_idx = blockIdx.y;

  // Check within range
  if(b_idx < batch_size && v_idx < vec_size && cont_update[b_idx])
  {
    const auto e_idx = num_inactive[b_idx];
    if (v_idx < e_idx)
    {
      d[b_idx][v_idx] = g[b_idx][inactive[b_idx][v_idx]];
    }
  }
}


void setup_cholesky_cuda(torch::Tensor g,
                         torch::Tensor d,
                         torch::Tensor inactive,
                         torch::Tensor num_inactive,
                         torch::Tensor cont_update)
{
  const auto batch_size = g.size(0);
  const auto vec_size = g.size(1);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);

  AT_DISPATCH_FLOATING_TYPES(g.type(), "setup_cholesky_cuda", ([&]
  {
    setup_cholesky_cuda_kernel<scalar_t><<<blocks, threads>>>(
      g.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size);
  }));
}
