// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void reduced_hessian_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> aux,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> aux2,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  torch::PackedTensorAccessor<bool, 1,torch::RestrictPtrTraits,size_t> cont_update,
  long batch_size, long vec_size)
{
  // Column index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_idx = blockIdx.y;

  // Check within range
  if(b_idx < batch_size && v_idx < vec_size && cont_update[b_idx])
  {
    const auto e_idx = num_inactive[b_idx];
    if (v_idx < e_idx-1)
    {
      aux[b_idx][v_idx] = 2.;
      if ((inactive[b_idx][v_idx+1]-inactive[b_idx][v_idx]) != 1)
        aux2[b_idx][v_idx] = 0.;
      else
        aux2[b_idx][v_idx] = -1.;
    }
    aux[b_idx][e_idx-1] = 2.;
  }
}


void reduced_hessian_cuda(torch::Tensor aux,
                          torch::Tensor aux2,
                          torch::Tensor inactive,
                          torch::Tensor num_inactive,
                          torch::Tensor cont_update)
{
  const auto batch_size = aux.size(0);
  const auto vec_size = aux.size(1);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);

  AT_DISPATCH_FLOATING_TYPES(aux.type(), "reduced_hessian_cuda", ([&]
  {
    reduced_hessian_cuda_kernel<scalar_t><<<blocks, threads>>>(
      aux.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      aux2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size);
  }));
}
