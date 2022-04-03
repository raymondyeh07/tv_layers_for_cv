// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void project_point_p_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> aux,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> w,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> delta,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  torch::PackedTensorAccessor<bool, 1,torch::RestrictPtrTraits,size_t> cont_find,
  long batch_size, long vec_size, long delta_size)
{
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int d_idx = blockIdx.z;

  // Check within range
  if(b_idx < batch_size && v_idx < vec_size
      && d_idx < delta_size && cont_find[b_idx])
  {
    const auto e_idx = num_inactive[b_idx];
    if (v_idx < e_idx)
    {
      const auto ind = inactive[b_idx][v_idx];
      aux[b_idx][ind][d_idx] = w[b_idx][ind] - delta[b_idx][d_idx]*d[b_idx][v_idx];
    }
  }
}


void project_point_p_cuda(torch::Tensor aux, torch::Tensor w,
                          torch::Tensor d, torch::Tensor delta,
                          torch::Tensor inactive,
                          torch::Tensor num_inactive, torch::Tensor cont_find)
{
  const auto batch_size = aux.size(0);
  const auto vec_size = aux.size(1);
  const auto delta_size = aux.size(2);

  const int threads = 32;
  const dim3 DimGrid(32, 32);
  const dim3 DimBlock((batch_size + threads - 1) / threads, (vec_size + threads - 1) / threads, delta_size);

  AT_DISPATCH_FLOATING_TYPES(aux.type(), "project_point_p_cuda", ([&]
  {
    project_point_p_cuda_kernel<scalar_t><<<DimBlock, DimGrid>>>(
      aux.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      w.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      delta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      cont_find.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size, delta_size);
  }));
}
