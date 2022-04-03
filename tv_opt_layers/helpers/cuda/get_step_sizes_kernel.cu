// Implements CUDA Kernel for get_step_sizes

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void get_step_sizes_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> step_sizes,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> w,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> lmbd,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  long batch_size, long vec_size, long lmbd_batch_size)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Column index
  const int v_idx = blockIdx.y;
  auto l_tmp = lmbd[0][0];

  // Check within range
  if(b_idx < batch_size && v_idx < vec_size)
  {
    const auto e_idx = num_inactive[b_idx];
    if (v_idx < e_idx && d[b_idx][v_idx] != 0.)
    {
      const auto ind = inactive[b_idx][v_idx];
      if (lmbd_batch_size > 1)
        l_tmp = lmbd[b_idx][0];
      step_sizes[b_idx][v_idx] = (w[b_idx][ind]+copysign(l_tmp,d[b_idx][v_idx]))/d[b_idx][v_idx];
    }
  }
}

void get_step_sizes_cuda(torch::Tensor step_sizes, torch::Tensor w,
                         torch::Tensor d, torch::Tensor lmbd,
                         torch::Tensor inactive,
                         torch::Tensor num_inactive)
{
  const auto batch_size = step_sizes.size(0);
  const auto vec_size = step_sizes.size(1);
  const auto lmbd_batch_size = lmbd.size(0);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);

  AT_DISPATCH_FLOATING_TYPES(step_sizes.type(), "get_step_sizes_cuda", ([&]
  {
    get_step_sizes_cuda_kernel<scalar_t><<<blocks, threads>>>(
      step_sizes.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      w.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      lmbd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size, lmbd_batch_size);
  }));
}
