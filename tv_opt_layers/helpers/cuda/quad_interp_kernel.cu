// Implements CUDA Kernel for quad_interp

#include <torch/extension.h>
//#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void quad_interp_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad0,
  torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> delta,
  torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> prev_delta,
  torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> improve,
  torch::PackedTensorAccessor<bool,1,torch::RestrictPtrTraits,size_t> cont_find,
  torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> max_step,
  long batch_size, double EPSILON)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_idx < batch_size && cont_find[b_idx])
  {
    const auto tmp = grad0[b_idx]*delta[b_idx];
    prev_delta[b_idx] = delta[b_idx];
    delta[b_idx] = 0.5*(tmp*delta[b_idx]) / (improve[b_idx] + tmp);
    /* If larger than maximum stepsize, clip */
    if(delta[b_idx] > max_step[b_idx])
    {
      delta[b_idx] = max_step[b_idx];
    }
    /* If too similar to previous stepsize or larger, cut in half */
    if((delta[b_idx]-prev_delta[b_idx]) >= -EPSILON)
    {
      delta[b_idx] = prev_delta[b_idx]/2.;
    }
    /* If negative or zero, stop! */
    if(delta[b_idx] < EPSILON)
    {
      cont_find[b_idx] = false;
    }
  }
}


void quad_interp_cuda(torch::Tensor grad0,
                      torch::Tensor delta, torch::Tensor prev_delta,
                      torch::Tensor improve, torch::Tensor cont_find, torch::Tensor max_step,
                      double EPSILON)
{
  const auto batch_size = grad0.size(0);
  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad0.type(), "quad_interp_cuda", ([&]
  {
    quad_interp_cuda_kernel<scalar_t><<<blocks, threads>>>(
      grad0.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      delta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      prev_delta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      improve.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      cont_find.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      max_step.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, EPSILON);
  }));
}
