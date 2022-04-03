// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void dpttrf_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> e,
  torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> vec_size,
  torch::PackedTensorAccessor<bool,1,torch::RestrictPtrTraits,size_t> cont_update,
  int kk)
{
  // Batch index
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k<kk && cont_update[k])
  {
    const auto n = vec_size[k];
    for(int i = 0; i < n-1; ++i)
    {
      const auto ei = e[k][i];
      e[k][i] = ei / d[k][i];
      d[k][i+1] -= e[k][i]*ei;
    }
  }
}


std::vector<torch::Tensor>  dpttrf_cuda(torch::Tensor d, torch::Tensor e,
                                        torch::Tensor vec_size,
                                        torch::Tensor cont_update)
{
  const auto batch_size = d.size(0);
  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(d.type(), "dpttrf_cuda", ([&]
  {
    dpttrf_cuda_kernel<scalar_t><<<blocks, threads>>>(
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      e.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      vec_size.packed_accessor<long, 1,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size);
  }));
  return {d, e};
}


