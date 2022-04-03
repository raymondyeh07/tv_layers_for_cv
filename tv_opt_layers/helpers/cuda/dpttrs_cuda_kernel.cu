// Implements CUDA Kernel for DPTTRS

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void dpttrs_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> e,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> b,
  torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> vec_size,
  torch::PackedTensorAccessor<bool,1,torch::RestrictPtrTraits,size_t> cont_update,
  int kk)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k<kk && cont_update[k])
  {
    const auto n = vec_size[k];
    for(int i = 1; i < n; ++i)
    {
      b[k][i] -= b[k][i-1]*e[k][i-1];
    }
    b[k][n-1] /= d[k][n-1];
    for (int i=n-2; i>-1; --i)
    {
      b[k][i] = b[k][i]/d[k][i] - b[k][i+1]*e[k][i];
    }
  }
}


std::vector<torch::Tensor>  dpttrs_cuda(torch::Tensor d, torch::Tensor e, torch::Tensor b,
                                        torch::Tensor vec_size, torch::Tensor cont_update)
{
  const auto batch_size = d.size(0);
  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(d.type(), "dpttrs_cuda", ([&]
  {
    dpttrs_cuda_kernel<scalar_t><<<blocks, threads>>>(
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      e.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      b.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      vec_size.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size);
  }));
  return {b};
}


