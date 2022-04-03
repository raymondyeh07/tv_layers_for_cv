// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void setup_l_s_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits,size_t> l_s,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> s_active,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> s_num_active,
  long batch_size, long row_size, long col_size)
{
  // Batch index
  const int r_idx = blockIdx.z;
  // Row index
  const int c_idx = blockIdx.y * blockDim.y + threadIdx.y;
  // Column index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Check within range
  if(b_idx < batch_size)
  {
    const auto max_s = s_num_active[b_idx];
    if (r_idx < row_size && c_idx < max_s)
    {
      const auto s_idx = s_active[b_idx][c_idx];
      if (r_idx <= s_idx)
        l_s[b_idx][r_idx][c_idx] = 1;
    }
  }
}


void setup_l_s_cuda(torch::Tensor l_s,
                    torch::Tensor s_active,
                    torch::Tensor s_num_active)
{
  const auto batch_size = l_s.size(0);
  const auto row_size = l_s.size(1);
  const auto col_size = l_s.size(2);

  const int threads = 32;
  const dim3 DimGrid(32, 32);
  const dim3 DimBlock((batch_size + threads - 1) / threads, (row_size + threads - 1) / threads, col_size);


  AT_DISPATCH_FLOATING_TYPES(l_s.type(), "setup_l_s_cuda", ([&]
  {
    setup_l_s_cuda_kernel<scalar_t><<<DimBlock, DimGrid>>>(
      l_s.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      s_active.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      s_num_active.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, row_size, col_size);
  }));
}
