// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void dual_to_primal_cont_p_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> w,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
  torch::PackedTensorAccessor<bool, 1,torch::RestrictPtrTraits,size_t> cont_update,
  long batch_size, long vec_size, long d_size)
{
  // Batch index
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Column index
  const int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int d_idx = blockIdx.z;

  // Check within range
  if(b_idx < batch_size && cont_update[b_idx] && v_idx <= vec_size && d_idx < d_size)
  {
    if (v_idx == 0)
      x[b_idx][0][d_idx] = y[b_idx][0]+w[b_idx][0][d_idx];
    else if (v_idx == vec_size)
      x[b_idx][vec_size][d_idx] = y[b_idx][vec_size]-w[b_idx][vec_size-1][d_idx];
    else
      x[b_idx][v_idx][d_idx] = y[b_idx][v_idx]-w[b_idx][v_idx-1][d_idx]+w[b_idx][v_idx][d_idx];
  }
}


void dual_to_primal_cont_p_cuda(torch::Tensor w, torch::Tensor x,
                                torch::Tensor y,
                                torch::Tensor cont_update
                               )
{
  const auto batch_size = x.size(0);
  const auto vec_size = x.size(1);
  const auto d_size = x.size(2);
  const auto nn = vec_size-1;

  const int threads = 32;
  const dim3 DimGrid(32, 32);
  const dim3 DimBlock((batch_size + threads - 1) / threads, (vec_size + threads - 1) / threads, d_size);

  AT_DISPATCH_FLOATING_TYPES(w.type(), "dual_to_primal_cont_p_cuda", ([&]
  {
    dual_to_primal_cont_p_cuda_kernel<scalar_t><<<DimBlock, DimGrid>>>(
      w.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      cont_update.packed_accessor<bool,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, nn, d_size);
  }));
}
