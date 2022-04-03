// Implements CUDA Kernel for DPTTRF

#include <torch/extension.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void get_gradient_cuda_kernel(
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad0,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> w,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> lmbd,
  torch::PackedTensorAccessor<long, 2,torch::RestrictPtrTraits,size_t> inactive,
  torch::PackedTensorAccessor<long, 1,torch::RestrictPtrTraits,size_t> num_inactive,
  long batch_size, long vec_size, long lmbd_batch_size, double EPSILON)
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
    const auto ind = inactive[b_idx][v_idx];
    if (lmbd_batch_size > 1)
      l_tmp = lmbd[b_idx][0];
    if (v_idx == 0 && ind ==0)
    {
      const bool c1 {!((fabsf(w[b_idx][0]-l_tmp)<EPSILON) && d[b_idx][0] > 0)};
      const bool c2 {!((fabsf(w[b_idx][0]-l_tmp)<EPSILON) && d[b_idx][0] < 0)};
      if (c1 && c2)
        grad0[b_idx][0] = -d[b_idx][0]*(2*w[b_idx][0] - w[b_idx][1] - y[b_idx][1] + y[b_idx][0]);
    }
    if (v_idx > 0 && v_idx < e_idx-1)
    {
      const bool c1 {!((fabsf(w[b_idx][ind]-l_tmp)<EPSILON) && d[b_idx][v_idx] > 0)};
      const bool c2 {!((fabsf(w[b_idx][ind]-l_tmp)<EPSILON) && d[b_idx][v_idx] < 0)};
      if (c1 && c2)
        grad0[b_idx][v_idx] = -d[b_idx][v_idx] * (2*w[b_idx][ind] - w[b_idx][ind+1] \
                              - w[b_idx][ind-1] - y[b_idx][ind+1] + y[b_idx][ind]);
    }
    if (v_idx == e_idx-1 && ind == vec_size-1)
    {
      const bool c1 {!((fabs(w[b_idx][ind]-l_tmp)<EPSILON) && d[b_idx][v_idx] > 0)};
      const bool c2 {!((fabs(w[b_idx][ind]-l_tmp)<EPSILON) && d[b_idx][v_idx] < 0)};
      if (c1 && c2)
        grad0[b_idx][v_idx] = -d[b_idx][v_idx] * (2*w[b_idx][ind] - w[b_idx][ind-1] \
                              - y[b_idx][ind+1] + y[b_idx][ind]);
    }
  }
}

void get_gradient_cuda(torch::Tensor grad0, torch::Tensor w,
                       torch::Tensor d, torch::Tensor y,
                       torch::Tensor lmbd,
                       torch::Tensor inactive,
                       torch::Tensor num_inactive, double EPSILON)
{
  const auto batch_size = grad0.size(0);
  const auto vec_size = grad0.size(1);
  const auto lmbd_batch_size = lmbd.size(0);

  const int threads = 1024;
  const dim3 blocks((batch_size + threads - 1) / threads, vec_size);

  AT_DISPATCH_FLOATING_TYPES(grad0.type(), "get_gradient_cuda", ([&]
  {
    get_gradient_cuda_kernel<scalar_t><<<blocks, threads>>>(
      grad0.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      w.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      d.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      lmbd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      inactive.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      num_inactive.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
      batch_size, vec_size, lmbd_batch_size, EPSILON);
  }));
}
