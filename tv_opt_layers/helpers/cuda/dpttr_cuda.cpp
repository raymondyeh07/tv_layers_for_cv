#include <torch/extension.h>

#include <vector>

// CUDA declarations

std::vector<torch::Tensor> dpttrf_cuda(torch::Tensor d, torch::Tensor e, torch::Tensor vec_size, torch::Tensor cont_update);
std::vector<torch::Tensor> dpttrs_cuda(torch::Tensor d, torch::Tensor e, torch::Tensor b, torch::Tensor vec_size, torch::Tensor cont_update);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> dpttrf(torch::Tensor d, torch::Tensor e, torch::Tensor vec_size, torch::Tensor cont_update)
{
  CHECK_INPUT(d);
  CHECK_INPUT(e);
  CHECK_INPUT(vec_size);
  CHECK_INPUT(cont_update);
  return dpttrf_cuda(d,e,vec_size,cont_update);
}

std::vector<torch::Tensor> dpttrs(torch::Tensor d, torch::Tensor e, torch::Tensor b,
                                  torch::Tensor vec_size, torch::Tensor cont_update)
{
  CHECK_INPUT(d);
  CHECK_INPUT(e);
  CHECK_INPUT(b);
  CHECK_INPUT(vec_size);
  CHECK_INPUT(cont_update);
  return dpttrs_cuda(d,e,b,vec_size, cont_update);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("factorize", &dpttrf, "DPTTRF (CUDA)");
  m.def("solve", &dpttrs, "DPTTRS (CUDA)");
}
