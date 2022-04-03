#include <torch/extension.h>

#include <vector>

// CUDA declarations
void check_s_active_cuda(torch::Tensor s, torch::Tensor s_active, torch::Tensor s_num_active);
void setup_l_s_cuda(torch::Tensor l_s, torch::Tensor s_active, torch::Tensor s_num_active);
void pad_ltl_diag_cuda(torch::Tensor ltl, torch::Tensor s_num_active);
void setup_sign_all_cuda(torch::Tensor sign_all,
                         torch::Tensor sign_z,
                         torch::Tensor s_active,
                         torch::Tensor s_num_active);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_s_active(torch::Tensor s,
                    torch::Tensor s_active,
                    torch::Tensor s_num_active)
{
  CHECK_INPUT(s);
  CHECK_INPUT(s_active);
  CHECK_INPUT(s_num_active);
  check_s_active_cuda(s, s_active, s_num_active);
}

void setup_l_s(torch::Tensor l_s, torch::Tensor s_active, torch::Tensor s_num_active)
{
  CHECK_INPUT(l_s);
  CHECK_INPUT(s_active);
  CHECK_INPUT(s_num_active);
  setup_l_s_cuda(l_s, s_active, s_num_active);
}

void pad_ltl_diag(torch::Tensor ltl, torch::Tensor s_num_active)
{
  CHECK_INPUT(ltl);
  CHECK_INPUT(s_num_active);
  pad_ltl_diag_cuda(ltl, s_num_active);
}

void setup_sign_all(torch::Tensor sign_all,
                    torch::Tensor sign_z,
                    torch::Tensor s_active,
                    torch::Tensor s_num_active)
{
  CHECK_INPUT(sign_all);
  CHECK_INPUT(sign_z);
  CHECK_INPUT(s_active);
  CHECK_INPUT(s_num_active);
  setup_sign_all_cuda(sign_all, sign_z, s_active, s_num_active);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("check_s_active",  &check_s_active,  "CHECK_S_ACTIVE (CUDA)");
  m.def("setup_l_s", &setup_l_s, "SETUP_L_S (CUDA)" );
  m.def("pad_ltl_diag", &pad_ltl_diag, "PAD_LTL_DIAG (CUDA)" );
  m.def("setup_sign_all", &setup_sign_all, "SETUP_SIGN_ALL (CUDA)");
}
