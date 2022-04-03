#include <torch/extension.h>

#include <vector>

// CUDA declarations
void reduced_hessian_cuda(torch::Tensor aux,
                          torch::Tensor aux2,
                          torch::Tensor inactive,
                          torch::Tensor num_inactive,
                          torch::Tensor cont_update);

void setup_cholesky_cuda(torch::Tensor g,
                         torch::Tensor d,
                         torch::Tensor inactive,
                         torch::Tensor num_inactive,
                         torch::Tensor cont_update);

void project_point_cuda(torch::Tensor aux, torch::Tensor w,
                        torch::Tensor d, torch::Tensor delta,
                        torch::Tensor inactive,
                        torch::Tensor num_inactive,
                        torch::Tensor cont_find);

void project_point_p_cuda(torch::Tensor aux, torch::Tensor w,
                          torch::Tensor d, torch::Tensor delta,
                          torch::Tensor inactive,
                          torch::Tensor num_inactive,
                          torch::Tensor cont_find);

void get_step_sizes_cuda(torch::Tensor step_sizes, torch::Tensor w,
                         torch::Tensor d, torch::Tensor lmbd,
                         torch::Tensor inactive, torch::Tensor num_inactive);

void get_gradient_cuda(torch::Tensor grad0, torch::Tensor w, torch::Tensor d,
                       torch::Tensor y, torch::Tensor lmbd, torch::Tensor inactive, torch::Tensor num_inactive,
                       double EPSILON);

void get_grd_cuda(torch::Tensor gRd, torch::Tensor g, torch::Tensor d,
                  torch::Tensor inactive, torch::Tensor num_inactive);

void check_inactive_cuda(torch::Tensor cond,
                         torch::Tensor inactive, torch::Tensor num_inactive);

void quad_interp_cuda(torch::Tensor grad0, torch::Tensor delta,
                      torch::Tensor prev_delta, torch::Tensor improve,
                      torch::Tensor cont_find, torch::Tensor max_step, double EPSILON);

void dual_to_primal_cuda(torch::Tensor w, torch::Tensor x, torch::Tensor y);

void dual_to_primal_cont_cuda(torch::Tensor w, torch::Tensor x, torch::Tensor y, torch::Tensor cont_update);

void dual_to_primal_cont_p_cuda(torch::Tensor w, torch::Tensor x, torch::Tensor y, torch::Tensor cont_update);

void assign_cont_update0_cuda(torch::Tensor src, torch::Tensor dst, torch::Tensor cont_update);

void assign_cont_update1_cuda(torch::Tensor src, torch::Tensor dst, torch::Tensor cont_update);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void reduced_hessian(torch::Tensor aux,
                     torch::Tensor aux2,
                     torch::Tensor inactive,
                     torch::Tensor num_inactive,
                     torch::Tensor cont_update)
{
  CHECK_INPUT(aux);
  CHECK_INPUT(aux2);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  CHECK_INPUT(cont_update);
  reduced_hessian_cuda(aux,aux2,inactive,num_inactive,cont_update);
}

void setup_cholesky(torch::Tensor g,torch::Tensor d,
                    torch::Tensor inactive,
                    torch::Tensor num_inactive,
                    torch::Tensor cont_update)
{
  CHECK_INPUT(g);
  CHECK_INPUT(d);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  CHECK_INPUT(cont_update);
  setup_cholesky_cuda(g,d,inactive,num_inactive, cont_update);
}

void project_point(torch::Tensor aux, torch::Tensor w,
                   torch::Tensor d, torch::Tensor delta,
                   torch::Tensor inactive,
                   torch::Tensor num_inactive,
                   torch::Tensor cont_find)
{
  CHECK_INPUT(aux);
  CHECK_INPUT(w);
  CHECK_INPUT(d);
  CHECK_INPUT(delta);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  CHECK_INPUT(cont_find);
  project_point_cuda(aux, w, d, delta, inactive, num_inactive, cont_find);
}

void project_point_p(torch::Tensor aux, torch::Tensor w,
                     torch::Tensor d, torch::Tensor delta,
                     torch::Tensor inactive,
                     torch::Tensor num_inactive,
                     torch::Tensor cont_find)
{
  CHECK_INPUT(aux);
  CHECK_INPUT(w);
  CHECK_INPUT(d);
  CHECK_INPUT(delta);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  CHECK_INPUT(cont_find);
  project_point_p_cuda(aux, w, d, delta, inactive, num_inactive, cont_find);
}


void get_step_sizes(torch::Tensor step_sizes, torch::Tensor w,
                    torch::Tensor d, torch::Tensor lmbd,
                    torch::Tensor inactive, torch::Tensor num_inactive)
{
  CHECK_INPUT(step_sizes);
  CHECK_INPUT(w);
  CHECK_INPUT(d);
  CHECK_INPUT(lmbd);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  get_step_sizes_cuda(step_sizes, w, d, lmbd, inactive, num_inactive);
}

void get_gradient(torch::Tensor grad0, torch::Tensor w, torch::Tensor d,
                  torch::Tensor y,
                  torch::Tensor lmbd,
                  torch::Tensor inactive, torch::Tensor num_inactive, double EPSILON)
{
  CHECK_INPUT(grad0);
  CHECK_INPUT(w);
  CHECK_INPUT(d);
  CHECK_INPUT(y);
  CHECK_INPUT(lmbd);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  get_gradient_cuda(grad0, w, d, y, lmbd, inactive, num_inactive, EPSILON);
}

void get_grd(torch::Tensor gRd, torch::Tensor g,
             torch::Tensor d,
             torch::Tensor inactive, torch::Tensor num_inactive)
{
  CHECK_INPUT(gRd);
  CHECK_INPUT(g);
  CHECK_INPUT(d);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  get_grd_cuda(gRd, g, d, inactive, num_inactive);
}

void check_inactive(torch::Tensor cond,
                    torch::Tensor inactive, torch::Tensor num_inactive)
{
  CHECK_INPUT(cond);
  CHECK_INPUT(inactive);
  CHECK_INPUT(num_inactive);
  check_inactive_cuda(cond, inactive, num_inactive);
}

void quad_interp(torch::Tensor grad0, torch::Tensor delta,
                 torch::Tensor prev_delta, torch::Tensor improve,
                 torch::Tensor cont_find, torch::Tensor max_step, double EPSILON)
{
  CHECK_INPUT(grad0);
  CHECK_INPUT(delta);
  CHECK_INPUT(prev_delta);
  CHECK_INPUT(improve);
  CHECK_INPUT(cont_find);
  CHECK_INPUT(max_step);
  quad_interp_cuda(grad0, delta, prev_delta, improve, cont_find, max_step, EPSILON);
}

void dual_to_primal(torch::Tensor w, torch::Tensor x, torch::Tensor y)
{
  CHECK_INPUT(w);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  dual_to_primal_cuda(w,x,y);
}

void dual_to_primal_cont(torch::Tensor w, torch::Tensor x, torch::Tensor y, torch::Tensor cont_update)
{
  CHECK_INPUT(w);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(cont_update);
  dual_to_primal_cont_cuda(w,x,y, cont_update);
}

void dual_to_primal_cont_p(torch::Tensor w, torch::Tensor x, torch::Tensor y, torch::Tensor cont_update)
{
  CHECK_INPUT(w);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(cont_update);
  dual_to_primal_cont_p_cuda(w,x,y, cont_update);
}

void assign_cont_update0(torch::Tensor src, torch::Tensor dst, torch::Tensor cont_update)
{
  CHECK_INPUT(src);
  CHECK_INPUT(dst);
  CHECK_INPUT(cont_update);
  assign_cont_update0_cuda(src, dst, cont_update);
}

void assign_cont_update1(torch::Tensor src, torch::Tensor dst, torch::Tensor cont_update)
{
  CHECK_INPUT(src);
  CHECK_INPUT(dst);
  CHECK_INPUT(cont_update);
  assign_cont_update1_cuda(src, dst, cont_update);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("reduced_hessian",        &reduced_hessian,       "REDUCED_HESSIAN (CUDA)");
  m.def("setup_cholesky",         &setup_cholesky,        "SETUP_CHOLESKY (CUDA)");
  m.def("project_point",          &project_point,         "PROJECT_POINT (CUDA)");
  m.def("project_point_p",        &project_point_p,       "PROJECT_POINT_P (CUDA)");
  m.def("get_step_sizes",         &get_step_sizes,        "GET_STEP_SIZES (CUDA)");
  m.def("get_gradient",           &get_gradient,          "GET_GRADIENT (CUDA)");
  m.def("get_grd",                &get_grd,               "GET_GRD (CUDA)");
  m.def("check_inactive",         &check_inactive,        "CHECK_INACTIVE (CUDA)");
  m.def("quad_interp",            &quad_interp,           "QUAD_INTERP (CUDA)");
  m.def("dual_to_primal",         &dual_to_primal,        "DUAL_TO_PRIMAL (CUDA)");
  m.def("dual_to_primal_cont",    &dual_to_primal_cont,   "DUAL_TO_PRIMAL_CONT (CUDA)");
  m.def("dual_to_primal_cont_p",  &dual_to_primal_cont_p, "DUAL_TO_PRIMAL_CONT_P (CUDA)");
  m.def("assign_cont_update0",    &assign_cont_update0,   "ASSIGN_CONT_UPDATE0 (CUDA)");
  m.def("assign_cont_update1",    &assign_cont_update1,   "ASSIGN_CONT_UPDATE1 (CUDA)");
}