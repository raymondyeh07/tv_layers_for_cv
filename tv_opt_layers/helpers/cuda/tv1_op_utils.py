"""Class Custom CUDA DPTTRF"""

import torch
import tv1_op_utils_cuda


def reduced_hessian(aux, aux2, inactive, num_inactive, cont_update):
  tv1_op_utils_cuda.reduced_hessian(
      aux, aux2, inactive, num_inactive, cont_update)


def setup_cholesky(g, d, inactive, num_inactive, cont_update):
  tv1_op_utils_cuda.setup_cholesky(g, d, inactive, num_inactive, cont_update)


def project_point(aux, w, d, delta, inactive, num_inactive, cont_find):
  tv1_op_utils_cuda.project_point(
      aux, w, d, delta, inactive, num_inactive, cont_find)


def project_point_p(aux, w, d, delta, inactive, num_inactive, cont_find):
  tv1_op_utils_cuda.project_point_p(
      aux, w, d, delta, inactive, num_inactive, cont_find)


def get_step_sizes(step_sizes, w, d, lmbd, inactive, num_inactive):
  tv1_op_utils_cuda.get_step_sizes(
      step_sizes, w, d, lmbd, inactive, num_inactive)


def get_gradient(grad0, w, d, y, lmbd, inactive, num_inactive, EPSILON):
  tv1_op_utils_cuda.get_gradient(
      grad0, w, d, y, lmbd, inactive, num_inactive, EPSILON)


def get_grd(grd, g, d, inactive, num_inactive):
  tv1_op_utils_cuda.get_grd(grd, g, d, inactive, num_inactive)


def check_inactive(cond, inactive, num_inactive):
  tv1_op_utils_cuda.check_inactive(cond, inactive, num_inactive)


def quad_interp(grad0, delta, prev_delta, improve, found, max_step, EPSILON):
  tv1_op_utils_cuda.quad_interp(
      grad0, delta, prev_delta, improve, found, max_step, EPSILON)


def dual_to_primal(w, x, y, cont_update=None):
  if cont_update is None:
    tv1_op_utils_cuda.dual_to_primal(w, x, y)
  else:
    if len(w.shape) == 3:
      tv1_op_utils_cuda.dual_to_primal_cont_p(w, x, y, cont_update)
    elif len(w.shape) == 2:
      tv1_op_utils_cuda.dual_to_primal_cont(w, x, y, cont_update)


def assign_cont_update(src, dst, cont_update):
  num_dim = len(src.shape)
  if num_dim == 1:
    tv1_op_utils_cuda.assign_cont_update0(src, dst, cont_update)
  elif num_dim == 2:
    tv1_op_utils_cuda.assign_cont_update1(src, dst, cont_update)