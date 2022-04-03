"""Class Custom CUDA DPTTRF"""

import torch
import tv1_backward_cuda


def check_s_active(s, s_active, s_num_active):
  tv1_backward_cuda.check_s_active(s, s_active, s_num_active)


def setup_l_s(l_s, s_active, s_num_active):
  tv1_backward_cuda.setup_l_s(l_s, s_active, s_num_active)


def pad_ltl_diag(ltl, s_num_active):
  tv1_backward_cuda.pad_ltl_diag(ltl, s_num_active)


def setup_sign_all(sign_all, sign_z, s_active, s_num_active):
  tv1_backward_cuda.setup_sign_all(sign_all, sign_z, s_active, s_num_active)