"""Implements some helper fuctions."""

import torch
from tv_opt_layers.helpers.cuda import tv1_op_utils

EPSILON_DOUBLE = 1e-10
EPSILON_SINGLE = 1e-5


def get_epsilon(x):
  if x.dtype == torch.float32:
    return EPSILON_SINGLE
  elif x.dtype == torch.float64:
    return EPSILON_DOUBLE
  else:
    raise TypeError('Need float32 or float64')


def projection(w, lmbd):
  if len(w.shape) == 3:
    tmp = lmbd.unsqueeze(-1).expand(-1, 1, 1)
  else:
    tmp = lmbd.expand(-1, 1)
  w.clamp_(-tmp, tmp)


def dual2primal(w, x, y, cont_find=None):
  if len(w.shape) == 3:
    y = y.unsqueeze(-1)
  if cont_find is not None:
    x[cont_find, 0] = y[cont_find, 0]+w[cont_find, 0]
    x[cont_find, 1:-1] = y[cont_find, 1:-1]-w[cont_find, :-1]+w[cont_find, 1:]
    x[cont_find, -1] = y[cont_find, -1]-w[cont_find, -1]
  else:
    x[:, 0] = y[:, 0]+w[:, 0]
    x[:, 1:-1] = y[:, 1:-1]-w[:, :-1]+w[:, 1:]
    x[:, -1] = y[:, -1]-w[:, -1]


def dual2primal_cuda(w, x, y):
  tv1_op_utils.dual_to_primal(w, x, y)


def primal2grad(x):
  return x[:, :-1]-x[:, 1:]


def primal2val(x):
  return 0.5*(torch.square(x).sum(1))


def grad2gap(g, w, lmbd):
  gap = torch.abs(g)*lmbd + w*g
  return gap.sum(-1)


def check_inactive(w, g, inactive, num_inactive, lmbd):
  EPSILON = get_epsilon(w)
  cond1 = torch.abs(w) < lmbd
  cond2 = torch.logical_and(torch.abs(w+lmbd) < EPSILON, g < -EPSILON)
  cond3 = torch.logical_and(torch.abs(w - lmbd) < EPSILON, g > EPSILON)
  cond123 = (cond1+cond2+cond3) > 0
  cond123 = torch.logical_or(torch.logical_or(cond1, cond2), cond3)
  num_inactive[:] = torch.sum(cond123, -1)
  bb_idx, vv_idx = cond123.nonzero(as_tuple=True)
  # This is a bottleneck.
  cc_idx = torch.cat([torch.arange(k, device=w.device) for k in num_inactive])
  inactive[bb_idx, cc_idx] = vv_idx
  return num_inactive


def check_inactive_cuda(w, g, inactive, num_inactive, lmbd):
  EPSILON = get_epsilon(w)
  cond1 = torch.abs(w) < lmbd
  cond2 = torch.logical_and(torch.abs(w+lmbd) < EPSILON, g < -EPSILON)
  cond3 = torch.logical_and(torch.abs(w - lmbd) < EPSILON, g > EPSILON)
  cond123 = (cond1+cond2+cond3) > 0
  tv1_op_utils.check_inactive(cond123, inactive, num_inactive)
