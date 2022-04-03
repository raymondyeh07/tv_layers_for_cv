"""Class Custom CUDA DPTTRF"""

import torch
import dpttr_cuda


def batch_factorize(d, e, n=None, cont_update=None):
  if n is None:
    n = torch.ones(d.shape[0], dtype=torch.long, device=d.device)*d.shape[1]
  if cont_update is None:
    cont_update = torch.ones(d.shape[0], dtype=bool, device=d.device)
  return dpttr_cuda.factorize(d, e, n, cont_update)


def batch_solve(d, e, b, n=None, cont_update=None):
  if n is None:
    n = torch.ones(d.shape[0], dtype=torch.long, device=d.device)*d.shape[1]
  if cont_update is None:
    cont_update = torch.ones(d.shape[0], dtype=bool, device=d.device)
  return dpttr_cuda.solve(d, e, b, n, cont_update)
