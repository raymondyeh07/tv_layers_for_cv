"""Implements an 2D TVL1  layer."""

import torch
import torch.nn as nn

from .l1_tv_1d_layer import L1TV1DLayer


class L1TV2DLayer(nn.Module):
  def __init__(self, lmbd_mode='learned', lmbd_init=-1, num_channels=1,
               lmbd_zero=-1, dtype=torch.float, num_iter=4):
    super(L1TV2DLayer, self).__init__()
    self.solve = L1TV1DLayer(lmbd_mode=lmbd_mode, lmbd_init=lmbd_init,
                             lmbd_zero=lmbd_zero, num_channels=num_channels,
                             dtype=dtype)
    self.num_iter = num_iter

  def forward(self, y, alpha=None):
    """
      y: (N, C, H, W)
    """
    X = y.clone()
    P = torch.zeros_like(y)
    Q = torch.zeros_like(y)
    for _ in range(self.num_iter):
      Z = self.solve(X+P)
      P += X-Z
      tmp = Z+Q
      X = self.solve(tmp, direction='col')
      Q += Z-X
    return X
