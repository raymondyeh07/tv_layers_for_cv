"""Implements an General (Sharp/Smooth) 2D TVL1  layer."""
from .l1_tv_2d_layer import L1TV2DLayer

import torch.nn as nn


class GeneralTV2DLayer(nn.Module):
  def __init__(self, num_channels=1,
               lmbd_mode='learned',
               lmbd_init=-1,
               lmbd_zero=-1,
               filt_mode='smooth', num_iter=4):
    super(GeneralTV2DLayer, self).__init__()
    self.lmbd_mode = lmbd_mode
    self.lmbd_init = lmbd_init
    self.lmdb_zero = lmbd_zero
    self.filt_mode = filt_mode
    self.tv_layer = L1TV2DLayer(lmbd_mode=lmbd_mode,
                                lmbd_init=lmbd_init,
                                lmbd_zero=lmbd_zero,
                                num_channels=num_channels,
                                num_iter=num_iter)

  def forward(self, x):
    x = x.contiguous()
    if self.filt_mode == 'smooth':
      return self.forward_smooth(x)
    elif self.filt_mode == 'sharp':
      return self.forward_sharp(x)

  def forward_smooth(self, x):
    return self.tv_layer(x)

  def forward_sharp(self, x):
    return 2.*x-self.tv_layer(x)
