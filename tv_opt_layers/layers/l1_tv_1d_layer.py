"""Implements an 1D TVL1  layer."""

import torch
import torch.nn as nn

from tv_opt_layers.ops.proximity_tv_cuda import ProxTV_l1_cuda


class L1TV1DLayer(nn.Module):
  def __init__(self, lmbd_mode='learned', lmbd_init=-1, lmbd_zero=-1, num_channels=1,
               direction='row', dtype=torch.float):
    """
      lmbd_zero: specifies lmbd when parameter = 0; this changes weight decay reg.
    """
    super(L1TV1DLayer, self).__init__()
    assert lmbd_mode in ['learned', 'fixed', 'given']
    self.lmbd_mode = lmbd_mode
    self.lmbd_offset = 0
    if lmbd_zero >= 0:
      self.lmbd_offset = lmbd_zero + \
          torch.log(-torch.expm1(-torch.tensor(lmbd_zero))+1e-8)
    self.num_channels = num_channels
    assert direction in ['row', 'col']
    self.direction = direction

    if lmbd_init < 0:  # random initialize.
      lmbd_init = torch.rand(num_channels)
    # Inverse of softplus for correct initialization.
    self.lmbd_init = lmbd_init + \
        torch.log(-torch.expm1(-torch.tensor(lmbd_init)))-self.lmbd_offset
    if lmbd_mode in ['learned', 'fixed']:
      self.lmbd = nn.parameter.Parameter(self.lmbd_init*torch.ones(1, num_channels, dtype=dtype),
                                         requires_grad=lmbd_mode == 'learned')
    else:
      self.lmbd = None
    self.solve = ProxTV_l1_cuda.apply

  def get_lmbd_val(self, input_lmbd=None):
    if input_lmbd is not None:
      return nn.functional.softplus(input_lmbd+self.lmbd_offset)
    else:
      return nn.functional.softplus(self.lmbd+self.lmbd_offset)

  def forward(self, y, input_lmbd=None, direction=None):
    """
      y: (N, C, H, W) tensor.
      input_lmbd: (N,1) tensor.
    """
    # Each channels are propcessed separately.
    if not direction:
      direction = self.direction

    if direction == 'col':
      y = y.transpose(-1, -2)

    N, C, H, W = y.shape

    y_flat = y.reshape(-1, W)

    if input_lmbd is not None:
      input_lmbd = input_lmbd.squeeze()
      if len(input_lmbd.shape) == 1:
        input_lmbd = input_lmbd.unsqueeze(-1).repeat(1, C, H).reshape(-1, 1)
      elif len(input_lmbd.shape) == 2:
        input_lmbd = input_lmbd.unsqueeze(-1).repeat(1, 1, H).reshape(-1, 1)
      else:
        assert False  # This should not happen.
    else:
      input_lmbd = self.lmbd
      if self.num_channels != 1:
        input_lmbd = input_lmbd.unsqueeze(-1).repeat(N, 1, H).reshape(-1, 1)
    ret_flat = self.solve(y_flat, self.get_lmbd_val(input_lmbd=input_lmbd))
    ret = ret_flat.reshape(N, C, H, W)

    if direction == 'col':
      ret = ret.transpose(-1, -2)
    return ret
