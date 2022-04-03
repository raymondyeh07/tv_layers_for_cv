"""Unit tests for checking gradient."""
import unittest

import functools
import torch
from torch.autograd import gradcheck

from tv_opt_layers.ops.proximity_tv_cuda import ProxTV_l1_cuda


class TestGradients(unittest.TestCase):
  def _test_backward(self, solve):
    for seed in range(10):
      torch.manual_seed(seed)
      N, C = 3, 5
      signal = torch.randn(N, C).double()
      signal.requires_grad = True
      signal = signal.cuda()
      lmbd = torch.ones(N, 1).cuda().double()
      lmbd.requires_grad = True
      gradcheck(solve, [signal, lmbd])

  def test_prox_tv_batched_backward(self):
    solve = ProxTV_l1_cuda.apply
    self._test_backward(solve)


if __name__ == '__main__':
  unittest.main()
