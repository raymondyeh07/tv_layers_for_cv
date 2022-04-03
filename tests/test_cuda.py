"""Unit tests for cuda helpers"""
import unittest
import numpy as np
import scipy
import scipy.linalg

import torch
import prox_tv
from tv_opt_layers.helpers.cuda import dpttr
from tv_opt_layers.ops import batched_tvl1_op


class TestTVOps(unittest.TestCase):
  def test_batched_tvl1(self):
    sigma = 0.5
    KK = 10
    NN = 5
    for seed in range(0, 10):
      torch.manual_seed(seed)
      y = torch.randn(KK, NN, device='cuda').double()
      lmbd = 1*torch.ones(KK, 1, device='cuda').double()
      out1 = batched_tvl1_op.pn_tv1(y, lmbd, sigma=sigma)
      out2 = []
      for k in range(KK):
        out = prox_tv.tv1_1d(y[k].cpu().numpy(), lmbd[k]
                             [0].item(), method='pn', sigma=sigma)
        out2.append(np.expand_dims(out, 0))
      out2 = np.concatenate(out2, 0)
      np.allclose(out2, out1.cpu().numpy())


class TestDPTTRS(unittest.TestCase):
  """Tests for factorizing and solving symmetric tridiagonal system."""

  def setUp(self):
    batch_size = 10
    input_size = 5
    # Create Positive Definite A.
    A = torch.randn(batch_size, input_size, input_size)
    A = A.bmm(A.transpose(-1, -2))/2 + input_size*torch.eye(input_size)

    self.d_in1 = torch.diagonal(A, dim1=-2, dim2=-1)
    self.e_in1 = torch.diagonal(A, offset=1, dim1=-2, dim2=-1)

  def _copy_from_self(self):
    d_in1 = self.d_in1.clone().double().cuda()
    e_in1 = self.e_in1.clone().double().cuda()
    d_in2 = np.array(d_in1.clone().detach().cpu().numpy())
    e_in2 = np.array(e_in1.clone().detach().cpu().numpy())
    return d_in1, d_in2, e_in1, e_in2

  def test_dpttrf_vs_lapack(self):
    d_in1, d_in2, e_in1, e_in2 = self._copy_from_self()
    dpttr.batch_factorize(d_in1, e_in1)
    for b_idx in range(d_in1.shape[0]):
      scipy.linalg.lapack.dpttrf(
          d_in2[b_idx:b_idx+1], e_in2[b_idx:b_idx+1], True, True)
    diff = d_in1.cpu().numpy()-d_in2
    diff = d_in1.cpu().numpy()-d_in2
    max_idx = np.abs(diff).mean(-1).argmax()
    assert np.allclose(d_in1.cpu().numpy(), d_in2)
    assert np.allclose(e_in1.cpu().numpy(), e_in2)

  def test_dpttrs_vs_lapack(self):
    d_in1, d_in2, e_in1, e_in2 = self._copy_from_self()
    b_in1 = torch.randn(d_in1.shape[0], d_in1.shape[1]).double().cuda()
    b_in2 = b_in1.clone().unsqueeze(-1).cpu().numpy()
    dpttr.batch_solve(d_in1, e_in1, b_in1)
    for b_idx in range(d_in1.shape[0]):
      info = scipy.linalg.lapack.dpttrs(
          d_in2[b_idx], e_in2[b_idx], b_in2[b_idx])
      assert np.allclose(b_in1[b_idx].cpu().numpy(), info[0][..., 0])


if __name__ == '__main__':
  unittest.main()
