"""Unit tests for cuda helpers."""
import unittest

import torch
import prox_tv
from tv_opt_layers.layers.l1_tv_1d_layer import L1TV1DLayer
from tv_opt_layers.layers.l1_tv_2d_layer import L1TV2DLayer


class TestTVLayers(unittest.TestCase):
  def test_forward_1d_identity(self):
    batch_size, num_channel, height, width = 2, 1, 1, 10
    x = torch.ones(batch_size, num_channel, height, width)
    x[:, :, :, width//2:] = 0
    solve = L1TV1DLayer(lmbd_mode='fixed', lmbd_init=0.0).cuda()
    y = solve(x.cuda())
    assert torch.allclose(x.cuda(), y.cuda())

  def test_backward_1d_identity(self):
    batch_size, num_channel, height, width = 2, 1, 1, 10
    x = torch.ones(batch_size, num_channel, height, width)
    x[:, :, :, width//2:] = 0
    solve = L1TV1DLayer(lmbd_mode='learned', lmbd_init=1).cuda()
    optimizer = torch.optim.Adam(solve.parameters(), lr=0.1)
    x = x.cuda()
    y = solve(x.cuda())
    init_loss = torch.mean(torch.abs(y-x))
    for _ in range(50):
      y = solve(x.cuda())
      loss = torch.mean(torch.abs(y-x))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    # Make sure lmbd decreases and loss goes down.
    assert torch.nn.functional.softplus(solve.lmbd)[0].item() < 1.0
    assert init_loss.item() > loss.item()

  def test_forward_2d_identity(self):
    batch_size, num_channel, height, width = 2, 1, 10, 10
    x = torch.ones(batch_size, num_channel, height, width)
    x[:, :, height//2:, :] = 0
    solve = L1TV2DLayer(lmbd_mode='fixed', lmbd_init=0.0).cuda()
    y = solve(x.cuda())
    assert torch.allclose(x.cuda(), y.cuda())

  def test_backward_2d_identity(self):
    batch_size, num_channel, height, width = 2, 1, 10, 10
    x = torch.ones(batch_size, num_channel, height, width)
    x[:, :, height//2:, :] = 0
    solve = L1TV2DLayer(lmbd_mode='learned', lmbd_init=1).cuda()
    optimizer = torch.optim.Adam(solve.parameters(), lr=0.1)
    x = x.cuda()
    y = solve(x.cuda())
    init_loss = torch.mean(torch.abs(y-x))
    for _ in range(50):
      y = solve(x.cuda())
      loss = torch.mean(torch.abs(y-x))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    # Make sure lmbd decreases and loss goes down.
    assert torch.nn.functional.softplus(solve.solve.lmbd)[0].item() < 1.0
    assert init_loss.item() > loss.item()

  def test_forward_2d_gauss_prox_tv(self):
    batch_size, num_channel, height, width = 2, 1, 10, 10
    x = torch.ones(batch_size, num_channel, height, width).double()
    x[:, :, height//2:, :] = 0
    solve = L1TV2DLayer(lmbd_mode='fixed', lmbd_init=0.1,
                        num_iter=1, dtype=x.dtype).cuda()
    x = x.cuda()
    y_cuda = solve(x, alpha=0.1)
    y_np = prox_tv.tv1_2d(x[0, 0].cpu().numpy(), 0.1, max_iters=1, method='pd')
    assert torch.allclose(torch.tensor(y_np).cuda(), y_cuda)


if __name__ == '__main__':
  unittest.main()
