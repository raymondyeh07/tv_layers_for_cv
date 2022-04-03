"""Implements TVL1 function with backward (CUDA)."""
import torch
from tv_opt_layers.ops.batched_tvl1_op import pn_tv1
from tv_opt_layers.helpers.cuda.tv1_backward import check_s_active, setup_l_s, pad_ltl_diag, setup_sign_all

from tv_opt_layers.ops.utils import get_epsilon


class ProxTV_l1_cuda(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, lbda):
    output_gpu = pn_tv1(x, lbda)
    output_gpu.requires_grad = True

    z_gpu = output_gpu.clone()
    z_gpu[:, :-1] -= output_gpu[:, 1:]

    EPSILON = get_epsilon(z_gpu)
    z_gpu[torch.abs(z_gpu) < EPSILON] = 0.
    ctx.save_for_backward(torch.sign(z_gpu))
    ctx.x_need_grad = x.requires_grad
    return output_gpu

  @staticmethod
  def backward(ctx, grad_output):
    batch_size, n_dim = grad_output.shape
    sign_z, = ctx.saved_tensors
    device = grad_output.device
    grad_output = grad_output
    sign_z = sign_z

    S = sign_z != 0
    S[:, -1] = True
    sign_z[:, -1] = 0

    S = S.cuda()
    s_active = torch.zeros(batch_size, n_dim, device=device, dtype=int)
    s_num_active = torch.zeros(batch_size, device=device, dtype=int)
    check_s_active(S, s_active, s_num_active)
    L_S_padded = torch.zeros(batch_size, n_dim, n_dim,
                             device=device, dtype=grad_output.dtype)
    setup_l_s(L_S_padded, s_active, s_num_active)
    LTL_all = L_S_padded.transpose(-1, -2).matmul(L_S_padded)
    pad_ltl_diag(LTL_all, s_num_active)
    grad_u = (grad_output.unsqueeze(-2)
              ).matmul(L_S_padded).squeeze().unsqueeze(-1)

    # Cholesky Solve.
    u_all = torch.linalg.cholesky(LTL_all)
    tmp0 = torch.cholesky_solve(grad_u, u_all)
    if ctx.x_need_grad:
      grad_x = L_S_padded.matmul(tmp0).squeeze()
    else:
      grad_x = None
    sign_all = torch.zeros(
        batch_size, n_dim, device=device, dtype=grad_output.dtype)
    setup_sign_all(sign_all, sign_z, s_active, s_num_active)
    grad_lbda = (sign_all.unsqueeze(-2)).matmul(tmp0).squeeze(-1)
    return (grad_x, grad_lbda)
