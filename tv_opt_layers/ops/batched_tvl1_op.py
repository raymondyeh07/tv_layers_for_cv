"""Batch solver for TV-L1 norm problem."""

import torch
from tv_opt_layers.helpers.cuda import dpttr
from tv_opt_layers.helpers.cuda import tv1_op_utils
from .utils import projection, primal2grad, primal2val, check_inactive_cuda, grad2gap, get_epsilon
from .find_step_size import find_step_size_iter, find_step_size_parallel

DEBUG = False
PARALLEL_DELTA = False

if DEBUG:
  import pdb

torch.set_printoptions(precision=8)


def pn_tv1(y, lmbd, sigma=0.05, STOP_PN=1e-6, MAX_ITERS_PN=100):
  """
  y(torch.Tensor): batch_size x n_dim
  lmbd(torch.Tensor): batch_size x 1, non-negative weight balancing two terms per example in a batch.
  sigma(double): tolerance for sufficient descent.
  """
  EPSILON = get_epsilon(y)
  # Factorize Hessian
  device = y.device
  bb = y.shape[0]  # Batch_size
  nn = y.shape[1]-1  # Vector Length
  dtype = y.dtype
  aux = 2.*torch.ones(1, nn, device=device, dtype=dtype)
  aux2 = -1.*torch.ones(1, nn, device=device, dtype=dtype)

  # Solve Choleski-like linear system to obtain unconstrained solution
  w = y[:, 1:]-y[:, :-1]
  w = w.contiguous()
  dpttr.batch_factorize(aux, aux2)
  aux = aux.expand(bb, -1).contiguous()
  aux2 = aux2.expand(bb, -1).contiguous()
  dpttr.batch_solve(aux, aux2, w)

  if DEBUG:
    # Compute maximum effective penalty
    lmbd_max = torch.max(torch.abs(w), -1)
    print("lmbda=%s, lmbdaMax=%s" % (lmbd[0].item(), lmbd_max[0][0].item()))
    print("aa=%s" % aux)
    print("bb=%s" % aux2)

  # Initial guess and gradient
  projection(w, lmbd)
  x = torch.zeros_like(y, dtype=dtype).contiguous()
  y = y.contiguous()
  tv1_op_utils.dual_to_primal(w, x, y)
  g = primal2grad(x)
  d = torch.zeros_like(g)
  fval0 = primal2val(x)

  # Identify inactive constraints at the starting point
  inactive = torch.zeros(bb, nn, device=device, dtype=torch.long)
  num_inactive = torch.zeros(bb, device=device, dtype=torch.long)
  check_inactive_cuda(w, g, inactive, num_inactive, lmbd)

  if DEBUG:
    print("---------Starting point--------")
    print("w=%s" % w[0])
    print("g=%s" % g[0])
    e_idx = num_inactive[0]
    print("inactive=%s" % inactive[0, :e_idx])
    print("fval=%s" % fval0[0])
    print("-------------------------------")

  # Solver loop
  stop = 9999999*torch.ones(bb, device=device, dtype=dtype)
  stop_prev = -99999999*torch.ones(bb, device=device, dtype=dtype)
  step_sizes = torch.zeros(bb, nn, device=device, dtype=dtype)
  gRd_elms = torch.zeros(bb, nn, device=device, dtype=dtype)
  delta = torch.ones(bb, device=device, dtype=dtype)
  prev_delta = torch.ones(bb, device=device, dtype=dtype)
  iters = 0

  # Continue update if any is true.
  cont_update = torch.logical_and(
      stop > STOP_PN, torch.abs(stop-stop_prev) > EPSILON)
  cont_update = torch.logical_and(cont_update, num_inactive != 0)

  if PARALLEL_DELTA:
    delta_res = 16
    aux_p = aux.unsqueeze(-1).repeat(1, 1, delta_res)
    x_p = x.unsqueeze(-1).repeat(1, 1, delta_res)
    delta_p = delta.unsqueeze(-1).repeat(1, delta_res)
    tmp = torch.pow(.5, torch.arange(0, delta_res, dtype=dtype, device=device))
    delta_p *= tmp

  while torch.any(cont_update) and iters < MAX_ITERS_PN:
    # Continue to search for step size.
    cont_find = torch.ones(bb, device=device, dtype=torch.bool)

    # Compute reduced Hessian (only inactive rows/columns)
    aux = torch.ones_like(aux)
    aux2.zero_()
    tv1_op_utils.reduced_hessian(
        aux, aux2, inactive, num_inactive, cont_update)

    if DEBUG:
      e_idx = num_inactive[0]
      print("alpha=%s" % aux[0, :e_idx])
      print("beta=%s" % aux2[0, :e_idx-1])

    # Factorize reduced Hessian
    dpttr.batch_factorize(aux, aux2, num_inactive, cont_update)
    if DEBUG:
      e_idx = num_inactive[0]
      print("c=%s" % aux[0, :e_idx])
      print("l=%s" % aux2[0, :e_idx-1])
    # Solve Choleski-like linear system to obtain Newton updating direction
    tv1_op_utils.setup_cholesky(g, d, inactive, num_inactive, cont_update)
    dpttr.batch_solve(aux, aux2, d, num_inactive, cont_update)
    if DEBUG:
      print("d=%s" % d[0, :e_idx])

    # Step size selection algorithm (quadratic interpolation)
    gRd_elms.zero_()
    tv1_op_utils.get_grd(gRd_elms, g, d, inactive, num_inactive)
    gRd = torch.sum(gRd_elms, -1)
    recomp = False
    delta = torch.ones_like(delta)
    cont_find[:] = cont_update[:]
    tv1_op_utils.assign_cont_update(w, aux, cont_find)

    if DEBUG:
      print("gRd=%s" % gRd[0])
      count = 0

    # Perform Search for Step Size.
    if PARALLEL_DELTA:
      cont_find = find_step_size_parallel(cont_find, cont_update,
                                          aux, aux2, w, d, delta, prev_delta, inactive, num_inactive,
                                          lmbd, fval0, x, y, sigma, gRd, step_sizes, EPSILON,
                                          aux_p, x_p,
                                          delta_p,
                                          DEBUG)
    else:
      cont_find = find_step_size_iter(cont_find, cont_update,
                                      aux, aux2, w, d, delta, prev_delta, inactive, num_inactive,
                                      lmbd, fval0, x, y, sigma, gRd, step_sizes, DEBUG)

    # Perform update
    tv1_op_utils.assign_cont_update(aux, w, cont_update)

    # Reconstruct gradient
    g = primal2grad(x)

    # Identify active and inactive constraints
    check_inactive_cuda(w, g, inactive, num_inactive, lmbd)

    # Compute stopping criterion
    stop_prev = stop
    stop = grad2gap(g, w, lmbd)

    iters += 1
    cont_update = torch.logical_and(
        stop > STOP_PN, torch.abs(stop-stop_prev) > EPSILON)
    cont_update = torch.logical_and(cont_update, num_inactive != 0)

    if DEBUG:
      print("---------End of iteration %d--------" % iters)
      print("w=%s" % w[0])
      print("g=%s" % g[0])
      e_idx = num_inactive[0]
      print('inactive=%s' % inactive[0, :e_idx])
      print("fVal=%s" % fval0[0])
      print("stop=%s" % stop[0])
      print("cont_update=%s" % cont_update)
  # End of loop and return
  return x
