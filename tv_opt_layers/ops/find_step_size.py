"""Implements Armijo Search and brute-force parallel search."""

import torch
from tv_opt_layers.helpers.cuda import tv1_op_utils
from .utils import projection, primal2val, get_epsilon


def find_step_size_parallel(cont_find, cont_update,
                            aux, aux2, w, d, delta, prev_delta, inactive, num_inactive,
                            lmbd, fval0, x, y, sigma, gRd, step_sizes,
                            EPSILON,
                            aux_p, x_p,
                            delta_p,
                            DEBUG=False):
  bb = y.shape[0]  # Batch_size

  # Compute maximum useful stepsize
  delta_res = delta_p.shape[-1]
  aux_p = aux.unsqueeze(-1).repeat(1, 1, delta_res)
  x_p = x.unsqueeze(-1).repeat(1, 1, delta_res)
  tv1_op_utils.project_point_p(
      aux_p, w, d, delta_p, inactive, num_inactive, cont_find)
  projection(aux_p, lmbd)

  # Get primal point
  tv1_op_utils.dual_to_primal(aux_p, x_p, y, cont_find)

  fval1 = primal2val(x_p)
  improve = fval0.unsqueeze(-1) - fval1
  rhs = sigma * delta_p * gRd.unsqueeze(-1)
  cont_find_p = improve > EPSILON
  cont_find_p = torch.logical_and(cont_find_p, improve < rhs+EPSILON)
  cont_find_p = torch.logical_and(cont_find_p, cont_update.unsqueeze(-1))
  best_delta_idx = torch.logical_not(cont_find_p).max(-1)[-1]
  bb_seq = torch.arange(bb)
  aux[:, :] = aux_p[bb_seq, :, best_delta_idx]
  x[:, :] = x_p[bb_seq, :, best_delta_idx]
  tv1_op_utils.assign_cont_update(
      fval1[bb_seq, best_delta_idx], fval0, cont_update)
  return cont_find


def find_step_size_iter(cont_find, cont_update,
                        aux, aux2, w, d, delta, prev_delta, inactive, num_inactive,
                        lmbd, fval0, x, y, sigma, gRd, step_sizes, DEBUG=False):
  device = y.device
  bb = y.shape[0]  # Batch_size
  nn = y.shape[1]-1  # Vector Length
  dtype = y.dtype
  EPSILON = get_epsilon(aux)
  recomp = False

  if DEBUG:
    count = 0
  while(torch.any(cont_find)):
    # Compute projected point after update
    tv1_op_utils.project_point(
        aux, w, d, delta, inactive, num_inactive, cont_find)
    projection(aux, lmbd)
    if DEBUG:
      e_idx = num_inactive[0]
      print("aux=%s" % aux[0, :e_idx])
      print("axu2=%s" % aux2[0, :e_idx-1])

    # Get primal point
    tv1_op_utils.dual_to_primal(aux, x, y, cont_find)

    # Compute new value of the objective function
    fval1 = primal2val(x)
    improve = fval0 - fval1
    if DEBUG:
      print('fval1=%s' % fval1[0])
      print('improve=%s' % improve[0])

    cont_find = torch.logical_and(cont_find, cont_update)
    # If zero improvement, the updating direction is not useful
    cont_find = torch.logical_and(cont_find, improve > EPSILON)
    # Compute right hand side of Armijo rule
    rhs = sigma * delta * gRd
    # Check if the rule is met
    cont_find = torch.logical_and(cont_find, improve < rhs+EPSILON)

    if torch.any(cont_find):
      if not recomp:
        # Compute maximum useful stepsize
        step_sizes.zero_()
        tv1_op_utils.get_step_sizes(
            step_sizes, w, d, lmbd, inactive, num_inactive)
        max_step, _ = torch.max(step_sizes, -1)
        if DEBUG:
          e_idx = num_inactive[0]
          print('d=%s' % d[0, :e_idx])
          print('maxStep=%s' % max_step[0])

        # Compute gradient w.r.t stepsize at the present position
        grad0_elms = torch.zeros(bb, nn, device=device, dtype=dtype)
        tv1_op_utils.get_gradient(
            grad0_elms, w, d, y, lmbd, inactive, num_inactive, EPSILON)
        grad0 = torch.sum(grad0_elms, -1)
        if DEBUG:
          print('grad0=%s' % grad0[0])
        recomp = True
      # Use quadratic interpolation to determine next stepsize
      tv1_op_utils.quad_interp(
          grad0, delta, prev_delta, improve, cont_find, max_step, EPSILON)
      # Re-adjust maximum allowed step
      max_step = delta

      if DEBUG:
        print("delta=%s" % delta[0])
        print("found=%s" % (not cont_find[0]))
        print("count=%s" % count)
        print("---")
        count += 1
  tv1_op_utils.assign_cont_update(fval1, fval0, cont_update)
  return cont_find
