import numpy as np
import nn_framework.ops as ops
from nn_framework.session import Session

eps = 1e-7
limit = 1e-7


def gradient_check(exp, feeds={}):
  variables = exp.variables()
  grads = [exp.deriv(var, ops.Ones(exp.shape)) for var in variables]

  sess = Session()
  sess.run([exp], feeds)  # initialize vars
  grads_computed = sess.run(grads, feeds)
  grads_approx = []

  for i, var in enumerate(variables):
    print('gradient check: variable %s/%s' % (i + 1, len(variables)), end='\r')

    val = np.copy(var.value)
    dval = np.zeros(val.shape)

    for r in range(val.shape[0]):
      for c in range(val.shape[1]):

        target = var.value[r, c]
        assert (target + eps < 0 and target - eps < 0) or (
            target + eps > 0 and
            target - eps > 0), "%s and %s have different sign" % (target + eps,
                                                                  target - eps)

        var.value = np.copy(val)
        var.value[r, c] -= eps
        [em] = sess.run([exp], feeds)

        var.value = np.copy(val)
        var.value[r, c] += eps
        [ep] = sess.run([exp], feeds)

        dval[r, c] = (ep - em) / (2 * eps)

    grads_approx.append(dval)

  assert len(grads_computed) == len(grads_approx)
  n_elements = 0
  for (a, b) in zip(grads_computed, grads_approx):
    assert a.shape == b.shape
    n_elements += np.prod(a.shape)

  grads_computed = np.hstack([grad.ravel() for grad in grads_computed])
  grads_approx = np.hstack([grad.ravel() for grad in grads_approx])

  assert grads_computed.shape == (n_elements, ) and grads_approx.shape == (
      n_elements, )

  check = np.linalg.norm(grads_approx - grads_computed) / (
      np.linalg.norm(grads_approx) + np.linalg.norm(grads_computed))
  assert check > 0 and check < limit, "%s should be < %s" % (check, limit)
  print('gradient check: %s' % check)
  return check
