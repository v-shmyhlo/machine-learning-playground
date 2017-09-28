import numpy as np
from nn_framework.op import Ones
from nn_framework.session import Session

e = 1e-7
limit = 1e-7
# ord = np.inf
ord = None


def gradient_check(exp, feeds):
  variables = exp.variables()
  grads = [exp.deriv(var, Ones(exp.shape)) for var in variables]

  sess = Session()
  grads_computed = sess.run(grads, feeds)
  grads_approx = []

  for var in variables:
    val = np.copy(var.value)
    dval = np.zeros(val.shape)

    for r in range(val.shape[0]):
      for c in range(val.shape[1]):
        var.value = np.copy(val)
        var.value[r, c] -= e
        [em] = sess.run([exp], feeds)

        var.value = np.copy(val)
        var.value[r, c] += e
        [ep] = sess.run([exp], feeds)

        dval[r, c] = (ep - em) / (2 * e)

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

  check = np.linalg.norm(
      grads_approx - grads_computed, ord=ord) / (np.linalg.norm(
          grads_approx, ord=ord) + np.linalg.norm(grads_computed, ord=ord))
  assert check > 0 and check < 1e-7, "%s should be < %s" % (check, limit)
  return check
