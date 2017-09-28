import numpy as np
import nn_framework.op as op
from nn_framework import framework as nn


class Sigmoid(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = x.shape
    self.dself_dx = self * (1 - self)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = 1 / (1 + np.exp(-self.x.eval(feeds, cache)))

    return cache[self]

  def deriv(self, var, dself):
    return self.x.deriv(var, self.dself_dx * dself)


class LogisticLoss(op.Op):
  def __init__(self, _sentinel=None, a=None, y=None):
    a.shape.assert_has_rank(2)
    y.shape.assert_has_rank(2)

    self.a, self.y, self.shape = op.elementwise_broadcast(a, y)
    self.dself_da = -(self.y / self.a) + (1 - self.y) / (1 - self.a)

  def variables(self):
    return self.a.variables().union(self.y.variables())

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      y = self.y.eval(feeds, cache)
      # assert np.all(a > 0) and np.all((1 - a) > 0)
      cache[self] = -np.log(a) * y + -np.log(1 - a) * (1 - y)

    return cache[self]

  def deriv(self, var, dself):
    # TODO: this is not full implementation, it ignores derivatives w.r.t. y
    return self.a.deriv(var, self.dself_da * dself)


class Maximum(op.ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      b = self.b.eval(feeds, cache)
      return np.maximum(a, b)

    return cache[self]

  def deriv(self, var, dself):
    dself_da = self.a > self.b
    dself_db = self.b > self.a

    da = self.a.deriv(var, dself_da * dself)
    db = self.b.deriv(var, dself_db * dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Mean(op.UnaryOp):
  def __init__(self, x):
    x.shape.assert_has_rank(2)
    self.x = x
    self.shape = op.make_shape(())

  def eval(self, feeds, cache):
    return np.mean(self.x.eval(feeds, cache))

  def deriv(self, var, dself):
    dself_dx = nn.ones(self.x.shape) / (self.x.shape[0] * self.x.shape[1])
    return self.x.deriv(var, dself * dself_dx)


class Dropout(op.UnaryOp):
  def __init__(self, x, keep_prob):
    self.x = x
    self.keep_prob = keep_prob
    self.keep = nn.random_uniform(x.shape) < self.keep_prob
    self.eval_op = (self.x * self.keep) / self.keep_prob
    self.shape = self.eval_op.shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.eval_op.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.x.deriv(var, (dself * self.keep) / self.keep_prob)
