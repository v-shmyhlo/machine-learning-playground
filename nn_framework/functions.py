from nn_framework.op import Op, UnaryOp, Const, make_shape, elementwise_broadcast
# from nn_framework.shape import make_shape
import numpy as np


class Group(Op):
  def __init__(self, expressions):
    self.expressions = expressions

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = [exp.eval(feeds, cache) for exp in self.expressions]

    return cache[self]


class Zeros(Op):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = np.zeros(self.shape.eval(feeds, cache))

    return cache[self]


class Sigmoid(UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = x.shape
    self.dself_dx = self * (Const(1) - self)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = 1 / (1 + np.exp(-self.x.eval(feeds, cache)))

    return cache[self]

  def deriv(self, var, dself):
    return self.x.deriv(var, self.dself_dx * dself)


class LogisticLoss(Op):
  def __init__(self, _sentinel=None, a=None, y=None):
    a.shape.assert_has_rank(2)
    y.shape.assert_has_rank(2)

    self.a, self.y, self.shape = elementwise_broadcast(a, y)
    self.dself_da = -(self.y / self.a) + (Const(1) - self.y) / (
        Const(1) - self.a)

  def variables(self):
    return self.a.variables().union(self.y.variables())

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      y = self.y.eval(feeds, cache)
      cache[self] = -(y * np.log(a) + (1 - y) * np.log(1 - a))

    return cache[self]

  def deriv(self, var, dself):
    # TODO: this is not full implementation, it ignores derivatives w.r.t. y
    return self.a.deriv(var, self.dself_da * dself)
