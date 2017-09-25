from nn_framework.op import Op, UnaryOp, Const
from nn_framework.shape import StaticShape
import numpy as np

class Ones(Op):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = np.ones(self.shape.eval(feeds, cache))

    return cache[self]

class ReLU(UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = x.shape

  def eval(self, feeds, cache):
    if not self in cache:
      x = self.x.eval(feeds, cache)
      return np.maximum(0, x)

    return cache[self]

  def deriv(self, var, dself):
    dself_dx = self.x > Const(0)
    return self.x.deriv(var, dself_dx * dself)

class Sigmoid(UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = x.shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = 1 / (1 + np.exp(-self.x.eval(feeds, cache)))

    return cache[self]

  def deriv(self, var, dself):
    dself_dx = self * (Const(1) - self)
    return self.x.deriv(var, dself_dx * dself)

class LogisticLoss(Op):
  def __init__(self, _sentinel=None, a=None, y=None):
    self.a = a
    self.y = y
    self.shape = StaticShape(())

  def variables(self):
    return self.a.variables().union(self.y.variables())

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      y = self.y.eval(feeds, cache)
      cache[self] = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

    return cache[self]

  def deriv(self, var, dself):
    # TODO: this is not full implementation, it ignores derivatives w.r.t. y
    dself_da = -(self.y / self.a) + (Const(1) - self.y) / (Const(1) - self.a)
    dself_da = dself_da / self.a.shape[1]
    return self.a.deriv(var, dself_da * dself)
