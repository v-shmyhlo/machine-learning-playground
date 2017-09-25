from nn_framework.op import Op, Const
from nn_framework.utils import Ones

class GradientDescentOptimizer(Op):
  def __init__(self, learning_rate, exp):
    self.learning_rate = learning_rate
    self.gradients = [(v, exp.deriv(v, Ones(exp.shape))) for v in exp.variables()]

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = [v.assign(v - Const(self.learning_rate) * dv).eval(feeds, cache) for (v, dv) in self.gradients]

    return cache[self]
