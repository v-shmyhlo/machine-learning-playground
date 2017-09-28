from nn_framework.op import Op, Const, Variable, Ones
from nn_framework.functions import Zeros, Group


class Optimizer(Op):
  def eval(self, feeds, cache):
    if not self in cache:
      updates = [upd.eval(feeds, cache) for upd in self.updates]
      increment = self.increment.eval(feeds, cache)
      cache[self] = (updates, increment)

    return cache[self]


class GradientDescentOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None):
    self.updates = [
        v.assign(v - learning_rate * f.deriv(v, Ones(f.shape)))
        for v in f.variables()
    ]
    self.increment = global_step.assign(global_step + Const(1))


def momentum_update(f, v, learning_rate, beta):
  dv = f.deriv(v, Ones(f.shape))
  avg = Variable(Zeros(dv.shape))
  avg_upd = avg.assign(beta * avg + (Const(1) - beta) * dv)
  upd = v.assign(v - learning_rate * avg)
  return Group((avg_upd, upd))


class MomentumOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None, beta=Const(0.9)):
    self.updates = [
        momentum_update(f, v, learning_rate, beta) for v in f.variables()
    ]
    self.increment = global_step.assign(global_step + Const(1))
