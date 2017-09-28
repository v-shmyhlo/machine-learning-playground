import nn_framework.op as op
from nn_framework import framework as nn


class Optimizer(op.Op):
  def eval(self, feeds, cache):
    if not self in cache:
      updates = [upd.eval(feeds, cache) for upd in self.updates]
      increment = self.increment.eval(feeds, cache)
      cache[self] = (updates, increment)

    return cache[self]


class GradientDescentOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None):
    self.updates = [
        v.assign(v - learning_rate * f.deriv(v, nn.ones(f.shape)))
        for v in f.variables()
    ]
    self.increment = global_step.assign(global_step + 1)


def momentum_update(f, v, learning_rate, beta):
  dv = f.deriv(v, nn.ones(f.shape))
  avg = nn.variable(nn.zeros(dv.shape))
  avg_upd = avg.assign(beta * avg + (1 - beta) * dv)
  upd = v.assign(v - learning_rate * avg)
  return nn.group((avg_upd, upd))


class MomentumOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None, beta=0.9):
    self.updates = [
        momentum_update(f, v, learning_rate, beta) for v in f.variables()
    ]
    self.increment = global_step.assign(global_step + 1)
