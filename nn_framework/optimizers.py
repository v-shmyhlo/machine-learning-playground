import nn_framework.op as op
from nn_framework import framework as nn


class Optimizer(op.Op):
  def _eval(self, feeds, cache):
    return self.step.eval(feeds, cache)


class GradientDescentOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None):
    variables = f.variables()
    gradients = [f.deriv(v, nn.ones(f.shape)) for v in variables]
    updates = [
        v.assign(v - learning_rate * dv)
        for (v, dv) in zip(variables, gradients)
    ]

    self.step = nn.group([
        nn.group(variables),
        nn.group(gradients),
        nn.group(updates),
        global_step.assign(global_step + 1)
    ])


def momentum_update(v, dv, learning_rate, beta):
  avg = nn.variable(nn.zeros(dv.shape))
  avg_upd = avg.assign(beta * avg + (1 - beta) * dv)
  upd = v.assign(v - learning_rate * avg)
  return nn.group((avg_upd, upd))


class MomentumOptimizer(Optimizer):
  def __init__(self, f, learning_rate, global_step=None, beta=0.9):
    variables = f.variables()
    gradients = [f.deriv(v, nn.ones(f.shape)) for v in variables]
    updates = [
        momentum_update(v, dv, learning_rate, beta)
        for (v, dv) in zip(variables, gradients)
    ]

    self.step = nn.group([
        nn.group(variables),
        nn.group(gradients),
        nn.group(updates),
        global_step.assign(global_step + 1)
    ])


def adam_update(v, dv, learning_rate, global_step, beta1, beta2, eps):
  t = global_step + 1
  avg = nn.variable(nn.zeros(dv.shape))
  s = nn.variable(nn.zeros(dv.shape))

  avg_upd = avg.assign(beta1 * avg + (1 - beta1) * dv)
  s_upd = s.assign(beta2 * s + (1 - beta2) * dv**2)

  avg_corr = avg / (1 - beta1**t)
  s_corr = s / (1 - beta2**t)

  upd_value = avg_corr / nn.sqrt(s_corr + eps)
  upd = v.assign(v - learning_rate * upd_value)
  return nn.group((avg_upd, s_upd, upd))


class AdamOptimizer(Optimizer):
  def __init__(self,
               f,
               learning_rate,
               global_step=None,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8):
    variables = f.variables()
    gradients = [f.deriv(v, nn.ones(f.shape)) for v in variables]
    updates = [
        adam_update(v, dv, learning_rate, global_step, beta1, beta2, eps)
        for (v, dv) in zip(variables, gradients)
    ]

    self.step = nn.group([
        nn.group(variables),
        nn.group(gradients),
        nn.group(updates),
        global_step.assign(global_step + 1)
    ])
