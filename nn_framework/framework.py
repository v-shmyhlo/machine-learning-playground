import numpy as np
import nn_framework.op as op
import nn_framework.functions as fn
from nn_framework.session import Session
from nn_framework.gradient_check import gradient_check
from nn_framework.optimizers import GradientDescentOptimizer, MomentumOptimizer, AdamOptimizer


def to_op(value):
  if isinstance(value, op.Op):
    return value
  else:
    return op.Const(np.array(value))


def const(x):
  return op.Const(x)


def variable(x):
  return op.Variable(to_op(x))


def placeholder(name, shape):
  return op.Placeholder(name, shape)


def ones(shape):
  return op.Ones(shape)


def zeros(shape):
  return op.Zeros(shape)


def add(a, b):
  return op.Add(to_op(a), to_op(b))


def sub(a, b):
  return op.Sub(to_op(a), to_op(b))


def mul(a, b):
  return op.Mul(to_op(a), to_op(b))


def div(a, b):
  return op.Div(to_op(a), to_op(b))


def pow(a, b):
  return op.Pow(to_op(a), to_op(b))


def matmul(a, b):
  return op.Matmul(to_op(a), to_op(b))


def gt(a, b):
  return op.Gt(to_op(a), to_op(b))


def ge(a, b):
  return op.Ge(to_op(a), to_op(b))


def eq(a, b):
  return op.Eq(to_op(a), to_op(b))


def neg(x):
  return op.Neg(to_op(x))


def transpose(x):
  return op.Transpose(to_op(x))


def logistic_loss(a, y):
  return fn.LogisticLoss(a=to_op(a), y=to_op(y))


def sigmoid_logistic_loss(_sentinel=None, logits=None, labels=None):
  return fn.SigmoidLogisticLoss(z=to_op(logits), y=to_op(labels))


def maximum(a, b):
  return fn.Maximum(to_op(a), to_op(b))


def relu(x):
  return maximum(0, x)


def leaky_relu(x):
  return maximum(0.1 * x, x)


def sigmoid(x):
  return fn.Sigmoid(to_op(x))


def mean(x):
  return fn.Mean(to_op(x))


def group(ops):
  return op.Group(ops)


def random_normal(shape):
  return op.RandomNormal(shape)


def random_uniform(shape):
  return op.RandomUniform(shape)


def dropout(x, keep_prob):
  return fn.Dropout(to_op(x), to_op(keep_prob))


def sqrt(x):
  return op.Sqrt(to_op(x))
