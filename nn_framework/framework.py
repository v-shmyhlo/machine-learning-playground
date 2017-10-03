import numpy as np
from nn_framework.op import is_op
import nn_framework.ops as ops
from nn_framework.session import Session
from nn_framework.gradient_check import gradient_check
from nn_framework.optimizers import GradientDescentOptimizer, MomentumOptimizer, AdamOptimizer


def _to_op(value):
  assert value is not None, "operation can't be None"
  if is_op(value):
    return value
  else:
    return const(np.array(value))


def _to_op1(f):
  return lambda x: f(_to_op(x))


def _to_op2(f):
  return lambda a, b: f(_to_op(a), _to_op(b))


def const(value):
  assert not is_op(value), "const value can't be operation"
  return ops.Const(value)


@_to_op1
def variable(x):
  return ops.Variable(x)


def placeholder(name, shape):
  return ops.Placeholder(name, shape)


def ones(shape):
  return ops.Ones(shape)


def zeros(shape):
  return ops.Zeros(shape)


@_to_op2
def add(a, b):
  return ops.Add(a, b)


@_to_op2
def sub(a, b):
  return ops.Sub(a, b)


@_to_op2
def mul(a, b):
  return ops.Mul(a, b)


@_to_op2
def div(a, b):
  return ops.Div(a, b)


@_to_op2
def pow(a, b):
  return ops.Pow(a, b)


@_to_op2
def matmul(a, b):
  return ops.Matmul(a, b)


@_to_op2
def gt(a, b):
  return ops.Gt(a, b)


@_to_op2
def ge(a, b):
  return ops.Ge(a, b)


@_to_op2
def eq(a, b):
  return ops.Eq(a, b)


@_to_op1
def neg(x):
  return ops.Neg(x)


@_to_op1
def transpose(x):
  return ops.Transpose(x)


def logistic_loss(a, y):
  return ops.LogisticLoss(a=_to_op(a), y=_to_op(y))


def sigmoid_logistic_loss(logits, labels):
  return ops.SigmoidLogisticLoss(z=_to_op(logits), y=_to_op(labels))


@_to_op2
def maximum(a, b):
  return ops.Maximum(a, b)


@_to_op1
def relu(x):
  return maximum(0, x)


def leaky_relu(x):
  return maximum(0.1 * x, x)


@_to_op1
def sigmoid(x):
  return ops.Sigmoid(x)


@_to_op1
def mean(x):
  return ops.Mean(x)


def group(xs):
  return ops.Group([_to_op(x) for x in xs])


def random_normal(shape):
  return ops.RandomNormal(shape)


def random_uniform(shape):
  return ops.RandomUniform(shape)


def dropout(x, keep_prob):
  return ops.Dropout(_to_op(x), _to_op(keep_prob))


@_to_op1
def sqrt(x):
  return ops.Sqrt(x)


@_to_op1
def sum0(x):
  return ops.Sum0(x)


@_to_op1
def log(x):
  return ops.Log(x)


def softmax_cross_entropy(logits, labels):
  logits, labels = _to_op(logits), _to_op(labels)
  return ops.SoftmaxCrossEntropy(z=logits, y=labels)


@_to_op2
def cross_entropy(s, y):
  return -sum0(y * log(s))


@_to_op1
def softmax(x):
  exp = np.e**x
  return exp / sum0(exp)


def mean1(x):
  return ops.Mean1(x)


def batch_norm(z, gamma, beta, eps=1e-8):
  mn = mean1(z)
  z = z - mn
  var = mean1(z**2)
  z_norm = z / sqrt(var + eps)
  return gamma * z_norm + beta
