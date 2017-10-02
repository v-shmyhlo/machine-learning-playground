import numpy as np
from nn_framework import framework as nn


def is_op(value):
  return isinstance(value, Op)


class Op(object):
  def __add__(self, b):
    return nn.add(self, b)

  def __sub__(self, b):
    return nn.sub(self, b)

  def __rsub__(self, a):
    return nn.sub(a, self)

  def __mul__(self, b):
    return nn.mul(self, b)

  def __rmul__(self, a):
    return nn.mul(a, self)

  def __truediv__(self, b):
    return nn.div(self, b)

  def __rtruediv__(self, a):
    return nn.div(a, self)

  def __pow__(self, b):
    return nn.pow(self, b)

  def __rpow__(self, a):
    return nn.pow(a, self)

  def __matmul__(self, b):
    return nn.matmul(self, b)

  def __gt__(self, b):
    return nn.gt(self, b)

  def __ge__(self, b):
    return nn.ge(self, b)

  def __neg__(self):
    return nn.neg(self)

  def t(self):
    return nn.transpose(self)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self._eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    if var is self:
      return dself
    else:
      return self._deriv(var, dself)


class UnaryOp(Op):
  def __init__(x):
    self.x = x

  def variables(self):
    return self.x.variables()

  def __str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.x)


class BinaryOp(Op):
  def variables(self):
    return self.a.variables().union(self.b.variables())

  def __str__(self):
    return '%s(%s, %s)' % (self.__class__.__name__, self.a, self.b)


class ElementwiseBinaryOp(BinaryOp):
  def __init__(self, a, b):
    self.a, self.b, self.shape = elementwise_broadcast(a, b)


class _Shape(object):
  def __init__(self, dims):
    self.dims = dims
    self.rank = len(dims)

  def __getitem__(self, i):
    return self.dims[i]

  def assert_has_rank(self, rank):
    assert self.rank == rank, '%s must have rank %s' % (self, rank)

  def eval(self, feeds, cache):
    return tuple([dim.eval(feeds, cache) for dim in self.dims])

  def __str__(self):
    return 'Shape([%s])' % ', '.join([dim.__str__() for dim in self.dims])


def make_dimension(dim):
  if isinstance(dim, _Dimension):
    return dim
  else:
    return _Dimension(dim)


def make_dynamic_dimension(dim, tensor, i):
  if isinstance(dim, _Dimension):
    return dim
  elif dim is None:
    return _DynamicDimension(tensor, i)
  else:
    return _Dimension(dim)


def dimension_broadcast(a, b):
  return _DimensionBroadcast(make_dimension(a), make_dimension(b))


class _Dimension(Op):
  def __init__(self, value):
    self.value = value
    self.shape = make_shape(())

  def _eval(self, feeds, cache):
    return self.value

  def __str__(self):
    return 'Dimension(%s)' % self.value


class _DynamicDimension(_Dimension):
  def __init__(self, tensor, i):
    super().__init__(None)
    self.tensor = tensor
    self.i = i

  def _eval(self, feeds, cache):
    return feeds[self.tensor].shape[self.i]


class _DimensionBroadcast(_Dimension):
  def __init__(self, a, b):
    super().__init__(None)
    self.a = a
    self.b = b

  def _eval(self, feeds, cache):
    a = self.a.eval(feeds, cache)
    b = self.b.eval(feeds, cache)

    if a == b:
      return a
    elif a == 1:
      return b
    elif b == 1:
      return a
    else:
      raise Exception("Can't broadcast dimension %s to dimension %s" % (a, b))


def make_shape(dims):
  return _Shape([make_dimension(dim) for dim in dims])


def make_dynamic_shape(dims, self):
  return _Shape(
      [make_dynamic_dimension(dim, self, i) for (i, dim) in enumerate(dims)])


def elementwise_shape_broadcast(a, b):
  if a.rank == 0 and b.rank == 0:
    return a

  if a.rank == 2 and b.rank == 0:
    return a

  if a.rank == 0 and b.rank == 2:
    return b

  if a.rank == 2 and b.rank == 2:
    return make_shape((dimension_broadcast(a[0], b[0]), dimension_broadcast(
        a[1], b[1])))

  raise Exception("Can't broadcast %s and %s" % (a, b))


def matmul_shape_broadcast(a, b):
  a.assert_has_rank(2)
  b.assert_has_rank(2)

  return make_shape((a[0], b[1]))


def transpose_shape_broadcast(shape):
  shape.assert_has_rank(2)

  return make_shape((shape[1], shape[0]))


def elementwise_broadcast(a, b):
  shape = elementwise_shape_broadcast(a.shape, b.shape)
  a_new = Broadcast(a, shape)
  b_new = Broadcast(b, shape)

  return a_new, b_new, shape


class Broadcast(UnaryOp):
  def __init__(self, x, shape):
    self.x = x
    self.shape = shape

  def _eval(self, feeds, cache):
    return np.broadcast_to(
        self.x.eval(feeds, cache), self.shape.eval(feeds, cache))

  def _deriv(self, var, dself):
    return self.x.deriv(var, SumToShape(dself, self.x.shape))


class SumToShape(Op):
  def __init__(self, tensor, shape):
    self.tensor = tensor
    self.shape = shape

  def _eval(self, feeds, cache):
    tensor = self.tensor.eval(feeds, cache)
    shape = self.shape.eval(feeds, cache)

    if tensor.shape != shape:
      axes = np.argwhere(np.array(tensor.shape) != np.array(shape)).ravel()
      result = tensor

      for axis in axes:
        result = np.sum(result, axis=axis, keepdims=True)

      assert result.shape == shape
      return result
    else:
      return tensor
