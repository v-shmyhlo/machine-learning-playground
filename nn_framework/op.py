import numpy as np
from nn_framework import framework as nn

# import nn_framework.framework as nn


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

  def __pow__(self, b):
    return nn.pow(self, b)

  def __matmul__(self, b):
    return nn.matmul(self, b)

  def __gt__(self, b):
    return nn.gt(self, b)

  def __neg__(self):
    return nn.neg(self)

  def t(self):
    return nn.transpose(self)


class UnaryOp(Op):
  def variables(self):
    return self.x.variables()


class Neg(UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = self.x.shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = -self.x.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.x.deriv(var, -dself)


class Transpose(UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = transpose_shape_broadcast(x.shape)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.x.eval(feeds, cache).T

    return cache[self]


class BinaryOp(Op):
  def variables(self):
    return self.a.variables().union(self.b.variables())


class ElementwiseBinaryOp(BinaryOp):
  def __init__(self, a, b):
    self.a, self.b, self.shape = elementwise_broadcast(a, b)


class Const(Op):
  def __init__(self, value):
    self.value = np.array(value)
    self.shape = make_shape(self.value.shape)

  def eval(self, feeds, cache):
    return self.value

  def deriv(self, var, dself):
    if var == self:
      return dself
    else:
      return None

  def variables(self):
    return set()

  def __str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.value)


class Variable(Op):
  def __init__(self, init_value):
    self.init_value = init_value
    self.shape = self.init_value.shape
    self.value = None

  def eval(self, feeds, cache):
    if self.value is None:
      self.value = self.init_value.eval(feeds, cache)

    return self.value

  def deriv(self, var, dself):
    if var == self:
      return dself
    else:
      return None

  def assign(self, exp):
    return Assign(self, exp)

  def variables(self):
    return set([self])

  def __str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.value)


class Placeholder(Op):
  def __init__(self, name, shape):
    self.name = name
    self.shape = make_dynamic_shape(shape, self)

  def eval(self, feeds, cache):
    return np.array(feeds[self])

  def deriv(self, var, dself):
    if var == self:
      return dself
    else:
      return None

  def variables(self):
    return set()


class Assign(Op):
  def __init__(self, var, exp):
    self.var = var
    self.exp = exp

  def eval(self, feeds, cache):
    if not self in cache:
      value = self.exp.eval(feeds, cache)
      assert value.shape == self.var.value.shape
      self.var.value = value
      cache[self] = self.var.value

    return cache[self]


class Add(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) + self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    da = self.a.deriv(var, dself)
    db = self.b.deriv(var, dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Sub(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) - self.b.eval(feeds, cache)
    return cache[self]

  def deriv(self, var, dself):
    da = self.a.deriv(var, dself)
    db = self.b.deriv(var, dself)

    if da is None and db is None:
      return None
    elif da is None:
      return -db
    elif db is None:
      return da
    else:
      return da - db


class Mul(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) * self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    da = self.a.deriv(var, self.b * dself)
    db = self.b.deriv(var, self.a * dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Div(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) / self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var):
    return (self.b * self.a.deriv(var) - self.a * self.b.deriv(var)) / (
        self.b**2)


class Pow(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache)**self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.a.deriv(var, (self.b * self.a)**(self.b - 1) * dself)


class Gt(ElementwiseBinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) > self.b.eval(feeds, cache)

    return cache[self]


class Matmul(ElementwiseBinaryOp):
  def __init__(self, a, b):
    a.shape.assert_has_rank(2)
    b.shape.assert_has_rank(2)

    self.a = a
    self.b = b
    self.shape = matmul_shape_broadcast(a.shape, b.shape)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) @ self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    dself_da = dself @ self.b.t()
    dself_db = dself.t() @ self.a
    da = self.a.deriv(var, dself_da)
    db = self.b.deriv(var, dself_db.t())

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Broadcast(UnaryOp):
  def __init__(self, x, shape):
    self.x = x
    self.shape = shape

  def eval(self, feeds, cache):
    return np.broadcast_to(
        self.x.eval(feeds, cache), self.shape.eval(feeds, cache))

  def deriv(self, var, dself):
    return self.x.deriv(var, SumToShape(dself, self.x.shape))


class SumToShape(Op):
  def __init__(self, tensor, shape):
    self.tensor = tensor
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      tensor = self.tensor.eval(feeds, cache)
      shape = self.shape.eval(feeds, cache)

      if tensor.shape != shape:
        axes = np.argwhere(np.array(tensor.shape) != np.array(shape)).ravel()
        result = tensor

        for axis in axes:
          result = np.sum(result, axis=axis, keepdims=True)

        assert result.shape == shape
        cache[self] = result
      else:
        cache[self] = tensor

    return cache[self]


def elementwise_broadcast(a, b):
  shape = elementwise_shape_broadcast(a.shape, b.shape)
  a_new = Broadcast(a, shape)
  b_new = Broadcast(b, shape)

  return a_new, b_new, shape


class Ones(Op):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = np.ones(self.shape.eval(feeds, cache))

    return cache[self]


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


class RandomNormal(Const):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      shape = self.shape.eval(feeds, cache)
      cache[self] = np.random.randn(*shape)
      assert cache[self].shape == shape

    return cache[self]


class RandomUniform(Const):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      shape = self.shape.eval(feeds, cache)
      cache[self] = np.random.rand(*shape)
      assert cache[self].shape == shape

    return cache[self]


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

  raise Exception('Can not broadcast %s and %s' % (a, b))


def matmul_shape_broadcast(a, b):
  a.assert_has_rank(2)
  b.assert_has_rank(2)

  return make_shape((a[0], b[1]))


def transpose_shape_broadcast(shape):
  shape.assert_has_rank(2)

  return make_shape((shape[1], shape[0]))


class _Shape(object):
  def __init__(self, dims):
    self.dims = dims
    self.rank = len(dims)

  def __getitem__(self, i):
    return self.dims[i]

  def assert_has_rank(self, rank):
    assert self.rank == rank, '%s must have rank %s' % (self, rank)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = tuple([dim.eval(feeds, cache) for dim in self.dims])

    return cache[self]

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

  def eval(self, feeds, cache):
    return self.value

  def __str__(self):
    return 'Dimension(%s)' % self.value


class _DynamicDimension(_Dimension):
  def __init__(self, tensor, i):
    super().__init__(None)
    self.tensor = tensor
    self.i = i

  def eval(self, feeds, cache):
    return feeds[self.tensor].shape[self.i]


class _DimensionBroadcast(_Dimension):
  def __init__(self, a, b):
    super().__init__(None)
    self.a = a
    self.b = b

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      b = self.b.eval(feeds, cache)

      if a == b:
        cache[self] = a
      elif a == 1:
        cache[self] = b
      elif b == 1:
        cache[self] = a
      else:
        raise Exception('Cannot broadcast dimension %s to dimension %s' % (a,
                                                                           b))

    return cache[self]
