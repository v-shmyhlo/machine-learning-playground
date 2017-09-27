import numpy as np
from nn_framework.shape import make_shape, make_dynamic_shape, elementwise_shape_broadcast, matmul_shape_broadcast, transpose_shape_broadcast


class Op(object):
  def __add__(self, b):
    return Add(self, b)

  def __sub__(self, b):
    return Sub(self, b)

  def __mul__(self, b):
    return Mul(self, b)

  def __truediv__(self, b):
    return Div(self, b)

  def __pow__(self, b):
    return Pow(self, b)

  def __neg__(self):
    return Neg(self)

  def __gt__(self, b):
    return Gt(self, b)

  def __matmul__(self, b):
    return Matmul(self, b)

  def t(self):
    return Transpose(self)


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


class BinaryElementwiseOp(BinaryOp):
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


class Variable(Op):
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

  def assign(self, exp):
    return Assign(self, exp)

  def variables(self):
    return set([self])


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


class Add(BinaryElementwiseOp):
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


class Sub(BinaryElementwiseOp):
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


class Mul(BinaryElementwiseOp):
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


class Div(BinaryElementwiseOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) / self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var):
    return (self.b * self.a.deriv(var) - self.a * self.b.deriv(var)) / (
        self.b**Const(2))


class Pow(BinaryElementwiseOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache)**self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.a.deriv(var, (self.b * self.a)**(self.b - Const(1)) * dself)


class Gt(BinaryElementwiseOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) > self.b.eval(feeds, cache)

    return cache[self]


class Matmul(BinaryElementwiseOp):
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
