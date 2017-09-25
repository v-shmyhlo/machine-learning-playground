import numpy as np
from nn_framework.shape import StaticShape, DynamicShape, ElementwiseShapeBroadcast

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

  def dot(self, b):
    return Dot(self, b)

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
    self.shape = BroadcastTranspose(x.shape)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.x.eval(feeds, cache).T

    return cache[self]

class BinaryOp(Op):
  def variables(self):
    return self.a.variables().union(self.b.variables())

class BinaryElementwiseOp(BinaryOp):
  def __init__(self, a, b):
    self.a, self.b, self.shape = broadcast_elementwise(a, b)

class Const(Op):
  def __init__(self, value):
    self.value = np.array(value)
    self.shape = StaticShape(self.value.shape)

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
    self.shape = StaticShape(self.value.shape)

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
  def __init__(self, name):
    self.name = name
    self.shape = DynamicShape(self)

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
    return (self.b * self.a.deriv(var) - self.a * self.b.deriv(var)) / (self.b ** Const(2))

class Pow(BinaryElementwiseOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) ** self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.a.deriv(var, (self.b * self.a) ** (self.b - Const(1)) * dself)

class Gt(BinaryElementwiseOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) > self.b.eval(feeds, cache)

    return cache[self]

class Dot(BinaryElementwiseOp):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.shape = BroadcastMatmul(a.shape, b.shape)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache).dot(self.b.eval(feeds, cache))

    return cache[self]

  def deriv(self, var, dself):
    dself_da = dself.dot(self.b.t())
    dself_db = dself.t().dot(self.a)
    da = self.a.deriv(var, dself_da)
    db = self.b.deriv(var, dself_db.t())

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db

class BroadcastMatmul(BinaryOp):
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      b = self.b.eval(feeds, cache)

      if a[1] != b[0]:
        raise Exception("Shape %s and %s dimensions do not agree" % (a, b))

      cache[self] = (a[0], b[1])

    return cache[self]

class BroadcastTranspose(object):
  def __init__(self, value):
    self.value = value

  def eval(self, feeds, cache):
    if not self in cache:
      value = self.value.eval(feeds, cache)
      cache[self] = (value[1], value[0])

    return cache[self]

class Broadcast(UnaryOp):
  def __init__(self, x, shape):
    self.x = x
    self.shape = shape

  def eval(self, feeds, cache):
    # TODO: i am ignoring broadcasting relying on numpy
    return self.x.eval(feeds, cache)

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
        axis = np.argmax(np.array(tensor.shape) != np.array(shape))
        cache[self] = np.sum(tensor, axis=axis, keepdims=True)
      else:
        cache[self] = tensor

    return cache[self]

def broadcast_elementwise(a, b):
  shape = ElementwiseShapeBroadcast(a.shape, b.shape)
  a_new = Broadcast(a, shape)
  b_new = Broadcast(b, shape)

  return a_new, b_new, shape
