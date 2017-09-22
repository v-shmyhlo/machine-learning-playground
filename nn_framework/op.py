import numpy as np

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

  def __getitem__(self, *keys):
    return GetItem(self, *keys)

  def t(self):
    return Transpose(self)

  def dot(self, b):
    return Dot(self, b)

class UnaryOp(Op):
  def __init__(self, x):
    self.x = x
    self.variables = x.variables

class BinaryOp(Op):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.variables = a.variables.union(b.variables)
    self.shape = BroadcastElementwise(a.shape, b.shape)

class Const(Op):
  def __init__(self, value):
    self.value = np.array(value)
    self.variables = set()
    self.shape = Shape(self.value.shape)

  def eval(self, feeds, cache):
    return self.value

  def deriv(self, var, dself):
    if var == self:
#       return Const(np.ones(self.value.shape))
      return dself
    else:
#       return Const(np.zeros(self.value.shape))
#       return Const(0)
      return None

class Variable(Op):
  def __init__(self, value):
    self.value = np.array(value)
    self.variables = set([self])
    self.shape = Shape(self.value.shape)

  def eval(self, feeds, cache):
    return self.value

  def deriv(self, var, dself):
    if var == self:
#       return Const(np.ones(self.value.shape))
      print('self: %s, dself: %s' % (self.eval({}, {}).shape, dself.eval({}, {}).shape))
      return dself
    else:
#       return Const(np.zeros(self.value.shape))
#       return Const(0)
      return None

  def assign(self, exp):
    return Assign(self, exp)

class Placeholder(Op):
  def __init__(self, name):
    self.name = name
    self.variables = set()
    self.shape = FeedShape(self)

  def eval(self, feeds, cache):
    return np.array(feeds[self])

  def deriv(self, var, dself):
    if var == self:
#       return Const(np.ones(self.value.shape))
      return dself
    else:
#       return Const(np.zeros(self.value.shape))
#       return Const(0)
      return None

class Ones(Op):
  def __init__(self, shape):
    self.shape = shape
    self.variables = shape.variables

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = np.ones(self.shape.eval(feeds, cache))

    return cache[self]

class Assign(Op):
  def __init__(self, var, exp):
    self.var = var
    self.exp = exp

  def eval(self, feeds, cache):
    if not self in cache:
      # print(self.var.shape.eval(feeds, cache), self.exp.eval(feeds, cache).shape)
      self.var.value = self.exp.eval(feeds, cache)
      cache[self] = self.var.value

    return cache[self]

class Add(BinaryOp):
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

class Sub(BinaryOp):
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

class Mul(BinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) * self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):

    da = self.a.deriv(var, dself * self.b)
    db = self.b.deriv(var, dself * self.a)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db

class Div(BinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) / self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var):
    return (self.b * self.a.deriv(var) - self.a * self.b.deriv(var)) / (self.b ** Const(2))

class Pow(BinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) ** self.b.eval(feeds, cache)

    return cache[self]

  def deriv(self, var):
    return ((self.b * self.a) ** (self.b - Const(1))) * self.a.deriv(var)

class Neg(UnaryOp):
  def __init__(self, x):
    super().__init__(x)
    self.shape = self.x.shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = -self.x.eval(feeds, cache)

    return cache[self]

  def deriv(self, var, dself):
    return self.x.deriv(var, -dself)

class Gt(BinaryOp):
  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.a.eval(feeds, cache) > self.b.eval(feeds, cache)

    return cache[self]

class Transpose(UnaryOp):
  def __init__(self, x):
    super().__init__(x)
    self.shape = BroadcastTranspose(x.shape)

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.x.eval(feeds, cache).T

    return cache[self]

class Dot(BinaryOp):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.variables = a.variables.union(b.variables)
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

class BroadcastElementwise(Op):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.variables = a.variables.union(b.variables)

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      b = self.b.eval(feeds, cache)

      if a == b:
        cache[self] = a

      if len(a) == 2 and len(b) == 2:
        if a[0] == b[0]:
          if a[1] == 1:
            cache[self] = (a[0], b[1])
          if b[1] == 1:
            cache[self] = (a[0], a[1])
        if a[1] == b[1]:
          if a[0] == 1:
            cache[self] = (b[0], a[1])
          if b[0] == 1:
            cache[self] = (a[0], a[1])

      if len(a) == 2 and len(b) == 0:
        cache[self] = a

      if len(a) == 0 and len(b) == 2:
        cache[self] = b

      if not self in cache:
        raise Exception("Can not broadcast %s and %s" % (a, b))

    return cache[self]

class BroadcastMatmul(object):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.variables = a.variables.union(b.variables)

  def eval(self, feeds, cache):
    if not self in cache:
      a = self.a.eval(feeds, cache)
      b = self.b.eval(feeds, cache)

      if a[1] != b[0]:
        raise Exception("Shape %s and %s dimensions do not agree" % (a, b))

      cache[self] = (a[0], b[1])

    return cache[self]

class ReLU(Op):
  def __init__(self, Z):
    self.Z = Z
    self.variables = Z.variables
    self.shape = Z.shape

  def eval(self, feeds, cache):
    if not self in cache:
      Z = self.Z.eval(feeds, cache)
      return np.maximum(0, Z)

    return cache[self]

  def deriv(self, var, dself):
    dself_dZ = self.Z > Const(0)
    return self.Z.deriv(var, dself_dZ * dself)

class Sigmoid(Op):
  def __init__(self, Z):
    self.Z = Z
    self.variables = Z.variables
    self.shape = Z.shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = 1 / (1 + np.exp(-self.Z.eval(feeds, cache)))

    return cache[self]

  def deriv(self, var, dself):
    dself_dZ = self * (Const(1) - self)
    return self.Z.deriv(var, dself_dZ * dself)

class LogisticLoss(Op):
  def __init__(self, _sentinel=None, A=None, Y=None):
    self.A = A
    self.Y = Y
    self.variables = A.variables.union(Y.variables)
    self.shape = Shape(())

  def eval(self, feeds, cache):
    if not self in cache:
      A = self.A.eval(feeds, cache)
      Y = self.Y.eval(feeds, cache)
      cache[self] = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
      # TODO: use mean as deriv

    return cache[self]

  def deriv(self, var, dself):
    # TODO: this is not full implementation, it ignores derivatives with respect to Y
    dself_da = -(self.Y / self.A) + (Const(1) - self.Y) / (Const(1) - self.A)
    dself_da = dself_da / self.A.shape[1]
    return self.A.deriv(var, dself_da * dself)

class GradientDescentOptimizer(Op):
  def __init__(self, learning_rate, exp):
    self.learning_rate = learning_rate
    self.gradients = [(v, exp.deriv(v, Ones(exp.shape))) for v in exp.variables]

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = [v.assign(v - Const(self.learning_rate) * dv).eval(feeds, cache) for (v, dv) in self.gradients]

    return cache[self]

class Shape(object):
  def __init__(self, value):
    self.value = value
    self.variables = set()

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.value

    return cache[self]

class FeedShape(object):
  def __init__(self, key):
    self.key = key
    self.variables = set()

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = feeds[self.key].shape

    return cache[self]

class BroadcastTranspose(object):
  def __init__(self, value):
    self.value = value
    self.variables = value.variables

  def eval(self, feeds, cache):
    if not self in cache:
      value = self.value.eval(feeds, cache)
      cache[self] = (value[1], value[0])

    return cache[self]

class GetItem(object):
  def __init__(self, value, *keys):
    self.value = value
    self.keys = keys
    self.variables = value.variables
    self.shape = Shape(())

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.value.eval(feeds, cache).__getitem__(*self.keys)

    return cache[self]
