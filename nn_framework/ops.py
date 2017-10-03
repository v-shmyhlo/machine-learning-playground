import nn_framework.op as op
import numpy as np


class Add(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) + self.b.eval(feeds, cache)

  def _deriv(self, var, dself):
    da = self.a.deriv(var, dself)
    db = self.b.deriv(var, dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Sub(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) - self.b.eval(feeds, cache)

  def _deriv(self, var, dself):
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


class Mul(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) * self.b.eval(feeds, cache)

  def _deriv(self, var, dself):
    da = self.a.deriv(var, self.b * dself)
    db = self.b.deriv(var, self.a * dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Div(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) / self.b.eval(feeds, cache)

  def _deriv(self, var, dself):  # TODO: check this
    dself_da = self.b / self.b**2
    dself_db = -self.a / self.b**2

    da = self.a.deriv(var, dself * dself_da)
    db = self.b.deriv(var, dself * dself_db)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Pow(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache)**self.b.eval(feeds, cache)

  def _deriv(self, var, dself):
    da = self.a.deriv(var, dself * self * (self.b / self.a))
    db = self.b.deriv(var, dself * self * Log(self.a))

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Gt(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) > self.b.eval(feeds, cache)


class Ge(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) >= self.b.eval(feeds, cache)


class Eq(op.ElementwiseBinaryOp):
  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) == self.b.eval(feeds, cache)


class Matmul(op.ElementwiseBinaryOp):
  def __init__(self, a, b):
    a.shape.assert_has_rank(2)
    b.shape.assert_has_rank(2)

    self.a = a
    self.b = b
    self.shape = op.matmul_shape_broadcast(a.shape, b.shape)

  def _eval(self, feeds, cache):
    return self.a.eval(feeds, cache) @ self.b.eval(feeds, cache)

  def _deriv(self, var, dself):
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


class Log(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = self.x.shape
    self.dself_dx = 1 / self.x

  def _eval(self, feeds, cache):
    return np.log(self.x.eval(feeds, cache))

  def _deriv(self, var, dself):
    return self.x.deriv(var, dself * self.dself_dx)


class Neg(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = self.x.shape

  def _eval(self, feeds, cache):
    return -self.x.eval(feeds, cache)

  def _deriv(self, var, dself):
    return self.x.deriv(var, -dself)


class Const(op.Op):
  def __init__(self, value):
    self.value = np.array(value)
    self.shape = op.make_shape(self.value.shape)

  def _eval(self, feeds, cache):
    return self.value

  def _deriv(self, var, dself):
    return None

  def variables(self):
    return set()

  def __str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.value)


class Zeros(op.Op):
  def __init__(self, shape):
    self.shape = op.make_shape(shape)

  def _eval(self, feeds, cache):
    return np.zeros(self.shape.eval(feeds, cache))


class RandomNormal(Const):
  def __init__(self, shape):
    self.shape = op.make_shape(shape)

  def _eval(self, feeds, cache):
    shape = self.shape.eval(feeds, cache)
    res = np.random.randn(*shape)
    assert res.shape == shape
    return res


class RandomUniform(Const):
  def __init__(self, shape):
    self.shape = op.make_shape(shape)

  def _eval(self, feeds, cache):
    shape = self.shape.eval(feeds, cache)
    res = np.random.rand(*shape)
    assert res.shape == shape
    return res


class Transpose(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = op.transpose_shape_broadcast(x.shape)

  def _eval(self, feeds, cache):
    return self.x.eval(feeds, cache).T

  def _deriv(self, var, dself):
    return self.x.deriv(var, dself.t())


class Variable(op.Op):
  def __init__(self, init_value):
    self.init_value = init_value
    self.shape = self.init_value.shape
    self.value = None

  def _eval(self, feeds, cache):
    if self.value is None:
      self.value = self.init_value.eval(feeds, cache)

    return self.value

  def _deriv(self, var, dself):
    return None

  def assign(self, exp):
    return Assign(self, exp)

  def variables(self):
    return set([self])

  def __str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.value)


class Placeholder(op.Op):
  def __init__(self, name, shape):
    self.name = name
    self.shape = op.make_dynamic_shape(shape, self)

  def _eval(self, feeds, cache):
    return np.array(feeds[self])

  def _deriv(self, var, dself):
    return None

  def variables(self):
    return set()

  def __str__(self):
    return '%s(%s, shape=%s)' % (self.__class__.__name__, self.name,
                                 self.shape)


class Ones(op.Op):
  def __init__(self, shape):
    self.shape = shape

  def _eval(self, feeds, cache):
    return np.ones(self.shape.eval(feeds, cache))


class Assign(op.Op):
  def __init__(self, var, exp):
    self.var = var
    self.exp = exp

  def _eval(self, feeds, cache):
    value = self.exp.eval(feeds, cache)
    assert self.var.value.shape == value.shape
    self.var.value = value
    return self.var.value


class Group(op.Op):
  def __init__(self, expressions):
    self.expressions = expressions

  def _eval(self, feeds, cache):
    return [exp.eval(feeds, cache) for exp in self.expressions]


class Sum0(op.UnaryOp):
  def __init__(self, x):
    x.shape.assert_has_rank(2)

    self.x = x
    self.shape = op.make_shape((1, self.x.shape[1]))
    self.dself_dx = Ones(self.x.shape)

  def _eval(self, feeds, cache):
    return np.sum(self.x.eval(feeds, cache), axis=0, keepdims=True)

  def _deriv(self, var, dself):
    return self.x.deriv(var, dself * self.dself_dx)


class Sqrt(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = self.x.shape

  def _eval(self, feeds, cache):
    return np.sqrt(self.x.eval(feeds, cache))


class Sigmoid(op.UnaryOp):
  def __init__(self, x):
    self.x = x
    self.shape = x.shape
    self.dself_dx = self * (1 - self)

  def _eval(self, feeds, cache):
    return 1 / (1 + np.exp(-self.x.eval(feeds, cache)))

  def _deriv(self, var, dself):
    return self.x.deriv(var, dself * self.dself_dx)


class LogisticLoss(op.Op):
  def __init__(self, _sentinel=None, a=None, y=None):
    a.shape.assert_has_rank(2)
    y.shape.assert_has_rank(2)

    self.a, self.y, self.shape = op.elementwise_broadcast(a, y)
    self.dself_da = -(self.y / self.a) + (1 - self.y) / (1 - self.a)

  def variables(self):
    return self.a.variables().union(self.y.variables())

  def _eval(self, feeds, cache):
    a = self.a.eval(feeds, cache)
    y = self.y.eval(feeds, cache)
    return -np.log(a) * y + -np.log(1 - a) * (1 - y)

  def _deriv(self, var, dself):
    # TODO: this is not full implementation, it ignores derivatives w.r.t. y
    return self.a.deriv(var, dself * self.dself_da)


class SigmoidLogisticLoss(op.Op):
  def __init__(self, _sentinel=None, z=None, y=None):
    z.shape.assert_has_rank(2)
    y.shape.assert_has_rank(2)

    self.z, self.y, self.shape = op.elementwise_broadcast(z, y)
    a = Sigmoid(self.z)
    self.dself_dz = a - self.y
    self.eval_op = LogisticLoss(a=a, y=self.y)

  def variables(self):
    return self.z.variables().union(self.y.variables())

  def _eval(self, feeds, cache):
    return self.eval_op.eval(feeds, cache)

  def _deriv(self, var, dself):
    return self.z.deriv(var, dself * self.dself_dz)


class Maximum(op.ElementwiseBinaryOp):
  def __init__(self, a, b):
    super().__init__(a, b)
    self.dself_da = self.a > self.b
    self.dself_db = self.b >= self.a  # TODO: gt or ge?

  def _eval(self, feeds, cache):
    a = self.a.eval(feeds, cache)
    b = self.b.eval(feeds, cache)
    return np.maximum(a, b)

  def _deriv(self, var, dself):
    da = self.a.deriv(var, self.dself_da * dself)
    db = self.b.deriv(var, self.dself_db * dself)

    if da is None:
      return db
    elif db is None:
      return da
    else:
      return da + db


class Mean(op.UnaryOp):
  def __init__(self, x):
    x.shape.assert_has_rank(2)
    self.x = x
    self.shape = op.make_shape(())
    self.dself_dx = Ones(self.x.shape) / (self.x.shape[0] * self.x.shape[1])

  def _eval(self, feeds, cache):
    return np.mean(self.x.eval(feeds, cache))

  def _deriv(self, var, dself):
    return self.x.deriv(var, dself * self.dself_dx)


class Dropout(op.UnaryOp):
  def __init__(self, x, keep_prob):
    self.x = x
    self.keep_prob = keep_prob
    self.keep = RandomUniform(x.shape) < self.keep_prob
    self.eval_op = (self.x * self.keep) / self.keep_prob
    self.shape = self.eval_op.shape

  def _eval(self, feeds, cache):
    return self.eval_op.eval(feeds, cache)

  def _deriv(self, var, dself):
    return self.x.deriv(var, (dself * self.keep) / self.keep_prob)


class SoftmaxCrossEntropy(op.Op):
  def __init__(self, z, y):
    self.z = z
    self.y = y

    exp = np.e**z
    a = exp / Sum0(exp)

    self.result = -Sum0(y * Log(a))
    self.shape = self.result.shape
    self.dself_dz = a - y

  def _eval(self, feeds, cache):
    return self.result.eval(feeds, cache)

  def _deriv(self, var, dself):
    return self.z.deriv(var, dself * self.dself_dz)

  def variables(self):
    return self.result.variables()
