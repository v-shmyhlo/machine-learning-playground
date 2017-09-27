def make_shape(dims):
  return _Shape([make_dimension(dim) for dim in dims])


def make_dynamic_shape(dims, self):
  return _Shape(
      [make_dynamic_dimension(dim, self, i) for (i, dim) in enumerate(dims)])


def elementwise_shape_broadcast(a, b):
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
    assert self.rank == rank, "Shape %s must have rank %s"

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = tuple([dim.eval(feeds, cache) for dim in self.dims])

    return cache[self]

  def __str__(self):
    return 'Shape([%s])' % ', '.join([dim.__str__() for dim in self.dims])


# class StaticShape(Shape):
#   def __init__(self, dims):
#     self.dims = dims

#   def eval(self, feeds, cache):
#     if not self in cache:
#       cache[self] = self.dims

#     return cache[self]

# class DynamicShape(Shape):
#   def __init__(self, tensor):
#     self.tensor = tensor

#   def eval(self, feeds, cache):
#     if not self in cache:
#       cache[self] = self.tensor.eval(feeds, cache).shape

#     return cache[self]

# class ElementwiseShapeBroadcast(Shape):
#   def __init__(self, a, b):
#     self.a = a
#     self.b = b

#   def eval(self, feeds, cache):
#     if not self in cache:
#       a = self.a.eval(feeds, cache)
#       b = self.b.eval(feeds, cache)

#       if a == b:
#         cache[self] = a

#       if len(a) == 2 and len(b) == 2:
#         if a[0] == b[0]:
#           if a[1] == 1:
#             cache[self] = (a[0], b[1])
#           if b[1] == 1:
#             cache[self] = (a[0], a[1])
#         if a[1] == b[1]:
#           if a[0] == 1:
#             cache[self] = (b[0], a[1])
#           if b[0] == 1:
#             cache[self] = (a[0], a[1])

#       if len(a) == 2 and len(b) == 0:
#         cache[self] = a

#       if len(a) == 0 and len(b) == 2:
#         cache[self] = b

#       if not self in cache:
#         raise Exception('Can not broadcast %s and %s' % (a, b))

#     return cache[self]

# class MatmulShapeBroadcast(Shape):
#   def __init__(self, a, b):
#     a.assert_has_rank(2)
#     b.assert_has_rank(2)

#     self.a = a
#     self.b = b
#     super().__init__((a[0], b[1]))

#   def eval(self, feeds, cache):
#     if not self in cache:
#       a = self.a.eval(feeds, cache)
#       b = self.b.eval(feeds, cache)

#       if a[1] != b[0]:
#         raise Exception("Shape %s and %s dimensions do not agree" % (a, b))

#       cache[self] = (a[0], b[1])

#     return cache[self]

# class GetItem(object):
#   def __init__(self, value, *keys):
#     self.value = value
#     self.keys = keys
#     self.shape = make_shape(())

#   def eval(self, feeds, cache):
#     if not self in cache:
#       cache[self] = self.value.eval(feeds, cache).__getitem__(*self.keys)

#     return cache[self]


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


class _Dimension(object):
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


# class StaticDimension(Dimension):
#   def __init__(self, value):
#     self.value = value

# class DynamicDimension(Dimension):
#   # def __init__(self):

#   def eval(self, feeds, cache):
#     self.shape.eval(index)
#     # self.tensor.eval().shape[self.index]
