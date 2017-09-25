class Shape(object):
  def __getitem__(self, *keys):
    return GetItem(self, *keys)

class StaticShape(Shape):
  def __init__(self, shape):
    self.shape = shape

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.shape

    return cache[self]

class DynamicShape(Shape):
  def __init__(self, tensor):
    self.tensor = tensor

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.tensor.eval(feeds, cache).shape

    return cache[self]

class ElementwiseShapeBroadcast(Shape):
  def __init__(self, a, b):
    self.a = a
    self.b = b

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
        raise Exception('Can not broadcast %s and %s' % (a, b))

    return cache[self]

class GetItem(object):
  def __init__(self, value, *keys):
    self.value = value
    self.keys = keys
    self.shape = StaticShape(())

  def eval(self, feeds, cache):
    if not self in cache:
      cache[self] = self.value.eval(feeds, cache).__getitem__(*self.keys)

    return cache[self]
