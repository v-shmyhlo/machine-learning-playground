class Session(object):
  def run(self, expressions, feeds={}):
    cache = {}
    return [exp.eval(feeds, cache) for exp in expressions]
