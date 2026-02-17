import math
import numpy

def randn(*shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
    array = numpy.random.randn(*shape) * std + mean
    return numpy.array(array, dtype=dtype)

def init_He(in_features, out_features, dtype="float32"):
    std = numpy.sqrt(2.0 / (in_features + out_features))
    s = numpy.random.normal(0, std, (in_features, out_features))
    return numpy.array(s, dtype=dtype)

def init_Xavier(in_features, out_features, dtype="float32"):
    v = math.sqrt(2/(in_features+out_features))
    return randn(in_features, out_features, std=v, dtype=dtype)
