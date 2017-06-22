import numpy

from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check


class meProp(function.Function):

    """meProp backprop regularization."""

    def __init__(self, k):
        self.k = k

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1, in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs(())
        return x[0],

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        indices = numpy.argpartition(
                cuda.to_cpu(-xp.absolute(gy[0])), self.k, axis=1)[:, :self.k]
        self.mask = numpy.zeros_like(gy[0], dtype=numpy.float32)
        for i in range(self.mask.shape[0]):
            self.mask[i][indices[i]] = 1
        #self.mask.ravel()[numpy.ravel_multi_index(indices.T, self.mask.shape)] = 1
        if xp != numpy:
            self.mask = cuda.to_gpu(self.mask)
        return gy[0] * self.mask,


def meprop(x, k=10, **kwargs):
    """meprop(x, k=10)
    Args:
        x (~chainer.Variable): Input variable.
        k (int): Top k parameter
    Returns:
        ~chainer.Variable: Output variable.
    See the paper by X. Sun: `meProp: Sparsified Back Propagation for \
    Accelerated Deep Learning with Reduced Overfitting 
    <https://arxiv.org/abs/1706.06197>`_.
    """
    argument.assert_kwargs_empty(kwargs)

    if configuration.config.train:
        return meProp(k)(x)
    return x
