from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maximum = array_api.max(Z, axis=1, keepdims=True)
        shifted = Z - maximum
        return shifted - array_api.log(array_api.sum(array_api.exp(shifted), axis=1, keepdims=True))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        probabilities = exp(node)
        row_sum = summation(out_grad, axes=(1,))
        return out_grad - probabilities * broadcast_to(reshape(row_sum, (node.shape[0], 1)), node.shape)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            maximum = array_api.max(Z)
            return array_api.log(array_api.sum(array_api.exp(Z - maximum))) + maximum
        maximum = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted = Z - maximum
        result = array_api.log(array_api.sum(array_api.exp(shifted), axis=self.axes, keepdims=True)) + maximum
        return array_api.squeeze(result, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        shape = input_tensor.shape
        if self.axes is None:
            axes = tuple(range(len(shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes % len(shape),)
        else:
            axes = tuple(axis % len(shape) for axis in self.axes)
        expanded_shape = tuple(1 if axis in axes else dimension for axis, dimension in enumerate(shape))
        output = broadcast_to(reshape(node, expanded_shape), shape)
        gradient = broadcast_to(reshape(out_grad, expanded_shape), shape)
        return gradient * exp(input_tensor - output)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
