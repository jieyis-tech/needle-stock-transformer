"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for parameter in self.params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.data + self.weight_decay * parameter.data
            previous = self.u.get(parameter)
            if previous is None:
                previous = ndl.init.zeros_like(parameter)
            update = self.momentum * previous + (1 - self.momentum) * gradient
            self.u[parameter] = update.data
            parameter.data = (parameter.data - self.lr * update).data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for parameter in self.params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.data + self.weight_decay * parameter.data
            previous_m = self.m.get(parameter)
            previous_v = self.v.get(parameter)
            if previous_m is None:
                previous_m = ndl.init.zeros_like(parameter)
                previous_v = ndl.init.zeros_like(parameter)
            moment = self.beta1 * previous_m + (1 - self.beta1) * gradient
            variance = self.beta2 * previous_v + (1 - self.beta2) * (gradient ** 2)
            self.m[parameter] = moment.data
            self.v[parameter] = variance.data
            corrected_m = moment / (1 - self.beta1 ** self.t)
            corrected_v = variance / (1 - self.beta2 ** self.t)
            parameter.data = (parameter.data - self.lr * corrected_m / (corrected_v ** 0.5 + self.eps)).data
        ### END YOUR SOLUTION
