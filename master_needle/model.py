"""Small MASTER architecture: market gate, T/S attention, temporal pooling.

Mirrors the structure of the user's existing MASTER, not its fitted weights or
data. Attention dropout is fixed to zero in this first implementation. Temporal
attention uses scale=1, cross-stock uses 1/sqrt(head_dim), matching that model.
"""
import math
import numpy as np
import needle as ndl
from needle import nn, ops, Tensor
from needle.autograd import TensorOp
from .attention import attention, dense_attention, dense_vjp_attention
from .backend import array_module


class Select(TensorOp):
    def __init__(self, index):
        self.index = index

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        data = out_grad.realize_cached_data()
        grad = array_module(data).zeros(node.inputs[0].shape, dtype=out_grad.dtype)
        grad[self.index] = data
        return Tensor.make_const(grad)


def select(x, index):
    return Select(index)(x)


def softmax(x, axis=-1):
    shape = list(x.shape)
    shape[axis] = 1
    lse = ops.logsumexp(x, axes=(axis,)).reshape(tuple(shape))
    return ops.exp(x - lse.broadcast_to(x.shape))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype="float32"):
        super().__init__()
        self.weight = nn.Parameter(np.ones(dim, dtype=dtype))
        self.bias = nn.Parameter(np.zeros(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        shape = x.shape[:-1] + (1,)
        mu = (x.sum(axes=(-1,)) / x.shape[-1]).reshape(shape).broadcast_to(x.shape)
        centered = x - mu
        var = ((centered * centered).sum(axes=(-1,)) / x.shape[-1]).reshape(shape)
        y = centered / ((var + self.eps) ** 0.5).broadcast_to(x.shape)
        return y * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, across_stocks=False, mode="tiled", block_size=64, dtype="float32"):
        super().__init__()
        if dim % heads:
            raise ValueError("model dimension must be divisible by heads")
        if mode not in ("dense", "dense_vjp", "tiled"):
            raise ValueError("mode must be dense, dense_vjp or tiled")
        self.dim, self.heads, self.across = dim, heads, across_stocks
        self.mode, self.block_size = mode, block_size
        self.qtrans, self.ktrans, self.vtrans = [nn.Linear(dim, dim, bias=False, dtype=dtype) for _ in range(3)]
        self.norm1, self.norm2 = LayerNorm(dim, dtype=dtype), LayerNorm(dim, dtype=dtype)
        self.ff1, self.ff2 = nn.Linear(dim, dim, dtype=dtype), nn.Linear(dim, dim, dtype=dtype)

    def forward(self, x):
        x = self.norm1(x)
        seq = x.transpose((0, 1)) if self.across else x
        b, n, d = seq.shape
        q, k, v = [f(seq).reshape((b, n, self.heads, d // self.heads)).transpose((1, 2)) for f in (self.qtrans, self.ktrans, self.vtrans)]
        scale = (d // self.heads) ** -0.5 if self.across else 1.0
        if self.mode == 'tiled':
            out = attention(q,k,v,self.block_size,scale)
        elif self.mode == 'dense_vjp':
            out = dense_vjp_attention(q,k,v,scale)
        else:
            out = dense_attention(q,k,v,scale)
        out = out.transpose((1, 2)).reshape((b, n, d))
        if self.across:
            out = out.transpose((0, 1))
        h = self.norm2(x + out)
        return h + self.ff2(ops.relu(self.ff1(h)))


class MASTER(nn.Module):
    def __init__(self, input_dim, market_dim=16, dim=32, t_heads=4, s_heads=2,
                 beta=5.0, max_len=64, mode="tiled", block_size=64, dtype="float32"):
        super().__init__()
        if beta <= 0 or dim % 2 or max_len < 1:
            raise ValueError("beta must be positive, dim even, max_len positive")
        self.input_dim, self.market_dim, self.dim = input_dim, market_dim, dim
        self.beta = beta
        self.gate = nn.Linear(market_dim, input_dim, dtype=dtype)
        self.embed = nn.Linear(input_dim, dim, dtype=dtype)
        self.temporal = AttentionBlock(dim, t_heads, False, mode, block_size, dtype)
        self.spatial = AttentionBlock(dim, s_heads, True, mode, block_size, dtype)
        self.pool = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.output = nn.Linear(dim, 1, dtype=dtype)
        pos = np.arange(max_len)[:, None]
        freq = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = np.zeros((max_len, dim), dtype=dtype)
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * freq), np.cos(pos * freq)
        self.position = Tensor(pe, requires_grad=False)

    def forward(self, x, market):
        if len(x.shape) != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("x must be (stocks, time, input_dim)")
        if market.shape != (self.market_dim,) or x.shape[1] > self.position.shape[0]:
            raise ValueError("Invalid market shape or sequence exceeds max_len")
        # HW2 MatMul VJP requires matrix operands, so keep the gate input 2-D.
        gate = softmax(self.gate(market.reshape((1, self.market_dim))) / self.beta).reshape((self.input_dim,)) * self.input_dim
        h = self.embed(x * gate.broadcast_to(x.shape))
        pe = select(self.position, (slice(0, x.shape[1]), slice(None)))
        h = self.spatial(self.temporal(h + pe.broadcast_to(h.shape)))
        z = self.pool(h)
        last = select(z, (slice(None), -1, slice(None))).reshape((x.shape[0], self.dim, 1))
        weights = softmax((z @ last).reshape(x.shape[:2]), axis=1).reshape((x.shape[0], x.shape[1], 1))
        pooled = (h * weights.broadcast_to(h.shape)).sum(axes=(1,))
        return self.output(pooled).reshape((x.shape[0],))


def named_parameters(module, prefix=""):
    """Stable names for independent reference tests and checkpoint serialization."""
    out = {}
    for name, value in module.__dict__.items():
        key = prefix + name
        if isinstance(value, nn.Parameter):
            out[key] = value
        elif isinstance(value, nn.Module):
            out.update(named_parameters(value, key + "."))
    return out
