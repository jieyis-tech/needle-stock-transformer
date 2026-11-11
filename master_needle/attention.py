"""Exact tiled attention as a Needle TensorOp, including a recomputed backward.

NumPy CPU or Needle CUDA/CuPy array backend; no PyTorch computation. FP32 CUDA
uses project kernels for supported dimensions/blocks; other GPU shapes use a
CuPy tiled reference. First-order derivatives only, no grad-grad/masks/dropout.
GPU inputs must be prevalidated finite (MASTER_VALIDATE_INPUTS=1 checks on call).
"""
import math
import os
import numpy as np
from needle import Tensor, ops
from needle.autograd import TensorOp
from .backend import array_module


def dense_attention(q, k, v, scale=None):
    """Ordinary Needle graph baseline; same leading batch dimensions required."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = (q @ k.transpose()) * scale
    shape = scores.shape[:-1] + (1,)
    lse = ops.logsumexp(scores, axes=(-1,)).reshape(shape)
    return ops.exp(scores - lse.broadcast_to(scores.shape)) @ v


class TiledAttention(TensorOp):
    def __init__(self, block_size=64, scale=None):
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size, self.scale = block_size, scale
        self.max_score_elements = 0

    def compute(self, q, k, v):
        xp = array_module(q)
        if not all(isinstance(a, xp.ndarray) for a in (q,k,v)):
            raise TypeError("Q/K/V must share the active array backend")
        if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("Expected (..., queries/keys, features) arrays")
        if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
            raise ValueError("Batch dimensions must match exactly")
        if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
            raise ValueError("Incompatible Q/K/V shapes")
        if min(q.shape + k.shape + v.shape) <= 0:
            raise ValueError("Empty attention dimensions are unsupported")
        if q.dtype not in (np.float32, np.float64) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError("Q/K/V must share float32 or float64 dtype")
        if (xp is np or os.environ.get('MASTER_VALIDATE_INPUTS')=='1') and not all(xp.isfinite(a).all() for a in (q,k,v)):
            raise ValueError("Q/K/V must contain finite values")
        scale = q.shape[-1] ** -0.5 if self.scale is None else float(self.scale)
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        self.effective_scale = scale
        self.cuda_impl = (xp is not np and q.dtype == np.float32
                          and self.block_size in (32,64,128,256)
                          and max(q.shape[-1],v.shape[-1]) <= self.block_size)
        if self.cuda_impl:
            from .cuda_attention import forward
            out,self.lse = forward(q,k,v,self.block_size,scale)
            self.max_score_elements = self.block_size
            return out
        if xp is not np:
            from .array_attention import forward
            out,self.lse=forward(q,k,v,self.block_size,scale)
            self.max_score_elements=math.prod(q.shape[:-2])*min(q.shape[-2],self.block_size)*min(k.shape[-2],self.block_size)
            return out
        batch = math.prod(q.shape[:-2])
        nq, nk, d, dv = q.shape[-2], k.shape[-2], q.shape[-1], v.shape[-1]
        qq, kk, vv = (a.reshape(batch, a.shape[-2], a.shape[-1]) for a in (q, k, v))
        out = xp.empty((batch, nq, dv), dtype=q.dtype)
        self.lse = xp.empty((batch, nq, 1), dtype=q.dtype)
        block = self.block_size
        for b in range(batch):
            for i in range(0, nq, block):
                qi = qq[b, i:i + block]
                m = xp.full((len(qi), 1), -np.inf, dtype=q.dtype)
                denom = xp.zeros_like(m)
                acc = xp.zeros((len(qi), dv), dtype=q.dtype)
                for j in range(0, nk, block):
                    s = (qi @ kk[b, j:j + block].T) * scale
                    self.max_score_elements = max(self.max_score_elements, s.size)
                    new_m = xp.maximum(m, s.max(axis=-1, keepdims=True))
                    alpha = xp.exp(m - new_m)
                    p = xp.exp(s - new_m)
                    acc = alpha * acc + p @ vv[b, j:j + block]
                    denom = alpha * denom + p.sum(axis=-1, keepdims=True)
                    m = new_m
                out[b, i:i + block] = acc / denom
                self.lse[b, i:i + block] = m + xp.log(denom)
        return out.reshape(q.shape[:-1] + (dv,))

    def gradient(self, out_grad, node):
        # dS = P * (dO V^T - rowsum(dO * O)); recompute P tile-by-tile.
        # Only Q/K/V/O and row log-normalizers survive forward; never full P.
        q, k, v = (x.realize_cached_data() for x in node.inputs)
        go, out = out_grad.realize_cached_data(), node.realize_cached_data()
        xp = array_module(q)
        if self.cuda_impl:
            from .cuda_attention import backward
            arrays = backward(q,k,v,go,out,self.lse,self.block_size,self.effective_scale)
            return tuple(Tensor.make_const(a) for a in arrays)
        if xp is not np:
            from .array_attention import backward
            arrays=backward(q,k,v,go,out,self.lse,self.block_size,self.effective_scale)
            return tuple(Tensor.make_const(a) for a in arrays)
        batch = math.prod(q.shape[:-2])
        arrays = [a.reshape(batch, a.shape[-2], a.shape[-1]) for a in (q, k, v, go, out)]
        qq, kk, vv, gg, oo = arrays
        dq, dk, dv = xp.zeros_like(qq), xp.zeros_like(kk), xp.zeros_like(vv)
        block, scale = self.block_size, self.effective_scale
        for b in range(batch):
            for i in range(0, q.shape[-2], block):
                qi, gi = qq[b, i:i + block], gg[b, i:i + block]
                delta = (gi * oo[b, i:i + block]).sum(-1, keepdims=True)
                for j in range(0, k.shape[-2], block):
                    kj, vj = kk[b, j:j + block], vv[b, j:j + block]
                    p = xp.exp((qi @ kj.T) * scale - self.lse[b, i:i + block])
                    ds = p * (gi @ vj.T - delta)
                    dq[b, i:i + block] += (ds @ kj) * scale
                    dk[b, j:j + block] += (ds.T @ qi) * scale
                    dv[b, j:j + block] += p.T @ gi
        return tuple(Tensor.make_const(a.reshape(x.shape)) for a, x in zip((dq, dk, dv), node.inputs))


def attention(q, k, v, block_size=64, scale=None):
    return TiledAttention(block_size, scale)(q, k, v)


class DenseAttentionVJP(TensorOp):
    """Dense, first-order-only control to isolate tiling from graph overhead.

Same Q/K/V contract as TiledAttention; intended for trusted benchmark inputs.
Saves dense probabilities and computes the analytical VJP without constructing
an extra differentiable backward graph, just like the tiled implementation.
"""
    def __init__(self, scale=None):
        self.scale = scale

    def compute(self, q, k, v):
        self.effective_scale = q.shape[-1] ** -0.5 if self.scale is None else self.scale
        xp = array_module(q)
        scores = (q @ k.swapaxes(-1,-2)) * self.effective_scale
        scores -= scores.max(-1,keepdims=True)
        xp.exp(scores,out=scores)
        scores /= scores.sum(-1,keepdims=True)
        self.probabilities = scores
        return scores @ v

    def gradient(self, out_grad, node):
        q,k,v = (x.realize_cached_data() for x in node.inputs)
        go, out = out_grad.realize_cached_data(),node.realize_cached_data()
        p = self.probabilities
        ds = p * (go @ v.swapaxes(-1,-2) - (go*out).sum(-1,keepdims=True))
        arrays = ((ds @ k)*self.effective_scale,
                  (ds.swapaxes(-1,-2) @ q)*self.effective_scale,
                  p.swapaxes(-1,-2) @ go)
        return tuple(Tensor.make_const(a) for a in arrays)


def dense_vjp_attention(q,k,v,scale=None):
    return DenseAttentionVJP(scale)(q,k,v)

