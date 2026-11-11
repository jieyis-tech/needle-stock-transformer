"""Batched tiled array reference for CUDA FP64 and unusual head dimensions.

This is a BLAS/array implementation, distinct from the project's FP32 kernels.
All leading batches are processed together, avoiding Python loops per stock.
"""
import math
from .backend import array_module


def forward(q,k,v,block,scale):
    xp=array_module(q);batch=math.prod(q.shape[:-2]);nq=q.shape[-2];nk=k.shape[-2]
    qq,kk,vv=[a.reshape(batch,a.shape[-2],a.shape[-1]) for a in (q,k,v)]
    out=xp.empty((batch,nq,v.shape[-1]),dtype=q.dtype);lse=xp.empty((batch,nq,1),dtype=q.dtype)
    for i in range(0,nq,block):
        qi=qq[:,i:i+block];m=xp.full(qi.shape[:-1]+(1,),-xp.inf,dtype=q.dtype)
        denom=xp.zeros_like(m);acc=xp.zeros(qi.shape[:-1]+(v.shape[-1],),dtype=q.dtype)
        for j in range(0,nk,block):
            s=(qi@kk[:,j:j+block].swapaxes(-1,-2))*scale
            new_m=xp.maximum(m,s.max(-1,keepdims=True));alpha=xp.exp(m-new_m);p=xp.exp(s-new_m)
            acc=alpha*acc+p@vv[:,j:j+block];denom=alpha*denom+p.sum(-1,keepdims=True);m=new_m
        out[:,i:i+block]=acc/denom;lse[:,i:i+block]=m+xp.log(denom)
    return out.reshape(q.shape[:-1]+(v.shape[-1],)),lse


def backward(q,k,v,go,out,lse,block,scale):
    xp=array_module(q);batch=math.prod(q.shape[:-2]);nq=q.shape[-2];nk=k.shape[-2]
    qq,kk,vv,gg,oo=[a.reshape(batch,a.shape[-2],a.shape[-1]) for a in (q,k,v,go,out)]
    dq,dk,dv=[xp.zeros_like(a) for a in (qq,kk,vv)]
    for i in range(0,nq,block):
        qi,gi=qq[:,i:i+block],gg[:,i:i+block]
        delta=(gi*oo[:,i:i+block]).sum(-1,keepdims=True)
        for j in range(0,nk,block):
            kj,vj=kk[:,j:j+block],vv[:,j:j+block]
            p=xp.exp((qi@kj.swapaxes(-1,-2))*scale-lse[:,i:i+block])
            ds=p*(gi@vj.swapaxes(-1,-2)-delta)
            dq[:,i:i+block]+=(ds@kj)*scale
            dk[:,j:j+block]+=(ds.swapaxes(-1,-2)@qi)*scale
            dv[:,j:j+block]+=p.swapaxes(-1,-2)@gi
    return tuple(a.reshape(original.shape) for a,original in zip((dq,dk,dv),(q,k,v)))
