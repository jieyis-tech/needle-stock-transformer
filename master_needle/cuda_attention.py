"""NVRTC-compiled project kernels, launched on CuPy's current CUDA stream."""
from pathlib import Path
import math
import numpy as np
from .backend import cupy_module

_kernels = None


def kernels():
    global _kernels
    if _kernels is None:
        cp = cupy_module()
        source = (Path(__file__).parent/'kernels'/'attention.cu').read_text()
        module = cp.RawModule(code=source,options=('--std=c++11',),
                              name_expressions=('attn_forward','attn_dq','attn_dkv'))
        _kernels = [module.get_function(n) for n in ('attn_forward','attn_dq','attn_dkv')]
    return _kernels


def forward(q,k,v,block,scale):
    cp = cupy_module()
    q,k,v = [cp.ascontiguousarray(a) for a in (q,k,v)]
    batch = math.prod(q.shape[:-2])
    nq,nk,d,dv = q.shape[-2],k.shape[-2],q.shape[-1],v.shape[-1]
    out = cp.empty(q.shape[:-1]+(dv,),dtype=cp.float32)
    lse = cp.empty((batch,nq,1),dtype=cp.float32)
    args=(q,k,v,out,lse,np.int32(nq),np.int32(nk),np.int32(d),np.int32(dv),np.float32(scale))
    kernels()[0]((nq,batch),(block,),args)
    return out,lse


def backward(q,k,v,go,out,lse,block,scale):
    cp = cupy_module()
    q,k,v,go,out = [cp.ascontiguousarray(a) for a in (q,k,v,go,out)]
    batch=math.prod(q.shape[:-2])
    nq,nk,d,dv=q.shape[-2],k.shape[-2],q.shape[-1],v.shape[-1]
    delta=(go*out).sum(-1)
    dq,dk,dvout=[cp.empty_like(a) for a in (q,k,v)]
    dims=(np.int32(nq),np.int32(nk),np.int32(d),np.int32(dv),np.float32(scale))
    kernels()[1]((nq,batch),(block,),(q,k,v,go,lse,delta,dq)+dims)
    kernels()[2]((nk,batch),(block,),(q,k,v,go,lse,delta,dk,dvout)+dims)
    return dq,dk,dvout
