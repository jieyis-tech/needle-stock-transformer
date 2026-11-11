"""Optional process-wide CUDA array backend for the local HW2 Needle.

The vendored autograd remains Needle's. CuPy provides ndarray allocation, basic
array operations and BLAS; attention kernels are project CUDA C++ code. This is
a new backend adapter, not a claim to have completed HW3's C++ NDArray homework.
Select the backend before constructing any Tensor. No mixed-device execution.
"""
from pathlib import Path
import os
import sys
import ctypes
import hashlib
import subprocess
import numpy as np

_cp = None
HOST_READS = 0
_dll_handles = []


def cupy_module():
    global _cp
    if _cp is None:
        dep = Path(__file__).resolve().parents[1]/'.deps'/'cuda'
        if dep.exists():
            # Windows NVRTC cannot reliably resolve non-ASCII header paths.
            # Keep actual dependencies inside the requested project; use an
            # ASCII junction only as their compiler-facing path, never an old
            # project checkout. Do not overwrite an existing junction.
            if os.name=='nt' and not str(dep).isascii():
                key=hashlib.sha256(str(dep.resolve()).encode()).hexdigest()[:12]
                alias=Path.home()/'.cache'/'needle_master_cuda'/key
                alias.parent.mkdir(parents=True,exist_ok=True)
                if not alias.exists():
                    env=dict(os.environ,NM_CUDA_ALIAS=str(alias),NM_CUDA_TARGET=str(dep))
                    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
                        'New-Item -ItemType Junction -Path $env:NM_CUDA_ALIAS -Value $env:NM_CUDA_TARGET | Out-Null'],
                        env=env,check=True,capture_output=True,text=True)
                if alias.resolve()!=dep.resolve():
                    raise RuntimeError('CUDA dependency junction points outside this project')
                dep=alias
            sys.path.insert(0,str(dep))
            os.environ.setdefault('CUDA_PATH',str(dep/'nvidia'/'cuda_runtime'))
            os.environ.setdefault('CUPY_CACHE_DIR',str(dep/'kernel_cache'))
            for directory in (dep/'nvidia').glob('*/bin'):
                if os.name == 'nt':
                    _dll_handles.append(os.add_dll_directory(str(directory)))
            if os.name == 'nt':
                for pattern in ('cuda_runtime/bin/cudart64_*.dll','cuda_nvrtc/bin/nvrtc-builtins*.dll',
                                'cuda_nvrtc/bin/nvrtc64_120_0.dll','cublas/bin/cublasLt64_*.dll','cublas/bin/cublas64_*.dll'):
                    for dll in (dep/'nvidia').glob(pattern):
                        _dll_handles.append(ctypes.WinDLL(str(dll)))
        import cupy
        _cp = cupy
    return _cp


def array_module(a):
    return np if isinstance(a,np.ndarray) else cupy_module()


class CUDADevice:
    def __repr__(self): return 'needle.cuda_cupy(0)'
    def __eq__(self,other): return isinstance(other,CUDADevice)
    def enabled(self): return True
    def zeros(self,*shape,dtype='float32'): return cupy_module().zeros(shape,dtype=dtype)
    def ones(self,*shape,dtype='float32'): return cupy_module().ones(shape,dtype=dtype)
    def empty(self,shape,dtype='float32'): return cupy_module().empty(shape,dtype=dtype)
    def full(self,shape,fill_value,dtype='float32'): return cupy_module().full(shape,fill_value,dtype=dtype)
    # Deterministic initialization shares the CPU random stream, one initial H2D copy.
    def rand(self,*shape): return cupy_module().asarray(np.random.rand(*shape))
    def randn(self,*shape): return cupy_module().asarray(np.random.randn(*shape))


CUDA = CUDADevice()


def enable_cuda():
    """Install array-api/device hooks, keeping original source files untouched."""
    import needle as ndl
    from needle import autograd
    from needle.ops import ops_mathematic,ops_logarithmic
    cp = cupy_module()
    if autograd.array_api is cp:
        return cp
    if autograd.TENSOR_COUNTER:
        raise RuntimeError('Enable CUDA before creating Needle tensors; use a fresh process')
    cp.cuda.Device(0).use()
    cp.get_default_memory_pool().set_limit(fraction=0.65)
    # Check allocation and actual JIT execution before switching global backend.
    assert float(cp.sum(cp.arange(4,dtype=cp.float32)).get()) == 6.0
    autograd.array_api = cp
    ops_mathematic.array_api = cp
    ops_logarithmic.array_api = cp
    ndl.Tensor._array_from_numpy = staticmethod(lambda a,device,dtype: cp.array(a,dtype=dtype))
    ndl.Tensor.device = property(lambda self: CUDA)
    def to_numpy(self):
        global HOST_READS
        HOST_READS += 1
        return cp.asnumpy(self.realize_cached_data())
    ndl.Tensor.numpy = to_numpy
    ndl.cuda = lambda: CUDA
    # Device-local zeros_like/ones_like use Tensor.device; initial random values
    # may be generated on CPU, then uploaded once by the Tensor constructor.
    ndl.all_devices = lambda: [CUDA]
    def first_order_backward(output,out_grad):
        # HW2's teaching driver stores gradients on constants and keeps a new
        # differentiable backward graph. A persistent position buffer can then
        # retain previous training graphs. This backend is explicitly first-order:
        # detach each VJP, skip constants, and preserve shared-input accumulation.
        pending={output:[out_grad.detach()]}
        for node in reversed(autograd.find_topo_sort([output])):
            if node not in pending:
                continue
            node.grad=autograd.sum_node_list(pending.pop(node)).detach()
            if node.op is None:
                continue
            for inp,g in zip(node.inputs,node.op.gradient_as_tuple(node.grad,node)):
                if inp.requires_grad:
                    pending.setdefault(inp,[]).append(g.detach())
    autograd.compute_gradient_of_variables=first_order_backward
    return cp

