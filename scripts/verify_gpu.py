"""GPU checks in a dedicated process; PyTorch is an independent CPU oracle."""
from pathlib import Path
import os
os.environ['MASTER_NEEDLE_BACKEND']='cuda'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OMP_NUM_THREADS']='1'
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ['CUPY_TF32']='0'
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
import json
import time
import numpy as np
import master_needle
import needle as ndl
from master_needle import backend
from master_needle.attention import attention,dense_vjp_attention
from master_needle.model import MASTER,named_parameters
from test_master import torch_master,torch


def main():
    cp=backend.cupy_module()
    rng=np.random.default_rng(61)
    metrics=[]
    start=time.perf_counter()
    for shape,nk,dv in [((2,3,7,4),9,5),((2,2,65,16),97,16),((1,1,1,1),1,1)]:
        for block in (32,64,128):
            arrays=[rng.normal(size=shape).astype('float32'),rng.normal(size=shape[:-2]+(nk,shape[-1])).astype('float32'),rng.normal(size=shape[:-2]+(nk,dv)).astype('float32')]
            grad=rng.normal(size=shape[:-1]+(dv,)).astype('float32')
            nodes=[ndl.Tensor(a) for a in arrays]
            before=backend.HOST_READS
            out=attention(*nodes,block_size=block)
            out.backward(ndl.Tensor(grad,requires_grad=False))
            assert backend.HOST_READS==before,'hidden CPU transfer in forward/backward'
            assert isinstance(out.realize_cached_data(),cp.ndarray) and out.op.cuda_impl
            refs=[torch.tensor(a,requires_grad=True) for a in arrays]
            expected=torch.softmax(refs[0]@refs[1].transpose(-1,-2)*shape[-1]**-.5,-1)@refs[2]
            expected.backward(torch.tensor(grad))
            err=float(np.max(np.abs(out.numpy()-expected.detach().numpy())))
            ge=max(float(np.max(np.abs(a.grad.numpy()-r.grad.numpy()))) for a,r in zip(nodes,refs))
            np.testing.assert_allclose(out.numpy(),expected.detach().numpy(),atol=2e-5,rtol=2e-4)
            for a,r in zip(nodes,refs):np.testing.assert_allclose(a.grad.numpy(),r.grad.numpy(),atol=3e-5,rtol=3e-4)
            metrics.append(dict(shape=shape,nk=nk,block=block,forward_max_abs=err,gradient_max_abs=ge))
    model_metrics=[]
    for mode in ('dense','dense_vjp','tiled'):
        np.random.seed(63)
        m=MASTER(6,market_dim=4,dim=8,t_heads=2,s_heads=2,mode=mode,block_size=32,dtype='float64')
        xa,ma=rng.normal(size=(11,5,6)),rng.normal(size=4)
        x,market=ndl.Tensor(xa),ndl.Tensor(ma)
        before=backend.HOST_READS
        y=m(x,market)
        y.sum().backward()
        assert backend.HOST_READS==before
        tx,tm=torch.tensor(xa,requires_grad=True),torch.tensor(ma,requires_grad=True)
        expected,params=torch_master(tx,tm,m)
        expected.sum().backward()
        np.testing.assert_allclose(y.numpy(),expected.detach().numpy(),atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(x.grad.numpy(),tx.grad.numpy(),atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(market.grad.numpy(),tm.grad.numpy(),atol=1e-8,rtol=1e-8)
        ge=0
        for name,p in named_parameters(m).items():
            a,b=p.grad.numpy(),params[name].grad.numpy()
            np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8,err_msg=name)
            ge=max(ge,float(np.max(np.abs(a-b))))
        model_metrics.append(dict(mode=mode,dtype='float64',forward_max_abs=float(np.max(np.abs(y.numpy()-expected.detach().numpy()))),parameter_gradient_max_abs=ge,hidden_tensor_host_reads=0))
    # FP32 full-model kernel comparison uses the same GPU math for the rest of
    # the model; this isolates attention from CPU/GPU LayerNorm rounding.
    fp32=[]
    xa,ma=rng.normal(size=(11,5,6)).astype('float32'),rng.normal(size=4).astype('float32')
    for mode in ('dense_vjp','tiled'):
        np.random.seed(63)
        m=MASTER(6,market_dim=4,dim=8,t_heads=2,s_heads=2,mode=mode,block_size=32)
        x,market=ndl.Tensor(xa),ndl.Tensor(ma)
        y=m(x,market);y.sum().backward()
        fp32.append((y.numpy(),[p.grad.numpy() for p in m.parameters()]))
    np.testing.assert_allclose(fp32[0][0],fp32[1][0],atol=5e-5,rtol=5e-4)
    fp32_ge=0;fp32_rel=0
    for a,b in zip(fp32[0][1],fp32[1][1]):
        # Normwise comparison avoids dividing by individual gradients near zero.
        rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-8))
        assert rel<3e-4 and float(np.max(np.abs(a-b)))<2e-3,(rel,float(np.max(np.abs(a-b))))
        fp32_rel=max(fp32_rel,rel)
        fp32_ge=max(fp32_ge,float(np.max(np.abs(a-b))))
    # Regression: constant position/input buffers must not retain earlier graphs.
    np.random.seed(69)
    m=MASTER(6,market_dim=4,dim=8,t_heads=2,block_size=32)
    opt=ndl.optim.Adam(m.parameters(),lr=.001)
    x=ndl.Tensor(rng.normal(size=(32,5,6)).astype('float32'),requires_grad=False)
    market=ndl.Tensor(np.zeros(4,dtype='float32'),requires_grad=False)
    live=[]
    for step in range(20):
        opt.reset_grad();out=m(x,market);loss=(out*out).sum()/32
        loss.backward();opt.step()
        assert getattr(m.position,'grad',None) is None
        assert getattr(x,'grad',None) is None
        del out,loss
        if step in (4,19):
            opt.reset_grad();cp.cuda.get_current_stream().synchronize()
            live.append(cp.get_default_memory_pool().used_bytes())
    assert live[1]-live[0]<2**20,live
    report=dict(success=True,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                attention=metrics,full_master=model_metrics,fp32_model_dense_vs_kernel_max_gradient_abs=fp32_ge,fp32_model_max_relative_l2_gradient_error=fp32_rel,
                constant_gradient_retention_check=dict(passed=True,live_pool_bytes_after_5_and_20_steps=live),seconds=time.perf_counter()-start,
                scope='Needle CUDA/CuPy array backend and project-owned NVRTC attention kernels; not HW3 binary backend')
    (ROOT/'results'/'gpu_correctness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
