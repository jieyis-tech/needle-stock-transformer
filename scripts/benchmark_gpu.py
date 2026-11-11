"""Full MASTER and attention-only GPU tradeoffs, fresh process per config.

Track CuPy pool live allocation peak (not total device memory). Excludes CUDA
context and library allocations outside this allocator. CUDA events AND wall
times recorded after two warmups, 5 independent timed iterations by default.
"""
from pathlib import Path
import sys
import os
os.environ['MASTER_NEEDLE_BACKEND']='cuda'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OMP_NUM_THREADS']='1'
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ['CUPY_TF32']='0'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import gc
import json
import subprocess
import time
import hashlib
from datetime import datetime,timezone
import numpy as np
import master_needle
import needle as ndl
from master_needle.backend import cupy_module
from master_needle.model import MASTER
from master_needle.attention import attention,dense_attention,dense_vjp_attention


def worker(args):
    cp=cupy_module();pool=cp.get_default_memory_pool()
    np.random.seed(101);rng=np.random.default_rng(101)
    n,t,d=args.stocks,args.steps,args.dim
    if args.task=='model':
        m=MASTER(10,dim=d,mode=args.mode,block_size=args.block)
        x=ndl.Tensor(rng.normal(size=(n,t,10)).astype('float32'),requires_grad=False)
        market=ndl.Tensor(rng.normal(size=16).astype('float32'),requires_grad=False)
        opt=ndl.optim.Adam(m.parameters(),lr=.001)
        def step():
            opt.reset_grad();x.grad=None;market.grad=None
            score=m(x,market);loss=(score*score).sum()/n
            loss.backward();opt.step()
            return loss
    else:
        q,k,v=[ndl.Tensor(rng.normal(size=(t,2,n,d//2)).astype('float32')) for _ in range(3)]
        def step():
            for a in (q,k,v):a.grad=None
            out=attention(q,k,v,args.block) if args.mode=='tiled' else (dense_vjp_attention(q,k,v) if args.mode=='dense_vjp' else dense_attention(q,k,v))
            loss=out.sum()/(t*n*d);loss.backward()
            return loss
    for _ in range(2):
        loss=step();cp.cuda.get_current_stream().synchronize();del loss
    if args.task=='model':
        opt.reset_grad();x.grad=None;market.grad=None
    else:
        for a in (q,k,v):a.grad=None
    gc.collect();pool.free_all_blocks()
    baseline=pool.used_bytes();peak=[baseline]
    def tracked_malloc(size):
        ptr=pool.malloc(size);peak[0]=max(peak[0],pool.used_bytes())
        return ptr
    cp.cuda.set_allocator(tracked_malloc)
    gpu_ms=[];wall_ms=[]
    for _ in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        ev0,ev1=cp.cuda.Event(),cp.cuda.Event()
        start=time.perf_counter();ev0.record();loss=step();ev1.record();ev1.synchronize()
        wall_ms.append((time.perf_counter()-start)*1000)
        gpu_ms.append(cp.cuda.get_elapsed_time(ev0,ev1));del loss
    cp.cuda.set_allocator(pool.malloc)
    return dict(task=args.task,mode=args.mode,stocks=n,time_steps=t,dim=d,block=args.block,dtype='float32',
                seed=101,warmups=2,repeats=args.repeats,wall_ms=wall_ms,cuda_event_ms=gpu_ms,
                wall_median_ms=float(np.median(wall_ms)),cuda_event_median_ms=float(np.median(gpu_ms)),
                pool_baseline_mib=baseline/2**20,pool_peak_live_mib=peak[0]/2**20,
                memory_scope='CuPy allocator live peak; excludes CUDA context and external library workspaces')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--task',default='model',choices=['model','attention'])
    p.add_argument('--mode',default='tiled',choices=['dense','dense_vjp','tiled']);p.add_argument('--stocks',type=int,default=128)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--dim',type=int,default=32);p.add_argument('--block',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.worker:
        print(json.dumps(worker(a)));return
    # All settings are predefined systems ablations, not stock-prediction tuning.
    configs=[]
    ns=(128,) if a.smoke else (128,512,1024)
    for n in ns:
        for mode in ('dense','dense_vjp','tiled'):
            configs.append(('model',mode,n,12,64))
    if not a.smoke:
        for block in (32,128):configs.append(('model','tiled',512,12,block))
        for steps in (4,24):
            for mode in ('dense_vjp','tiled'):configs.append(('model',mode,512,steps,64))
        for mode in ('dense_vjp','tiled'):configs.append(('attention',mode,1024,12,64))
    rows=[]
    for task,mode,n,t,block in configs:
        cmd=[sys.executable,__file__,'--worker','--task',task,'--mode',mode,'--stocks',str(n),'--steps',str(t),'--block',str(block),'--repeats',str(a.repeats)]
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        row=json.loads(result.stdout.strip().splitlines()[-1]);rows.append(row)
        print(f"{task} {mode} N={n} T={t} B={block}: {row['pool_peak_live_mib']:.1f} MiB, {row['wall_median_ms']:.2f} ms",flush=True)
    cp=cupy_module()
    report=dict(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,numpy=np.__version__,
                rows=rows,smoke=a.smoke,backend='Needle CUDA array adapter, CuPy BLAS, project attention CUDA kernels',
                project_root=str(ROOT),measured_at_utc=datetime.now(timezone.utc).isoformat(),
                cuda_source_sha256=hashlib.sha256((ROOT/'master_needle/kernels/attention.cu').read_bytes()).hexdigest(),
                backend_source_sha256=hashlib.sha256((ROOT/'master_needle/backend.py').read_bytes()).hexdigest())
    path=ROOT/'results'/('gpu_benchmark_smoke.json' if a.smoke else 'gpu_benchmark.json')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
