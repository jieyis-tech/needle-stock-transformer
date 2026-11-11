"""Fresh-process, single-BLAS-thread CPU benchmark; no PyTorch dependency.

Reports real process working-set peaks separately from unique forward-graph
array storage. Process peaks include Python/NumPy and are not GPU memory.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import gc
import json
import os
import subprocess
import time
import threading
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
import master_needle
import needle as ndl
from master_needle.attention import attention, dense_attention, dense_vjp_attention


def graph_arrays(output):
    visited, buffers = set(), {}
    pending = [output]
    def record(a):
        if isinstance(a,np.ndarray):
            while isinstance(a.base,np.ndarray):
                a = a.base
            buffers[id(a)] = a.nbytes
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        record(node.cached_data)
        if node.op is not None:
            record(getattr(node.op,'lse',None))
            record(getattr(node.op,'probabilities',None))
        pending.extend(node.inputs)
    return sum(buffers.values()),len(visited)


def worker(mode,n,block):
    rng = np.random.default_rng(42)
    q,k,v = [ndl.Tensor(rng.normal(size=(4,2,n,16)).astype('float32')) for _ in range(3)]
    grad = ndl.Tensor(rng.normal(size=(4,2,n,16)).astype('float32'),requires_grad=False)
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.001):
            peak[0] = max(peak[0],process.memory_info().rss)
    watcher = threading.Thread(target=sample,daemon=True)
    watcher.start()
    start = time.perf_counter()
    if mode=='tiled':
        out = attention(q,k,v,block_size=block)
    elif mode=='dense_vjp':
        out = dense_vjp_attention(q,k,v)
    else:
        out = dense_attention(q,k,v)
    forward = time.perf_counter()-start
    graph_bytes,nodes = graph_arrays(out)
    start = time.perf_counter()
    out.backward(grad)
    backward = time.perf_counter()-start
    stop.set()
    watcher.join()
    mem = process.memory_info()
    peak[0] = max(peak[0],mem.rss,getattr(mem,'peak_wset',0))
    assert all(np.isfinite(a.grad.numpy()).all() for a in (q,k,v))
    return dict(mode=mode,stocks=n,time_steps=4,heads=2,head_dim=16,block_size=block,
                dtype='float32',seed=42,blas_threads=1,
                forward_seconds=forward,backward_seconds=backward,
                forward_graph_array_mib=graph_bytes/2**20,forward_graph_nodes=nodes,
                baseline_rss_mib=baseline/2**20,peak_rss_total_mib=peak[0]/2**20,
                output_sum=float(out.numpy().sum()),q_grad_sum=float(q.grad.numpy().sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker',choices=['dense','dense_vjp','tiled'])
    p.add_argument('--stocks',type=int,default=512)
    p.add_argument('--block',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3)
    args = p.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            print(json.dumps(worker(args.worker,args.stocks,args.block)))
        return
    rows = []
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='SEQUENTIAL')
    for n in (256,512,1024):
        for mode in ('dense','dense_vjp','tiled'):
            for repeat in range(args.repeats):
                r = subprocess.run([sys.executable,__file__,'--worker',mode,'--stocks',str(n),'--block',str(args.block)],
                                   env=env,capture_output=True,text=True,check=True,timeout=60)
                row = json.loads(r.stdout.strip().splitlines()[-1])
                row['repeat'] = repeat
                rows.append(row)
            recent = rows[-args.repeats:]
            print(f"{mode:5s} N={n}: forward graph {recent[0]['forward_graph_array_mib']:.2f} MiB, "
                  f"peak RSS median {np.median([x['peak_rss_total_mib'] for x in recent]):.1f} MiB, "
                  f"fwd+bwd median {np.median([x['forward_seconds']+x['backward_seconds'] for x in recent]):.3f}s",flush=True)
    # Provenance and all individual runs, rather than a best-run comparison.
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    report = dict(backend='Needle HW2 NumPy CPU',python=sys.version,numpy=np.__version__,
                  methodology='fresh child per measurement, no warmup, 3 independent repeats by default; medians; Windows working set',
                  rows=rows)
    (out/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
