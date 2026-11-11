"""Train-only precision diagnostic; never selects parameters from test metrics."""
from pathlib import Path
import sys
import os
os.environ['MASTER_NEEDLE_BACKEND']='cuda'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import json
import numpy as np
import master_needle
import needle as ndl
from master_needle.backend import cupy_module
from master_needle.model import MASTER,select,named_parameters
from train_real_gpu import correlation_loss


def main():
    p=np.load(ROOT/'data/real_panel_v1/panel.npz');cp=cupy_module()
    days=np.random.default_rng(42).permutation(np.where(p['split']=='train')[0])[:16]
    results={}
    for dtype,modes in [('float32',['dense_vjp','tiled','tiled']),('float64',['dense_vjp','tiled'])]:
        runs=[]
        for mode in modes:
            np.random.seed(42);m=MASTER(10,dim=32,mode=mode,block_size=64,dtype=dtype)
            opt=ndl.optim.Adam(m.parameters(),lr=.001);losses=[];grads=[]
            for i in days:
                ids=np.where(p['score_mask'][i])[0];ix=np.where(np.isfinite(p['y'][i,ids]))[0]
                x=ndl.Tensor(p['X'][i,ids].astype(dtype),requires_grad=False)
                market=ndl.Tensor(p['market'][i].astype(dtype),requires_grad=False)
                y=ndl.Tensor(p['y'][i,ids][ix].astype(dtype),requires_grad=False)
                opt.reset_grad();score=m(x,market);loss=correlation_loss(select(score,ix),y);loss.backward()
                norm=cp.sqrt(sum(cp.sum(z.grad.realize_cached_data()**2) for z in m.parameters()))
                factor=cp.minimum(1.0,1.0/cp.maximum(norm,1e-12))
                for z in m.parameters():z.grad.cached_data*=factor
                if not losses:grads=[z.grad.numpy().copy() for z in m.parameters()]
                opt.step();losses.append(float(loss.numpy()))
            runs.append(dict(mode=mode,losses=losses,parameters=[z.numpy().copy() for z in m.parameters()],first_grads=grads))
        gd=max(float(np.max(np.abs(a-b))) for a,b in zip(runs[0]['first_grads'],runs[1]['first_grads']))
        pdiff=max(float(np.max(np.abs(a-b))) for a,b in zip(runs[0]['parameters'],runs[1]['parameters']))
        ld=max(abs(a-b) for a,b in zip(runs[0]['losses'],runs[1]['losses']))
        report=dict(first_step_max_gradient_abs=gd,after_16_steps_max_parameter_abs=pdiff,max_loss_difference=ld,
                    dense_loss_curve=runs[0]['losses'],tiled_loss_curve=runs[1]['losses'])
        if dtype=='float32':
            repeat=max(float(np.max(np.abs(a-b))) for a,b in zip(runs[1]['parameters'],runs[2]['parameters']))
            assert repeat==0,repeat
            report['same_kernel_exact_replay_max_parameter_abs']=repeat
        results[dtype]=report
    assert results['float64']['after_16_steps_max_parameter_abs']<1e-6,results
    (ROOT/'results/training_numerics.json').write_text(json.dumps(dict(train_days_only=[str(p['dates'][i]) for i in days],results=results),indent=2),encoding='utf-8')
    print(json.dumps({dtype:{k:v for k,v in r.items() if not k.endswith('curve')} for dtype,r in results.items()},indent=2))


if __name__=='__main__':main()
