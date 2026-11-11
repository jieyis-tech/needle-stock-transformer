"""Fixed small-panel MASTER experiment. No tuning and no post-2017 data.

All known T-day scoring stocks enter MASTER together. Missing future labels
mask only the loss/metrics, never the inter-stock-attention/scoring universe.
"""
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
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import master_needle
import needle as ndl
from master_needle.backend import cupy_module
from master_needle.model import MASTER,select,named_parameters


def correlation_loss(s,y):
    n=s.shape[0]
    a=s-(s.sum()/n).broadcast_to(s.shape)
    b=y-(y.sum()/n).broadcast_to(y.shape)
    corr=(a*b).sum()/(((a*a).sum()*(b*b).sum()+1e-12)**.5)
    return -corr+1.0


def rank_ic(a,b):
    a=pd.Series(a).rank(method='average').to_numpy();b=pd.Series(b).rank(method='average').to_numpy()
    if a.std()==0 or b.std()==0:return float('nan')
    return float(np.corrcoef(a,b)[0,1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    prefix='real_training_smoke_v' if a.smoke else 'real_training_v'
    version=1
    while (ROOT/'results'/f'{prefix}{version}').exists():version+=1
    out=ROOT/'results'/f'{prefix}{version}'
    out.mkdir(parents=True)
    paneldir=ROOT/'data'/'real_panel_v1'
    panel=np.load(paneldir/'panel.npz');manifest=json.loads((paneldir/'manifest.json').read_text())
    config=dict(seed=42,epochs=1 if a.smoke else 3,dim=32,block_size=64,lr=.001,clip_grad_norm=1.0,
                modes=['dense','dense_vjp','tiled'],dropout=0,loss='1-Pearson within each labelled cross-section',
                selection='fixed protocol; no epoch/model/hyperparameter selection from validation/test',
                panel_fingerprint=manifest['fingerprint_sha256'],panel_manifest=str(paneldir/'manifest.json'),
                smoke=a.smoke,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    X,M,Y,S=panel['X'],panel['market'],panel['y'],panel['score_mask']
    tr,va,te=[np.where(panel['split']==k)[0] for k in ('train','validation','test')]
    if a.smoke:tr,va,te=tr[:8],va[:3],te[:3]
    cp=cupy_module();GX,GM,GY=[cp.asarray(v) for v in (X,M,np.nan_to_num(Y))]
    ids={int(i):np.where(S[i])[0] for i in range(len(X))}
    valid={int(i):np.where(np.isfinite(Y[i,ids[i]]))[0] for i in range(len(X))}
    def inputs(i):
        return ndl.Tensor.make_const(GX[i,ids[i]]),ndl.Tensor.make_const(GM[i])
    results=[]
    for mode in config['modes']:
        np.random.seed(42);rng=np.random.default_rng(42)
        model=MASTER(X.shape[-1],market_dim=M.shape[-1],dim=32,mode=mode,block_size=64)
        opt=ndl.optim.Adam(model.parameters(),lr=config['lr'])
        curve=[];elapsed=[]
        start=time.perf_counter()
        for epoch in range(config['epochs']):
            losses=[];tic=time.perf_counter()
            for i in rng.permutation(tr):
                opt.reset_grad();x,m=inputs(int(i));scores=model(x,m)
                # Future availability is consulted here, AFTER all T-day stocks were scored.
                ix=valid[int(i)];sc=select(scores,ix);y=ndl.Tensor.make_const(GY[i,ids[int(i)][ix]])
                loss=correlation_loss(sc,y);loss.backward()
                params=model.parameters()
                norm=cp.sqrt(sum(cp.sum(p.grad.realize_cached_data()**2) for p in params))
                factor=cp.minimum(1.0,config['clip_grad_norm']/cp.maximum(norm,1e-12))
                for parameter in params:
                    parameter.grad.cached_data*=factor
                opt.step();losses.append(loss.realize_cached_data())
                del x,m,scores,sc,y,loss,params
            cp.cuda.get_current_stream().synchronize();elapsed.append(time.perf_counter()-tic)
            mean=float(cp.mean(cp.stack(losses)).get());curve.append(mean)
            if not np.isfinite(mean):raise FloatingPointError('non-finite training loss')
            print(f'{mode} epoch {epoch+1}: loss={mean:.6f}, {elapsed[-1]:.2f}s',flush=True)
        model.eval()
        scores_all=np.full_like(Y,np.nan)
        summary={}
        for name,days in [('validation',va),('test',te)]:
            daily=[];scored=labelled=missing=0
            for i in days:
                x,m=inputs(int(i));scores=model(x,m).numpy()
                assert len(scores)==len(ids[int(i)]) and np.isfinite(scores).all()
                scores_all[i,ids[int(i)]]=scores
                ok=np.isfinite(Y[i,ids[int(i)]])
                daily.append(rank_ic(scores[ok],Y[i,ids[int(i)][ok]]))
                scored+=len(scores);labelled+=int(ok.sum());missing+=int((~ok).sum())
            summary[name]=dict(mean_daily_rankic=float(np.nanmean(daily)),days=len(days),
                               scored_stock_days=scored,labelled_stock_days=labelled,unlabelled_but_scored=missing,daily_rankic=daily)
        np.savez_compressed(out/f'{mode}_scores.npz',scores=scores_all,dates=panel['dates'],codes=panel['codes'],score_mask=S,split=panel['split'])
        np.savez_compressed(out/f'{mode}_weights.npz',**{name:p.numpy() for name,p in named_parameters(model).items()})
        result=dict(mode=mode,train_days=len(tr),train_loss_curve=curve,epoch_seconds=elapsed,total_seconds=time.perf_counter()-start,metrics=summary)
        results.append(result)
        (out/'summary.json').write_text(json.dumps(dict(protocol=config,results=results),indent=2),encoding='utf-8')
        print(f"{mode}: validation RankIC={summary['validation']['mean_daily_rankic']:.5f}, test RankIC={summary['test']['mean_daily_rankic']:.5f}",flush=True)
    # Report actual differences; do not require GPU FP32 Adam trajectories to be bitwise identical.
    base=np.load(out/'dense_vjp_scores.npz')['scores']
    tile=np.load(out/'tiled_scores.npz')['scores']
    finite=np.isfinite(base)&np.isfinite(tile)
    diff=dict(max_abs=float(np.max(np.abs(base[finite]-tile[finite]))),rms=float(np.sqrt(np.mean((base[finite]-tile[finite])**2))))
    final=dict(protocol=config,results=results,dense_vjp_vs_tiled_scores=diff,
               limitations='Systems integration demonstration; fixed 128-stock historical cohort, daily-input variant, no costs/limits/ST filters or investment claim')
    (out/'summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print('Saved:',out,flush=True)


if __name__=='__main__':main()
