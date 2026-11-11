"""Bounded synthetic training check; no stock data or investment claim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import master_needle
import needle as ndl
from master_needle.model import MASTER


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    x = ndl.Tensor(rng.normal(size=(24,6,9)).astype('float32'),requires_grad=False)
    market = ndl.Tensor(rng.normal(size=16).astype('float32'),requires_grad=False)
    # Fixed cross-sectional toy target, used only to show differentiable learning.
    target = ndl.Tensor((x.numpy()[:,-1,0] + .3*x.numpy()[:,:,1].mean(1)).astype('float32'),requires_grad=False)
    model = MASTER(9,dim=16,block_size=8)
    optimizer = ndl.optim.Adam(model.parameters(),lr=0.003)
    losses = []
    start = time.perf_counter()
    for step in range(40):
        optimizer.reset_grad()
        delta = model(x,market) - target
        loss = (delta*delta).sum() / 24
        loss.backward()
        losses.append(float(loss.numpy()))
        if not all(np.isfinite(p.grad.numpy()).all() for p in model.parameters()):
            raise FloatingPointError('non-finite gradient')
        optimizer.step()
    delta = model(x,market) - target
    final = float(((delta*delta).sum()/24).numpy())
    result = dict(task='synthetic same-batch fit, not out-of-sample evidence', backend='Needle HW2 NumPy CPU',
                  stocks=24,time_steps=6,input_dim=9,dim=16,steps=40,seed=42,
                  initial_loss=losses[0],final_loss=final,seconds=time.perf_counter()-start,loss_curve=losses)
    assert final < losses[0]*0.5, result
    out = Path(__file__).resolve().parents[1]/'results'
    out.mkdir(exist_ok=True)
    (out/'smoke_train.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='loss_curve'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
