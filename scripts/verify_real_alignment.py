"""Real-panel state/label timing audit and full-model FP64 oracle comparison."""
from pathlib import Path
import os
os.environ['MASTER_NEEDLE_BACKEND']='cuda'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OMP_NUM_THREADS']='1'
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import json
import hashlib
import numpy as np
import master_needle
import needle as ndl
from master_needle.model import MASTER,select,named_parameters
from test_master import torch,torch_master


def main():
    p=np.load(ROOT/'data/real_panel_v1/panel.npz')
    manifest=json.loads((ROOT/'data/real_panel_v1/manifest.json').read_text())
    h=hashlib.sha256()
    for name in ('X','market','y','score_mask','dates','label_dates','codes'):h.update(p[name].tobytes())
    assert h.hexdigest()==manifest['fingerprint_sha256']
    for earlier,later in [('train','validation'),('validation','test')]:
        assert max(p['label_dates'][p['split']==earlier])<min(p['dates'][p['split']==later])
    assert np.array_equal(p['score_mask'],p['X'][:,:,-1,-1].astype(bool))
    audit={}
    for name in ('train','validation','test'):
        mask=p['split']==name;s=p['score_mask'][mask];y=p['y'][mask]
        audit[name]=dict(scored=int(s.sum()),unlabelled_scored=int((s&~np.isfinite(y)).sum()))
        assert audit[name]['scored']==manifest['waterfall'][name]['score_stock_days']
    i=0;ids=np.where(p['score_mask'][i])[0];label=np.isfinite(p['y'][i,ids]);ix=np.where(label)[0]
    np.random.seed(42)
    m=MASTER(10,dim=32,block_size=64,dtype='float64')
    xa,ma=p['X'][i,ids].astype('float64'),p['market'][i].astype('float64')
    ya=p['y'][i,ids][label].astype('float64')
    scores=m(ndl.Tensor(xa),ndl.Tensor(ma));s=select(scores,ix);y=ndl.Tensor(ya,requires_grad=False)
    a=s-(s.sum()/len(ix)).broadcast_to(s.shape);b=y-(y.sum()/len(ix)).broadcast_to(y.shape)
    loss=-(a*b).sum()/(((a*a).sum()*(b*b).sum()+1e-12)**.5)+1.0
    loss.backward()
    ts,params=torch_master(torch.tensor(xa,requires_grad=True),torch.tensor(ma,requires_grad=True),m)
    ss=ts[label];yy=torch.tensor(ya);aa=ss-ss.mean();bb=yy-yy.mean()
    tloss=1-(aa*bb).sum()/torch.sqrt((aa*aa).sum()*(bb*bb).sum()+1e-12);tloss.backward()
    np.testing.assert_allclose(scores.numpy(),ts.detach().numpy(),atol=1e-8,rtol=1e-8)
    ge=0
    for name,param in named_parameters(m).items():
        actual,ref=param.grad.numpy(),params[name].grad.numpy()
        np.testing.assert_allclose(actual,ref,atol=1e-7,rtol=1e-7,err_msg=name)
        ge=max(ge,float(np.max(np.abs(actual-ref))))
    report=dict(passed=True,panel_sha256=h.hexdigest(),split_label_date_audit=audit,
                real_sample_date=str(p['dates'][0]),scored_stocks=len(ids),loss_label_stocks=len(ix),
                master_fp64_max_output_abs=float(np.max(np.abs(scores.numpy()-ts.detach().numpy()))),
                master_fp64_max_parameter_gradient_abs=ge,first_loss=float(loss.numpy()),
                limitations='FP64 full-model reference; FP32 CUDA kernel separately validated. Does not imply identical FP32 Adam trajectories.')
    (ROOT/'results/real_alignment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1);main()
