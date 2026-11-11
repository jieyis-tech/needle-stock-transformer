"""One-click, bounded mentor demo. No downloads or original result overwrites.

The parent tries CUDA in a fresh child process; a missing CUDA runtime may fall
back to CPU, but a correctness failure never silently falls back. PyTorch is
used only as an independent attention oracle. Model execution is Needle.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]


def worker(backend):
    os.environ['MASTER_NEEDLE_BACKEND']=backend
    os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    os.environ['CUPY_TF32']='0'
    sys.path.insert(0,str(ROOT))
    try:
        import master_needle
        import needle as ndl
        import numpy as np
        from master_needle.attention import attention
        from master_needle.model import MASTER
        from master_needle.backend import cupy_module
        xp=cupy_module() if backend=='cuda' else np
    except Exception as exc:
        print('RESULT_JSON '+json.dumps(dict(status='backend_unavailable',backend=backend,error=f'{type(exc).__name__}: {exc}')))
        return 3
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    print('Backend: '+('Needle CUDA on '+xp.cuda.runtime.getDeviceProperties(0)['name'].decode() if backend=='cuda' else 'Needle NumPy CPU'),flush=True)
    start=time.perf_counter()
    rng=np.random.default_rng(42)
    arrays=[rng.normal(size=(2,2,17,8)).astype('float32') for _ in range(3)]
    g=rng.normal(size=(2,2,17,8)).astype('float32')
    nodes=[ndl.Tensor(a) for a in arrays]
    with threadpool_limits(limits=1):
        out=attention(*nodes,block_size=32)
        out.backward(ndl.Tensor(g,requires_grad=False))
        refs=[torch.tensor(a,requires_grad=True) for a in arrays]
        expected=torch.softmax((refs[0]@refs[1].transpose(-1,-2))*(8**-.5),-1)@refs[2]
        expected.backward(torch.tensor(g))
        forward_error=float(np.max(np.abs(out.numpy()-expected.detach().numpy())))
        grad_error=max(float(np.max(np.abs(n.grad.numpy()-r.grad.numpy()))) for n,r in zip(nodes,refs))
        np.testing.assert_allclose(out.numpy(),expected.detach().numpy(),atol=2e-5,rtol=2e-4)
        for n,r in zip(nodes,refs):np.testing.assert_allclose(n.grad.numpy(),r.grad.numpy(),atol=3e-5,rtol=3e-4)
        print(f'[PASS] Attention output: max absolute error = {forward_error:.2e}',flush=True)
        print(f'[PASS] Q/K/V gradients: max absolute error = {grad_error:.2e}',flush=True)

        panel_path=ROOT/'data/real_panel_v1/panel.npz'
        np.random.seed(42)
        model=MASTER(10,dim=16,block_size=32)
        if panel_path.exists():
            with np.load(panel_path) as p:
                ids=np.where(p['score_mask'][0])[0][:32]
                xx=p['X'][0,ids];mm=p['market'][0];date=str(p['dates'][0])
            scores=model(ndl.Tensor(xx,requires_grad=False),ndl.Tensor(mm,requires_grad=False)).numpy()
            source='real historical panel'
        else:
            xx=rng.normal(size=(32,12,10)).astype('float32');mm=np.zeros(16,dtype='float32');date=None
            scores=model(ndl.Tensor(xx,requires_grad=False),ndl.Tensor(mm,requires_grad=False)).numpy()
            source='synthetic inputs (real panel not present)'
        assert scores.shape==(32,) and np.isfinite(scores).all()
        print(f'[PASS] Model forward: 32 stocks x 12 days -> 32 scores ({source})',flush=True)

        # A synthetic same-batch fit only verifies that parameters can learn.
        np.random.seed(42);rng=np.random.default_rng(42)
        x_array=rng.normal(size=(24,6,9)).astype('float32')
        market_array=rng.normal(size=16).astype('float32')
        y_array=(x_array[:,-1,0]+.3*x_array[:,:,1].mean(1)).astype('float32')
        x=ndl.Tensor(x_array,requires_grad=False);market=ndl.Tensor(market_array,requires_grad=False)
        target=ndl.Tensor(y_array,requires_grad=False)
        train_model=MASTER(9,dim=16,block_size=32)
        optimizer=ndl.optim.Adam(train_model.parameters(),lr=.003)
        losses=[]
        for _ in range(40):
            optimizer.reset_grad()
            delta=train_model(x,market)-target;loss=(delta*delta).sum()/24
            losses.append(float(loss.numpy()));loss.backward()
            assert all(bool(xp.isfinite(p.grad.realize_cached_data()).all()) for p in train_model.parameters())
            optimizer.step()
        delta=train_model(x,market)-target;final=float(((delta*delta).sum()/24).numpy())
        assert np.isfinite(final) and final<losses[0]*.5,(losses[0],final)
        print(f'[PASS] Short synthetic training: loss {losses[0]:.4f} -> {final:.4f}',flush=True)
    result=dict(status='passed',backend=backend,seconds=time.perf_counter()-start,
                forward_max_abs=forward_error,gradient_max_abs=grad_error,
                model_input_source=source,real_signal_date=date,model_stocks=32,model_time_steps=12,
                synthetic_initial_loss=losses[0],synthetic_final_loss=final,synthetic_updates=40,
                scope='One small correctness case, one forward pass, synthetic same-batch learning. Not a stock-performance or full numerical-equivalence test.')
    print('RESULT_JSON '+json.dumps(result))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--no-pause',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args.backend)
    logdir=Path(__file__).parent/'demo_runs';logdir.mkdir(exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log=[]
    def say(s):print(s,flush=True);log.append(s)
    say('NEEDLE TRANSFORMER - SHORT CHECK-IN DEMO')
    say('Checks: attention output, backward gradients, model forward, short learning run.')
    say('First CUDA launch may take longer while compiling. No data download or full training.')
    choices=['cuda','cpu'] if args.backend=='auto' else [args.backend]
    result=None
    for backend in choices:
        say('\nStarting '+backend.upper()+' check...')
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
        proc=subprocess.run([sys.executable,'-X','utf8',__file__,'--worker','--backend',backend],
                            cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        result=None
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT_JSON '):result=json.loads(line[12:])
            else:say(line)
        if proc.stderr.strip():log.append(proc.stderr)
        if result and result['status']=='backend_unavailable' and args.backend=='auto' and backend=='cuda':
            say('CUDA is unavailable. Switching explicitly to CPU; this run will not claim a GPU test.')
            log.append(result['error']);continue
        if proc.returncode!=0:
            say('\nDEMO FAILED. See the log below; a correctness failure is never hidden by CPU fallback.')
            if proc.stderr.strip():say(proc.stderr.strip())
            result=result or dict(status='failed',backend=backend,error=proc.stderr[-2000:])
        break
    result=result or dict(status='failed',error='No result returned')
    report=logdir/f'demo_{stamp}.json';report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    say('\nALL CHECKS PASSED.' if result['status']=='passed' else '\nCHECKS DID NOT PASS.')
    say('Synthetic loss is a learning check, not evidence of stock prediction quality.')
    say('Result: '+str(report))
    (logdir/f'demo_{stamp}.txt').write_text('\n'.join(log)+'\n',encoding='utf-8')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'Unexpected demo error: {type(exc).__name__}: {exc}',flush=True)
        raise SystemExit(1)
