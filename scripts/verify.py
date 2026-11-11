"""Run numerical checks and write machine-readable evidence."""
from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
import hashlib
import json
import platform
import time
import unittest
from threadpoolctl import threadpool_limits
import test_master


def main():
    manifest = json.loads((ROOT/'vendor/PROVENANCE.json').read_text(encoding='utf-8'))
    for file,digest in manifest['files'].items():
        actual = hashlib.sha256((ROOT/'vendor/needle'/file).read_bytes()).hexdigest()
        assert actual == digest, 'Vendored source changed: '+file
    test_master.torch.set_num_threads(1)
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_master)
    with threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun,successful=result.wasSuccessful(),
                  failures=[str(x[0]) for x in result.failures],errors=[str(x[0]) for x in result.errors],
                  seconds=time.perf_counter()-start,metrics=test_master.METRICS,
                  python=platform.python_version(),numpy=test_master.np.__version__,torch=test_master.torch.__version__,
                  vendor_files_verified=len(manifest['files']),
                  scope='CPU float32/float64 attention and dropout-free MASTER; first-order derivatives')
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/'correctness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    main()
