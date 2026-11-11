$ErrorActionPreference = 'Stop'
$projectPython = 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '1'
Push-Location $PSScriptRoot
try {
    & $projectPython -X utf8 scripts/verify.py
    if ($LASTEXITCODE -ne 0) { throw 'Correctness checks failed' }
    & $projectPython -X utf8 scripts/smoke_train.py
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic smoke failed' }
    & $projectPython -X utf8 scripts/benchmark.py
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
    & $projectPython -X utf8 scripts/write_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
} finally {
    Pop-Location
}
