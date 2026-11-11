$ErrorActionPreference = 'Stop'
$projectPython = 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe'
Push-Location $PSScriptRoot
try {
    & $projectPython -X utf8 scripts/verify_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU correctness failed' }
    & $projectPython -X utf8 scripts/verify_real_alignment.py
    if ($LASTEXITCODE -ne 0) { throw 'Real-panel alignment failed' }
    & $projectPython -X utf8 scripts/check_training_numerics.py
    if ($LASTEXITCODE -ne 0) { throw 'Training precision diagnostic failed' }
} finally { Pop-Location }
