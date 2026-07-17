$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $ProjectRoot
& "$ProjectRoot\.venv-edge\Scripts\python.exe" -m edge_runtime.main --config-dir edge_runtime/config
