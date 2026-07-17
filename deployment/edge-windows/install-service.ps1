[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$WinSW = "$PSScriptRoot\WinSW-x64.exe"
)

$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this installer from an elevated PowerShell session."
}

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Venv = Join-Path $ProjectRoot ".venv-edge"
if (-not (Test-Path $WinSW -PathType Leaf)) {
    throw "WinSW-x64.exe is required at $WinSW. Download a pinned WinSW release and verify its checksum."
}

& $Python -3.10 -m venv $Venv
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r "$PSScriptRoot\requirements.txt"
& $VenvPython "$ProjectRoot\scripts\verify_pretrained_models.py" --require-detector

$Wrapper = Join-Path $PSScriptRoot "edge-runtime.exe"
Copy-Item $WinSW $Wrapper -Force
& $Wrapper install
& $Wrapper start
Write-Host "ConstructionSafetyEdge installed and started."
